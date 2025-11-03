"""
Helper functions for saving and retrieving analysis results to/from database.
Replaces global variables in ML modules.
"""
from flask import g
from app.models import db, AnalysisResult, AnalysisPlot, Dataset
from datetime import datetime
import time
import json
import numpy as np


def make_json_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {make_json_serializable(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


class AnalysisSession:
    """
    Context manager for analysis operations.
    Replaces global variables like classification_Reports, accuracies, etc.
    """

    def __init__(self, analysis_type, dataset_id=None, parameters=None):
        self.analysis_type = analysis_type
        self.dataset_id = dataset_id or self._get_current_dataset_id()
        self.parameters = parameters or {}
        self.results = {}
        self.plots = []
        self.start_time = None
        self.analysis_result = None

    def _get_current_dataset_id(self):
        """Get current dataset ID from g.user."""
        dataset = g.user.get_current_dataset()
        if not dataset:
            raise ValueError("No current dataset found for user")
        return dataset.id

    def __enter__(self):
        """Start analysis session."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Save results to database when context exits."""
        execution_time = int((time.time() - self.start_time) * 1000)  # ms

        if exc_type is not None:
            # Error occurred
            self.analysis_result = AnalysisResult(
                user_id=g.user.id,
                dataset_id=self.dataset_id,
                analysis_type=self.analysis_type,
                status='failed',
                error_message=str(exc_val),
                execution_time_ms=execution_time
            )
            self.analysis_result.set_parameters(self.parameters)
            db.session.add(self.analysis_result)
            db.session.commit()
            return False  # Re-raise exception

        # Success - save results
        self.analysis_result = AnalysisResult(
            user_id=g.user.id,
            dataset_id=self.dataset_id,
            analysis_type=self.analysis_type,
            status='completed',
            execution_time_ms=execution_time
        )
        self.analysis_result.set_parameters(make_json_serializable(self.parameters))
        self.analysis_result.set_results(make_json_serializable(self.results))

        db.session.add(self.analysis_result)
        db.session.flush()  # Get ID before adding plots

        # Save plot references
        for plot_data in self.plots:
            plot = AnalysisPlot(
                analysis_result_id=self.analysis_result.id,
                plot_type=plot_data['type'],
                filename=plot_data['filename'],
                file_path=plot_data['path']
            )
            db.session.add(plot)

        db.session.commit()
        return True

    def add_metric(self, name, value):
        """Add a metric to results."""
        if 'metrics' not in self.results:
            self.results['metrics'] = {}
        self.results['metrics'][name] = value

    def add_classification_report(self, report_dict):
        """Add classification report."""
        self.results['classification_report'] = report_dict

    def add_confusion_matrix(self, matrix):
        """Add confusion matrix."""
        self.results['confusion_matrix'] = matrix.tolist() if hasattr(matrix, 'tolist') else matrix

    def add_data_table(self, name, data):
        """Add data table (e.g., PCA components, loadings)."""
        if 'tables' not in self.results:
            self.results['tables'] = {}
        # Convert DataFrame to dict if needed
        if hasattr(data, 'to_dict'):
            self.results['tables'][name] = data.to_dict()
        else:
            self.results['tables'][name] = data

    def add_plot(self, plot_type, filename, file_path):
        """Register a plot file."""
        self.plots.append({
            'type': plot_type,
            'filename': filename,
            'path': file_path
        })

    def get_result_id(self):
        """Get the database ID of saved result."""
        return self.analysis_result.id if self.analysis_result else None


# Convenience functions for specific analysis types

def save_classification_result(analysis_type, accuracy, report_dict, confusion_matrix,
                               test_size=None, k_folds=None, scale=False, encode=False):
    """
    Save classification analysis result.
    Replaces global variables in logistic.py, linear_svc.py, etc.
    """
    parameters = {
        'test_size': test_size,
        'k_folds': k_folds,
        'scaling': scale,
        'encoding': encode
    }

    with AnalysisSession(analysis_type, parameters=parameters) as session:
        session.add_metric('accuracy', accuracy)
        session.add_classification_report(report_dict)
        session.add_confusion_matrix(confusion_matrix)

        return session.get_result_id()


def save_pca_result(n_components, explained_variance, components_df, loadings_df, plot_paths):
    """
    Save PCA analysis result.
    """
    parameters = {'n_components': n_components}

    with AnalysisSession('pca', parameters=parameters) as session:
        session.add_metric('explained_variance', explained_variance.tolist())
        session.add_data_table('components', components_df)
        session.add_data_table('loadings', loadings_df)

        # Register plots
        for plot_name, plot_path in plot_paths.items():
            session.add_plot(plot_name, f"{plot_name}.png", plot_path)

        return session.get_result_id()


def save_regression_result(analysis_type, metrics_dict, coefficients=None, residuals=None, plot_paths=None):
    """
    Save regression analysis result (PCR, MLR, etc.).
    """
    parameters = metrics_dict.get('parameters', {})

    with AnalysisSession(analysis_type, parameters=parameters) as session:
        # Add all metrics
        for key, value in metrics_dict.items():
            if key != 'parameters':
                session.add_metric(key, value)

        if coefficients is not None:
            session.add_data_table('coefficients', coefficients)

        if residuals is not None:
            session.add_data_table('residuals', residuals)

        # Register plots
        if plot_paths:
            for plot_name, plot_path in plot_paths.items():
                session.add_plot(plot_name, f"{plot_name}.png", plot_path)

        return session.get_result_id()


def save_clustering_result(algorithm, n_clusters, labels, metrics_dict, plot_paths=None):
    """
    Save clustering analysis result.
    """
    parameters = {
        'algorithm': algorithm,
        'n_clusters': n_clusters
    }

    with AnalysisSession('clustering', parameters=parameters) as session:
        session.add_data_table('labels', labels.tolist() if hasattr(labels, 'tolist') else labels)

        for key, value in metrics_dict.items():
            session.add_metric(key, value)

        if plot_paths:
            for plot_name, plot_path in plot_paths.items():
                session.add_plot(plot_name, f"{plot_name}.png", plot_path)

        return session.get_result_id()


def get_latest_analysis(analysis_type=None, dataset_id=None):
    """
    Get latest analysis result for current user.
    """
    query = AnalysisResult.query.filter_by(user_id=g.user.id)

    if analysis_type:
        query = query.filter_by(analysis_type=analysis_type)

    if dataset_id:
        query = query.filter_by(dataset_id=dataset_id)

    return query.order_by(AnalysisResult.created_at.desc()).first()


def get_analysis_history(analysis_type=None, limit=10):
    """
    Get analysis history for current user.
    """
    query = AnalysisResult.query.filter_by(user_id=g.user.id)

    if analysis_type:
        query = query.filter_by(analysis_type=analysis_type)

    return query.order_by(AnalysisResult.created_at.desc()).limit(limit).all()


def get_analysis_plots(analysis_result_id):
    """
    Get all plots for an analysis result.
    """
    return AnalysisPlot.query.filter_by(analysis_result_id=analysis_result_id).all()


# Example usage in ML modules:
"""
# OLD WAY (with global variables):
classification_Reports = []
accuracies = []
confusion_Matrix = []

def logisticReg(...):
    # ... training code ...
    accuracies.append(acc)
    classification_Reports.append(report)
    confusion_Matrix.append(matrix)
    return [accuracies, classification_Reports, confusion_Matrix]


# NEW WAY (with database):
from app.analysis_helpers import save_classification_result

def logisticReg(...):
    # ... training code ...
    result_id = save_classification_result(
        analysis_type='logistic_regression',
        accuracy=acc,
        report_dict=report,
        confusion_matrix=matrix,
        test_size=size,
        scale=scale_val == 1,
        encode=encode_val == 1
    )
    return result_id

# Later, to retrieve:
from app.analysis_helpers import get_latest_analysis

result = get_latest_analysis('logistic_regression')
metrics = result.get_results()
accuracy = metrics['metrics']['accuracy']
report = metrics['classification_report']
"""
