"""
Naive Bayes module - REFACTORED to use database instead of global variables.
"""
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from flask import session, g
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    make_scorer,
)
from app.config import get_clean_path, get_upload_path
from app.analysis_helpers import AnalysisSession


def naiveBayes(value, choice, scale_val, encode_val):
    """
    Perform Naive Bayes analysis.

    Args:
        value: Test size (%) for choice=1, or k-folds for choice=2
        choice: 1=train/test split, 2=k-fold CV, 0=external test set
        scale_val: 1=apply scaling, 0=no scaling
        encode_val: 1=apply label encoding, 0=no encoding

    Returns:
        dict with analysis results ID and formatted results for display
    """
    # Read data
    if session["ext"] == "csv":
        df = pd.read_csv(get_clean_path("clean.csv"))
    elif session["ext"] == "json":
        df = pd.read_json(get_clean_path("clean.json"))

    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    target_names = list(df.iloc[:, -1].unique())

    le = LabelEncoder()
    sc = StandardScaler()

    # Prepare parameters for database
    parameters = {
        'method': 'train_test_split' if choice == 1 else 'k_fold_cv' if choice == 2 else 'external_test',
        'test_size': value / 100 if choice == 1 else None,
        'k_folds': value if choice == 2 else None,
        'scaling': scale_val == 1,
        'encoding': encode_val == 1,
    }

    # Choice 1: Train/Test Split
    if choice == 1:
        size = value / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=size, random_state=40
        )

        if scale_val == 1:
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

        if encode_val == 1:
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

        clf = GaussianNB().fit(X_train, y_train)
        pred_vals = clf.predict(X_test)

        acc = accuracy_score(y_test, pred_vals)
        report = classification_report(
            y_test,
            pred_vals,
            target_names=target_names,
            output_dict=True,
        )
        matrix = confusion_matrix(y_test, pred_vals)

        # Save to database
        with AnalysisSession('naive_bayes', parameters=parameters) as db_session:
            db_session.add_metric('accuracy', round(acc * 100, 2))
            db_session.add_classification_report(report)
            db_session.add_confusion_matrix(matrix)

        # Get ID after session is committed
        analysis_id = db_session.get_result_id()

        # Return formatted data for display (backward compatibility)
        data = pd.DataFrame(report).transpose()
        matrix_data = pd.DataFrame(matrix).transpose()

        return {
            'analysis_id': analysis_id,
            'accuracy': round(acc * 100, 2),
            'report': data,
            'matrix': matrix_data,
            'format': 'single'  # For template to know which format
        }

    # Choice 2: K-Fold Cross Validation
    elif choice == 2:
        k = value
        kfold = KFold(n_splits=k, random_state=7, shuffle=True)
        model = GaussianNB()

        if scale_val == 1:
            X = sc.fit_transform(X)

        if encode_val == 1:
            y = le.fit_transform(y)

        # Store results for each fold
        fold_results = []

        def fold_scorer(y_true, y_pred):
            """Custom scorer for k-fold that stores results."""
            acc = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred, output_dict=True)
            matrix = confusion_matrix(y_true, y_pred)

            fold_results.append({
                'accuracy': round(acc * 100, 2),
                'report': report,
                'matrix': matrix.tolist()
            })
            return acc

        predicted = cross_val_score(
            model,
            X,
            y,
            cv=kfold,
            scoring=make_scorer(fold_scorer),
        )

        # Save to database
        with AnalysisSession('naive_bayes', parameters=parameters) as db_session:
            # Average accuracy across folds
            avg_accuracy = round(np.mean([f['accuracy'] for f in fold_results]), 2)
            db_session.add_metric('accuracy_mean', avg_accuracy)
            db_session.add_metric('accuracy_std', round(np.std([f['accuracy'] for f in fold_results]), 2))

            # Store all fold results
            db_session.results['folds'] = fold_results
            db_session.add_metric('n_folds', k)

        # Get ID after session is committed
        analysis_id = db_session.get_result_id()

        # Format for display (backward compatibility)
        accuracies = [f['accuracy'] for f in fold_results]
        reports = [[pd.DataFrame(f['report']).transpose().to_html(
            classes=["table", "table-bordered", "table-striped", "table-hover", "thead-light"]
        ).strip("\n")] for f in fold_results]
        matrices = [[pd.DataFrame(f['matrix']).transpose().to_html(
            classes=["table", "table-bordered", "table-striped", "table-hover", "thead-light"]
        )] for f in fold_results]

        return {
            'analysis_id': analysis_id,
            'accuracies': accuracies,
            'reports': reports,
            'matrices': matrices,
            'format': 'kfold'  # For template
        }

    # Choice 0: External Test Set
    elif choice == 0:
        if scale_val == 1:
            X = sc.fit_transform(X)

        if encode_val == 1:
            y = le.fit_transform(y)

        clf = GaussianNB().fit(X, y)

        # Load external test set
        if session["ext"] == "csv":
            df_test = pd.read_csv(get_upload_path("test.csv"))
        elif session["ext"] == "json":
            df_test = pd.read_json(get_upload_path("test.json"))

        X_test = df_test.iloc[:, 1:-1]
        y_test = df_test.iloc[:, -1]

        if scale_val == 1:
            X_test = sc.transform(X_test)
        if encode_val == 1:
            y_test = le.transform(y_test)

        pred_vals = clf.predict(X_test)
        acc = accuracy_score(y_test, pred_vals)
        report = classification_report(
            y_test,
            pred_vals,
            target_names=target_names,
            output_dict=True,
        )
        matrix = confusion_matrix(y_test, pred_vals)

        # Save to database
        with AnalysisSession('naive_bayes', parameters=parameters) as db_session:
            db_session.add_metric('accuracy', round(acc * 100, 2))
            db_session.add_classification_report(report)
            db_session.add_confusion_matrix(matrix)

        # Get ID after session is committed
        analysis_id = db_session.get_result_id()

        # Return formatted data
        data = pd.DataFrame(report).transpose()
        matrix_data = pd.DataFrame(matrix).transpose()

        return {
            'analysis_id': analysis_id,
            'accuracy': round(acc * 100, 2),
            'report': data,
            'matrix': matrix_data,
            'format': 'single'
        }
