"""
Multiple Linear Regression routes
"""
from flask import Blueprint, render_template, request, session, flash, redirect, url_for, jsonify, send_file
import os
import pandas as pd
import numpy as np
import time
from werkzeug.utils import secure_filename
from app.config import get_clean_path, get_temp_path, get_upload_path, get_unified_path, TEMP_DIR, CLEAN_DIR, UPLOAD_DIR, UNIFIED_DIR
from app.chemalize.utils import read_dataset, clean_temp_folder, check_dataset, get_dataset_info, ensure_temp_dir
from app.chemalize.preprocessing import generic_preprocessing as gp

from app.chemalize.modules import mlr


mlr_bp = Blueprint('mlr', __name__)

@mlr_bp.route("/mlr_analysis")
def mlr_analysis():
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))

    # Check if coming from GA - use GA preprocessed data
    from_ga = session.get('from_ga', False)
    ga_data_path = session.get('ga_preprocessed_data_path')

    if from_ga and ga_data_path and os.path.exists(ga_data_path):
        # Use GA preprocessed data
        df = read_dataset(ga_data_path)
    else:
        # Use original clean data
        clean_path = get_clean_path(session["csv_name"])
        df = read_dataset(clean_path)

    info = get_dataset_info(df)

    # Get all data columns (avoiding the name 'columns' to prevent conflict)
    all_data_columns = df.columns.tolist()

    # Add MLR-specific parameters from session, excluding target_var
    # since it's already in info from get_dataset_info()
    mlr_params = {
        'all_data_columns': all_data_columns,
        'include_intercept': session.get('include_intercept', True),
        'selected_features': session.get('selected_features', []),
        'test_size': session.get('test_size', 0.2),
        'mlr_performed': session.get('mlr_performed', False),

        # Split method parameters
        'split_method': session.get('split_method', 'random'),
        'shuffle': session.get('shuffle', True),
        'random_state': session.get('random_state', 42),
        'strat_test_size': session.get('strat_test_size', 0.2),
        'strat_bins': session.get('strat_bins', 5),
        'time_column': session.get('time_column', ''),
        'time_test_size': session.get('time_test_size', 0.2),
        'n_folds': session.get('n_folds', 5),
        'shuffle_kfold': session.get('shuffle_kfold', True),

        # Other options
        'scale_data': session.get('scale_data', False),
        'check_assumptions': session.get('check_assumptions', True),
        'detect_outliers': session.get('detect_outliers', False),

        # Prediction mode parameters
        'prediction_performed': session.get('prediction_performed', False)
    }
    
    if session.get('mlr_performed'):
        # Add MLR results if analysis was performed
        mlr_results = {
            'train_r2': session.get('train_r2'),
            'adj_r2': session.get('adj_r2'),
            'test_r2': session.get('test_r2'),
            'q2_loo': session.get('q2_loo'),
            'q2_test': session.get('q2_test'),
            'train_rmse': session.get('train_rmse'),
            'test_rmse': session.get('test_rmse'),
            'rmse_loo': session.get('rmse_loo'),
            'train_mae': session.get('train_mae'),
            'test_mae': session.get('test_mae'),
            'f_statistic': session.get('f_statistic'),
            'f_pvalue': session.get('f_pvalue'),
            'aic': session.get('aic'),
            'bic': session.get('bic'),
            'dw_stat': session.get('dw_stat'),
            'vif_values': session.get('vif_values'),
            'ccc_ext': session.get('ccc_ext'),
            'coefficients': session.get('coefficients'),
            'std_errors': session.get('std_errors'),
            't_values': session.get('t_values'),
            'p_values': session.get('p_values'),
            'feature_names': session.get('feature_names'),
            'mlr_pred_actual_plot': session.get('mlr_pred_actual_plot'),
            'mlr_residuals_plot': session.get('mlr_residuals_plot'),
            'mlr_residuals_hist': session.get('mlr_residuals_hist'),
            'mlr_qq_plot': session.get('mlr_qq_plot'),
            'mlr_williams_plot': session.get('mlr_williams_plot'),
            'AD_train': session.get('AD_train'),
            'AD_test': session.get('AD_test'),
            'h_star': session.get('h_star'),
            'williams_outliers': session.get('williams_outliers', []),
            # Cross-validation specific metrics
            'cv_train_r2_mean': session.get('cv_train_r2_mean'),
            'cv_test_r2_mean': session.get('cv_test_r2_mean'),
            'cv_train_rmse_mean': session.get('cv_train_rmse_mean'),
            'cv_test_rmse_mean': session.get('cv_test_rmse_mean'),
            # GA-specific info
            'from_ga': session.get('from_ga', False),
            'ga_model_rank': session.get('ga_model_rank'),
            'ga_best_score': session.get('ga_best_score')
        }
        return render_template('mlr_analysis.html', title='MLR Analysis', active="analyze", zip=zip, **info, **mlr_params, **mlr_results)

    if session.get('prediction_performed'):
        # Add prediction results if custom model was applied
        pred_results = {
            'pred_column_name': session.get('pred_column_name'),
            'pred_intercept': session.get('pred_intercept'),
            'pred_coefficients': session.get('pred_coefficients'),
            'pred_include_intercept': session.get('pred_include_intercept'),
            'pred_n_samples': session.get('pred_n_samples'),
            'pred_mean': session.get('pred_mean'),
            'pred_std': session.get('pred_std'),
            'pred_min': session.get('pred_min'),
            'pred_max': session.get('pred_max'),
            'pred_preview': session.get('pred_preview')
        }
        return render_template('mlr_analysis.html', title='MLR Analysis', active="analyze", zip=zip, **info, **mlr_params, **pred_results)

    return render_template('mlr_analysis.html', title='MLR Analysis', active="analyze", zip=zip, **info, **mlr_params)


@mlr_bp.route("/perform_mlr", methods=['POST'])
def perform_mlr():
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))
    
    # Get basic form parameters
    target_var = request.form.get('target_var')
    include_intercept = 'include_intercept' in request.form
    selected_features = request.form.getlist('selected_features')
    scale_data = 'scale_data' in request.form
    check_assumptions = 'check_assumptions' in request.form
    detect_outliers = 'detect_outliers' in request.form
    
    # Get split method and its parameters
    split_method = request.form.get('split_method', 'random')
    
    # Parameters for random split
    test_size = float(request.form.get('test_size', 0.2))
    shuffle = 'shuffle' in request.form
    random_state = int(request.form.get('random_state', 42))
    
    # Parameters for stratified split
    strat_test_size = float(request.form.get('strat_test_size', 0.2))
    strat_bins = int(request.form.get('strat_bins', 5))
    
    # Parameters for time-based split
    time_column = request.form.get('time_column', '')
    time_test_size = float(request.form.get('time_test_size', 0.2))
    
    # Parameters for k-fold CV
    n_folds = int(request.form.get('n_folds', 5))
    shuffle_kfold = 'shuffle_kfold' in request.form
    
    # Parameters for one vs n split
    n_parts = int(request.form.get('n_parts', 3))
    shuffle_onevn = 'shuffle_onevn' in request.form
    random_state_onevn = int(request.form.get('random_state_onevn', 42))
    
    # Parameters for systematic sampling (nowy kod)
    systematic_step = int(request.form.get('systematic_step', 3))
    include_last_point = 'include_last_point' in request.form
    
    if not target_var:
        flash('Please select a target variable!', 'danger')
        return redirect(url_for('mlr.mlr_analysis'))
    
    if not selected_features:
        flash('Please select at least one feature!', 'danger')
        return redirect(url_for('mlr.mlr_analysis'))
    
    # Save parameters to session
    session['target_var'] = target_var
    session['include_intercept'] = include_intercept
    session['selected_features'] = selected_features
    session['scale_data'] = scale_data
    session['check_assumptions'] = check_assumptions
    session['detect_outliers'] = detect_outliers
    
    # Save split method parameters
    session['split_method'] = split_method
    session['test_size'] = test_size
    session['shuffle'] = shuffle
    session['random_state'] = random_state
    session['strat_test_size'] = strat_test_size
    session['strat_bins'] = strat_bins
    session['time_column'] = time_column
    session['time_test_size'] = time_test_size
    session['n_folds'] = n_folds
    session['shuffle_kfold'] = shuffle_kfold
    session['n_parts'] = n_parts
    session['shuffle_onevn'] = shuffle_onevn
    session['random_state_onevn'] = random_state_onevn
    session['systematic_step'] = systematic_step
    session['include_last_point'] = include_last_point

    # Mark session as modified
    session.modified = True

    # Perform MLR using the module
    # Check if coming from GA - use GA preprocessed data
    from_ga = session.get('from_ga', False)
    ga_data_path = session.get('ga_preprocessed_data_path')

    if from_ga and ga_data_path and os.path.exists(ga_data_path):
        # Use GA preprocessed data
        df = read_dataset(ga_data_path)
    else:
        # Use original clean data
        clean_path = get_clean_path(session["csv_name"])
        df = read_dataset(clean_path)
    
    try:
        # Create temp directory if it doesn't exist
        os.makedirs(ensure_temp_dir(), exist_ok=True)
        
        # Prepare split parameters based on selected method
        split_params = {}
        if split_method == 'random':
            split_params = {
                'test_size': test_size,
                'shuffle': shuffle,
                'random_state': random_state
            }
        elif split_method == 'stratified':
            split_params = {
                'test_size': strat_test_size,
                'n_bins': strat_bins
            }
        elif split_method == 'time':
            split_params = {
                'time_column': time_column,
                'test_size': time_test_size
            }
        elif split_method == 'kfold':
            split_params = {
                'n_folds': n_folds,
                'shuffle': shuffle_kfold
            }
        elif split_method == 'one_vs_n':
            split_params = {
                'n_parts': n_parts,
                'shuffle': shuffle_onevn,
                'random_state': random_state_onevn
            }
        elif split_method == 'systematic':
            split_params = {
                'step': systematic_step,
                'include_last_point': include_last_point
            }
        # LOOCV doesn't need additional parameters

        # Check for predefined indices from GA
        predefined_train_idx = session.get('predefined_train_idx')
        predefined_test_idx = session.get('predefined_test_idx')

        # Call the MLR module with the split method and parameters
        results = mlr.perform_mlr(
            df,
            target_var=target_var,
            selected_features=selected_features,
            include_intercept=include_intercept,
            split_method=split_method,
            split_params=split_params,
            scale_data=scale_data,
            check_assumptions=check_assumptions,
            detect_outliers=detect_outliers,
            temp_path=ensure_temp_dir(),
            predefined_train_idx=predefined_train_idx if split_method == 'predefined' else None,
            predefined_test_idx=predefined_test_idx if split_method == 'predefined' else None
        )
        
        # Store results in session
        session['mlr_performed'] = True

        # Add timestamp to prevent browser caching of images
        timestamp = int(time.time() * 1000)

        # Process results before storing in session
        for key, value in results.items():
            # Check if the value looks like an image file path
            if isinstance(value, str) and (value.endswith('.png') or value.endswith('.jpg')):
                # Extract filename and convert to URL with timestamp
                filename = os.path.basename(value)
                session[key] = url_for('utils.serve_temp_image', filename=filename, t=timestamp)
            else:
                # Store other values as-is
                session[key] = value

        # Mark session as modified to ensure it's saved
        session.modified = True

        flash('MLR analysis completed successfully!', 'success')
    except Exception as e:
        flash(f'Error performing MLR: {str(e)}', 'danger')
    
    return redirect(url_for('mlr.mlr_analysis'))


@mlr_bp.route("/download_mlr_model")
def download_mlr_model():
    if not check_dataset() or not session.get('mlr_performed'):
        flash('No MLR analysis results available', 'danger')
        return redirect(url_for('mlr.mlr_analysis'))
    
    try:
        clean_path = get_clean_path(session["csv_name"])
        temp_file = mlr.generate_model_file(
            clean_path, 
            session['target_var'], 
            session['selected_features'],
            include_intercept=session.get('include_intercept', True),
            temp_path=ensure_temp_dir()
        )
        return send_file(temp_file, as_attachment=True, download_name='mlr_model_summary.csv')
    except Exception as e:
        flash(f'Error generating model file: {str(e)}', 'danger')
        return redirect(url_for('mlr.mlr_analysis'))


@mlr_bp.route("/download_mlr_report")
def download_mlr_report():
    if not check_dataset() or not session.get('mlr_performed'):
        flash('No MLR analysis results available', 'danger')
        return redirect(url_for('mlr.mlr_analysis'))

    try:
        clean_path = get_clean_path(session["csv_name"])
        temp_dir = ensure_temp_dir()

        # Gather metrics from session
        metrics = {
            'train_r2': session.get('train_r2'),
            'adj_r2': session.get('adj_r2'),
            'test_r2': session.get('test_r2'),
            'q2_loo': session.get('q2_loo'),
            'q2_test': session.get('q2_test'),
            'train_rmse': session.get('train_rmse'),
            'test_rmse': session.get('test_rmse'),
            'rmse_loo': session.get('rmse_loo'),
            'train_mae': session.get('train_mae'),
            'test_mae': session.get('test_mae'),
            'f_statistic': session.get('f_statistic'),
            'f_pvalue': session.get('f_pvalue'),
            'aic': session.get('aic'),
            'bic': session.get('bic'),
            'dw_stat': session.get('dw_stat'),
            'ccc_ext': session.get('ccc_ext'),
            'AD_train': session.get('AD_train'),
            'AD_test': session.get('AD_test'),
            'h_star': session.get('h_star')
        }

        # Gather plot paths (use the temp directory where plots were saved)
        plot_paths = {
            'pred_actual': os.path.join(temp_dir, 'mlr_pred_actual_plot.png'),
            'williams': os.path.join(temp_dir, 'mlr_williams_plot.png'),
            'residuals_hist': os.path.join(temp_dir, 'mlr_residuals_hist.png'),
            'qq': os.path.join(temp_dir, 'mlr_qq_plot.png')
        }

        temp_file = mlr.generate_report(
            clean_path,
            session['target_var'],
            session['selected_features'],
            include_intercept=session.get('include_intercept', True),
            temp_path=temp_dir,
            metrics=metrics,
            plot_paths=plot_paths
        )
        return send_file(temp_file, as_attachment=True, download_name='mlr_report.pdf')
    except Exception as e:
        flash(f'Error generating report: {str(e)}', 'danger')
        return redirect(url_for('mlr.mlr_analysis'))


@mlr_bp.route("/download_mlr_predictions")
def download_mlr_predictions():
    if not check_dataset() or not session.get('mlr_performed'):
        flash('No MLR analysis results available', 'danger')
        return redirect(url_for('mlr.mlr_analysis'))
    
    try:
        clean_path = get_clean_path(session["csv_name"])
        temp_file = mlr.generate_predictions_file(
            clean_path, 
            session['target_var'], 
            session['selected_features'],
            include_intercept=session.get('include_intercept', True),
            temp_path=ensure_temp_dir()
        )
        return send_file(temp_file, as_attachment=True, download_name='mlr_predictions.csv')
    except Exception as e:
        flash(f'Error generating predictions file: {str(e)}', 'danger')
        return redirect(url_for('mlr.mlr_analysis'))


@mlr_bp.route("/download_williams_outliers")
def download_williams_outliers():
    if not check_dataset() or not session.get('mlr_performed'):
        flash('No MLR analysis results available', 'danger')
        return redirect(url_for('mlr.mlr_analysis'))

    # Check if there are outliers
    outliers = session.get('williams_outliers', [])
    h_star = session.get('h_star')

    if h_star is None:
        flash('Williams Plot data not available', 'danger')
        return redirect(url_for('mlr.mlr_analysis'))

    try:
        clean_path = get_clean_path(session["csv_name"])
        temp_file = mlr.generate_outliers_file(
            outliers,
            h_star,
            clean_path,
            session['target_var'],
            session['selected_features'],
            temp_path=ensure_temp_dir()
        )
        return send_file(temp_file, as_attachment=True, download_name='williams_outliers.csv')
    except Exception as e:
        flash(f'Error generating outliers file: {str(e)}', 'danger')
        return redirect(url_for('mlr.mlr_analysis'))


@mlr_bp.route("/reset_mlr_analysis")
def reset_mlr_analysis():
    # Reset MLR-specific session variables
    for key in [
        'mlr_performed', 'train_r2', 'test_r2', 'adj_r2', 'q2_loo', 'q2_test',
        'train_rmse', 'test_rmse', 'rmse_loo', 'train_mae', 'test_mae',
        'f_statistic', 'f_pvalue', 'aic', 'bic', 'dw_stat', 'vif_values',
        'ccc_ext', 'coefficients', 'std_errors', 't_values', 'p_values',
        'feature_names', 'mlr_pred_actual_plot', 'mlr_residuals_plot',
        'mlr_residuals_hist', 'mlr_qq_plot', 'cv_train_r2_mean', 'cv_test_r2_mean',
        'cv_train_rmse_mean', 'cv_test_rmse_mean', 'mlr_williams_plot',
        'AD_train', 'AD_test', 'h_star', 'williams_outliers',
        # Prediction mode variables
        'prediction_performed', 'pred_column_name', 'pred_intercept',
        'pred_coefficients', 'pred_include_intercept', 'pred_n_samples',
        'pred_mean', 'pred_std', 'pred_min', 'pred_max', 'pred_preview'
    ]:
        if key in session:
            session.pop(key)

    # Keep target_var and previously selected features for convenience
    flash('MLR analysis reset. Configure a new analysis.', 'info')
    return redirect(url_for('mlr.mlr_analysis'))


@mlr_bp.route("/apply_custom_model", methods=['POST'])
def apply_custom_model():
    """Apply a custom model with user-defined coefficients for prediction"""
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))

    # Get form parameters
    prediction_name = request.form.get('prediction_name', 'Y_pred')
    include_intercept = 'pred_include_intercept' in request.form
    intercept_value = float(request.form.get('intercept_value', 0)) if include_intercept else 0
    selected_features = request.form.getlist('pred_selected_features')

    if not selected_features:
        flash('Please select at least one feature!', 'danger')
        return redirect(url_for('mlr.mlr_analysis'))

    # Get coefficients for each selected feature
    coefficients = {}
    for feature in selected_features:
        coef_value = request.form.get(f'coefficient_{feature}', '')
        if coef_value == '' or coef_value is None:
            flash(f'Please enter a coefficient for {feature}!', 'danger')
            return redirect(url_for('mlr.mlr_analysis'))
        try:
            coefficients[feature] = float(coef_value)
        except ValueError:
            flash(f'Invalid coefficient value for {feature}!', 'danger')
            return redirect(url_for('mlr.mlr_analysis'))

    # Read dataset
    clean_path = get_clean_path(session["csv_name"])
    df = read_dataset(clean_path)

    try:
        # Apply custom model
        results = mlr.apply_custom_model(
            df=df,
            coefficients=coefficients,
            intercept=intercept_value if include_intercept else None,
            prediction_column_name=prediction_name,
            temp_path=ensure_temp_dir()
        )

        # Store results in session
        session['prediction_performed'] = True
        session['pred_column_name'] = prediction_name
        session['pred_intercept'] = intercept_value if include_intercept else 0
        session['pred_coefficients'] = coefficients
        session['pred_include_intercept'] = include_intercept
        session['pred_n_samples'] = results['n_samples']
        session['pred_mean'] = results['mean']
        session['pred_std'] = results['std']
        session['pred_min'] = results['min']
        session['pred_max'] = results['max']
        session['pred_preview'] = results['preview']

        # Mark session as modified to ensure it's saved
        session.modified = True

        flash(f'Custom model applied successfully! {results["n_samples"]} predictions calculated.', 'success')
    except Exception as e:
        flash(f'Error applying custom model: {str(e)}', 'danger')

    return redirect(url_for('mlr.mlr_analysis'))


@mlr_bp.route("/download_custom_predictions")
def download_custom_predictions():
    """Download predictions from custom model"""
    if not check_dataset() or not session.get('prediction_performed'):
        flash('No prediction results available', 'danger')
        return redirect(url_for('mlr.mlr_analysis'))

    try:
        clean_path = get_clean_path(session["csv_name"])

        # Get coefficients and intercept from session
        coefficients = session.get('pred_coefficients', {})
        intercept = session.get('pred_intercept', 0) if session.get('pred_include_intercept') else None
        prediction_name = session.get('pred_column_name', 'Y_pred')

        temp_file = mlr.generate_custom_predictions_file(
            data_path=clean_path,
            coefficients=coefficients,
            intercept=intercept,
            prediction_column_name=prediction_name,
            temp_path=ensure_temp_dir()
        )
        return send_file(temp_file, as_attachment=True, download_name=f'{prediction_name}_predictions.csv')
    except Exception as e:
        flash(f'Error generating predictions file: {str(e)}', 'danger')
        return redirect(url_for('mlr.mlr_analysis'))

# Clustering Analysis Routes

