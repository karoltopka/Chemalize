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
from app.utils import read_dataset, clean_temp_folder, check_dataset, get_dataset_info, ensure_temp_dir
from app.preprocessing import generic_preprocessing as gp

from app.modules import mlr


mlr_bp = Blueprint('mlr', __name__)

@mlr_bp.route("/mlr_analysis")
def mlr_analysis():
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))
    
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
        'detect_outliers': session.get('detect_outliers', False)
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
            # Cross-validation specific metrics
            'cv_train_r2_mean': session.get('cv_train_r2_mean'),
            'cv_test_r2_mean': session.get('cv_test_r2_mean'),
            'cv_train_rmse_mean': session.get('cv_train_rmse_mean'),
            'cv_test_rmse_mean': session.get('cv_test_rmse_mean')
        }
        return render_template('mlr_analysis.html', title='MLR Analysis', active="analyze", zip=zip, **info, **mlr_params, **mlr_results)
    
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
    
    # Perform MLR using the module
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
            temp_path=ensure_temp_dir()
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
        temp_file = mlr.generate_report(
            clean_path, 
            session['target_var'], 
            session['selected_features'],
            include_intercept=session.get('include_intercept', True),
            temp_path=ensure_temp_dir()
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
        'cv_train_rmse_mean', 'cv_test_rmse_mean' 'mlr_williams_plot',
        'AD_train', 'AD_test', 'h_star'
    ]:
        if key in session:
            session.pop(key)
    
    # Keep target_var and previously selected features for convenience
    flash('MLR analysis reset. Configure a new analysis.', 'info')
    return redirect(url_for('mlr.mlr_analysis'))

# Clustering Analysis Routes

