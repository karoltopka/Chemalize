"""
Principal Component Regression routes
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

from app.chemalize.modules import pcr


pcr_bp = Blueprint('pcr', __name__)

@pcr_bp.route("/pcr_analysis")
def pcr_analysis():
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))
    
    clean_path = get_clean_path(session["csv_name"])
    df = read_dataset(clean_path)
    info = get_dataset_info(df)
    
    # Add any additional PCR-specific parameters from session
    pcr_params = {k: session.get(k) for k in [
        'pcr_n_components', 'test_size', 'pcr_scale_data', 'optimize_components', 
        'compare_with_linear', 'show_pca_variance', 'show_pred_actual', 
        'show_residuals', 'pcr_performed'
    ]}
    
    if session.get('pcr_performed'):
        # Add PCR results if analysis was performed
        pcr_results = {
            'pcr_train_r2': session.get('pcr_train_r2'),
            'pcr_test_r2': session.get('pcr_test_r2'),
            'pcr_train_rmse': session.get('pcr_train_rmse'),
            'pcr_test_rmse': session.get('pcr_test_rmse'),
            'pcr_test_mae': session.get('pcr_test_mae'),
            'lr_train_r2': session.get('lr_train_r2'),
            'lr_test_r2': session.get('lr_test_r2'),
            'lr_train_rmse': session.get('lr_train_rmse'),
            'lr_test_rmse': session.get('lr_test_rmse'),
            'lr_test_mae': session.get('lr_test_mae'),
            'total_variance_explained': session.get('total_variance_explained'),
            'pcr_variance_plot': session.get('pcr_variance_plot'),
            'pcr_pred_actual_plot': session.get('pcr_pred_actual_plot'),
            'pcr_residuals_plot': session.get('pcr_residuals_plot'),
            'pcr_optimization_plot': session.get('pcr_optimization_plot')
        }
        return render_template('pcr_analysis.html', title='PCR Analysis', active="analyze", **info, **pcr_params, **pcr_results)
    
    return render_template('pcr_analysis.html', title='PCR Analysis', active="analyze", **info, **pcr_params)


@pcr_bp.route("/perform_pcr", methods=['POST'])
def perform_pcr():
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))
    
    if not session.get('target_var'):
        flash('Please select a target variable first!', 'danger')
        return redirect(url_for('pcr.pcr_analysis'))
    
    # Get form parameters
    pcr_n_components = int(request.form.get('pcr_n_components', 2))
    test_size = float(request.form.get('test_size', 0.2))
    pcr_scale_data = 'pcr_scale_data' in request.form
    optimize_components = 'optimize_components' in request.form
    compare_with_linear = 'compare_with_linear' in request.form
    show_pca_variance = 'show_pca_variance' in request.form
    show_pred_actual = 'show_pred_actual' in request.form
    show_residuals = 'show_residuals' in request.form
    
    # Save parameters to session
    session['pcr_n_components'] = pcr_n_components
    session['test_size'] = test_size
    session['pcr_scale_data'] = pcr_scale_data
    session['optimize_components'] = optimize_components
    session['compare_with_linear'] = compare_with_linear
    session['show_pca_variance'] = show_pca_variance
    session['show_pred_actual'] = show_pred_actual
    session['show_residuals'] = show_residuals
    
    # Perform PCR using the module
    clean_path = get_clean_path(session["csv_name"])
    df = read_dataset(clean_path)
    target_var = session['target_var']
    
    try:
        # Create temp directory if it doesn't exist
        os.makedirs(ensure_temp_dir(), exist_ok=True)
        
        # Call the PCR module
        results = pcr.perform_pcr(
            df,
            target_var=target_var,
            n_components=pcr_n_components,
            test_size=test_size,
            scale_data=pcr_scale_data,
            optimize_components=optimize_components,
            compare_with_linear=compare_with_linear,
            show_variance=show_pca_variance,
            show_pred_actual=show_pred_actual,
            show_residuals=show_residuals,
            temp_path=ensure_temp_dir()
        )
        
        # Store results in session
        session['pcr_performed'] = True
        session['pcr_train_r2'] = results.get('pcr_train_r2')
        session['pcr_test_r2'] = results.get('pcr_test_r2')
        session['pcr_train_rmse'] = results.get('pcr_train_rmse')
        session['pcr_test_rmse'] = results.get('pcr_test_rmse')
        session['pcr_test_mae'] = results.get('pcr_test_mae')
        
        if compare_with_linear:
            session['lr_train_r2'] = results.get('lr_train_r2')
            session['lr_test_r2'] = results.get('lr_test_r2')
            session['lr_train_rmse'] = results.get('lr_train_rmse')
            session['lr_test_rmse'] = results.get('lr_test_rmse')
            session['lr_test_mae'] = results.get('lr_test_mae')
        
        session['total_variance_explained'] = results.get('total_variance_explained')

        # Add timestamp to prevent browser caching of images
        timestamp = int(time.time() * 1000)

        # Convert image paths to URLs
        # PCR variance plot
        if results.get('pcr_variance_plot'):
            variance_filename = os.path.basename(results.get('pcr_variance_plot'))
            session['pcr_variance_plot'] = url_for('utils.serve_temp_image', filename=variance_filename, t=timestamp)
        else:
            session['pcr_variance_plot'] = None

        # PCR pred vs actual plot
        if results.get('pcr_pred_actual_plot'):
            pred_actual_filename = os.path.basename(results.get('pcr_pred_actual_plot'))
            session['pcr_pred_actual_plot'] = url_for('utils.serve_temp_image', filename=pred_actual_filename, t=timestamp)
        else:
            session['pcr_pred_actual_plot'] = None

        # PCR residuals plot
        if results.get('pcr_residuals_plot'):
            residuals_filename = os.path.basename(results.get('pcr_residuals_plot'))
            session['pcr_residuals_plot'] = url_for('utils.serve_temp_image', filename=residuals_filename, t=timestamp)
        else:
            session['pcr_residuals_plot'] = None

        # Optimization plot (if enabled)
        if optimize_components and results.get('optimization_plot'):
            optimization_filename = os.path.basename(results.get('optimization_plot'))
            session['pcr_optimization_plot'] = url_for('utils.serve_temp_image', filename=optimization_filename, t=timestamp)
        
        flash('PCR analysis completed successfully!', 'success')
    except Exception as e:
        flash(f'Error performing PCR: {str(e)}', 'danger')
    
    return redirect(url_for('pcr.pcr_analysis'))


@pcr_bp.route("/download_pcr_predictions")
def download_pcr_predictions():
    if not check_dataset() or not session.get('pcr_performed'):
        flash('No PCR analysis results available', 'danger')
        return redirect(url_for('pcr.pcr_analysis'))
    
    try:
        clean_path = get_clean_path(session["csv_name"])
        temp_file = pcr.generate_predictions_file(clean_path, session['target_var'], temp_path=ensure_temp_dir())
        return send_file(temp_file, as_attachment=True, download_name='pcr_predictions.csv')
    except Exception as e:
        flash(f'Error generating predictions file: {str(e)}', 'danger')
        return redirect(url_for('pcr.pcr_analysis'))


@pcr_bp.route("/download_pcr_model")
def download_pcr_model():
    if not check_dataset() or not session.get('pcr_performed'):
        flash('No PCR analysis results available', 'danger')
        return redirect(url_for('pcr.pcr_analysis'))
    
    try:
        clean_path = get_clean_path(session["csv_name"])
        temp_file = pcr.generate_model_file(clean_path, session['target_var'], temp_path=ensure_temp_dir())
        return send_file(temp_file, as_attachment=True, download_name='pcr_model_coefficients.csv')
    except Exception as e:
        flash(f'Error generating model file: {str(e)}', 'danger')
        return redirect(url_for('pcr.pcr_analysis'))


@pcr_bp.route("/download_pcr_report")
def download_pcr_report():
    if not check_dataset() or not session.get('pcr_performed'):
        flash('No PCR analysis results available', 'danger')
        return redirect(url_for('pcr.pcr_analysis'))
    
    try:
        clean_path = get_clean_path(session["csv_name"])
        temp_file = pcr.generate_report(clean_path, session['target_var'], temp_path=ensure_temp_dir())
        return send_file(temp_file, as_attachment=True, download_name='pcr_report.pdf')
    except Exception as e:
        flash(f'Error generating report: {str(e)}', 'danger')
        return redirect(url_for('pcr.pcr_analysis'))
# MLR Analysis Routes

