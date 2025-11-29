"""
Principal Component Analysis routes
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

from app.chemalize.modules import pca


pca_bp = Blueprint('pca', __name__)

@pca_bp.route("/pca_analysis")
def pca_analysis():
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))
    
    clean_path = get_clean_path(session["csv_name"])
    df = read_dataset(clean_path)
    info = get_dataset_info(df)
    
    # Add any additional PCA-specific parameters from session
    pca_params = {k: session.get(k) for k in [
        'n_components', 'scale_data', 'show_variance', 'show_scatter',
        'show_loading', 'show_biplot', 'pc_color_by', 'pca_performed',
        'pc_x_axis', 'pc_y_axis', 'pc_loadings_select', 'feature_selection_method',
        'top_n_features', 'loading_threshold', 'show_top_features_plot',
        'show_feature_importance', 'export_feature_importance', 'export_selected_features'
    ]}
    
    if session.get('pca_performed'):
        # Add PCA results if analysis was performed
        pca_results = {
            'pca_summary': session.get('pca_summary', []),
            'pca_variance_plot': session.get('pca_variance_plot'),
            'pca_scatter_plot': session.get('pca_scatter_plot'),
            'pca_loadings_plot': session.get('pca_loadings_plot'),
            'pca_biplot': session.get('pca_biplot'),
            'pca_feature_importance_plot': session.get('pca_feature_importance_plot'),
            'pca_selected_scatter_plot': session.get('pca_selected_scatter_plot'),
            'pca_selected_biplot': session.get('pca_selected_biplot'),
            'selected_features_summary': session.get('selected_features_summary'),
            'selected_features_count': session.get('selected_features_count'),
            'features_per_pc': session.get('features_per_pc'),
            'feature_selection_method_display': session.get('feature_selection_method_display')
        }
        return render_template('pca_analysis.html', title='PCA Analysis', active="analyze", **info, **pca_params, **pca_results)
    
    return render_template('pca_analysis.html', title='PCA Analysis', active="analyze", **info, **pca_params)


@pca_bp.route("/perform_pca", methods=['POST'])
def perform_pca():
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))

    # --- Podstawowe parametry z formularza ---
    n_components = int(request.form.get('n_components', 2))
    n_components = max(2, n_components)  # minimalnie 2, żeby mieć sensowny scatter/biplot

    scale_data = 'scale_data' in request.form
    show_variance = 'show_variance' in request.form
    show_scatter = 'show_scatter' in request.form
    show_loading = 'show_loading' in request.form
    show_biplot = 'show_biplot' in request.form
    pc_color_by = request.form.get('pc_color_by', '')

    # --- Zaawansowane parametry z formularza ---
    pc_x_axis = int(request.form.get('pc_x_axis', 1))
    pc_y_axis = int(request.form.get('pc_y_axis', 2))

    # lista komponentów dla loadings (multi-select) -> na inty i w zakresie
    pc_loadings_select = request.form.getlist('pc_loadings_select')
    pc_loadings_select = [int(x) for x in pc_loadings_select] if pc_loadings_select else [1, 2]
    # unikalne + w zakresie 1..n_components
    pc_loadings_select = sorted({c for c in pc_loadings_select if 1 <= c <= n_components})
    if not pc_loadings_select:
        pc_loadings_select = [1, 2]

    feature_selection_method = request.form.get('feature_selection_method', 'all')
    top_n_features = int(request.form.get('top_n_features', 5))
    loading_threshold = float(request.form.get('loading_threshold', 0.3))
    show_top_features_plot = 'show_top_features_plot' in request.form
    show_feature_importance = 'show_feature_importance' in request.form
    export_feature_importance = 'export_feature_importance' in request.form
    export_selected_features = 'export_selected_features' in request.form

    # --- Walidacja i korekty osi PC (NIE rysujemy PC1 x PC1) ---
    # zbij w zakres 1..n_components
    pc_x_axis = min(max(1, pc_x_axis), n_components)
    pc_y_axis = min(max(1, pc_y_axis), n_components)

    if pc_x_axis == pc_y_axis:
        # wybierz sąsiedni komponent; preferuj +1, a jak się nie da to 1/2
        if pc_y_axis < n_components:
            pc_y_axis += 1
        else:
            pc_y_axis = 1 if pc_x_axis != 1 else 2
        flash('Y-axis PC było równe X-axis PC – automatycznie zmieniono, aby uniknąć wykresu PC×PC.', 'info')

    # --- Zapis parametrów do session (po korektach!) ---
    session['n_components'] = n_components
    session['scale_data'] = scale_data
    session['show_variance'] = show_variance
    session['show_scatter'] = show_scatter
    session['show_loading'] = show_loading
    session['show_biplot'] = show_biplot
    session['pc_color_by'] = pc_color_by
    session['pc_x_axis'] = pc_x_axis
    session['pc_y_axis'] = pc_y_axis
    session['pc_loadings_select'] = pc_loadings_select
    session['feature_selection_method'] = feature_selection_method
    session['top_n_features'] = top_n_features
    session['loading_threshold'] = loading_threshold
    session['show_top_features_plot'] = show_top_features_plot
    session['show_feature_importance'] = show_feature_importance
    session['export_feature_importance'] = export_feature_importance
    session['export_selected_features'] = export_selected_features

    # --- Wykonanie PCA ---
    clean_path = get_clean_path(session["csv_name"])
    df = read_dataset(clean_path)

    try:
        os.makedirs(ensure_temp_dir(), exist_ok=True)

        results = pca.perform_enhanced_pca(
            df,
            n_components=n_components,
            scale_data=scale_data,
            show_variance=show_variance,
            show_scatter=show_scatter,
            show_loading=show_loading,
            show_biplot=show_biplot,
            color_by=pc_color_by,
            pc_x_axis=pc_x_axis,
            pc_y_axis=pc_y_axis,
            pc_loadings_select=pc_loadings_select,
            feature_selection_method=feature_selection_method,
            top_n_features=top_n_features,
            loading_threshold=loading_threshold,
            show_top_features_plot=show_top_features_plot,
            show_feature_importance=show_feature_importance,
            temp_path=ensure_temp_dir()
        )

        session['pca_performed'] = True
        session['pca_summary'] = results.get('summary', [])

        # Add timestamp to prevent browser caching of images
        timestamp = int(time.time() * 1000)  # milliseconds for better uniqueness

        def create_plot_url(plot_path):
            if plot_path and os.path.exists(plot_path):
                return url_for('utils.serve_temp_image', filename=os.path.basename(plot_path), t=timestamp)
            return None

        session['pca_variance_plot'] = create_plot_url(results.get('variance_plot'))
        session['pca_scatter_plot'] = create_plot_url(results.get('scatter_plot'))
        session['pca_loadings_plot'] = create_plot_url(results.get('loadings_plot'))
        session['pca_biplot'] = create_plot_url(results.get('biplot'))
        session['pca_feature_importance_plot'] = create_plot_url(results.get('feature_importance_plot'))
        session['pca_selected_scatter_plot'] = create_plot_url(results.get('selected_scatter_plot'))
        session['pca_selected_biplot'] = create_plot_url(results.get('selected_biplot'))

        session['selected_features_summary'] = results.get('selected_features_summary')
        session['selected_features_count'] = results.get('selected_features_count')
        session['features_per_pc'] = results.get('features_per_pc')
        session['feature_selection_method_display'] = results.get('feature_selection_method_display')

        flash('Enhanced PCA analysis completed successfully!', 'success')

    except Exception as e:
        # typowy błąd: zbyt duży obraz (matplotlib)
        flash(f'Error performing PCA: {str(e)}', 'danger')

    return redirect(url_for('pca.pca_analysis'))


@pca_bp.route("/download_pca_components")
def download_pca_components():
    if not check_dataset() or not session.get('pca_performed'):
        flash('No PCA analysis results available', 'danger')
        return redirect(url_for('pca.pca_analysis'))
    
    # Generate and return the file
    try:
        clean_path = get_clean_path(session["csv_name"])
        temp_file = pca.generate_components_file(clean_path, temp_path=ensure_temp_dir())
        return send_file(temp_file, as_attachment=True, download_name='pca_components.csv')
    except Exception as e:
        flash(f'Error generating components file: {str(e)}', 'danger')
        return redirect(url_for('pca.pca_analysis'))


@pca_bp.route("/download_pca_loadings")
def download_pca_loadings():
    if not check_dataset() or not session.get('pca_performed'):
        flash('No PCA analysis results available', 'danger')
        return redirect(url_for('pca.pca_analysis'))
    
    try:
        clean_path = get_clean_path(session["csv_name"])
        temp_file = pca.generate_loadings_file(clean_path, temp_path=ensure_temp_dir())
        return send_file(temp_file, as_attachment=True, download_name='pca_loadings.csv')
    except Exception as e:
        flash(f'Error generating loadings file: {str(e)}', 'danger')
        return redirect(url_for('pca.pca_analysis'))


@pca_bp.route("/download_pca_report")
def download_pca_report():
    if not check_dataset() or not session.get('pca_performed'):
        flash('No PCA analysis results available', 'danger')
        return redirect(url_for('pca.pca_analysis'))
    
    try:
        clean_path = get_clean_path(session["csv_name"])
        temp_file = pca.generate_enhanced_report(clean_path, temp_path=ensure_temp_dir())
        return send_file(temp_file, as_attachment=True, download_name='enhanced_pca_report.pdf')
    except Exception as e:
        flash(f'Error generating report: {str(e)}', 'danger')
        return redirect(url_for('pca.pca_analysis'))

# PCR Analysis Routes

