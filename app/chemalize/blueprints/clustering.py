"""
Clustering analysis routes
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

from app.chemalize.modules import clustering


clustering_bp = Blueprint('clustering', __name__)

@clustering_bp.route("/clustering_analysis")
def clustering_analysis():
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))
    
    clean_path = get_clean_path(session["csv_name"])
    df = read_dataset(clean_path)
    info = get_dataset_info(df)
    
    # Get any clustering-specific parameters from session
    clustering_params = {}
    
    # Method-specific parameters
    method = session.get('clustering_method', 'kmeans')
    
    if method == 'kmeans':
        clustering_params['n_clusters'] = session.get('n_clusters', 3)
    elif method == 'dbscan':
        clustering_params['eps'] = session.get('eps', 0.5)
        clustering_params['min_samples'] = session.get('min_samples', 5)
    elif method == 'hierarchical':
        # Use h_n_clusters instead of n_clusters for hierarchical clustering
        clustering_params['h_n_clusters'] = session.get('h_n_clusters', 3)
        clustering_params['linkage_method'] = session.get('linkage_method', 'ward')
    
    # Common parameters
    clustering_params['method'] = method
    clustering_params['scale_data'] = session.get('scale_data', True)
    clustering_params['pca_visualization'] = session.get('pca_visualization', True)
    
    # Add performed status and numeric columns for feature selection
    clustering_params['clustering_performed'] = session.get('clustering_performed', False)
    clustering_params['numeric_columns'] = df.select_dtypes(include=[np.number]).columns.tolist()
    # Add index_column and label_density to clustering_params
    clustering_params['index_column'] = session.get('index_column', None)
    clustering_params['label_density'] = session.get('label_density', 10)
    clustering_params['columns'] = df.columns.tolist()  # Make sure all columns are available for selection

    clustering_params['feature_selection'] = session.get('feature_selection', 'all')
    clustering_params['selected_features'] = session.get('selected_features', None)
    
    # Combine all parameters
    render_params = {**info, **clustering_params}
    
    # If clustering was performed, add the results
    if session.get('clustering_performed'):
        # Add clustering results
        result_params = {k: session.get(k) for k in [
            'silhouette', 'calinski', 'n_noise', 'inertia',
            'cluster_plot', 'profile_plot', 'size_plot', 
            'elbow_plot', 'eps_plot', 'dendrogram'
        ] if session.get(k) is not None}
        
        render_params.update(result_params)
    
    return render_template('clustering_analysis.html', 
                           title='Clustering Analysis',
                           **render_params)

@clustering_bp.route("/perform_clustering", methods=['POST'])
def perform_clustering():
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))
    
    # Get form parameters
    method = request.form.get('clustering_method', 'kmeans')
    scale_data = 'scale_data' in request.form
    pca_visualization = 'pca_visualization' in request.form
    
    # Get feature selection parameters
    feature_selection = request.form.get('feature_selection', 'all')
    selected_features = request.form.getlist('selected_features') if feature_selection == 'select' else None
    
    # Get the index column parameter
    index_column = request.form.get('index_column', '')
    label_density = int(request.form.get('label_density', 10))

    # Store them in the session
    session['feature_selection'] = feature_selection
    session['selected_features'] = selected_features
    session['index_column'] = index_column if index_column else None
    session['label_density'] = label_density
        
    # Method-specific parameters - get the right parameters based on the method
    if method == 'kmeans':
        n_clusters = int(request.form.get('n_clusters', 3))
        h_n_clusters = None  # Not used for K-means
    elif method == 'hierarchical':
        n_clusters = 3  # Default value, not used
        h_n_clusters = int(request.form.get('h_n_clusters', 3))
    else:  # DBSCAN
        n_clusters = 3  # Default value, not used
        h_n_clusters = None  # Not used for DBSCAN
    
    eps = float(request.form.get('eps', 0.5))
    min_samples = int(request.form.get('min_samples', 5))
    linkage_method = request.form.get('linkage_method', 'ward')
    
    # Save parameters to session
    session['clustering_method'] = method
    
    # Store method-specific parameters
    if method == 'kmeans':
        session['n_clusters'] = n_clusters
    elif method == 'hierarchical':
        session['h_n_clusters'] = h_n_clusters
        session['linkage_method'] = linkage_method
    elif method == 'dbscan':
        session['eps'] = eps
        session['min_samples'] = min_samples
    
    # Store common parameters
    session['scale_data'] = scale_data
    session['pca_visualization'] = pca_visualization
    
    # Perform clustering using the module
    clean_path = get_clean_path(session["csv_name"])
    df = read_dataset(clean_path)
    
    try:
        # Create temp directory if it doesn't exist
        os.makedirs(ensure_temp_dir(), exist_ok=True)
        
        # Call the clustering module with both n_clusters and h_n_clusters
        results = clustering.perform_clustering(
            df,
            method=method,
            n_clusters=n_clusters,
            h_n_clusters=h_n_clusters,
            eps=eps,
            min_samples=min_samples,
            linkage_method=linkage_method,
            scale_data=scale_data,
            pca_visualization=pca_visualization,
            temp_path=ensure_temp_dir(),
            index_column=index_column if index_column else None,
            label_density=label_density,
            feature_selection=feature_selection,
            selected_features=selected_features
        )
        
        # Store results in session
        session['clustering_performed'] = True

        # Add timestamp to prevent browser caching of images
        timestamp = int(time.time() * 1000)

        # Handle numeric/text data normally
        for key, value in results.items():
            # Check if this is potentially an image path (look for known image keys or .png extension)
            if key.endswith('_plot') or (isinstance(value, str) and value.endswith('.png')):
                # Convert filesystem path to URL with timestamp
                if value:
                    filename = os.path.basename(value)
                    session[key] = url_for('utils.serve_temp_image', filename=filename, t=timestamp)
            else:
                # For non-image data, store directly
                session[key] = value
        
        flash('Clustering analysis completed successfully!', 'success')
    except Exception as e:
        flash(f'Error performing clustering: {str(e)}', 'danger')
    
    return redirect(url_for('clustering.clustering_analysis'))


@clustering_bp.route("/download_clustering_results")
def download_clustering_results():
    if not check_dataset() or not session.get('clustering_performed'):
        flash('No clustering analysis results available', 'danger')
        return redirect(url_for('clustering.clustering_analysis'))
    
    try:
        clean_path = get_clean_path(session["csv_name"])
        temp_file = clustering.generate_results_file(clean_path, temp_path=ensure_temp_dir())
        return send_file(temp_file, as_attachment=True, download_name='clustering_results.csv')
    except Exception as e:
        flash(f'Error generating results file: {str(e)}', 'danger')
        return redirect(url_for('clustering.clustering_analysis'))


@clustering_bp.route("/download_clustering_report")
def download_clustering_report():
    if not check_dataset() or not session.get('clustering_performed'):
        flash('No clustering analysis results available', 'danger')
        return redirect(url_for('clustering.clustering_analysis'))
    
    try:
        clean_path = get_clean_path(session["csv_name"])
        temp_file = clustering.generate_report(
            clean_path, 
            method=session.get('clustering_method', 'kmeans'),
            temp_path=ensure_temp_dir()
        )
        return send_file(temp_file, as_attachment=True, download_name='clustering_report.pdf')
    except Exception as e:
        flash(f'Error generating report: {str(e)}', 'danger')
        return redirect(url_for('clustering.clustering_analysis'))


