"""
Clustering analysis routes
"""
from flask import Blueprint, render_template, request, session, flash, redirect, url_for, jsonify, send_file
import os
import pandas as pd
import numpy as np
import time
from werkzeug.utils import secure_filename
from app.config import get_clean_path, get_temp_path, get_upload_path, get_unified_path, TEMP_DIR, CLEAN_DIR, UPLOAD_DIR, UNIFIED_DIR, DESCRIPTOR_GROUPS_FILE
from app.chemalize.utils import read_dataset, clean_temp_folder, check_dataset, get_dataset_info, ensure_temp_dir
from app.chemalize.preprocessing import generic_preprocessing as gp
from app.chemalize.utils.descriptor_groups import parse_descriptor_groups, filter_dataframe_by_groups

from app.chemalize.modules import clustering
from app.chemalize.visualization import coloring


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
    # Default grouping column for two-way HCA: prefer selected index column, fall back to Cluster
    default_grouping_col = session.get('index_column') or 'Cluster'
    clustering_params['twoway_hca_grouping_column'] = session.get('twoway_hca_grouping_column', default_grouping_col)

    # Build grouping column options: only categorical (non-numeric) columns
    # This includes string columns, object columns, and categorical dtype
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    grouping_columns_options = []
    index_column = clustering_params['index_column']

    # Add index column first if it exists
    if index_column and index_column in df.columns and index_column not in grouping_columns_options:
        grouping_columns_options.append(index_column)

    # Add all categorical columns
    for col in categorical_columns:
        if col not in grouping_columns_options:
            grouping_columns_options.append(col)

    clustering_params['grouping_columns_options'] = grouping_columns_options
    clustering_params['categorical_columns'] = categorical_columns

    # Get feature selection settings from session (needed for default HCA variables)
    feature_selection = session.get('feature_selection', 'all')
    selected_features = session.get('selected_features', None)

    # Default HCA X-axis variables: reuse last selection, otherwise match single-HCA set
    default_hca_variables = session.get('twoway_hca_variables', None)
    if not default_hca_variables:
        if feature_selection in ('select', 'groups') and selected_features:
            default_hca_variables = selected_features
        else:
            default_hca_variables = clustering_params['numeric_columns']
    # Keep only numeric variables to mirror single HCA input
    if default_hca_variables:
        default_hca_variables = [col for col in default_hca_variables if col in clustering_params['numeric_columns']]
    clustering_params['hca_default_variables'] = default_hca_variables

    clustering_params['feature_selection'] = session.get('feature_selection', 'all')
    clustering_params['selected_features'] = session.get('selected_features', None)

    # Add group file information if available
    uploaded_groups = session.get('clustering_uploaded_groups', None)
    selected_groups = session.get('clustering_selected_groups', None)
    clustering_params['uploaded_groups'] = uploaded_groups
    clustering_params['selected_groups'] = selected_groups

    # Combine all parameters
    render_params = {**info, **clustering_params}
    
    # If clustering was performed, add the results
    if session.get('clustering_performed'):
        # Add clustering results
        result_params = {k: session.get(k) for k in [
            'silhouette', 'calinski', 'n_noise', 'inertia',
            'cluster_plot', 'profile_plot', 'size_plot',
            'elbow_plot', 'eps_plot', 'dendrogram',
            'twoway_hca_plot', 'twoway_hca_variables',
            'twoway_hca_row_linkage', 'twoway_hca_col_linkage',
            'twoway_hca_grouping_column', 'twoway_hca_height_scale', 'twoway_hca_width_scale',
            'twoway_hca_dendro_color_column', 'twoway_hca_custom_colors', 'twoway_hca_show_zeros',
            'twoway_hca_x_axis_font_size', 'twoway_hca_y_axis_font_size', 'twoway_hca_legend_font_size',
            'twoway_hca_endpoint_loaded', 'twoway_hca_endpoint_key_column',
            'twoway_hca_endpoint_columns', 'twoway_hca_endpoint_column_types',
            'twoway_hca_endpoint_selected_column', 'twoway_hca_endpoint_reverse_colors'
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
    
    # Handle feature selection
    feature_selection = request.form.get('feature_selection', 'all')

    if feature_selection == 'groups':
        # Get selected groups and build feature list
        selected_groups = request.form.getlist('selected_groups')

        if selected_groups and 'clustering_uploaded_groups' in session:
            groups_dict = session['clustering_uploaded_groups']

            # Get current dataset to check which features actually exist
            clean_path = get_clean_path(session["csv_name"])
            df = read_dataset(clean_path)
            available_columns = set(df.columns)

            # Build list of features from selected groups
            all_group_features = []
            for group_id in selected_groups:
                if group_id in groups_dict:
                    all_group_features.extend(groups_dict[group_id]['descriptors'])

            # Remove duplicates while preserving order
            all_group_features = list(dict.fromkeys(all_group_features))

            # Filter to only include features that exist in the dataset
            selected_features = [feat for feat in all_group_features if feat in available_columns]

            # Warn if some features were not found
            missing_features = [feat for feat in all_group_features if feat not in available_columns]
            if missing_features:
                flash(f'Warning: {len(missing_features)} features from selected groups not found in dataset (e.g., {missing_features[0]}). Using {len(selected_features)} available features.', 'warning')
            else:
                flash(f'Using {len(selected_features)} features from {len(selected_groups)} selected groups.', 'info')

            session['clustering_selected_groups'] = selected_groups
        else:
            selected_features = None
            if 'clustering_uploaded_groups' not in session:
                flash('No groups file uploaded. Please upload a groups file first. Using all features.', 'warning')
                feature_selection = 'all'
            elif not selected_groups:
                flash('No groups selected. Please select at least one group. Using all features.', 'warning')
                feature_selection = 'all'
    elif feature_selection == 'select':
        # Manual feature selection
        selected_features = request.form.getlist('selected_features')
    else:
        # Use all features
        selected_features = None

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

        # Filter dataframe to only include selected features (if feature selection is active)
        if feature_selection == 'select' and selected_features:
            # Manual feature selection - filter to selected features
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

            # Build columns to keep (avoid duplicates)
            cols_to_keep = list(non_numeric_cols)  # Start with non-numeric

            # Add selected features (only if not already in list)
            for feat in selected_features:
                if feat not in cols_to_keep and feat in df.columns:
                    cols_to_keep.append(feat)

            # Add index column (only if not already in list)
            if index_column and index_column in df.columns and index_column not in cols_to_keep:
                cols_to_keep.append(index_column)

            # Filter dataframe
            df = df[cols_to_keep]

        elif feature_selection == 'groups' and selected_features:
            # Group-based feature selection - filter to features from selected groups
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

            # Build columns to keep (avoid duplicates)
            cols_to_keep = list(non_numeric_cols)  # Start with non-numeric

            # Add selected features (only if not already in list)
            for feat in selected_features:
                if feat not in cols_to_keep and feat in df.columns:
                    cols_to_keep.append(feat)

            # Add index column (only if not already in list)
            if index_column and index_column in df.columns and index_column not in cols_to_keep:
                cols_to_keep.append(index_column)

            # Filter dataframe
            df = df[cols_to_keep]

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


@clustering_bp.route("/upload_clustering_groups", methods=['POST'])
def upload_clustering_groups():
    """Handle upload of variable groups file for clustering"""
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))

    if 'group_file' not in request.files:
        flash('No file uploaded', 'danger')
        return redirect(url_for('clustering.clustering_analysis'))

    group_file = request.files['group_file']

    if group_file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('clustering.clustering_analysis'))

    try:
        # Read and parse the group file
        file_content = group_file.read().decode('utf-8')

        # Save to temp file for parsing
        temp_path = ensure_temp_dir()
        temp_group_file = os.path.join(temp_path, 'clustering_groups.txt')
        with open(temp_group_file, 'w', encoding='utf-8') as f:
            f.write(file_content)

        # Parse groups using the same parser as Alvadesk PCA
        groups_dict = parse_descriptor_groups(temp_group_file)

        # Store in session for display
        session['clustering_uploaded_groups'] = groups_dict
        session['clustering_groups_file_path'] = temp_group_file

        flash(f'Group file uploaded successfully! Found {len(groups_dict)} groups. Please select the groups you want to use below.', 'success')
    except Exception as e:
        flash(f'Error parsing group file: {str(e)}', 'danger')

    return redirect(url_for('clustering.clustering_analysis'))


@clustering_bp.route("/download_example_groups")
def download_example_groups():
    """Download example groups file (Alvadesk descriptor groups)"""
    try:
        return send_file(
            DESCRIPTOR_GROUPS_FILE,
            as_attachment=True,
            download_name='example_variable_groups.txt',
            mimetype='text/plain'
        )
    except Exception as e:
        flash(f'Error downloading example file: {str(e)}', 'danger')
        return redirect(url_for('clustering.clustering_analysis'))


@clustering_bp.route("/generate_twoway_hca", methods=['POST'])
def generate_twoway_hca():
    """Generate interactive two-way HCA heatmap using Plotly"""
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    if not check_dataset() or not session.get('clustering_performed'):
        msg = 'No clustering analysis results available. Please run clustering first.'
        if is_ajax:
            return jsonify({'success': False, 'error': msg}), 400
        flash(msg, 'danger')
        return redirect(url_for('clustering.clustering_analysis'))

    try:
        # Get user-selected variables for column clustering
        selected_variables = request.form.getlist('hca_variables')

        if not selected_variables or len(selected_variables) < 2:
            msg = 'Please select at least 2 variables for column clustering'
            if is_ajax:
                return jsonify({'success': False, 'error': msg}), 400
            flash(msg, 'warning')
            return redirect(url_for('clustering.clustering_analysis'))

        # Get linkage methods
        row_linkage = request.form.get('hca_linkage_rows', 'ward')
        col_linkage = request.form.get('hca_linkage_cols', 'ward')

        # Get dendrogram scale parameters from form sliders (%)
        height_scale = int(request.form.get('hca_row_height', 100))
        width_scale = int(request.form.get('hca_col_width', 100))

        # Get dendrogram color column (optional)
        row_color_column = request.form.get('hca_dendro_color_column', '').strip()
        if row_color_column == '':
            row_color_column = None

        # Get custom colors for dendrogram categories (JSON string)
        custom_colors_str = request.form.get('hca_custom_colors', '').strip()
        custom_colors = None
        if custom_colors_str:
            try:
                import json
                custom_colors = json.loads(custom_colors_str)
            except (json.JSONDecodeError, ValueError):
                custom_colors = None

        # Get show zeros option
        show_zeros = request.form.get('hca_show_zeros') == '1'

        # Get endpoint column (from external endpoint file)
        hca_endpoint_column = request.form.get('hca_endpoint_column', '').strip()
        endpoint_reverse_colors = request.form.get('hca_endpoint_reverse_colors') == '1'
        endpoint_column = None
        endpoint_data = None
        endpoint_is_numeric = False
        endpoint_mapping_message = None

        # Get axis font sizes (optional)
        x_axis_font_size_str = request.form.get('hca_x_axis_font_size', '').strip()
        y_axis_font_size_str = request.form.get('hca_y_axis_font_size', '').strip()
        legend_font_size_str = request.form.get('hca_legend_font_size', '').strip()
        x_axis_font_size = int(x_axis_font_size_str) if x_axis_font_size_str else None
        y_axis_font_size = int(y_axis_font_size_str) if y_axis_font_size_str else None
        legend_font_size = int(legend_font_size_str) if legend_font_size_str else None

        # Load the clustering results (which includes the Cluster column)
        results_path = os.path.join(ensure_temp_dir(), 'clustering_results.csv')

        if not os.path.exists(results_path):
            msg = 'Clustering results file not found. Please run clustering analysis first.'
            if is_ajax:
                return jsonify({'success': False, 'error': msg}), 400
            flash(msg, 'danger')
            return redirect(url_for('clustering.clustering_analysis'))

        # Read the results dataframe
        results_df = pd.read_csv(results_path)

        # Pick grouping column for Y axis: explicit selection -> index column -> Cluster
        grouping_column = request.form.get('hca_grouping_column', '').strip()
        index_column = session.get('index_column', None)
        if not grouping_column:
            grouping_column = index_column if index_column else 'Cluster'

        # Validate grouping column exists in the results
        if grouping_column not in results_df.columns:
            fallback_grouping = 'Cluster' if 'Cluster' in results_df.columns else None
            if fallback_grouping and grouping_column != fallback_grouping:
                msg = f"Selected grouping column '{grouping_column}' not found in clustering results. Using '{fallback_grouping}' instead."
                if is_ajax:
                    grouping_column = fallback_grouping
                else:
                    flash(msg, 'warning')
                grouping_column = fallback_grouping
            else:
                msg = f"Grouping column '{grouping_column}' not found in clustering results."
                if is_ajax:
                    return jsonify({'success': False, 'error': msg}), 400
                flash(msg, 'danger')
                return redirect(url_for('clustering.clustering_analysis'))

        # Load endpoint data if selected
        if hca_endpoint_column and session.get('twoway_hca_endpoint_loaded'):
            endpoint_file_path = session.get('twoway_hca_endpoint_file_path')
            endpoint_key_column = session.get('twoway_hca_endpoint_key_column')
            if endpoint_file_path and endpoint_key_column and os.path.exists(endpoint_file_path):
                df_endpoint = coloring.load_coloring_file(endpoint_file_path)
                if df_endpoint is not None:
                    color_result = coloring.prepare_color_data(
                        results_df, df_endpoint, endpoint_key_column, hca_endpoint_column
                    )
                    if color_result['success']:
                        endpoint_column = hca_endpoint_column
                        endpoint_data = color_result['data']
                        endpoint_is_numeric = color_result['is_numeric']
                        session['twoway_hca_endpoint_selected_column'] = hca_endpoint_column
                        matched = color_result.get('matched', 0)
                        total = color_result.get('total', 0)
                        match_ratio = color_result.get('match_ratio', 0.0)
                        endpoint_mapping_message = (
                            f" Endpoint mapping: {matched}/{total} ({match_ratio * 100:.1f}%)."
                        )
                    else:
                        endpoint_mapping_message = (
                            f" Endpoint mapping failed: {color_result.get('message', 'unknown error')}."
                        )

        # Generate the two-way HCA heatmap
        heatmap_html = clustering.generate_twoway_hca_heatmap(
            df=results_df,
            selected_variables=selected_variables,
            grouping_column=grouping_column,
            row_linkage=row_linkage,
            col_linkage=col_linkage,
            temp_path=ensure_temp_dir(),
            height_scale=height_scale,
            width_scale=width_scale,
            row_color_column=row_color_column,
            show_zeros=show_zeros,
            custom_colors=custom_colors,
            x_axis_font_size=x_axis_font_size,
            y_axis_font_size=y_axis_font_size,
            legend_font_size=legend_font_size,
            endpoint_column=endpoint_column,
            endpoint_data=endpoint_data,
            endpoint_is_numeric=endpoint_is_numeric,
            endpoint_reverse_colors=endpoint_reverse_colors
        )

        # Store the heatmap HTML in session
        session['twoway_hca_plot'] = heatmap_html
        session['twoway_hca_variables'] = selected_variables
        session['twoway_hca_row_linkage'] = row_linkage
        session['twoway_hca_col_linkage'] = col_linkage
        session['twoway_hca_grouping_column'] = grouping_column
        session['twoway_hca_height_scale'] = height_scale
        session['twoway_hca_width_scale'] = width_scale
        session['twoway_hca_dendro_color_column'] = row_color_column
        session['twoway_hca_x_axis_font_size'] = x_axis_font_size
        session['twoway_hca_y_axis_font_size'] = y_axis_font_size
        session['twoway_hca_legend_font_size'] = legend_font_size
        session['twoway_hca_endpoint_reverse_colors'] = endpoint_reverse_colors
        # Save custom colors per column for persistence
        if custom_colors and row_color_column:
            saved_colors = session.get('twoway_hca_custom_colors', {})
            saved_colors[row_color_column] = custom_colors
            session['twoway_hca_custom_colors'] = saved_colors
        session['twoway_hca_show_zeros'] = show_zeros

        success_msg = f"Two-way HCA heatmap generated successfully with {len(selected_variables)} variables, grouped by '{grouping_column}'."
        if row_color_column:
            success_msg += f" Dendrogram colored by '{row_color_column}'."
        if endpoint_mapping_message:
            success_msg += endpoint_mapping_message
        if is_ajax:
            return jsonify({
                'success': True,
                'plot_html': heatmap_html,
                'grouping_column': grouping_column,
                'variables': selected_variables,
                'message': success_msg
            })
        flash(success_msg, 'success')

    except Exception as e:
        error_msg = f'Error generating two-way HCA heatmap: {str(e)}'
        if is_ajax:
            return jsonify({'success': False, 'error': error_msg}), 500
        flash(error_msg, 'danger')

    return redirect(url_for('clustering.clustering_analysis'))


@clustering_bp.route("/upload_twoway_hca_endpoint", methods=['POST'])
def upload_twoway_hca_endpoint():
    """Upload an external endpoint file for the Two-Way HCA heatmap."""
    if not check_dataset() or not session.get('clustering_performed'):
        return jsonify({'success': False, 'error': 'No clustering results available'}), 400

    if 'endpoint_file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400

    endpoint_file = request.files['endpoint_file']
    if endpoint_file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    try:
        temp_path = ensure_temp_dir()
        filename = secure_filename(endpoint_file.filename)
        file_path = os.path.join(temp_path, 'hca_endpoint_' + filename)
        endpoint_file.save(file_path)

        # Load the endpoint file
        df_endpoint = coloring.load_coloring_file(file_path)
        if df_endpoint is None:
            return jsonify({'success': False, 'error': 'Could not read file. Supported formats: CSV, XLSX.'}), 400

        # Load clustering results for key column detection
        results_path = os.path.join(temp_path, 'clustering_results.csv')
        if not os.path.exists(results_path):
            return jsonify({'success': False, 'error': 'Clustering results not found. Run clustering first.'}), 400
        results_df = pd.read_csv(results_path)

        # Auto-detect key column
        detection = coloring.detect_key_column(results_df, df_endpoint)

        # Pick best key candidate:
        # 1) auto-detected key (if strong),
        # 2) top scored common candidate,
        # 3) best overlap among common columns.
        key_column = detection.get('key_column')
        candidates_info = detection.get('candidates_info', [])
        common_cols = detection.get('common_columns', [])

        if not key_column and candidates_info:
            key_column = candidates_info[0].get('column')

        if (not key_column) and common_cols:
            ranked_common = sorted(
                common_cols,
                key=lambda c: coloring.check_column_overlap(results_df, df_endpoint, c),
                reverse=True
            )
            key_column = ranked_common[0]

        # Final guard: key must exist in both datasets; otherwise keep as None.
        if key_column and (key_column not in results_df.columns or key_column not in df_endpoint.columns):
            key_column = None

        validation = coloring.validate_coloring_setup(results_df, df_endpoint, key_column)

        if not validation['valid'] and common_cols:
            ranked_common = sorted(
                common_cols,
                key=lambda c: coloring.check_column_overlap(results_df, df_endpoint, c),
                reverse=True
            )
            for candidate_key in ranked_common:
                candidate_validation = coloring.validate_coloring_setup(results_df, df_endpoint, candidate_key)
                if candidate_validation['valid']:
                    key_column = candidate_key
                    validation = candidate_validation
                    break

        # Get available columns and their types
        columns = validation.get('available_color_columns', [])
        column_types = validation.get('column_type_map', {})

        if not columns:
            # Fallback: list all non-key columns
            columns = [c for c in df_endpoint.columns if c != key_column]
            column_types = {}
            for col in columns:
                if pd.api.types.is_numeric_dtype(df_endpoint[col]):
                    column_types[col] = 'numeric'
                else:
                    column_types[col] = 'categorical'

        key_overlap = 0.0
        if key_column and key_column in results_df.columns and key_column in df_endpoint.columns:
            key_overlap = coloring.check_column_overlap(results_df, df_endpoint, key_column)

        # Store in session
        session['twoway_hca_endpoint_file_path'] = file_path
        session['twoway_hca_endpoint_loaded'] = True
        session['twoway_hca_endpoint_key_column'] = key_column
        session['twoway_hca_endpoint_columns'] = columns
        session['twoway_hca_endpoint_column_types'] = column_types

        return jsonify({
            'success': True,
            'columns': columns,
            'column_types': column_types,
            'key_column': key_column,
            'auto_detected': detection.get('found', False),
            'pca_candidates': detection.get('pca_candidates', []),
            'common_columns': detection.get('common_columns', []),
            'key_overlap': key_overlap,
            'message': f'File loaded: {len(df_endpoint)} rows, {len(df_endpoint.columns)} columns. '
                       f'Key column: {key_column or "not detected"} '
                       f'(overlap: {key_overlap * 100:.1f}%).'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f'Error processing file: {str(e)}'}), 500


@clustering_bp.route("/confirm_twoway_hca_key_column", methods=['POST'])
def confirm_twoway_hca_key_column():
    """Manually override the key column for endpoint matching."""
    if not check_dataset() or not session.get('clustering_performed'):
        return jsonify({'success': False, 'error': 'No clustering results available'}), 400

    key_column = request.form.get('key_column', '').strip()
    if not key_column:
        return jsonify({'success': False, 'error': 'No key column specified'}), 400

    endpoint_file_path = session.get('twoway_hca_endpoint_file_path')
    if not endpoint_file_path or not os.path.exists(endpoint_file_path):
        return jsonify({'success': False, 'error': 'Endpoint file not found. Please re-upload.'}), 400

    try:
        temp_path = ensure_temp_dir()
        df_endpoint = coloring.load_coloring_file(endpoint_file_path)
        if df_endpoint is None:
            return jsonify({'success': False, 'error': 'Could not re-read endpoint file.'}), 400

        results_path = os.path.join(temp_path, 'clustering_results.csv')
        results_df = pd.read_csv(results_path)

        if key_column not in results_df.columns:
            return jsonify({'success': False, 'error': f"Key column '{key_column}' not found in clustering results."}), 400
        if key_column not in df_endpoint.columns:
            return jsonify({'success': False, 'error': f"Key column '{key_column}' not found in endpoint file."}), 400

        # Validate with the manually selected key column
        validation = coloring.validate_coloring_setup(results_df, df_endpoint, key_column)

        columns = validation.get('available_color_columns', [])
        column_types = validation.get('column_type_map', {})

        if not columns:
            columns = [c for c in df_endpoint.columns if c != key_column]
            column_types = {}
            for col in columns:
                if pd.api.types.is_numeric_dtype(df_endpoint[col]):
                    column_types[col] = 'numeric'
                else:
                    column_types[col] = 'categorical'

        key_overlap = coloring.check_column_overlap(results_df, df_endpoint, key_column)

        # Update session
        session['twoway_hca_endpoint_key_column'] = key_column
        session['twoway_hca_endpoint_columns'] = columns
        session['twoway_hca_endpoint_column_types'] = column_types

        return jsonify({
            'success': True,
            'columns': columns,
            'column_types': column_types,
            'key_column': key_column,
            'key_overlap': key_overlap,
            'message': f'Key column updated to: {key_column} (overlap: {key_overlap * 100:.1f}%)'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f'Error: {str(e)}'}), 500


@clustering_bp.route("/get_column_categories", methods=['POST'])
def get_column_categories():
    """Get unique categories for a column (for dynamic color picker)"""
    if not check_dataset() or not session.get('clustering_performed'):
        return jsonify({'success': False, 'error': 'No clustering results available'}), 400

    try:
        column_name = request.form.get('column_name', '').strip()
        if not column_name:
            return jsonify({'success': False, 'error': 'No column specified'}), 400

        # Load the clustering results
        results_path = os.path.join(ensure_temp_dir(), 'clustering_results.csv')
        if not os.path.exists(results_path):
            return jsonify({'success': False, 'error': 'Results file not found'}), 400

        results_df = pd.read_csv(results_path)

        if column_name not in results_df.columns:
            return jsonify({'success': False, 'error': f'Column {column_name} not found'}), 400

        # Get unique values, sorted
        unique_values = sorted(results_df[column_name].dropna().unique(), key=str)
        categories = [str(v) for v in unique_values]

        # Get previously saved colors if any
        saved_colors = session.get('twoway_hca_custom_colors', {})

        return jsonify({
            'success': True,
            'categories': categories,
            'saved_colors': saved_colors.get(column_name, {})
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
