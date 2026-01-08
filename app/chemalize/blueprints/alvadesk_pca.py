"""
Alvadesk PCA Analysis with Descriptor Group Selection
Allows users to select specific descriptor groups from Descriptors_group.txt
and perform PCA only on those descriptors with visualization capabilities.
"""
from flask import Blueprint, render_template, request, session, flash, redirect, url_for, jsonify, send_file
import os
import pandas as pd
import numpy as np
import time
import json
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename

from app.config import (
    get_clean_path,
    get_temp_path,
    TEMP_DIR,
    DESCRIPTOR_GROUPS_FILE
)
from app.chemalize.utils import read_dataset, check_dataset, get_dataset_info, ensure_temp_dir
from app.chemalize.utils.descriptor_groups import (
    parse_descriptor_groups,
    filter_dataframe_by_groups,
    get_group_summary,
    get_descriptors_for_groups
)
from app.chemalize.visualization import coloring
from app.nocache import nocache

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from app.utils.watermark import add_watermark_matplotlib_after_plot


alvadesk_pca_bp = Blueprint('alvadesk_pca', __name__)


@alvadesk_pca_bp.route("/alvadesk", methods=["GET"])
@nocache
def alvadesk_main():
    """Main AlvaDesk page with available analysis methods."""
    if not check_dataset():
        flash("Please upload a dataset first!", "danger")
        return redirect(url_for("preprocessing.preprocess"))
    
    clean_path = get_clean_path(session["csv_name"])
    df = read_dataset(clean_path)
    info = get_dataset_info(df)
    
    return render_template("alvadesk_main.html", 
                          title="AlvaDesk Analysis",
                          active="alvadesk",
                          **info)



@alvadesk_pca_bp.route("/alvadesk_pca_analysis", methods=['GET', 'POST'])
@nocache
def alvadesk_pca_analysis():
    """Main route for Alvadesk PCA analysis with descriptor group selection."""

    if not check_dataset():
        flash('Please upload a dataset first!', 'danger')
        return redirect(url_for('preprocessing.preprocess'))

    # Parse descriptor groups
    try:
        groups_dict = parse_descriptor_groups(DESCRIPTOR_GROUPS_FILE)
        # Sort groups numerically by ID
        groups_dict = dict(sorted(groups_dict.items(), key=lambda x: int(x[0])))
    except Exception as e:
        flash(f'Error loading descriptor groups: {str(e)}', 'danger')
        return redirect(url_for('main.home'))

    # Load dataset info
    clean_path = get_clean_path(session["csv_name"])

    # Check if file exists before trying to read
    if not os.path.exists(clean_path):
        flash('Dataset file not found. Please upload a dataset.', 'danger')
        return redirect(url_for('preprocessing.preprocess'))

    df = read_dataset(clean_path)
    info = get_dataset_info(df)

    # Check if any PCA has been performed
    pca_performed = session.get('alvadesk_pca_performed', False)

    # Prepare template variables
    template_vars = {
        'title': 'Alvadesk PCA Analysis',
        'active': 'alvadesk_pca',
        'descriptor_groups': groups_dict,
        'pca_performed': pca_performed,
        **info
    }

    # If PCA was performed, add results
    if pca_performed:
        template_vars.update({
            'selected_groups': session.get('alvadesk_selected_groups', []),
            'pca_summary': session.get('alvadesk_pca_summary', []),
            'pca_n_components': session.get('alvadesk_pca_n_components', 0),
            'pca_variance_plot': session.get('alvadesk_pca_variance_plot'),
            'pca_loading_plot': session.get('alvadesk_pca_loadings_plot'),
            'group_summary': session.get('alvadesk_group_summary', {}),
            'filtered_descriptors_count': session.get('alvadesk_filtered_descriptors_count', 0),
            # For visualization
            'pca_color_options': session.get('alvadesk_pca_color_options', []),
            'color_file_loaded': session.get('alvadesk_color_file_loaded', False),
        })

    return render_template('alvadesk_pca_analysis.html', **template_vars)


@alvadesk_pca_bp.route("/perform_alvadesk_pca", methods=['POST'])
def perform_alvadesk_pca():
    """Perform PCA on selected descriptor groups."""

    if not check_dataset():
        flash('Please upload a dataset first!', 'danger')
        return redirect(url_for('preprocessing.preprocess'))

    # Get selected descriptor groups
    selected_groups = request.form.getlist('descriptor_groups')

    if not selected_groups:
        flash('Please select at least one descriptor group!', 'warning')
        return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))

    # Get PCA parameters
    n_components = int(request.form.get('n_components', 2))
    n_components = max(2, n_components)
    scale_data = 'scale_data' in request.form

    # Load dataset
    clean_path = get_clean_path(session["csv_name"])
    df = read_dataset(clean_path)

    try:
        # Parse descriptor groups
        groups_dict = parse_descriptor_groups(DESCRIPTOR_GROUPS_FILE)

        # Filter dataframe by selected groups
        filtered_df, descriptor_columns = filter_dataframe_by_groups(
            df, groups_dict, selected_groups, keep_non_descriptors=True
        )

        if len(descriptor_columns) == 0:
            flash('No matching descriptors found in the dataset for selected groups!', 'danger')
            return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))

        # Get group summary
        group_summary = get_group_summary(groups_dict, selected_groups)

        # Adjust n_components if necessary
        max_components = min(len(descriptor_columns), len(filtered_df))
        if n_components > max_components:
            n_components = max_components
            flash(f'Number of components adjusted to {n_components} based on data dimensions', 'info')

        # Prepare data for PCA - only descriptor columns
        X = filtered_df[descriptor_columns].select_dtypes(include=[np.number])

        # Handle missing values
        if X.isnull().any().any():
            X = X.fillna(X.mean())

        # Scale data if requested
        if scale_data:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values

        # Perform PCA
        pca_model = PCA(n_components=n_components)
        principal_components = pca_model.fit_transform(X_scaled)

        # Create PC DataFrame with all original non-descriptor columns
        pc_columns = [f'PC{i+1}' for i in range(n_components)]
        pc_df = pd.DataFrame(principal_components, columns=pc_columns, index=filtered_df.index)

        # Add non-descriptor columns to PC dataframe
        non_descriptor_cols = [col for col in filtered_df.columns if col not in descriptor_columns]
        for col in non_descriptor_cols:
            pc_df[col] = filtered_df[col].values

        # Save PC components to temp
        temp_path = ensure_temp_dir()
        pca_components_file = os.path.join(temp_path, 'alvadesk_pca_components.csv')
        pc_df.to_csv(pca_components_file, index=False)

        # Prepare summary
        variance_explained = pca_model.explained_variance_ratio_ * 100
        cumulative_variance = np.cumsum(variance_explained)

        pca_summary = []
        for i in range(n_components):
            pca_summary.append({
                'component': i + 1,
                'explained_variance': round(variance_explained[i], 2),
                'cumulative_variance': round(cumulative_variance[i], 2)
            })

        # Compute component loadings dataframe
        # IMPORTANT: Loadings should be correlations between original variables and PCs
        # Formula: loadings = eigenvectors * sqrt(eigenvalues)
        # This gives loadings in [-1, 1] range representing correlations
        loadings_raw = pca_model.components_ * np.sqrt(pca_model.explained_variance_)[:, np.newaxis]

        # If data was NOT scaled, we need to normalize by standard deviations
        # to get proper correlations
        if not scale_data:
            std_devs = np.std(X.values if hasattr(X, 'values') else X, axis=0, ddof=1)
            loadings_raw = loadings_raw / std_devs[np.newaxis, :]

        # Create loadings dataframe with proper correlations
        loadings_df = pd.DataFrame(
            loadings_raw.T,
            index=descriptor_columns,
            columns=pc_columns
        )

        # Clip to [-1, 1] range to ensure valid correlations
        loadings_df = loadings_df.clip(-1, 1)

        # Save loadings to CSV for biplot generation
        loadings_file = os.path.join(temp_path, 'alvadesk_pca_loadings.csv')
        loadings_df.to_csv(loadings_file)

        # Save eigenvalues to CSV
        eigenvalues_df = pd.DataFrame({
            'Component': pc_columns,
            'Eigenvalue': pca_model.explained_variance_
        })
        eigenvalues_file = os.path.join(temp_path, 'alvadesk_pca_eigenvalues.csv')
        eigenvalues_df.to_csv(eigenvalues_file, index=False)

        # Save explained variance to CSV
        explained_variance_df = pd.DataFrame({
            'Component': pc_columns,
            'Explained_Variance_%': variance_explained,
            'Cumulative_Variance_%': cumulative_variance
        })
        explained_variance_file = os.path.join(temp_path, 'alvadesk_pca_explained_variance.csv')
        explained_variance_df.to_csv(explained_variance_file, index=False)

        # Generate variance plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, n_components + 1), variance_explained, alpha=0.7, label='Individual')
        plt.plot(range(1, n_components + 1), cumulative_variance, 'ro-', label='Cumulative')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained (%)')
        plt.title('Alvadesk PCA - Variance Explained by Each Component')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        add_watermark_matplotlib_after_plot(plt.gcf())
        variance_plot_path = os.path.join(temp_path, 'alvadesk_pca_variance.png')
        plt.savefig(variance_plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Generate component loading plot (top contributors per component)
        num_components_plot = min(3, n_components)
        if num_components_plot > 0:
            fig, axes = plt.subplots(num_components_plot, 1, figsize=(12, 4 * num_components_plot))
            if num_components_plot == 1:
                axes = [axes]

            for idx in range(num_components_plot):
                component_label = pc_columns[idx]
                component_series = loadings_df[component_label]
                sorted_series = component_series.reindex(component_series.abs().sort_values(ascending=False).index)
                top_loadings = sorted_series.head(min(15, len(sorted_series)))

                axes[idx].axvline(0, color='#9ca3af', linewidth=0.8)
                colors = ['#2563eb' if val >= 0 else '#ef4444' for val in top_loadings.iloc[::-1]]
                axes[idx].barh(
                    top_loadings.index[::-1],
                    top_loadings.iloc[::-1],
                    color=colors
                )
                axes[idx].set_title(f'Top Loadings for {component_label}')
                axes[idx].set_xlabel('Loading Weight')
                axes[idx].set_ylabel('Descriptor')
                axes[idx].grid(True, axis='x', linewidth=0.3, alpha=0.4)

            plt.tight_layout()
            add_watermark_matplotlib_after_plot(fig)
            loadings_plot_path = os.path.join(temp_path, 'alvadesk_pca_loadings.png')
            plt.savefig(loadings_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            loadings_plot_path = None

        # Store results in session
        session['alvadesk_pca_performed'] = True
        session['alvadesk_selected_groups'] = selected_groups
        session['alvadesk_pca_summary'] = pca_summary
        session['alvadesk_pca_n_components'] = n_components
        session['alvadesk_pca_scale_data'] = scale_data
        session['alvadesk_group_summary'] = group_summary
        session['alvadesk_filtered_descriptors_count'] = len(descriptor_columns)

        # Create plot URL with timestamp
        timestamp = int(time.time() * 1000)
        session['alvadesk_pca_variance_plot'] = url_for('utils.serve_temp_image',
                                                         filename='alvadesk_pca_variance.png',
                                                         t=timestamp)
        if loadings_plot_path:
            session['alvadesk_pca_loadings_plot'] = url_for(
                'utils.serve_temp_image',
                filename='alvadesk_pca_loadings.png',
                t=timestamp + 1
            )
        else:
            session.pop('alvadesk_pca_loadings_plot', None)

        # Store color options for visualization
        session['alvadesk_pca_color_options'] = non_descriptor_cols

        # Save PCA model and related data for projection of new compounds
        pca_model_data = {
            'pca_model': pca_model,
            'scaler': scaler if scale_data else None,
            'descriptor_columns': descriptor_columns,
            'scale_data': scale_data,
            'n_components': n_components
        }
        pca_model_file = os.path.join(temp_path, 'alvadesk_pca_model.joblib')
        joblib.dump(pca_model_data, pca_model_file)
        session['alvadesk_pca_model_saved'] = True

        flash(f'Alvadesk PCA completed successfully using {len(descriptor_columns)} descriptors from {len(selected_groups)} groups!', 'success')

    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f'Error performing Alvadesk PCA: {str(e)}', 'danger')

    return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))


@alvadesk_pca_bp.route("/alvadesk_pca_visualize", methods=['POST'])
@nocache
def alvadesk_pca_visualize():
    """Generate PCA visualization with coloring and filtering options."""

    try:
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

        if not session.get('alvadesk_pca_performed'):
            error_msg = "Alvadesk PCA analysis has not been performed yet. Please run PCA first."
            if is_ajax:
                return jsonify({"status": "error", "message": error_msg})
            flash(error_msg, "warning")
            return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))

        # Get visualization parameters
        pc_x = int(request.form.get("pc_x", 1))
        pc_y = int(request.form.get("pc_y", 2))
        hover_column = request.form.get("hover_column", "")

        use_external_coloring = request.form.get("use_external_coloring") == "1"

        # Load PCA components
        temp_path = ensure_temp_dir()
        pca_file = os.path.join(temp_path, 'alvadesk_pca_components.csv')

        if not os.path.exists(pca_file):
            error_msg = "PCA components file not found. Please run PCA analysis again."
            if is_ajax:
                return jsonify({"status": "error", "message": error_msg})
            flash(error_msg, "danger")
            return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))

        pc_df = pd.read_csv(pca_file)

        # Normalize column names: remove newlines, tabs, and extra whitespace
        pc_df.columns = pc_df.columns.str.replace(r'[\n\r\t]+', ' ', regex=True).str.strip()

        # Validate PC columns exist
        pc_x_col = f'PC{pc_x}'
        pc_y_col = f'PC{pc_y}'

        if pc_x_col not in pc_df.columns or pc_y_col not in pc_df.columns:
            error_msg = f"Selected principal components not found in PCA results."
            if is_ajax:
                return jsonify({"status": "error", "message": error_msg})
            flash(error_msg, "danger")
            return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))

        # Prepare hover text
        hover_text = None
        if hover_column and hover_column in pc_df.columns:
            hover_text = pc_df[hover_column].tolist()

        # Initialize plot data
        plot_data = []
        color_data = None
        color_categories = None
        color_by = None
        external_data = {}

        if use_external_coloring:
            # External coloring from uploaded file
            key_column = request.form.get("external_key_column")
            color_column = request.form.get("external_color_column")

            if not key_column:
                error_msg = "External key column missing"
                if is_ajax:
                    return jsonify({"status": "error", "message": error_msg})
                flash(error_msg, "danger")
                return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))

            color_file_path = session.get('alvadesk_color_file_path')
            if not color_file_path or not os.path.exists(color_file_path):
                error_msg = "Color file not found. Please upload it again."
                if is_ajax:
                    return jsonify({"status": "error", "message": error_msg})
                flash(error_msg, "danger")
                return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))

            df_color = coloring.load_coloring_file(color_file_path)

            # Prepare coloring if color column is specified
            if color_column:
                color_result = coloring.prepare_color_data(pc_df, df_color, key_column, color_column)

                if not color_result['success']:
                    error_msg = color_result['message']
                    if is_ajax:
                        return jsonify({"status": "error", "message": error_msg})
                    flash(error_msg, "danger")
                    return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))

                color_series = color_result['data']
                is_numeric = color_result['is_numeric']

                global_indices = list(range(len(pc_df)))

                if is_numeric:
                    # Numeric coloring - separate N/A points for transparency
                    na_mask = color_series.isna()
                    valid_mask = ~na_mask

                    if na_mask.any():
                        # Split into valid data and N/A data
                        valid_pc_df = pc_df[valid_mask]
                        valid_color = color_series[valid_mask]

                        # Main trace with valid numeric colors
                        plot_data = [{
                            'x': valid_pc_df[pc_x_col].tolist(),
                            'y': valid_pc_df[pc_y_col].tolist(),
                            'indices': valid_pc_df.index.tolist(),
                            'hover_text': [hover_text[i] for i in valid_pc_df.index.tolist()] if hover_text else []
                        }]
                        color_data = valid_color.tolist()

                        # N/A trace with transparency
                        subset_na = pc_df[na_mask]
                        subset_na_indices = subset_na.index.tolist()
                        na_data = {
                            'label': 'N/A',
                            'data': [{'x': row[pc_x_col], 'y': row[pc_y_col]} for _, row in subset_na.iterrows()],
                            'color': '#CCCCCC',  # Gray color
                            'opacity': 0.4,  # 40% opacity
                            'indices': subset_na_indices
                        }
                        if hover_text:
                            na_data['hover_text'] = [hover_text[i] for i in subset_na_indices]
                        plot_data.append(na_data)
                    else:
                        # No N/A values - standard numeric coloring
                        plot_data = [{
                            'x': pc_df[pc_x_col].tolist(),
                            'y': pc_df[pc_y_col].tolist(),
                            'indices': global_indices,
                            'hover_text': hover_text or []
                        }]
                        color_data = color_series.tolist()

                    color_by = color_column
                else:
                    # Categorical coloring
                    unique_categories = color_series.dropna().unique()
                    n_categories = len(unique_categories)
                    color_palette = coloring.generate_distinct_colors(n_categories)
                    color_map = {cat: color_palette[i] for i, cat in enumerate(unique_categories)}
                    color_categories = unique_categories.tolist()

                    # Add categories with values
                    for category in unique_categories:
                        mask = color_series == category
                        subset = pc_df[mask]
                        subset_indices = subset.index.tolist()
                        category_data = {
                            'label': str(category),
                            'data': [{'x': row[pc_x_col], 'y': row[pc_y_col]} for _, row in subset.iterrows()],
                            'color': color_map[category],
                            'indices': subset_indices  # Add indices for external data mapping
                        }
                        if hover_text:
                            category_data['hover_text'] = [hover_text[i] for i in subset_indices]
                        plot_data.append(category_data)

                    # Add N/A category for missing values (gray color with transparency)
                    na_mask = color_series.isna()
                    if na_mask.any():
                        subset_na = pc_df[na_mask]
                        subset_na_indices = subset_na.index.tolist()
                        na_data = {
                            'label': 'N/A',
                            'data': [{'x': row[pc_x_col], 'y': row[pc_y_col]} for _, row in subset_na.iterrows()],
                            'color': '#CCCCCC',  # Gray color
                            'opacity': 0.4,  # 40% opacity
                            'indices': subset_na_indices
                        }
                        if hover_text:
                            na_data['hover_text'] = [hover_text[i] for i in subset_na_indices]
                        plot_data.append(na_data)

                    color_by = color_column
            else:
                # No coloring - basic plot
                plot_data = [{
                    'x': pc_df[pc_x_col].tolist(),
                    'y': pc_df[pc_y_col].tolist(),
                    'indices': list(range(len(pc_df))),
                    'hover_text': hover_text or []
                }]

            # Build external_data dictionary for filtering
            available_columns = [col for col in df_color.columns if col != key_column]
            for col_name in available_columns:
                try:
                    col_result = coloring.prepare_color_data(pc_df, df_color, key_column, col_name)
                    if col_result['success']:
                        col_series = col_result['data']
                        external_data[col_name] = col_series.tolist()
                except Exception as e:
                    print(f"Warning: Could not prepare data for column {col_name}: {e}")
                    continue
        else:
            # Internal coloring from PCA data
            color_by = request.form.get("color_by", "")

            if color_by and color_by in pc_df.columns:
                if pc_df[color_by].dtype == 'object' or pd.api.types.is_categorical_dtype(pc_df[color_by]):
                    # Categorical
                    color_series = pc_df[color_by]
                    unique_categories = color_series.dropna().unique()

                    # Add categories with values
                    for category in unique_categories:
                        mask = color_series == category
                        subset = pc_df[mask]
                        subset_indices = subset.index.tolist()
                        category_data = {
                            'label': str(category),
                            'data': [{'x': row[pc_x_col], 'y': row[pc_y_col]} for _, row in subset.iterrows()],
                            'indices': subset_indices  # Add indices for data mapping
                        }
                        if hover_text:
                            category_data['hover_text'] = [hover_text[i] for i in subset_indices]
                        plot_data.append(category_data)

                    # Add N/A category for missing values (gray color with transparency)
                    na_mask = color_series.isna()
                    if na_mask.any():
                        subset_na = pc_df[na_mask]
                        subset_na_indices = subset_na.index.tolist()
                        na_data = {
                            'label': 'N/A',
                            'data': [{'x': row[pc_x_col], 'y': row[pc_y_col]} for _, row in subset_na.iterrows()],
                            'color': '#CCCCCC',  # Gray color
                            'opacity': 0.4,  # 40% opacity
                            'indices': subset_na_indices
                        }
                        if hover_text:
                            na_data['hover_text'] = [hover_text[i] for i in subset_na_indices]
                        plot_data.append(na_data)

                    color_categories = unique_categories.tolist()
                else:
                    # Numeric - separate N/A points for transparency
                    color_series = pc_df[color_by]
                    na_mask = color_series.isna()
                    valid_mask = ~na_mask

                    if na_mask.any():
                        # Split into valid data and N/A data
                        valid_pc_df = pc_df[valid_mask]
                        valid_color = color_series[valid_mask]

                        # Main trace with valid numeric colors
                        plot_data = [{
                            'x': valid_pc_df[pc_x_col].tolist(),
                            'y': valid_pc_df[pc_y_col].tolist(),
                            'indices': valid_pc_df.index.tolist(),
                            'hover_text': [hover_text[i] for i in valid_pc_df.index.tolist()] if hover_text else []
                        }]
                        color_data = valid_color.tolist()

                        # N/A trace with transparency
                        subset_na = pc_df[na_mask]
                        subset_na_indices = subset_na.index.tolist()
                        na_data = {
                            'label': 'N/A',
                            'data': [{'x': row[pc_x_col], 'y': row[pc_y_col]} for _, row in subset_na.iterrows()],
                            'color': '#CCCCCC',  # Gray color
                            'opacity': 0.4,  # 40% opacity
                            'indices': subset_na_indices
                        }
                        if hover_text:
                            na_data['hover_text'] = [hover_text[i] for i in subset_na_indices]
                        plot_data.append(na_data)
                    else:
                        # No N/A values - standard numeric coloring
                        plot_data = [{
                            'x': pc_df[pc_x_col].tolist(),
                            'y': pc_df[pc_y_col].tolist(),
                            'indices': list(range(len(pc_df))),
                            'hover_text': hover_text or []
                        }]
                        color_data = pc_df[color_by].tolist()
            else:
                # No coloring
                plot_data = [{
                    'x': pc_df[pc_x_col].tolist(),
                    'y': pc_df[pc_y_col].tolist(),
                    'indices': list(range(len(pc_df))),
                    'hover_text': hover_text or []
                }]

        # Get variance explained
        pca_summary = session.get('alvadesk_pca_summary', [])
        variance_x = pca_summary[pc_x-1]['explained_variance'] if pc_x <= len(pca_summary) else 0
        variance_y = pca_summary[pc_y-1]['explained_variance'] if pc_y <= len(pca_summary) else 0

        success_msg = f"Alvadesk PCA scatter plot created: PC{pc_x} vs PC{pc_y}"
        if use_external_coloring and color_by:
            success_msg += f" with external coloring from column '{color_by}'"

        # Get raw PC data for filtering
        pc_x_raw = pc_df[pc_x_col].tolist()
        pc_y_raw = pc_df[pc_y_col].tolist()

        if is_ajax:
            response_data = {
                "status": "success",
                "message": success_msg,
                "pc_x": pc_x,
                "pc_y": pc_y,
                "variance_x": round(variance_x, 2),
                "variance_y": round(variance_y, 2),
                "plot_data": plot_data,
                "color_data": color_data,
                "color_categories": color_categories,
                "color_by": color_by,
                "hover_column": hover_column,
                "pc_x_raw": pc_x_raw,
                "pc_y_raw": pc_y_raw,
                "external_data": external_data,
                "use_external_coloring": use_external_coloring
            }

            return jsonify(response_data)

        flash(success_msg, "success")
        return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))

    except Exception as e:
        import traceback
        traceback.print_exc()

        error_msg = f"Visualization error: {str(e)}"
        if is_ajax:
            return jsonify({"status": "error", "message": error_msg})
        flash(error_msg, "danger")
        return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))


@alvadesk_pca_bp.route("/save_alvadesk_colored_plot", methods=['POST'])
def save_alvadesk_colored_plot():
    """Create a static colored PCA plot using the current visualization state."""

    if not session.get('alvadesk_pca_performed'):
        return jsonify({"status": "error", "message": "Please run Alvadesk PCA analysis first."}), 400

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"status": "error", "message": "Invalid request payload."}), 400

    plot_data = payload.get('plot_data') or []
    if not plot_data:
        return jsonify({"status": "error", "message": "No plot data provided."}), 400

    try:
        pc_x = payload.get('pc_x', 1)
        pc_y = payload.get('pc_y', 2)
        variance_x = payload.get('variance_x', 0)
        variance_y = payload.get('variance_y', 0)
        color_by = payload.get('color_by')
        title = payload.get('title') or f"PC{pc_x} vs PC{pc_y}"
        color_categories = payload.get('color_categories')
        color_data = payload.get('color_data')
        filter_info = payload.get('filter_info')

        fig, ax = plt.subplots(figsize=(12, 9))
        points_plotted = 0

        if color_categories:
            palette = plt.cm.get_cmap('tab20', max(len(plot_data), 1))
            for idx, category in enumerate(plot_data):
                points = category.get('data', [])
                if not points:
                    continue
                xs = [float(point.get('x', 0.0)) for point in points]
                ys = [float(point.get('y', 0.0)) for point in points]
                if not xs:
                    continue
                raw_color = category.get('color')
                if raw_color:
                    color = raw_color
                else:
                    rgba = palette(idx % palette.N)
                    color = (rgba[0], rgba[1], rgba[2])
                ax.scatter(xs, ys, s=40, label=category.get('label') or f'Group {idx + 1}',
                           color=color, edgecolors='white', linewidths=0.3)
                points_plotted += len(xs)
            if points_plotted and len(plot_data) > 1:
                ax.legend(loc='best', fontsize=8, frameon=True)
        else:
            dataset = plot_data[0]
            xs = [float(x) for x in dataset.get('x', [])]
            ys = [float(y) for y in dataset.get('y', [])]
            points_plotted = len(xs)
            if color_data:
                colors = []
                for value in color_data[:len(xs)]:
                    try:
                        colors.append(float(value))
                    except (TypeError, ValueError):
                        colors.append(float('nan'))
                scatter = ax.scatter(xs, ys, c=colors, cmap='viridis', s=40,
                                     edgecolors='white', linewidths=0.3)
                cbar = fig.colorbar(scatter, ax=ax)
                cbar.set_label(color_by or 'Value')
            else:
                ax.scatter(xs, ys, s=40, color='#2563eb', edgecolors='white', linewidths=0.3)

        ax.set_xlabel(f'PC{pc_x} ({variance_x:.2f}% variance)')
        ax.set_ylabel(f'PC{pc_y} ({variance_y:.2f}% variance)')
        ax.set_title(title)
        ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.4)
        ax.set_aspect('equal', adjustable='box')

        if filter_info:
            filtered_points = filter_info.get('filtered_points')
            total_points = filter_info.get('total_points')
            column = filter_info.get('column')
            info_lines = []
            if filtered_points is not None and total_points is not None:
                info_lines.append(f'Filtered: {filtered_points}/{total_points} points')
            if column:
                info_lines.append(f'Column: {column}')
            if info_lines:
                ax.text(
                    0.02,
                    0.02,
                    '\n'.join(info_lines),
                    transform=ax.transAxes,
                    fontsize=9,
                    color='#374151',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.65, edgecolor='none')
                )

        # Draw biplot arrows if provided
        biplot_arrows = payload.get('biplot_arrows')
        print(f"DEBUG: biplot_arrows received: {biplot_arrows is not None}")
        if biplot_arrows:
            print(f"DEBUG: arrows_by_pc keys: {biplot_arrows.get('arrows_by_pc', {}).keys()}")
            arrows_by_pc = biplot_arrows.get('arrows_by_pc', {})
            show_labels = biplot_arrows.get('show_labels', False)

            # Get data limits for scaling
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            max_x = max(abs(xlim[0]), abs(xlim[1]))
            max_y = max(abs(ylim[0]), abs(ylim[1]))

            # Find max loading for scaling
            max_loading = 0
            for pc_arrows in arrows_by_pc.values():
                for arrow in pc_arrows:
                    loading_mag = np.sqrt(arrow['x']**2 + arrow['y']**2)
                    if loading_mag > max_loading:
                        max_loading = loading_mag

            # Scale factor
            target_max_length = min(max_x, max_y) * 0.8
            scale_factor = target_max_length / max_loading if max_loading > 0 else 1

            # Color mapping for PCs
            pc_colors = {
                pc_x: (1.0, 0.0, 0.0, 0.2),    # Red with alpha
                pc_y: (0.0, 0.0, 1.0, 0.2)     # Blue with alpha
            }
            pc_colors_head = {
                pc_x: (1.0, 0.0, 0.0, 0.35),   # Red head
                pc_y: (0.0, 0.0, 1.0, 0.35)    # Blue head
            }

            # Draw arrows for each PC
            arrow_count = 0
            print(f"DEBUG: arrows_by_pc type: {type(arrows_by_pc)}")
            print(f"DEBUG: arrows_by_pc content: {arrows_by_pc}")

            for pc_str, arrows in arrows_by_pc.items():
                print(f"DEBUG: Processing pc_str='{pc_str}' (type: {type(pc_str)})")
                pc_num = int(pc_str)
                shaft_color = pc_colors.get(pc_num, (1.0, 0.0, 1.0, 0.2))
                head_color = pc_colors_head.get(pc_num, (1.0, 0.0, 1.0, 0.35))
                print(f"DEBUG: Drawing {len(arrows)} arrows for PC{pc_num}, shaft_color={shaft_color}")

                for arrow in arrows:
                    arrow_count += 1
                    arrow_x = arrow['x'] * scale_factor
                    arrow_y = arrow['y'] * scale_factor

                    print(f"DEBUG: Drawing arrow '{arrow.get('name', 'unknown')}' at ({arrow_x:.3f}, {arrow_y:.3f})")

                    # Calculate arrowhead size
                    arrow_length = np.sqrt(arrow_x**2 + arrow_y**2)
                    head_length = min(arrow_length * 0.08, target_max_length * 0.015)
                    head_width = head_length * 0.6

                    # Use FancyArrowPatch for better rendering
                    arrow_patch = FancyArrowPatch(
                        (0, 0), (arrow_x, arrow_y),
                        arrowstyle=f'->,head_width={head_width},head_length={head_length}',
                        color=shaft_color[:3],  # RGB only (no alpha in color)
                        alpha=shaft_color[3],   # Alpha separately
                        linewidth=1.2,
                        zorder=10  # Draw on top of data points
                    )
                    ax.add_patch(arrow_patch)

                    # Add label if requested
                    if show_labels:
                        loading_mag = np.sqrt(arrow['x']**2 + arrow['y']**2)
                        label_text = f"{arrow['name']}\n({loading_mag:.3f})"
                        label_color = (head_color[0], head_color[1], head_color[2])  # RGB without alpha
                        ax.text(arrow_x * 1.1, arrow_y * 1.1, label_text,
                               fontsize=7, color=label_color, ha='center', va='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                       alpha=0.7, edgecolor='none'))

            print(f"DEBUG: Total arrows drawn: {arrow_count}")

        temp_path = ensure_temp_dir()
        filename = f"alvadesk_pca_colored_{int(time.time() * 1000)}.png"
        save_path = os.path.join(temp_path, filename)
        fig.tight_layout()
        add_watermark_matplotlib_after_plot(fig)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        image_url = url_for('utils.serve_temp_image', filename=filename, t=int(time.time() * 1000))
        return jsonify({
            "status": "success",
            "image_url": image_url,
            "points": points_plotted
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@alvadesk_pca_bp.route("/upload_alvadesk_color_file", methods=['POST'])
def upload_alvadesk_color_file():
    """Upload external color file for Alvadesk PCA visualization."""

    try:
        if 'color_file' not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"})

        file = request.files['color_file']

        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"})

        # Save file temporarily
        temp_path = ensure_temp_dir()
        color_file_path = os.path.join(temp_path, 'alvadesk_color_' + secure_filename(file.filename))
        file.save(color_file_path)

        # Save path in session
        session['alvadesk_color_file_path'] = color_file_path

        # Load color file
        df_color = coloring.load_coloring_file(color_file_path)

        if df_color is None:
            return jsonify({
                "status": "error",
                "message": "Failed to load color file. Please check the file format."
            })

        # Load PCA data
        pca_file = os.path.join(temp_path, 'alvadesk_pca_components.csv')
        if not os.path.exists(pca_file):
            return jsonify({
                "status": "error",
                "message": "PCA data not found. Please run Alvadesk PCA analysis first."
            })

        df_pca = pd.read_csv(pca_file)

        # Normalize column names: remove newlines, tabs, and extra whitespace
        df_pca.columns = df_pca.columns.str.replace(r'[\n\r\t]+', ' ', regex=True).str.strip()

        # Detect key column
        detection = coloring.detect_key_column(df_pca, df_color)

        # Validate setup
        validation = coloring.validate_coloring_setup(
            df_pca,
            df_color,
            detection['key_column'] if detection['found'] else None
        )

        # Prepare color column info
        color_column_info = {}
        if validation['available_color_columns']:
            for col in validation['available_color_columns']:
                is_numeric = pd.api.types.is_numeric_dtype(df_color[col])
                n_unique = df_color[col].nunique()
                color_column_info[col] = {
                    'is_numeric': is_numeric,
                    'n_unique': n_unique
                }

        # Prepare merged data for all columns
        merged_data = {}
        if detection['found'] and detection['key_column']:
            key_column = detection['key_column']
            available_columns = [col for col in df_color.columns if col != key_column]

            for col_name in available_columns:
                try:
                    col_result = coloring.prepare_color_data(df_pca, df_color, key_column, col_name)
                    if col_result['success']:
                        col_series = col_result['data']
                        merged_data[col_name] = col_series.tolist()
                except Exception as e:
                    print(f"Warning: Could not prepare merged data for column {col_name}: {e}")
                    continue

        # Save in session
        session['alvadesk_color_file_loaded'] = True
        session['alvadesk_color_detection'] = detection
        session['alvadesk_color_validation'] = validation
        session['alvadesk_color_column_info'] = color_column_info
        session['alvadesk_color_merged_data'] = merged_data

        response_data = {
            "status": "success",
            "auto_detected": detection['found'],
            "key_column": detection['key_column'],
            "common_columns": detection.get('common_columns', []),
            "available_color_columns": validation['available_color_columns'],
            "color_column_info": color_column_info,
            "merged_data": merged_data,
            "message": validation['recommendations'][0] if validation['recommendations'] else "File uploaded successfully",
            "warnings": validation['warnings']
        }

        return jsonify(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Error processing color file: {str(e)}"
        })


@alvadesk_pca_bp.route("/confirm_alvadesk_key_column", methods=['POST'])
def confirm_alvadesk_key_column():
    """Confirm key column selection for external coloring."""

    try:
        key_column = request.form.get('key_column')

        if not key_column:
            return jsonify({"status": "error", "message": "No key column provided"})

        color_file_path = session.get('alvadesk_color_file_path')
        if not color_file_path:
            return jsonify({
                "status": "error",
                "message": "Color file not found. Please upload the file again."
            })

        # Load files
        df_color = coloring.load_coloring_file(color_file_path)
        temp_path = ensure_temp_dir()
        pca_file = os.path.join(temp_path, 'alvadesk_pca_components.csv')
        df_pca = pd.read_csv(pca_file)

        # Normalize column names: remove newlines, tabs, and extra whitespace
        df_pca.columns = df_pca.columns.str.replace(r'[\n\r\t]+', ' ', regex=True).str.strip()

        # Validate with selected key column
        validation = coloring.validate_coloring_setup(df_pca, df_color, key_column)

        if not validation['valid']:
            return jsonify({"status": "error", "message": "Invalid key column selection"})

        # Prepare color column info
        color_column_info = {}
        for col in validation['available_color_columns']:
            is_numeric = pd.api.types.is_numeric_dtype(df_color[col])
            n_unique = df_color[col].nunique()
            color_column_info[col] = {
                'is_numeric': is_numeric,
                'n_unique': n_unique
            }

        # Prepare merged data
        merged_data = {}
        available_columns = [col for col in df_color.columns if col != key_column]

        for col_name in available_columns:
            try:
                col_result = coloring.prepare_color_data(df_pca, df_color, key_column, col_name)
                if col_result['success']:
                    col_series = col_result['data']
                    merged_data[col_name] = col_series.tolist()
            except Exception as e:
                print(f"Warning: Could not prepare merged data for column {col_name}: {e}")
                continue

        # Update session
        session['alvadesk_color_key_column'] = key_column
        session['alvadesk_color_validation'] = validation
        session['alvadesk_color_column_info'] = color_column_info
        session['alvadesk_color_merged_data'] = merged_data

        return jsonify({
            "status": "success",
            "key_column": key_column,
            "available_color_columns": validation['available_color_columns'],
            "color_column_info": color_column_info,
            "merged_data": merged_data,
            "message": f"Key column '{key_column}' confirmed successfully"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Error confirming key column: {str(e)}"
        })


@alvadesk_pca_bp.route("/download_alvadesk_pca_scores")
def download_alvadesk_pca_scores():
    """Download Alvadesk PCA scores (projected data) as CSV."""

    if not session.get('alvadesk_pca_performed'):
        flash('No Alvadesk PCA analysis results available', 'danger')
        return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))

    try:
        temp_path = ensure_temp_dir()
        pca_file = os.path.join(temp_path, 'alvadesk_pca_components.csv')

        if not os.path.exists(pca_file):
            flash('PCA scores file not found', 'danger')
            return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))

        return send_file(pca_file, as_attachment=True, download_name='alvadesk_pca_scores.csv')

    except Exception as e:
        flash(f'Error downloading file: {str(e)}', 'danger')
        return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))


@alvadesk_pca_bp.route("/download_alvadesk_pca_eigenvalues")
def download_alvadesk_pca_eigenvalues():
    """Download Alvadesk PCA eigenvalues as CSV."""

    if not session.get('alvadesk_pca_performed'):
        flash('No Alvadesk PCA analysis results available', 'danger')
        return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))

    try:
        temp_path = ensure_temp_dir()
        eigenvalues_file = os.path.join(temp_path, 'alvadesk_pca_eigenvalues.csv')

        if not os.path.exists(eigenvalues_file):
            flash('Eigenvalues file not found', 'danger')
            return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))

        return send_file(eigenvalues_file, as_attachment=True, download_name='alvadesk_pca_eigenvalues.csv')

    except Exception as e:
        flash(f'Error downloading file: {str(e)}', 'danger')
        return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))


@alvadesk_pca_bp.route("/download_alvadesk_pca_explained_variance")
def download_alvadesk_pca_explained_variance():
    """Download Alvadesk PCA explained variance as CSV."""

    if not session.get('alvadesk_pca_performed'):
        flash('No Alvadesk PCA analysis results available', 'danger')
        return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))

    try:
        temp_path = ensure_temp_dir()
        explained_variance_file = os.path.join(temp_path, 'alvadesk_pca_explained_variance.csv')

        if not os.path.exists(explained_variance_file):
            flash('Explained variance file not found', 'danger')
            return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))

        return send_file(explained_variance_file, as_attachment=True, download_name='alvadesk_pca_explained_variance.csv')

    except Exception as e:
        flash(f'Error downloading file: {str(e)}', 'danger')
        return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))


@alvadesk_pca_bp.route("/download_alvadesk_pca_loadings")
def download_alvadesk_pca_loadings():
    """Download Alvadesk PCA loadings as CSV (PC on X-axis, variables on Y-axis)."""

    if not session.get('alvadesk_pca_performed'):
        flash('No Alvadesk PCA analysis results available', 'danger')
        return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))

    try:
        temp_path = ensure_temp_dir()
        loadings_file = os.path.join(temp_path, 'alvadesk_pca_loadings.csv')

        if not os.path.exists(loadings_file):
            flash('Loadings file not found', 'danger')
            return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))

        return send_file(loadings_file, as_attachment=True, download_name='alvadesk_pca_loadings.csv')

    except Exception as e:
        flash(f'Error downloading file: {str(e)}', 'danger')
        return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))


@alvadesk_pca_bp.route("/download_alvadesk_pca_all_metrics")
def download_alvadesk_pca_all_metrics():
    """Download all Alvadesk PCA metrics in one Excel file with 4 sheets."""

    if not session.get('alvadesk_pca_performed'):
        flash('No Alvadesk PCA analysis results available', 'danger')
        return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))

    try:
        temp_path = ensure_temp_dir()

        # Load all data files
        scores_file = os.path.join(temp_path, 'alvadesk_pca_components.csv')
        eigenvalues_file = os.path.join(temp_path, 'alvadesk_pca_eigenvalues.csv')
        explained_variance_file = os.path.join(temp_path, 'alvadesk_pca_explained_variance.csv')
        loadings_file = os.path.join(temp_path, 'alvadesk_pca_loadings.csv')

        # Check if all files exist
        if not all(os.path.exists(f) for f in [scores_file, eigenvalues_file, explained_variance_file, loadings_file]):
            flash('Some PCA metric files not found. Please run PCA analysis again.', 'danger')
            return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))

        # Read all data
        scores_df = pd.read_csv(scores_file)
        eigenvalues_df = pd.read_csv(eigenvalues_file)
        explained_variance_df = pd.read_csv(explained_variance_file)
        loadings_df = pd.read_csv(loadings_file, index_col=0)

        # Create Excel file with multiple sheets
        excel_file = os.path.join(temp_path, 'alvadesk_pca_all_metrics.xlsx')

        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            scores_df.to_excel(writer, sheet_name='PCA_Scores', index=False)
            eigenvalues_df.to_excel(writer, sheet_name='Eigenvalues', index=False)
            explained_variance_df.to_excel(writer, sheet_name='Explained_Variance', index=False)
            loadings_df.to_excel(writer, sheet_name='Loadings')

        return send_file(excel_file, as_attachment=True, download_name='alvadesk_pca_all_metrics.xlsx')

    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f'Error downloading file: {str(e)}', 'danger')
        return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))


@alvadesk_pca_bp.route("/generate_biplot_data", methods=['POST'])
def generate_biplot_data():
    """Generate biplot data with top N arrows for each visible PC component."""

    if not session.get('alvadesk_pca_performed'):
        return jsonify({
            "status": "error",
            "message": "Please run Alvadesk PCA analysis first."
        }), 400

    try:
        # Get parameters from request
        pc_x = int(request.form.get("pc_x", 1))
        pc_y = int(request.form.get("pc_y", 2))
        top_n_arrows = int(request.form.get("top_n_arrows", 10))

        # Limit top_n_arrows to reasonable range
        top_n_arrows = max(1, min(top_n_arrows, 100))

        # Load loadings data
        temp_path = ensure_temp_dir()
        loadings_file = os.path.join(temp_path, 'alvadesk_pca_loadings.csv')

        if not os.path.exists(loadings_file):
            return jsonify({
                "status": "error",
                "message": "Loadings file not found. Please run PCA analysis again."
            }), 404

        loadings_df = pd.read_csv(loadings_file, index_col=0)

        # Validate PC columns
        pc_x_col = f'PC{pc_x}'
        pc_y_col = f'PC{pc_y}'

        if pc_x_col not in loadings_df.columns or pc_y_col not in loadings_df.columns:
            return jsonify({
                "status": "error",
                "message": f"Selected principal components not found in loadings data."
            }), 400

        # Get top N descriptors for EACH visible PC component
        arrows_by_pc = {}

        # For PC X axis
        pc_x_top = loadings_df[pc_x_col].abs().nlargest(top_n_arrows)
        arrows_by_pc[pc_x] = []
        for descriptor in pc_x_top.index:
            arrows_by_pc[pc_x].append({
                'name': descriptor,
                'x': float(loadings_df.loc[descriptor, pc_x_col]),
                'y': float(loadings_df.loc[descriptor, pc_y_col]),
                'loading_value': float(loadings_df.loc[descriptor, pc_x_col]),
                'pc': pc_x
            })

        # For PC Y axis (only if different from X)
        if pc_y != pc_x:
            pc_y_top = loadings_df[pc_y_col].abs().nlargest(top_n_arrows)
            arrows_by_pc[pc_y] = []
            for descriptor in pc_y_top.index:
                arrows_by_pc[pc_y].append({
                    'name': descriptor,
                    'x': float(loadings_df.loc[descriptor, pc_x_col]),
                    'y': float(loadings_df.loc[descriptor, pc_y_col]),
                    'loading_value': float(loadings_df.loc[descriptor, pc_y_col]),
                    'pc': pc_y
                })

        # Also calculate combined magnitude for reference
        loadings_df['magnitude'] = np.sqrt(
            loadings_df[pc_x_col]**2 + loadings_df[pc_y_col]**2
        )
        top_combined = loadings_df.nlargest(top_n_arrows, 'magnitude')

        combined_arrows = []
        for descriptor, row in top_combined.iterrows():
            combined_arrows.append({
                'name': descriptor,
                'x': float(row[pc_x_col]),
                'y': float(row[pc_y_col]),
                'magnitude': float(row['magnitude'])
            })

        return jsonify({
            "status": "success",
            "arrows_by_pc": arrows_by_pc,
            "combined_arrows": combined_arrows,
            "pc_x": pc_x,
            "pc_y": pc_y,
            "top_n": top_n_arrows,
            "total_descriptors": len(loadings_df)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Error generating biplot data: {str(e)}"
        }), 500


@alvadesk_pca_bp.route("/alvadesk_pca_project_new", methods=['POST'])
@nocache
def alvadesk_pca_project_new():
    """Project new compounds onto existing PCA space."""

    if not session.get('alvadesk_pca_performed'):
        return jsonify({
            "status": "error",
            "message": "Please run Alvadesk PCA analysis first."
        }), 400

    if not session.get('alvadesk_pca_model_saved'):
        return jsonify({
            "status": "error",
            "message": "PCA model not saved. Please run PCA analysis again."
        }), 400

    try:
        # Check if file was uploaded
        if 'new_compounds_file' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No file uploaded."
            }), 400

        file = request.files['new_compounds_file']
        if file.filename == '':
            return jsonify({
                "status": "error",
                "message": "No file selected."
            }), 400

        # Load the saved PCA model
        temp_path = ensure_temp_dir()
        pca_model_file = os.path.join(temp_path, 'alvadesk_pca_model.joblib')

        if not os.path.exists(pca_model_file):
            return jsonify({
                "status": "error",
                "message": "PCA model file not found. Please run PCA analysis again."
            }), 404

        pca_model_data = joblib.load(pca_model_file)
        pca_model = pca_model_data['pca_model']
        scaler = pca_model_data['scaler']
        descriptor_columns = pca_model_data['descriptor_columns']
        scale_data = pca_model_data['scale_data']
        n_components = pca_model_data['n_components']

        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        new_file_path = os.path.join(temp_path, f'new_compounds_{filename}')
        file.save(new_file_path)

        # Read the new compounds file
        if filename.lower().endswith('.xlsx') or filename.lower().endswith('.xls'):
            new_df = pd.read_excel(new_file_path)
        else:
            # Try different separators
            try:
                new_df = pd.read_csv(new_file_path)
            except:
                new_df = pd.read_csv(new_file_path, sep='\t')

        # Check if all required descriptor columns are present
        missing_cols = [col for col in descriptor_columns if col not in new_df.columns]

        if missing_cols:
            return jsonify({
                "status": "error",
                "message": f"Missing {len(missing_cols)} required descriptor columns. First missing: {missing_cols[:5]}",
                "missing_columns": missing_cols[:20]
            }), 400

        # Get label column if specified
        label_column = request.form.get('label_column', '')

        # Get color column (must match the column used for coloring original PCA)
        color_by = request.form.get('color_by', '') or ''
        # Get filter column (must match the column used for filtering original PCA, if any)
        filter_by = request.form.get('filter_by', '') or ''

        # Get color map from original visualization for unified colors
        original_color_map = None
        color_map_json = request.form.get('color_map', '')
        if color_map_json:
            try:
                original_color_map = json.loads(color_map_json)
            except:
                pass

        # Get color range from original visualization for unified numeric scale
        original_color_range = None
        color_range_json = request.form.get('color_range', '')
        if color_range_json:
            try:
                original_color_range = json.loads(color_range_json)
            except:
                pass

        # Prepare data for projection
        X_new = new_df[descriptor_columns].select_dtypes(include=[np.number])

        # Handle missing values
        if X_new.isnull().any().any():
            X_new = X_new.fillna(X_new.mean())

        # Scale data if the original PCA used scaling
        if scale_data and scaler is not None:
            X_new_scaled = scaler.transform(X_new)
        else:
            X_new_scaled = X_new.values

        # Project onto PCA space
        new_pc_scores = pca_model.transform(X_new_scaled)

        # Create PC DataFrame
        pc_columns = [f'PC{i+1}' for i in range(n_components)]
        new_pc_df = pd.DataFrame(new_pc_scores, columns=pc_columns, index=new_df.index)

        # Add label column if specified
        if label_column and label_column in new_df.columns:
            new_pc_df['label'] = new_df[label_column].values

        # Optional: extract color and filter values if columns are specified
        # Priority 1: separate color file for new compounds (if provided)
        # Priority 2: columns present directly in new compounds file
        color_values = None
        filter_values = None
        df_color_new = None

        # Try dedicated color/filter file for new compounds once, if needed
        if color_by or filter_by:
            color_file = request.files.get('new_compounds_color_file')
            if color_file and color_file.filename:
                color_filename = secure_filename(color_file.filename)
                color_file_path = os.path.join(temp_path, f'new_compounds_color_{color_filename}')
                color_file.save(color_file_path)

                # Read color file
                if color_filename.lower().endswith('.xlsx') or color_filename.lower().endswith('.xls'):
                    df_color_new = pd.read_excel(color_file_path)
                else:
                    try:
                        df_color_new = pd.read_csv(color_file_path)
                    except Exception:
                        df_color_new = pd.read_csv(color_file_path, sep='\t')

                # Validate: same number of rows
                if len(df_color_new) != len(new_df):
                    return jsonify({
                        "status": "error",
                        "message": "New compounds color file must have the same number of rows as the new compounds file."
                    }), 400

        # Resolve color values
        if color_by:
            if df_color_new is not None:
                if color_by not in df_color_new.columns:
                    return jsonify({
                        "status": "error",
                        "message": f"New compounds color file must contain the column '{color_by}' used for coloring the original PCA plot."
                    }), 400
                color_values = df_color_new[color_by].tolist()
            elif color_by in new_df.columns:
                color_values = new_df[color_by].tolist()
            else:
                # If the color column is missing everywhere, ignore coloring for new compounds
                color_by = ''

        # Resolve filter values (same filter column as in Step 2, if provided)
        if filter_by:
            if df_color_new is not None and filter_by in df_color_new.columns:
                filter_values = df_color_new[filter_by].tolist()
            elif filter_by in new_df.columns:
                filter_values = new_df[filter_by].tolist()

        # Determine color type (categorical, numeric, or none)
        color_type = 'none'
        color_categories = None
        color_map = None

        if color_by and color_values is not None:
            color_series = pd.Series(color_values)

            # Use existing function to check if numeric
            from app.chemalize.visualization.coloring import _coerce_numeric_series
            coerced, is_numeric = _coerce_numeric_series(color_series)

            if is_numeric:
                color_type = 'numeric'
                # Update color_values to numeric
                color_values = coerced.tolist()
            else:
                color_type = 'categorical'
                # Get unique categories
                color_categories = color_series.dropna().unique().tolist()

                # Use original color map if provided, otherwise generate new one
                if original_color_map:
                    color_map = original_color_map
                else:
                    # Generate color map matching original PCA visualization
                    from app.chemalize.visualization.coloring import generate_distinct_colors
                    n_categories = len(color_categories)
                    color_palette = generate_distinct_colors(n_categories)
                    color_map = {cat: color_palette[i] for i, cat in enumerate(color_categories)}

        # Save projected scores
        new_scores_file = os.path.join(temp_path, 'alvadesk_pca_new_compounds.csv')
        new_pc_df.to_csv(new_scores_file, index=False)

        # Store in session
        session['alvadesk_pca_new_projected'] = True
        session['alvadesk_pca_new_label_column'] = label_column if label_column and label_column in new_df.columns else None
        session['alvadesk_pca_new_count'] = len(new_df)

        # Return data for plotting - structured by category for unified legend
        plot_data = []

        if color_type == 'categorical':
            # Group by category for unified legend
            for category in color_categories:
                category_points = []
                for i, (idx, row) in enumerate(new_pc_df.iterrows()):
                    if color_values[i] == category:
                        point = {col: float(row[col]) for col in pc_columns}
                        if 'label' in new_pc_df.columns:
                            point['label'] = str(row['label'])
                        else:
                            point['label'] = f'New_{idx}'

                        if filter_values is not None:
                            point['filter_value'] = filter_values[i]

                        category_points.append(point)

                if category_points:
                    plot_data.append({
                        'category': category,
                        'color': color_map[category],
                        'points': category_points
                    })

            # Add N/A category if present
            na_points = []
            for i, (idx, row) in enumerate(new_pc_df.iterrows()):
                if pd.isna(color_values[i]):
                    point = {col: float(row[col]) for col in pc_columns}
                    if 'label' in new_pc_df.columns:
                        point['label'] = str(row['label'])
                    else:
                        point['label'] = f'New_{idx}'

                    if filter_values is not None:
                        point['filter_value'] = filter_values[i]

                    na_points.append(point)

            if na_points:
                plot_data.append({
                    'category': 'N/A',
                    'color': '#CCCCCC',
                    'opacity': 0.4,
                    'points': na_points
                })

        elif color_type == 'numeric':
            # Single group with color values
            points = []
            numeric_colors = []

            for i, (idx, row) in enumerate(new_pc_df.iterrows()):
                point = {col: float(row[col]) for col in pc_columns}
                if 'label' in new_pc_df.columns:
                    point['label'] = str(row['label'])
                else:
                    point['label'] = f'New_{idx}'

                if filter_values is not None:
                    point['filter_value'] = filter_values[i]

                # Only add if not N/A
                if not pd.isna(color_values[i]):
                    points.append(point)
                    numeric_colors.append(color_values[i])

            if points:
                plot_data.append({
                    'category': None,
                    'points': points,
                    'color_values': numeric_colors
                })

            # Handle N/A values separately
            na_points = []
            for i, (idx, row) in enumerate(new_pc_df.iterrows()):
                if pd.isna(color_values[i]):
                    point = {col: float(row[col]) for col in pc_columns}
                    if 'label' in new_pc_df.columns:
                        point['label'] = str(row['label'])
                    else:
                        point['label'] = f'New_{idx}'

                    if filter_values is not None:
                        point['filter_value'] = filter_values[i]

                    na_points.append(point)

            if na_points:
                plot_data.append({
                    'category': 'N/A',
                    'color': '#CCCCCC',
                    'opacity': 0.4,
                    'points': na_points
                })

        else:  # color_type == 'none'
            # No coloring - all points in single group
            points = []
            for i, (idx, row) in enumerate(new_pc_df.iterrows()):
                point = {col: float(row[col]) for col in pc_columns}
                if 'label' in new_pc_df.columns:
                    point['label'] = str(row['label'])
                else:
                    point['label'] = f'New_{idx}'

                if filter_values is not None:
                    point['filter_value'] = filter_values[i]

                points.append(point)

            plot_data.append({
                'category': None,
                'points': points
            })

        return jsonify({
            "status": "success",
            "message": f"Successfully projected {len(new_df)} new compounds.",
            "n_compounds": len(new_df),
            "plot_data": plot_data,
            "pc_columns": pc_columns,
            "label_column": label_column if label_column and label_column in new_df.columns else None,
            "color_by": color_by or None,
            "color_type": color_type,           # 'categorical', 'numeric', or 'none'
            "color_categories": color_categories,  # list of category names (categorical only)
            "color_map": color_map,             # {category: hex_color} (categorical only)
            "color_range": original_color_range,  # {min: float, max: float} (numeric only)
            "filter_by": filter_by or None
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Error projecting new compounds: {str(e)}"
        }), 500


@alvadesk_pca_bp.route("/alvadesk_pca_get_projected", methods=['GET'])
@nocache
def alvadesk_pca_get_projected():
    """Get projected new compounds data for visualization."""

    if not session.get('alvadesk_pca_new_projected'):
        return jsonify({
            "status": "error",
            "message": "No new compounds have been projected yet."
        }), 400

    try:
        temp_path = ensure_temp_dir()
        new_scores_file = os.path.join(temp_path, 'alvadesk_pca_new_compounds.csv')

        if not os.path.exists(new_scores_file):
            return jsonify({
                "status": "error",
                "message": "Projected scores file not found."
            }), 404

        new_pc_df = pd.read_csv(new_scores_file)
        n_components = session.get('alvadesk_pca_n_components', 2)
        pc_columns = [f'PC{i+1}' for i in range(n_components)]

        # Prepare plot data
        plot_data = []
        for idx, row in new_pc_df.iterrows():
            point = {col: float(row[col]) for col in pc_columns if col in new_pc_df.columns}
            if 'label' in new_pc_df.columns:
                point['label'] = str(row['label'])
            else:
                point['label'] = f'New_{idx}'
            plot_data.append(point)

        return jsonify({
            "status": "success",
            "plot_data": plot_data,
            "n_compounds": len(plot_data),
            "label_column": session.get('alvadesk_pca_new_label_column')
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Error retrieving projected data: {str(e)}"
        }), 500


@alvadesk_pca_bp.route("/alvadesk_pca_clear_projected", methods=['POST'])
@nocache
def alvadesk_pca_clear_projected():
    """Clear projected new compounds data."""

    session.pop('alvadesk_pca_new_projected', None)
    session.pop('alvadesk_pca_new_label_column', None)
    session.pop('alvadesk_pca_new_count', None)

    # Remove the file
    temp_path = ensure_temp_dir()
    new_scores_file = os.path.join(temp_path, 'alvadesk_pca_new_compounds.csv')
    if os.path.exists(new_scores_file):
        os.remove(new_scores_file)

    return jsonify({
        "status": "success",
        "message": "Projected compounds cleared."
    })


@alvadesk_pca_bp.route("/download_alvadesk_pca_projected", methods=['GET'])
@nocache
def download_alvadesk_pca_projected():
    """Download projected new compounds scores."""

    if not session.get('alvadesk_pca_new_projected'):
        flash('No new compounds have been projected yet.', 'warning')
        return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))

    temp_path = ensure_temp_dir()
    new_scores_file = os.path.join(temp_path, 'alvadesk_pca_new_compounds.csv')

    if not os.path.exists(new_scores_file):
        flash('Projected scores file not found.', 'danger')
        return redirect(url_for('alvadesk_pca.alvadesk_pca_analysis'))

    return send_file(
        new_scores_file,
        as_attachment=True,
        download_name='alvadesk_pca_new_compounds_projected.csv',
        mimetype='text/csv'
    )


@alvadesk_pca_bp.route("/alvadesk_pca_analyze_group_correlations", methods=['POST'])
def alvadesk_pca_analyze_group_correlations():
    """Analyze correlation between descriptor group combinations and target variable."""

    if not check_dataset():
        return jsonify({
            "status": "error",
            "message": "Please upload a dataset first."
        }), 400

    try:
        from scipy.stats import pearsonr
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        from itertools import combinations as iter_combinations

        # Get parameters
        target_variable = request.form.get('target_variable')
        combination_size = request.form.get('combination_size')

        if not target_variable:
            return jsonify({
                "status": "error",
                "message": "Please select a target variable."
            }), 400

        try:
            combination_size = int(combination_size)
        except (ValueError, TypeError):
            return jsonify({
                "status": "error",
                "message": "Invalid combination size."
            }), 400

        if combination_size < 1 or combination_size > 34:
            return jsonify({
                "status": "error",
                "message": "Combination size must be between 1 and 34."
            }), 400

        # Check if external color file is loaded
        color_file_path = session.get('alvadesk_color_file_path')
        if not color_file_path or not os.path.exists(color_file_path):
            return jsonify({
                "status": "error",
                "message": "Please upload external color file first."
            }), 400

        # Load PCA dataset (with descriptors)
        clean_path = get_clean_path(session["csv_name"])
        df_pca = read_dataset(clean_path)

        # Load external color file
        df_external = coloring.load_coloring_file(color_file_path)

        if df_external is None:
            return jsonify({
                "status": "error",
                "message": "Failed to load external color file."
            }), 400

        # Check if target variable exists and is numeric
        if target_variable not in df_external.columns:
            return jsonify({
                "status": "error",
                "message": f"Target variable '{target_variable}' not found in external file."
            }), 400

        if not pd.api.types.is_numeric_dtype(df_external[target_variable]):
            return jsonify({
                "status": "error",
                "message": f"Target variable '{target_variable}' must be numeric."
            }), 400

        # Get key column for merging
        key_column = session.get('alvadesk_color_key_column')
        if not key_column:
            detection = session.get('alvadesk_color_detection', {})
            if detection and detection.get('found'):
                key_column = detection.get('key_column')

        if not key_column or key_column not in df_pca.columns or key_column not in df_external.columns:
            return jsonify({
                "status": "error",
                "message": "Cannot determine key column for merging datasets."
            }), 400

        # Merge datasets - use suffixes to handle duplicate columns
        df_merged = pd.merge(
            df_pca,
            df_external[[key_column, target_variable]],
            on=key_column,
            how='inner',
            suffixes=('_pca', '_ext')
        )

        if len(df_merged) == 0:
            return jsonify({
                "status": "error",
                "message": "No matching rows found between PCA data and external file."
            }), 400

        # Get target values - check for suffixed column name if original doesn't exist
        if target_variable in df_merged.columns:
            target_col = target_variable
        elif f"{target_variable}_ext" in df_merged.columns:
            target_col = f"{target_variable}_ext"
        else:
            return jsonify({
                "status": "error",
                "message": f"Target variable '{target_variable}' not found after merging datasets."
            }), 400

        # Get target values and remove NaN
        y = df_merged[target_col].copy()
        valid_indices = ~y.isna()
        y = y[valid_indices]
        df_valid = df_merged[valid_indices].copy()

        if len(y) < 10:
            return jsonify({
                "status": "error",
                "message": "Not enough valid samples (need at least 10)."
            }), 400

        # Parse descriptor groups (all 34 groups)
        groups_dict = parse_descriptor_groups(DESCRIPTOR_GROUPS_FILE)
        groups_dict = dict(sorted(groups_dict.items(), key=lambda x: int(x[0])))

        # Generate group combinations
        group_ids = list(groups_dict.keys())
        group_combinations = list(iter_combinations(group_ids, combination_size))

        # Analyze each group combination
        results = []

        for group_combo in group_combinations:
            # Collect all descriptors from these groups
            all_descriptors = []
            group_names = []

            for group_id in group_combo:
                group_info = groups_dict[group_id]
                group_names.append(group_info['name'])
                all_descriptors.extend(group_info['descriptors'])

            # Find available descriptors in dataset
            available_descriptors = [d for d in all_descriptors if d in df_valid.columns]

            if len(available_descriptors) == 0:
                continue

            # Get descriptor data (only numeric)
            X_combo = df_valid[available_descriptors].select_dtypes(include=[np.number])
            X_combo = X_combo.dropna(axis=1)

            if X_combo.shape[1] == 0:
                continue

            # Calculate correlations
            correlations = []
            for desc in X_combo.columns:
                try:
                    corr, pval = pearsonr(X_combo[desc], y)
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                except:
                    continue

            if len(correlations) == 0:
                continue

            avg_correlation = float(np.mean(correlations))
            max_correlation = float(np.max(correlations))

            # Calculate R² using linear regression
            try:
                model = LinearRegression()
                model.fit(X_combo, y)
                y_pred = model.predict(X_combo)
                r2 = float(r2_score(y, y_pred))
            except:
                r2 = 0.0

            results.append({
                'group_ids': [int(gid) for gid in group_combo],
                'group_names': group_names,
                'n_descriptors': X_combo.shape[1],
                'avg_correlation': round(avg_correlation, 4),
                'max_correlation': round(max_correlation, 4),
                'r2_score': round(r2, 4)
            })

        # Sort by average correlation (descending)
        results.sort(key=lambda x: x['avg_correlation'], reverse=True)

        # Store in session
        session['alvadesk_group_correlation_analysis'] = results
        session['alvadesk_group_correlation_target'] = target_variable

        message = f"Analyzed {len(results)} group combinations (size={combination_size}) against '{target_variable}'."

        return jsonify({
            "status": "success",
            "message": message,
            "results": results,
            "total_combinations": len(results)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Error analyzing group combinations: {str(e)}"
        }), 500
