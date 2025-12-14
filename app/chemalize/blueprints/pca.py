"""
Principal Component Analysis routes
"""
from flask import Blueprint, render_template, request, session, flash, redirect, url_for, jsonify, send_file
import os
import pandas as pd
import numpy as np
import time
from scipy.stats import pearsonr
import math
from itertools import combinations
import json
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
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Add any additional PCA-specific parameters from session
    pca_params = {k: session.get(k) for k in [
        'n_components', 'scale_data', 'show_variance', 'show_scatter',
        'show_loading', 'show_biplot', 'pc_color_by', 'pca_performed',
        'pc_x_axis', 'pc_y_axis', 'pc_loadings_select', 'feature_selection_method',
        'top_n_features', 'loading_threshold', 'show_top_features_plot',
        'show_feature_importance', 'export_feature_importance', 'export_selected_features',
        'top_n_arrows'
    ]}
    
    # Custom descriptor groups data
    custom_groups = session.get('custom_descriptor_groups', None)
    selected_group_ids = session.get('pca_selected_groups', [])
    correlation_results = session.get('pca_correlation_results', [])
    correlation_target = session.get('pca_correlation_target')
    correlated_features_used = session.get('pca_correlated_features_used', [])
    correlated_target_used = session.get('pca_correlated_target')
    feature_mode = session.get('pca_feature_mode', 'none')

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
        return render_template('pca_analysis.html', title='PCA Analysis', active="analyze",
                             custom_groups=custom_groups, selected_group_ids=selected_group_ids,
                             correlation_results=correlation_results, correlation_target=correlation_target,
                             correlated_features_used=correlated_features_used, correlated_target_used=correlated_target_used,
                             feature_mode=feature_mode,
                             numeric_columns=numeric_columns,
                             **info, **pca_params, **pca_results)

    return render_template('pca_analysis.html', title='PCA Analysis', active="analyze",
                         custom_groups=custom_groups, selected_group_ids=selected_group_ids,
                         correlation_results=correlation_results, correlation_target=correlation_target,
                         correlated_features_used=correlated_features_used, correlated_target_used=correlated_target_used,
                         feature_mode=feature_mode,
                         numeric_columns=numeric_columns,
                         **info, **pca_params)


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

    # Liczba strzałek na biplocie (top N features by loading magnitude)
    top_n_arrows_str = request.form.get('top_n_arrows', '')
    top_n_arrows = int(top_n_arrows_str) if top_n_arrows_str and top_n_arrows_str.strip() else None

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
    session['top_n_arrows'] = top_n_arrows

    # --- Wykonanie PCA ---
    clean_path = get_clean_path(session["csv_name"])
    df = read_dataset(clean_path)

    # Store identifier columns to add back after PCA
    identifier_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    print(f"\n🔍 PCA: Found {len(identifier_cols)} identifier columns: {identifier_cols}")

    # ===========================
    # Check for Correlation-based Feature Selection FIRST
    # ===========================
    # We need to check this BEFORE descriptor groups filtering
    # because correlation features should take precedence
    correlated_features_raw = request.form.get('correlated_features_selected', '')
    use_correlated_features = request.form.get('use_correlated_features') == '1'

    # ===========================
    # Custom Descriptor Groups Filtering
    # ===========================
    selected_groups_str = request.form.get('selected_descriptor_groups', '')
    descriptor_groups_used = False

    # Skip descriptor groups if using correlation features
    if selected_groups_str and use_correlated_features:
        flash('Note: Descriptor groups are ignored when using Endpoint Correlation features.', 'info')

    if selected_groups_str and not use_correlated_features:
        selected_group_ids = selected_groups_str.split(',')
        custom_groups = session.get('custom_descriptor_groups')

        if custom_groups and selected_group_ids:
            from app.chemalize.utils.descriptor_groups import filter_dataframe_by_groups, get_group_summary

            try:
                # Filter dataframe to only include descriptors from selected groups
                df_filtered, descriptor_columns_kept = filter_dataframe_by_groups(
                    df, custom_groups, selected_group_ids, keep_non_descriptors=True
                )

                # Get summary for logging
                summary = get_group_summary(custom_groups, selected_group_ids)

                print(f"\n📊 DESCRIPTOR GROUPS FILTERING:")
                print(f"   Selected groups: {summary['num_groups']}")
                print(f"   Group names: {', '.join(summary['group_names'])}")
                print(f"   Descriptors used: {summary['total_descriptors']}")
                print(f"   Descriptor columns kept: {len(descriptor_columns_kept)}")
                print(f"   Original DataFrame shape: {df.shape}")
                print(f"   Filtered DataFrame shape: {df_filtered.shape}")

                # Use filtered dataframe for PCA
                df = df_filtered
                descriptor_groups_used = True

                # Save info to session
                session['pca_descriptor_groups_used'] = True
                session['pca_selected_groups'] = selected_group_ids
                session['pca_groups_summary'] = summary

                flash(f"PCA performed on {summary['num_groups']} descriptor groups ({summary['total_descriptors']} descriptors)", 'info')

            except Exception as e:
                print(f"❌ Error filtering by descriptor groups: {e}")
                flash(f"Warning: Could not apply descriptor groups filter: {str(e)}", 'warning')
                descriptor_groups_used = False
        else:
            print("⚠️  Selected groups provided but no custom groups found in session")

    if not descriptor_groups_used:
        session['pca_descriptor_groups_used'] = False

    # ===========================
    # Correlation-based Feature Selection
    # ===========================
    # (correlated_features_raw and use_correlated_features already read above)
    correlation_target_var = request.form.get('correlation_target_var', '')
    session['pca_correlated_features_used'] = []
    session['pca_correlated_target'] = correlation_target_var if correlation_target_var else None

    if use_correlated_features and correlated_features_raw:
        # Try to load JSON list first (handles names with commas), fallback to CSV split
        correlated_features = []
        try:
            parsed = json.loads(correlated_features_raw)
            if isinstance(parsed, list):
                correlated_features = [str(feat).strip() for feat in parsed if str(feat).strip()]
        except Exception:
            correlated_features = [feat.strip() for feat in correlated_features_raw.split(',') if feat.strip()]
        numeric_cols_after_filters = df.select_dtypes(include=[np.number]).columns.tolist()
        missing_features = [f for f in correlated_features if f not in df.columns]
        non_numeric_features = [f for f in correlated_features if (f in df.columns and f not in numeric_cols_after_filters)]
        usable_correlated_features = [f for f in correlated_features if f in numeric_cols_after_filters]

        if len(usable_correlated_features) >= 2:
            non_numeric_cols_current = df.select_dtypes(exclude=[np.number]).columns.tolist()
            df = df[non_numeric_cols_current + usable_correlated_features]
            session['pca_correlated_features_used'] = usable_correlated_features
            session['pca_correlated_target'] = correlation_target_var
            # Override feature_selection_method to use ALL correlated features
            feature_selection_method = 'all'
            session['feature_selection_method'] = 'all'
            session['pca_feature_mode'] = 'correlation'

            if missing_features or non_numeric_features:
                info_chunks = []
                if missing_features:
                    info_chunks.append(f"brak: {', '.join(missing_features)}")
                if non_numeric_features:
                    info_chunks.append(f"nienumeryczne: {', '.join(non_numeric_features)}")
                flash(f'Using {len(usable_correlated_features)}/{len(correlated_features)} correlated features after filters ({"; ".join(info_chunks)}).', 'warning')
            else:
                flash(f'Using {len(usable_correlated_features)} correlated features for PCA.', 'info')
        else:
            flash('Not enough correlated numeric features available after filtering. Using all features instead.', 'warning')
            session['pca_feature_mode'] = 'none'
    elif descriptor_groups_used:
        session['pca_feature_mode'] = 'groups'
    else:
        session['pca_feature_mode'] = 'none'

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
            top_n_arrows=top_n_arrows,
            temp_path=ensure_temp_dir()
        )

        session['pca_performed'] = True
        session['pca_summary'] = results.get('summary', [])

        # ✨ ADD IDENTIFIER COLUMNS TO PCA COMPONENTS FILE ✨
        pca_components_path = os.path.join(ensure_temp_dir(), 'pca_components.csv')
        if os.path.exists(pca_components_path) and identifier_cols:
            print(f"\n✨ Adding identifier columns to PCA file...")
            pca_df = pd.read_csv(pca_components_path)
            print(f"   Before: {list(pca_df.columns)}")

            # Add each identifier column
            for col in identifier_cols:
                if col in df.columns and col not in pca_df.columns:
                    pca_df[col] = df[col].reset_index(drop=True)
                    print(f"   ✓ Added: {col}")

            # Save back
            pca_df.to_csv(pca_components_path, index=False)
            print(f"   After: {list(pca_df.columns)}")
            print(f"✅ Saved PCA file with {len(pca_df.columns)} total columns\n")

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


@pca_bp.route("/upload_descriptor_groups", methods=["POST"])
def upload_descriptor_groups():
    """
    Upload and parse a custom descriptor groups file.
    Returns parsed groups information.
    """
    try:
        from app.chemalize.utils.descriptor_groups import parse_descriptor_groups, get_group_summary

        if 'descriptor_groups_file' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No file uploaded"
            })

        file = request.files['descriptor_groups_file']

        if file.filename == '':
            return jsonify({
                "status": "error",
                "message": "No file selected"
            })

        # Save file temporarily
        temp_dir = ensure_temp_dir()
        groups_file_path = os.path.join(temp_dir, 'custom_descriptor_groups.txt')
        file.save(groups_file_path)

        # Parse the file
        try:
            groups_dict = parse_descriptor_groups(groups_file_path)
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Failed to parse file: {str(e)}"
            })

        if not groups_dict:
            return jsonify({
                "status": "error",
                "message": "No groups found in file. Please check the file format."
            })

        # Save groups to session
        session['custom_descriptor_groups'] = groups_dict
        session['custom_groups_file_path'] = groups_file_path

        # Calculate summary
        all_group_ids = list(groups_dict.keys())
        summary = get_group_summary(groups_dict, all_group_ids)

        return jsonify({
            "status": "success",
            "groups": groups_dict,
            "num_groups": summary['num_groups'],
            "total_descriptors": summary['total_descriptors'],
            "message": f"Successfully loaded {summary['num_groups']} descriptor groups"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Upload error: {str(e)}"
        })


# PCR Analysis Routes


@pca_bp.route("/pca_correlations", methods=['POST'])
def pca_correlations():
    """
    Calculate Pearson correlations between numeric variables and a selected endpoint.
    Returns a ranked list for interactive selection on the PCA page.
    """
    if not check_dataset():
        return jsonify({
            "status": "error",
            "message": "Please upload a dataset first."
        }), 400

    try:
        target_var = request.form.get('target_var', '').strip()
        try:
            max_results = int(request.form.get('max_results', 300))
        except (TypeError, ValueError):
            max_results = 300

        if not target_var:
            return jsonify({
                "status": "error",
                "message": "Please choose an endpoint variable."
            }), 400

        clean_path = get_clean_path(session["csv_name"])
        df = read_dataset(clean_path)

        if target_var not in df.columns:
            return jsonify({
                "status": "error",
                "message": f"Column '{target_var}' not found in dataset."
            }), 400

        if not pd.api.types.is_numeric_dtype(df[target_var]):
            return jsonify({
                "status": "error",
                "message": f"Endpoint '{target_var}' must be numeric to compute correlations."
            }), 400

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        candidate_features = [col for col in numeric_cols if col != target_var]

        results = []
        for feat in candidate_features:
            pair_df = df[[feat, target_var]].dropna()
            if len(pair_df) < 3:
                continue
            try:
                corr, pval = pearsonr(pair_df[feat], pair_df[target_var])
            except Exception:
                continue
            if np.isnan(corr):
                continue
            results.append({
                "feature": feat,
                "correlation": float(corr),
                "abs_correlation": float(abs(corr)),
                "p_value": float(pval),
                "n": int(len(pair_df))
            })

        results.sort(key=lambda x: x['abs_correlation'], reverse=True)
        if max_results and len(results) > max_results:
            results = results[:max_results]

        session['pca_correlation_results'] = results
        session['pca_correlation_target'] = target_var

        return jsonify({
            "status": "success",
            "results": results,
            "target": target_var,
            "total_features": len(results)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Error computing correlations: {str(e)}"
        }), 500


@pca_bp.route("/pca_combo_correlations", methods=['POST'])
def pca_combo_correlations():
    """
    Calculate correlations for combinations of variables of a chosen size against the endpoint.
    Composite is built as the mean of z-scored variables in the combination.
    """
    if not check_dataset():
        return jsonify({
            "status": "error",
            "message": "Please upload a dataset first."
        }), 400

    try:
        target_var = request.form.get('target_var', '').strip()
        combo_size_raw = request.form.get('combo_size', '2')
        pool_limit_raw = request.form.get('pool_limit', '25')
        top_n_raw = request.form.get('top_n', '50')
        feature_pool_raw = request.form.get('feature_pool', '')

        try:
            combo_size = int(combo_size_raw)
        except (TypeError, ValueError):
            combo_size = 2

        try:
            pool_limit = int(pool_limit_raw)
        except (TypeError, ValueError):
            pool_limit = 25

        try:
            top_n = int(top_n_raw)
        except (TypeError, ValueError):
            top_n = 50

        combo_size = max(2, combo_size)
        pool_limit = max(5, min(100, pool_limit))
        top_n = max(1, min(200, top_n))

        if not target_var:
            return jsonify({
                "status": "error",
                "message": "Please choose an endpoint variable first."
            }), 400

        clean_path = get_clean_path(session["csv_name"])
        df = read_dataset(clean_path)

        if target_var not in df.columns:
            return jsonify({
                "status": "error",
                "message": f"Column '{target_var}' not found in dataset."
            }), 400

        if not pd.api.types.is_numeric_dtype(df[target_var]):
            return jsonify({
                "status": "error",
                "message": f"Endpoint '{target_var}' must be numeric to compute correlations."
            }), 400

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        candidate_features = [col for col in numeric_cols if col != target_var]

        # If user passed a pool, respect intersection
        if feature_pool_raw:
            pool_from_form = [c.strip() for c in feature_pool_raw.split(',') if c.strip()]
            candidate_features = [c for c in pool_from_form if c in candidate_features]

        if len(candidate_features) < combo_size:
            return jsonify({
                "status": "error",
                "message": "Not enough numeric features to build combinations."
            }), 400

        # Pre-filter features by absolute correlation with target (top pool_limit)
        feature_corrs = []
        y_full = df[target_var]
        for feat in candidate_features:
            pair_df = df[[feat, target_var]].dropna()
            if len(pair_df) < 3:
                continue
            try:
                corr, _ = pearsonr(pair_df[feat], pair_df[target_var])
            except Exception:
                continue
            if np.isnan(corr):
                continue
            feature_corrs.append((feat, abs(corr)))

        feature_corrs.sort(key=lambda x: x[1], reverse=True)
        filtered_features = [f for f, _ in feature_corrs[:pool_limit]]

        if len(filtered_features) < combo_size:
            return jsonify({
                "status": "error",
                "message": "Not enough numeric features after filtering."
            }), 400

        total_combos = math.comb(len(filtered_features), combo_size)
        if total_combos > 5000:
            return jsonify({
                "status": "error",
                "message": f"Too many combinations ({total_combos}). Reduce pool size or combination size."
            }), 400

        results = []
        y = df[target_var]

        for combo in combinations(filtered_features, combo_size):
            combo_df = df[list(combo)].dropna()
            if combo_df.shape[0] < 3:
                continue
            # z-score each feature to balance scales
            z_scored = (combo_df - combo_df.mean()) / combo_df.std(ddof=0)
            composite = z_scored.mean(axis=1)
            aligned_y = y.loc[composite.index]
            if aligned_y.isna().all():
                continue
            try:
                corr, pval = pearsonr(composite, aligned_y)
            except Exception:
                continue
            if np.isnan(corr):
                continue
            results.append({
                "features": list(combo),
                "label": " + ".join(combo),
                "correlation": float(corr),
                "abs_correlation": float(abs(corr)),
                "p_value": float(pval),
                "n": int(len(composite))
            })

        results.sort(key=lambda x: x['abs_correlation'], reverse=True)
        results = results[:top_n]

        session['pca_correlation_combos'] = results
        session['pca_correlation_combos_target'] = target_var

        return jsonify({
            "status": "success",
            "results": results,
            "target": target_var,
            "total_combinations": len(results)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Error computing combination correlations: {str(e)}"
        }), 500
