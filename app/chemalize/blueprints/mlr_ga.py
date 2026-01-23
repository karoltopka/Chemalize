"""
Multiple Linear Regression with Genetic Algorithm for Variable Selection
"""
from flask import Blueprint, render_template, request, session, flash, redirect, url_for, jsonify, send_file, Response
import os
import pandas as pd
import numpy as np
import time
import math
import json
import queue
import threading
from werkzeug.utils import secure_filename
from app.config import get_clean_path, get_temp_path, get_upload_path, get_unified_path, get_user_id, TEMP_DIR, CLEAN_DIR, UPLOAD_DIR, UNIFIED_DIR
from app.chemalize.utils import read_dataset, clean_temp_folder, check_dataset, get_dataset_info, ensure_temp_dir
from app.chemalize.preprocessing import generic_preprocessing as gp

from app.chemalize.modules import mlr
from app.chemalize.modules.ga_variable_selection import (
    GeneticAlgorithmSelector,
    preprocess_for_ga,
    plot_y_histogram
)


mlr_ga_bp = Blueprint('mlr_ga', __name__)

# Global dictionary to store latest progress for each session (for polling)
ga_latest_progress = {}
ga_progress_lock = threading.Lock()
ga_threads = {}  # Store running GA threads


@mlr_ga_bp.route("/mlr_ga_analysis")
def mlr_ga_analysis():
    """Main page for MLR with Genetic Algorithm variable selection"""
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))

    clean_path = get_clean_path(session["csv_name"])
    df = read_dataset(clean_path)
    info = get_dataset_info(df)

    # Get only numeric columns for GA analysis
    df_numeric = df.select_dtypes(include=[np.number])
    all_data_columns = df_numeric.columns.tolist()

    # Get current step
    current_step = session.get('mlr_ga_step', 'target_selection')

    # Parameters for different steps
    mlr_ga_params = {
        'all_data_columns': all_data_columns,
        'current_step': current_step,

        # Step 1: Target selection and Y analysis
        'mlr_ga_target_var': session.get('mlr_ga_target_var', ''),
        'mlr_ga_y_analyzed': session.get('mlr_ga_y_analyzed', False),
        'y_transformation': session.get('mlr_ga_y_transformation', 'none'),
        'y_histogram_plot': session.get('mlr_ga_y_histogram_plot', ''),
        'y_stats': session.get('mlr_ga_y_stats', {}),
        'y_histogram_plot_transformed': session.get('mlr_ga_y_histogram_plot_transformed', ''),
        'y_stats_transformed': session.get('mlr_ga_y_stats_transformed', {}),

        # Step 2: Split method
        'split_method': session.get('mlr_ga_split_method', 'random'),
        'test_size': session.get('mlr_ga_test_size', 0.2),
        'shuffle': session.get('mlr_ga_shuffle', True),
        'random_state': session.get('mlr_ga_random_state', 42),
        'strat_test_size': session.get('mlr_ga_strat_test_size', 0.2),
        'strat_bins': session.get('mlr_ga_strat_bins', 5),
        'time_column': session.get('mlr_ga_time_column', ''),
        'time_test_size': session.get('mlr_ga_time_test_size', 0.2),
        'n_folds': session.get('mlr_ga_n_folds', 5),
        'shuffle_kfold': session.get('mlr_ga_shuffle_kfold', True),
        'systematic_step': session.get('mlr_ga_systematic_step', 3),
        'include_last_point': session.get('mlr_ga_include_last_point', False),

        # Step 3: Standardization, variance check, and internal CV type
        'autoscale': session.get('mlr_ga_autoscale', True),
        'remove_zero_variance': session.get('mlr_ga_remove_zero_variance', True),
        'remove_low_variance': session.get('mlr_ga_remove_low_variance', False),
        'variance_threshold': session.get('mlr_ga_variance_threshold', 0.01),
        'internal_cv_type': session.get('mlr_ga_internal_cv_type', 'kfold'),
        'sorted_step': session.get('mlr_ga_sorted_step', 5),
        'sorted_iterations': session.get('mlr_ga_sorted_iterations', 5),

        # Step 4: GA parameters
        'n_variables': session.get('mlr_ga_n_variables', ''),
        'correlation_threshold': session.get('mlr_ga_correlation_threshold', 0.95),
        'mutation_rate': session.get('mlr_ga_mutation_rate', 0.1),
        'random_models_ratio': session.get('mlr_ga_random_models_ratio', 0.1),
        'use_validation': session.get('mlr_ga_use_validation', False),
        'population_size': session.get('mlr_ga_population_size', 50),
        'n_iterations': session.get('mlr_ga_n_iterations', 100),
        'max_retries': session.get('mlr_ga_max_retries', 3),
        'cv_folds': session.get('mlr_ga_cv_folds', 5),
        'cv_folds_validation': session.get('mlr_ga_cv_folds_validation', 3),
        'test_normality': session.get('mlr_ga_test_normality', True),
        'normality_alpha': session.get('mlr_ga_normality_alpha', 0.05),
        'n_best_models': session.get('mlr_ga_n_best_models', 50),
        'check_ad': session.get('mlr_ga_check_ad', True),
        'ad_threshold': session.get('mlr_ga_ad_threshold', 100.0),

        # Results
        'ga_completed': session.get('mlr_ga_completed', False),
        'best_models': session.get('mlr_ga_best_models', []),
        'selected_features': session.get('mlr_ga_selected_features', []),
        'selected_model_rank': session.get('mlr_ga_selected_model_rank', 1),
        'removed_features': session.get('mlr_ga_removed_features', []),
    }

    # If analysis was performed, add MLR results
    if session.get('mlr_ga_analysis_performed'):
        mlr_results = {
            'train_r2': session.get('mlr_ga_train_r2'),
            'adj_r2': session.get('mlr_ga_adj_r2'),
            'test_r2': session.get('mlr_ga_test_r2'),
            'q2_loo': session.get('mlr_ga_q2_loo'),
            'q2_test': session.get('mlr_ga_q2_test'),
            'train_rmse': session.get('mlr_ga_train_rmse'),
            'test_rmse': session.get('mlr_ga_test_rmse'),
            'rmse_loo': session.get('mlr_ga_rmse_loo'),
            'train_mae': session.get('mlr_ga_train_mae'),
            'test_mae': session.get('mlr_ga_test_mae'),
            'f_statistic': session.get('mlr_ga_f_statistic'),
            'f_pvalue': session.get('mlr_ga_f_pvalue'),
            'aic': session.get('mlr_ga_aic'),
            'bic': session.get('mlr_ga_bic'),
            'dw_stat': session.get('mlr_ga_dw_stat'),
            'vif_values': session.get('mlr_ga_vif_values'),
            'ccc_ext': session.get('mlr_ga_ccc_ext'),
            'coefficients': session.get('mlr_ga_coefficients'),
            'std_errors': session.get('mlr_ga_std_errors'),
            't_values': session.get('mlr_ga_t_values'),
            'p_values': session.get('mlr_ga_p_values'),
            'feature_names': session.get('mlr_ga_feature_names'),
            'mlr_pred_actual_plot': session.get('mlr_ga_pred_actual_plot'),
            'mlr_residuals_plot': session.get('mlr_ga_residuals_plot'),
            'mlr_residuals_hist': session.get('mlr_ga_residuals_hist'),
            'mlr_qq_plot': session.get('mlr_ga_qq_plot'),
            'mlr_williams_plot': session.get('mlr_ga_williams_plot'),
            'AD_train': session.get('mlr_ga_AD_train'),
            'AD_test': session.get('mlr_ga_AD_test'),
            'h_star': session.get('mlr_ga_h_star'),
            'ga_fitness_plot': session.get('mlr_ga_fitness_plot'),
            'ga_best_score': session.get('mlr_ga_best_score'),
        }
        return render_template('mlr_ga_analysis.html', title='MLR with GA Variable Selection',
                             active="analyze", zip=zip, **info, **mlr_ga_params, **mlr_results)

    return render_template('mlr_ga_analysis.html', title='MLR with GA Variable Selection',
                         active="analyze", zip=zip, **info, **mlr_ga_params)


@mlr_ga_bp.route("/mlr_ga_step1", methods=['POST'])
def mlr_ga_step1():
    """Step 1: Analyze target variable and select transformation"""
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))

    target_var = request.form.get('target_var')
    y_transformation = request.form.get('y_transformation', 'none')

    # If no target_var provided, this is a re-analyze request - clear the analyzed flag
    if not target_var:
        session['mlr_ga_y_analyzed'] = False
        session.pop('mlr_ga_y_histogram_plot', None)
        session.pop('mlr_ga_y_histogram_plot_transformed', None)
        return redirect(url_for('mlr_ga.mlr_ga_analysis'))

    # Read data
    clean_path = get_clean_path(session["csv_name"])
    df = read_dataset(clean_path)

    # Check if target variable exists
    if target_var not in df.columns:
        flash(f'Target variable {target_var} not found in dataset!', 'danger')
        return redirect(url_for('mlr_ga.mlr_ga_analysis'))

    # Get target variable (original)
    y_original = df[target_var].copy()

    try:
        temp_path = ensure_temp_dir()

        # Always plot original histogram
        plot_path_original, y_stats_original = plot_y_histogram(
            y_original,
            y_name=f"{target_var} (Original)",
            temp_path=temp_path
        )

        # Store only filename (without path) - will use utils.serve_temp_image endpoint
        plot_filename_original = os.path.basename(plot_path_original)
        session['mlr_ga_y_histogram_plot'] = plot_filename_original
        session['mlr_ga_y_stats'] = y_stats_original

        # Apply transformation if requested and plot transformed
        y_transformed = None
        plot_path_transformed = None
        y_stats_transformed = None

        if y_transformation != 'none':
            if y_transformation == 'log':
                if (y_original <= 0).any():
                    flash('Log transformation requires all positive values!', 'danger')
                    return redirect(url_for('mlr_ga.mlr_ga_analysis'))
                y_transformed = np.log(y_original)
            elif y_transformation == 'sqrt':
                if (y_original < 0).any():
                    flash('Square root transformation requires non-negative values!', 'danger')
                    return redirect(url_for('mlr_ga.mlr_ga_analysis'))
                y_transformed = np.sqrt(y_original)
            elif y_transformation == 'square':
                y_transformed = y_original ** 2
            elif y_transformation == 'inverse':
                if (y_original == 0).any():
                    flash('Inverse transformation requires non-zero values!', 'danger')
                    return redirect(url_for('mlr_ga.mlr_ga_analysis'))
                y_transformed = 1 / y_original

            # Plot transformed histogram
            if y_transformed is not None:
                plot_path_transformed, y_stats_transformed = plot_y_histogram(
                    y_transformed,
                    y_name=f"{target_var} ({y_transformation})",
                    temp_path=temp_path
                )
                # Store only filename (without path)
                plot_filename_transformed = os.path.basename(plot_path_transformed)
                session['mlr_ga_y_histogram_plot_transformed'] = plot_filename_transformed
                session['mlr_ga_y_stats_transformed'] = y_stats_transformed
        else:
            # No transformation - clear transformed histogram
            session.pop('mlr_ga_y_histogram_plot_transformed', None)
            session.pop('mlr_ga_y_stats_transformed', None)

        # Store in session
        session['mlr_ga_target_var'] = target_var
        session['mlr_ga_y_transformation'] = y_transformation
        session['mlr_ga_y_analyzed'] = True  # Mark that analysis is done
        # Stay in target_selection step to show histogram
        session['mlr_ga_step'] = 'target_selection'

        if y_transformation != 'none':
            flash(f'Target variable analyzed with {y_transformation} transformation!', 'success')
        else:
            flash('Target variable analyzed successfully!', 'success')

    except Exception as e:
        flash(f'Error analyzing target variable: {str(e)}', 'danger')
        import traceback
        print(traceback.format_exc())

    return redirect(url_for('mlr_ga.mlr_ga_analysis'))


@mlr_ga_bp.route("/mlr_ga_detect_outliers_step1", methods=['POST'])
def mlr_ga_detect_outliers_step1():
    """Step 1: Detect outliers in target variable and show preview"""
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))

    try:
        from app.chemalize.modules.ga_variable_selection import detect_and_remove_y_outliers, plot_y_histogram

        # Get parameters
        outlier_method = request.form.get('outlier_method', 'iqr')
        outlier_threshold = float(request.form.get('outlier_threshold', 1.5))

        # Read data
        clean_path = get_clean_path(session["csv_name"])
        df = read_dataset(clean_path)

        target_var = session.get('mlr_ga_target_var')
        y_transformation = session.get('mlr_ga_y_transformation', 'none')

        if not target_var or target_var not in df.columns:
            flash('Target variable not set or not found!', 'danger')
            return redirect(url_for('mlr_ga.mlr_ga_analysis'))

        # Detect outliers (but don't remove yet)
        outlier_info = detect_and_remove_y_outliers(
            df, target_var, method=outlier_method, threshold=outlier_threshold
        )

        n_outliers = len(outlier_info['removed_samples'])
        total_samples = len(df)
        outlier_pct = (n_outliers / total_samples) * 100 if total_samples > 0 else 0

        # Generate PREVIEW histograms (before and after)
        # 1. Current state (before removal)
        y_before = df[target_var].copy()
        if y_transformation != 'none':
            if y_transformation == 'log':
                y_before = np.log(y_before)
            elif y_transformation == 'sqrt':
                y_before = np.sqrt(y_before)
            elif y_transformation == 'square':
                y_before = y_before ** 2
            elif y_transformation == 'inverse':
                y_before = 1 / y_before

        # 2. Preview after removal (using keep_indices)
        keep_indices = outlier_info['keep_indices']
        df_preview = df.loc[keep_indices].copy()
        y_after = df_preview[target_var].copy()
        if y_transformation != 'none':
            if y_transformation == 'log':
                y_after = np.log(y_after)
            elif y_transformation == 'sqrt':
                y_after = np.sqrt(y_after)
            elif y_transformation == 'square':
                y_after = y_after ** 2
            elif y_transformation == 'inverse':
                y_after = 1 / y_after

        # Generate both histograms
        temp_path = ensure_temp_dir()
        plot_path_before, y_stats_before = plot_y_histogram(
            y_before,
            y_name=f"{target_var} (Current)",
            temp_path=temp_path
        )

        plot_path_after, y_stats_after = plot_y_histogram(
            y_after,
            y_name=f"{target_var} (Preview After Removal)",
            temp_path=temp_path
        )

        # Store preview histograms
        plot_filename_before = os.path.basename(plot_path_before)
        plot_filename_after = os.path.basename(plot_path_after)

        session['mlr_ga_step1_preview_before'] = plot_filename_before
        session['mlr_ga_step1_preview_after'] = plot_filename_after
        session['mlr_ga_step1_preview_stats_before'] = y_stats_before
        session['mlr_ga_step1_preview_stats_after'] = y_stats_after

        # Store detection results in session
        session['mlr_ga_step1_outliers_detected'] = True
        session['mlr_ga_step1_n_outliers'] = n_outliers
        session['mlr_ga_step1_total_samples'] = total_samples
        session['mlr_ga_step1_outlier_pct'] = outlier_pct
        session['mlr_ga_step1_outlier_method'] = outlier_method
        session['mlr_ga_step1_outlier_threshold'] = outlier_threshold
        session['mlr_ga_step1_outlier_info'] = outlier_info

        if n_outliers > 0:
            flash(f'Detected {n_outliers} outlier(s) ({outlier_pct:.1f}%). Review the preview below.', 'warning')
        else:
            flash('No outliers detected! Your data looks good.', 'success')

    except Exception as e:
        flash(f'Error detecting outliers: {str(e)}', 'danger')
        import traceback
        traceback.print_exc()

    return redirect(url_for('mlr_ga.mlr_ga_analysis'))


@mlr_ga_bp.route("/mlr_ga_remove_outliers_step1", methods=['POST'])
def mlr_ga_remove_outliers_step1():
    """Step 1: Confirm and apply outlier removal"""
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))

    try:
        outlier_info = session.get('mlr_ga_step1_outlier_info')
        if not outlier_info:
            flash('No outlier detection results found! Please detect outliers first.', 'danger')
            return redirect(url_for('mlr_ga.mlr_ga_analysis'))

        # Get indices to keep
        keep_indices = outlier_info['keep_indices']

        # Store removal results
        session['mlr_ga_step1_outliers_removed'] = True
        session['mlr_ga_step1_n_removed'] = len(outlier_info['removed_samples'])
        session['mlr_ga_step1_remaining_samples'] = len(keep_indices)
        session['mlr_ga_step1_normality_before'] = outlier_info['normality_before']
        session['mlr_ga_step1_normality_after'] = outlier_info['normality_after']
        session['mlr_ga_step1_removed_samples'] = outlier_info['removed_samples']

        # Store cleaned indices for use in later steps (keep_indices is already a list)
        session['mlr_ga_step1_clean_indices'] = keep_indices

        # Clear detection results to allow re-detection
        session.pop('mlr_ga_step1_outliers_detected', None)
        session.pop('mlr_ga_step1_preview_before', None)
        session.pop('mlr_ga_step1_preview_after', None)

        flash(f'Successfully removed {len(outlier_info["removed_samples"])} outlier(s)!', 'success')

    except Exception as e:
        flash(f'Error removing outliers: {str(e)}', 'danger')
        import traceback
        traceback.print_exc()

    return redirect(url_for('mlr_ga.mlr_ga_analysis'))


@mlr_ga_bp.route("/download_step1_outliers")
def download_step1_outliers():
    """Download CSV with removed Y outliers from Step 1"""
    if not check_dataset():
        flash('No dataset found', 'danger')
        return redirect(url_for('mlr_ga.mlr_ga_analysis'))

    removed_samples = session.get('mlr_ga_step1_removed_samples', [])

    if not removed_samples:
        flash('No outliers were removed in Step 1', 'info')
        return redirect(url_for('mlr_ga.mlr_ga_analysis'))

    try:
        # Create DataFrame from removed samples
        outliers_df = pd.DataFrame(removed_samples)

        # Reorder columns: Sample_Index first, Y_Value, then rest
        cols = outliers_df.columns.tolist()
        priority_cols = []

        if 'Sample_Index' in cols:
            priority_cols.append('Sample_Index')
        if 'Y_Value' in cols:
            priority_cols.append('Y_Value')

        # Add ID columns if present
        id_candidates = [col for col in cols if col.lower() in ['id', 'sample_id', 'name', 'compound', 'sample_name', 'sample']]
        priority_cols.extend([col for col in id_candidates if col not in priority_cols])

        # Add target variable if present
        target_var = session.get('mlr_ga_target_var')
        if target_var and target_var in cols and target_var not in priority_cols:
            priority_cols.append(target_var)

        # Add remaining columns
        remaining_cols = [col for col in cols if col not in priority_cols]
        final_cols = priority_cols + remaining_cols

        outliers_df = outliers_df[final_cols]

        # Save to temp CSV
        temp_path = ensure_temp_dir()
        temp_file = os.path.join(temp_path, 'step1_y_outliers_removed.csv')
        outliers_df.to_csv(temp_file, index=False)

        return send_file(temp_file, as_attachment=True, download_name='step1_y_outliers_removed.csv')

    except Exception as e:
        flash(f'Error generating outliers file: {str(e)}', 'danger')
        return redirect(url_for('mlr_ga.mlr_ga_analysis'))


@mlr_ga_bp.route("/mlr_ga_detect_outliers_step2", methods=['POST'])
def mlr_ga_detect_outliers_step2():
    """Step 2: Detect outliers in train and test sets separately"""
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))

    try:
        from app.chemalize.modules.ga_variable_selection import detect_and_remove_y_outliers

        # Get parameters
        outlier_method = request.form.get('outlier_method', 'iqr')
        outlier_threshold = float(request.form.get('outlier_threshold', 1.5))

        # Read data
        clean_path = get_clean_path(session["csv_name"])
        df = read_dataset(clean_path)

        # Apply Step 1 outlier removal if it was performed
        step1_clean_indices = session.get('mlr_ga_step1_clean_indices')
        if step1_clean_indices:
            df = df.loc[step1_clean_indices].reset_index(drop=True)

        target_var = session.get('mlr_ga_target_var')
        if not target_var or target_var not in df.columns:
            flash('Target variable not set or not found!', 'danger')
            return redirect(url_for('mlr_ga.mlr_ga_analysis'))

        # Get split indices
        train_idx = session.get('mlr_ga_step2_train_idx', [])
        test_idx = session.get('mlr_ga_step2_test_idx', [])

        if not train_idx or not test_idx:
            flash('Split not configured! Please configure split first.', 'danger')
            return redirect(url_for('mlr_ga.mlr_ga_analysis'))

        # Create train and test dataframes (keep original indices for mapping)
        df_train = df.iloc[train_idx].copy().reset_index(drop=False)
        df_train_original_indices = df_train['index'].tolist()
        df_train = df_train.drop(columns=['index'])

        df_test = df.iloc[test_idx].copy().reset_index(drop=False)
        df_test_original_indices = df_test['index'].tolist()
        df_test = df_test.drop(columns=['index'])

        # Detect outliers in train set
        outlier_info_train = detect_and_remove_y_outliers(
            df_train, target_var, method=outlier_method, threshold=outlier_threshold
        )

        # Detect outliers in test set
        outlier_info_test = detect_and_remove_y_outliers(
            df_test, target_var, method=outlier_method, threshold=outlier_threshold
        )

        n_outliers_train = len(outlier_info_train['removed_samples'])
        n_outliers_test = len(outlier_info_test['removed_samples'])
        total_train = len(df_train)
        total_test = len(df_test)

        # Generate PREVIEW histograms showing current vs after removal
        from app.chemalize.modules.ga_variable_selection import plot_y_histogram_split

        y_transformation = session.get('mlr_ga_y_transformation', 'none')
        y = df[target_var].copy()

        # Apply transformation if needed
        if y_transformation != 'none':
            if y_transformation == 'log':
                y = np.log(y)
            elif y_transformation == 'sqrt':
                y = np.sqrt(y)
            elif y_transformation == 'square':
                y = y ** 2
            elif y_transformation == 'inverse':
                y = 1 / y

        # 1. Current state histogram
        y_train_current = y.iloc[train_idx]
        y_test_current = y.iloc[test_idx]

        # 2. Preview after removal histogram
        # Use keep_indices to filter train and test
        train_keep_positions = outlier_info_train['keep_indices']
        test_keep_positions = outlier_info_test['keep_indices']

        # Map positions to absolute indices
        train_keep_absolute = [df_train_original_indices[i] for i in train_keep_positions]
        test_keep_absolute = [df_test_original_indices[i] for i in test_keep_positions]

        y_train_preview = y.iloc[train_keep_absolute]
        y_test_preview = y.iloc[test_keep_absolute]

        # Generate both preview histograms
        temp_path = ensure_temp_dir()
        plot_path_current = plot_y_histogram_split(
            y_train_current, y_test_current,
            y_name=f"{target_var} (Current Split)",
            temp_path=temp_path
        )

        plot_path_preview = plot_y_histogram_split(
            y_train_preview, y_test_preview,
            y_name=f"{target_var} (Preview After Removal)",
            temp_path=temp_path
        )

        # Store preview histograms
        session['mlr_ga_step2_preview_current'] = os.path.basename(plot_path_current)
        session['mlr_ga_step2_preview_after'] = os.path.basename(plot_path_preview)

        # Store detection results in session
        session['mlr_ga_step2_outliers_detected'] = True
        session['mlr_ga_step2_train_n_outliers'] = n_outliers_train
        session['mlr_ga_step2_test_n_outliers'] = n_outliers_test
        session['mlr_ga_step2_train_total'] = total_train
        session['mlr_ga_step2_test_total'] = total_test
        session['mlr_ga_step2_train_outlier_pct'] = (n_outliers_train / total_train * 100) if total_train > 0 else 0
        session['mlr_ga_step2_test_outlier_pct'] = (n_outliers_test / total_test * 100) if total_test > 0 else 0
        session['mlr_ga_step2_outlier_method'] = outlier_method
        session['mlr_ga_step2_outlier_threshold'] = outlier_threshold
        session['mlr_ga_step2_train_outlier_info'] = outlier_info_train
        session['mlr_ga_step2_test_outlier_info'] = outlier_info_test
        session['mlr_ga_step2_train_original_indices'] = df_train_original_indices
        session['mlr_ga_step2_test_original_indices'] = df_test_original_indices

        total_outliers = n_outliers_train + n_outliers_test
        if total_outliers > 0:
            flash(f'Detected {total_outliers} outlier(s): {n_outliers_train} in train, {n_outliers_test} in test. Review the preview below.', 'warning')
        else:
            flash('No outliers detected in either set!', 'success')

    except Exception as e:
        flash(f'Error detecting outliers: {str(e)}', 'danger')
        import traceback
        traceback.print_exc()

    return redirect(url_for('mlr_ga.mlr_ga_analysis'))


@mlr_ga_bp.route("/mlr_ga_remove_outliers_step2", methods=['POST'])
def mlr_ga_remove_outliers_step2():
    """Step 2: Confirm and apply outlier removal from train/test sets"""
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))

    try:
        # Get user selection
        remove_train = 'remove_train_outliers' in request.form
        remove_test = 'remove_test_outliers' in request.form

        if not remove_train and not remove_test:
            flash('Please select at least one set to remove outliers from.', 'warning')
            return redirect(url_for('mlr_ga.mlr_ga_analysis'))

        outlier_info_train = session.get('mlr_ga_step2_train_outlier_info')
        outlier_info_test = session.get('mlr_ga_step2_test_outlier_info')
        train_original_indices = session.get('mlr_ga_step2_train_original_indices', [])
        test_original_indices = session.get('mlr_ga_step2_test_original_indices', [])

        if not outlier_info_train or not outlier_info_test:
            flash('No outlier detection results found! Please detect outliers first.', 'danger')
            return redirect(url_for('mlr_ga.mlr_ga_analysis'))

        # Get original split indices
        train_idx = session.get('mlr_ga_step2_train_idx', [])
        test_idx = session.get('mlr_ga_step2_test_idx', [])

        # Collect all removed samples
        all_removed_samples = []

        # Update train indices if removing train outliers
        if remove_train and len(outlier_info_train['removed_samples']) > 0:
            # keep_indices from outlier detection are indices in the reset dataframe (0, 1, 2, ...)
            train_keep_indices_positions = outlier_info_train['keep_indices']
            # Map to original indices using the mapping we stored
            train_keep_indices_absolute = [train_original_indices[i] for i in train_keep_indices_positions]
            # Update train_idx
            train_idx = train_keep_indices_absolute
            all_removed_samples.extend(outlier_info_train['removed_samples'])

        # Update test indices if removing test outliers
        if remove_test and len(outlier_info_test['removed_samples']) > 0:
            # keep_indices from outlier detection are indices in the reset dataframe (0, 1, 2, ...)
            test_keep_indices_positions = outlier_info_test['keep_indices']
            # Map to original indices using the mapping we stored
            test_keep_indices_absolute = [test_original_indices[i] for i in test_keep_indices_positions]
            # Update test_idx
            test_idx = test_keep_indices_absolute
            all_removed_samples.extend(outlier_info_test['removed_samples'])

        # Store updated indices
        session['mlr_ga_step2_train_idx'] = train_idx
        session['mlr_ga_step2_test_idx'] = test_idx

        # Store removal results
        session['mlr_ga_step2_outliers_removed'] = True
        session['mlr_ga_step2_total_removed'] = len(all_removed_samples)
        session['mlr_ga_step2_train_remaining'] = len(train_idx)
        session['mlr_ga_step2_test_remaining'] = len(test_idx)
        session['mlr_ga_step2_removed_samples'] = all_removed_samples

        # Store final cleaned indices for use in preprocessing (combine train and test)
        final_clean_indices = sorted(train_idx + test_idx)
        session['mlr_ga_step2_clean_indices'] = final_clean_indices

        # Clear detection results to allow re-detection
        session.pop('mlr_ga_step2_outliers_detected', None)
        session.pop('mlr_ga_step2_preview_current', None)
        session.pop('mlr_ga_step2_preview_after', None)

        flash(f'Successfully removed {len(all_removed_samples)} outlier(s)!', 'success')

    except Exception as e:
        flash(f'Error removing outliers: {str(e)}', 'danger')
        import traceback
        traceback.print_exc()

    return redirect(url_for('mlr_ga.mlr_ga_analysis'))


@mlr_ga_bp.route("/download_step2_outliers")
def download_step2_outliers():
    """Download CSV with removed Y outliers from Step 2"""
    if not check_dataset():
        flash('No dataset found', 'danger')
        return redirect(url_for('mlr_ga.mlr_ga_analysis'))

    removed_samples = session.get('mlr_ga_step2_removed_samples', [])

    if not removed_samples:
        flash('No outliers were removed in Step 2', 'info')
        return redirect(url_for('mlr_ga.mlr_ga_analysis'))

    try:
        # Create DataFrame from removed samples
        outliers_df = pd.DataFrame(removed_samples)

        # Reorder columns: Sample_Index first, Y_Value, then rest
        cols = outliers_df.columns.tolist()
        priority_cols = []

        if 'Sample_Index' in cols:
            priority_cols.append('Sample_Index')
        if 'Y_Value' in cols:
            priority_cols.append('Y_Value')

        # Add ID columns if present
        id_candidates = [col for col in cols if col.lower() in ['id', 'sample_id', 'name', 'compound', 'sample_name', 'sample']]
        priority_cols.extend([col for col in id_candidates if col not in priority_cols])

        # Add target variable if present
        target_var = session.get('mlr_ga_target_var')
        if target_var and target_var in cols and target_var not in priority_cols:
            priority_cols.append(target_var)

        # Add remaining columns
        remaining_cols = [col for col in cols if col not in priority_cols]
        final_cols = priority_cols + remaining_cols

        outliers_df = outliers_df[final_cols]

        # Save to temp CSV
        temp_path = ensure_temp_dir()
        temp_file = os.path.join(temp_path, 'step2_split_outliers_removed.csv')
        outliers_df.to_csv(temp_file, index=False)

        return send_file(temp_file, as_attachment=True, download_name='step2_split_outliers_removed.csv')

    except Exception as e:
        flash(f'Error generating outliers file: {str(e)}', 'danger')
        return redirect(url_for('mlr_ga.mlr_ga_analysis'))


@mlr_ga_bp.route("/mlr_ga_advance_to_preprocessing", methods=['POST'])
def mlr_ga_advance_to_preprocessing():
    """Advance from split_selection to preprocessing after reviewing split histograms"""
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))

    # Move to next step
    session['mlr_ga_step'] = 'preprocessing'
    flash('Proceed to preprocessing.', 'success')

    return redirect(url_for('mlr_ga.mlr_ga_analysis'))


@mlr_ga_bp.route("/mlr_ga_advance_step", methods=['POST'])
def mlr_ga_advance_step():
    """Advance from target_selection to split_selection after viewing histogram"""
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))

    # Move to next step
    session['mlr_ga_step'] = 'split_selection'
    flash('Proceed to split selection.', 'success')

    return redirect(url_for('mlr_ga.mlr_ga_analysis'))


@mlr_ga_bp.route("/mlr_ga_back_to_target", methods=['POST'])
def mlr_ga_back_to_target():
    """Go back to target selection step"""
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))

    session['mlr_ga_step'] = 'target_selection'
    flash('Returned to target variable selection.', 'info')
    return redirect(url_for('mlr_ga.mlr_ga_analysis'))


@mlr_ga_bp.route("/mlr_ga_back_to_split", methods=['POST'])
def mlr_ga_back_to_split():
    """Go back to split selection step"""
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))

    session['mlr_ga_step'] = 'split_selection'
    flash('Returned to split method configuration.', 'info')
    return redirect(url_for('mlr_ga.mlr_ga_analysis'))


@mlr_ga_bp.route("/mlr_ga_back_to_preprocessing", methods=['POST'])
def mlr_ga_back_to_preprocessing():
    """Go back to preprocessing step"""
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))

    session['mlr_ga_step'] = 'preprocessing'
    flash('Returned to preprocessing configuration.', 'info')
    return redirect(url_for('mlr_ga.mlr_ga_analysis'))


@mlr_ga_bp.route("/mlr_ga_back_to_ga_params", methods=['POST'])
def mlr_ga_back_to_ga_params():
    """Go back to GA parameters step"""
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))

    session['mlr_ga_step'] = 'ga_parameters'
    flash('Returned to GA parameters configuration.', 'info')
    return redirect(url_for('mlr_ga.mlr_ga_analysis'))


@mlr_ga_bp.route("/mlr_ga_back_to_model_selection", methods=['POST'])
def mlr_ga_back_to_model_selection():
    """Go back to model selection step (after GA results)"""
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))

    # Check if GA results exist
    if not session.get('mlr_ga_best_models'):
        flash('No GA results available. Please run GA first.', 'warning')
        session['mlr_ga_step'] = 'ga_parameters'
        return redirect(url_for('mlr_ga.mlr_ga_analysis'))

    session['mlr_ga_step'] = 'model_selection'
    flash('Returned to model selection.', 'info')
    return redirect(url_for('mlr_ga.mlr_ga_analysis'))


@mlr_ga_bp.route("/mlr_ga_step2", methods=['POST'])
def mlr_ga_step2():
    """Step 2: Configure split method and generate train/test histograms"""
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))

    try:
        from app.chemalize.modules.ga_variable_selection import plot_y_histogram_split
        from sklearn.model_selection import train_test_split

        # Get split method parameters
        split_method = request.form.get('split_method', 'random')

        # Store parameters based on split method
        session['mlr_ga_split_method'] = split_method

        if split_method == 'random':
            session['mlr_ga_test_size'] = float(request.form.get('test_size', 0.2))
            session['mlr_ga_shuffle'] = 'shuffle' in request.form
            session['mlr_ga_random_state'] = int(request.form.get('random_state', 42))

        elif split_method == 'stratified':
            session['mlr_ga_strat_test_size'] = float(request.form.get('strat_test_size', 0.2))
            session['mlr_ga_strat_bins'] = int(request.form.get('strat_bins', 5))

        elif split_method == 'time':
            session['mlr_ga_time_column'] = request.form.get('time_column', '')
            session['mlr_ga_time_test_size'] = float(request.form.get('time_test_size', 0.2))

        elif split_method == 'kfold':
            session['mlr_ga_n_folds'] = int(request.form.get('n_folds', 5))
            session['mlr_ga_shuffle_kfold'] = 'shuffle_kfold' in request.form

        elif split_method == 'systematic':
            session['mlr_ga_systematic_step'] = int(request.form.get('systematic_step', 3))
            session['mlr_ga_include_last_point'] = 'include_last_point' in request.form

        # For K-Fold and LOOCV, skip split histograms and go directly to preprocessing
        if split_method in ['kfold', 'loocv']:
            session['mlr_ga_step'] = 'preprocessing'
            flash('Split method configured! K-Fold/LOOCV does not support train/test outlier removal.', 'info')
            return redirect(url_for('mlr_ga.mlr_ga_analysis'))

        # For other split methods, perform split and generate histograms
        # Read data
        clean_path = get_clean_path(session["csv_name"])
        df = read_dataset(clean_path)

        # Apply Step 1 outlier removal if it was performed
        step1_clean_indices = session.get('mlr_ga_step1_clean_indices')
        if step1_clean_indices:
            df = df.loc[step1_clean_indices].reset_index(drop=True)

        target_var = session.get('mlr_ga_target_var')
        y_transformation = session.get('mlr_ga_y_transformation', 'none')

        if not target_var or target_var not in df.columns:
            flash('Target variable not set or not found!', 'danger')
            return redirect(url_for('mlr_ga.mlr_ga_analysis'))

        y = df[target_var].copy()

        # Apply transformation if needed
        if y_transformation != 'none':
            if y_transformation == 'log':
                if (y <= 0).any():
                    flash('Log transformation requires all positive values!', 'danger')
                    return redirect(url_for('mlr_ga.mlr_ga_analysis'))
                y = np.log(y)
            elif y_transformation == 'sqrt':
                if (y < 0).any():
                    flash('Square root transformation requires non-negative values!', 'danger')
                    return redirect(url_for('mlr_ga.mlr_ga_analysis'))
                y = np.sqrt(y)
            elif y_transformation == 'square':
                y = y ** 2
            elif y_transformation == 'inverse':
                if (y == 0).any():
                    flash('Inverse transformation requires non-zero values!', 'danger')
                    return redirect(url_for('mlr_ga.mlr_ga_analysis'))
                y = 1 / y

        # Perform split based on method
        if split_method == 'random':
            test_size = session.get('mlr_ga_test_size', 0.2)
            shuffle = session.get('mlr_ga_shuffle', True)
            random_state = session.get('mlr_ga_random_state', 42)
            train_idx, test_idx = train_test_split(
                range(len(y)), test_size=test_size, shuffle=shuffle, random_state=random_state
            )
        elif split_method == 'stratified':
            from sklearn.model_selection import StratifiedShuffleSplit
            test_size = session.get('mlr_ga_strat_test_size', 0.2)
            n_bins = session.get('mlr_ga_strat_bins', 5)
            # Create bins for stratification
            y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            train_idx, test_idx = next(sss.split(np.zeros(len(y)), y_binned))
        elif split_method == 'time':
            # Time-based split: last N% goes to test
            test_size = session.get('mlr_ga_time_test_size', 0.2)
            split_point = int(len(y) * (1 - test_size))
            train_idx = list(range(split_point))
            test_idx = list(range(split_point, len(y)))
        elif split_method == 'systematic':
            # Sorted systematic sampling: sort by Y, then every nth observation goes to test
            # This ensures equal distribution of Y values in both train and test sets
            # Min and max Y values always go to TEST set (endpoints for validation)
            step = session.get('mlr_ga_systematic_step', 3)

            # Get indices sorted by Y value
            sorted_indices = y.argsort().tolist() if hasattr(y, 'argsort') else list(np.argsort(y))

            # Keep min (first) and max (last) for TEST
            min_idx = sorted_indices[0]
            max_idx = sorted_indices[-1]
            middle_indices = sorted_indices[1:-1]  # exclude first and last

            # Take every nth from middle indices for test, add min/max
            test_idx = [min_idx, max_idx] + [middle_indices[i] for i in range(0, len(middle_indices), step)]
            train_idx = [idx for idx in middle_indices if idx not in test_idx]
        else:
            flash(f'Unsupported split method: {split_method}', 'danger')
            return redirect(url_for('mlr_ga.mlr_ga_analysis'))

        # Get train and test sets
        y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        y_test = y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx]

        # Store split indices
        session['mlr_ga_step2_train_idx'] = train_idx
        session['mlr_ga_step2_test_idx'] = test_idx

        # Generate split histogram
        temp_path = ensure_temp_dir()
        plot_path = plot_y_histogram_split(
            y_train, y_test,
            y_name=target_var,
            temp_path=temp_path
        )

        # Store histogram filename
        plot_filename = os.path.basename(plot_path)
        session['mlr_ga_split_histogram'] = plot_filename
        session['mlr_ga_split_configured'] = True

        flash(f'Split configured! Review train/test distributions below.', 'success')

    except Exception as e:
        flash(f'Error configuring split: {str(e)}', 'danger')
        import traceback
        traceback.print_exc()

    return redirect(url_for('mlr_ga.mlr_ga_analysis'))


@mlr_ga_bp.route("/mlr_ga_step3", methods=['POST'])
def mlr_ga_step3():
    """Step 3: Configure standardization and variance check"""
    print("DEBUG: mlr_ga_step3 called")
    print(f"DEBUG: request.form = {request.form}")

    if not check_dataset():
        print("DEBUG: No dataset found, redirecting to preprocess")
        return redirect(url_for('preprocessing.preprocess'))

    try:
        # Get preprocessing parameters
        autoscale = 'autoscale' in request.form
        remove_zero_variance = 'remove_zero_variance' in request.form
        remove_low_variance = 'remove_low_variance' in request.form

        print(f"DEBUG: autoscale={autoscale}, remove_zero_variance={remove_zero_variance}, remove_low_variance={remove_low_variance}")

        session['mlr_ga_autoscale'] = autoscale
        session['mlr_ga_remove_zero_variance'] = remove_zero_variance
        session['mlr_ga_remove_low_variance'] = remove_low_variance

        # Parse variance threshold with error handling
        variance_threshold_str = request.form.get('variance_threshold', '0.01')
        print(f"DEBUG: variance_threshold_str = '{variance_threshold_str}'")

        try:
            variance_threshold = float(variance_threshold_str) if variance_threshold_str else 0.01
            session['mlr_ga_variance_threshold'] = variance_threshold
            print(f"DEBUG: variance_threshold = {variance_threshold}")
        except (ValueError, TypeError) as e:
            print(f"DEBUG: Error parsing variance_threshold: {e}")
            session['mlr_ga_variance_threshold'] = 0.01

        # Parse internal CV type parameters
        internal_cv_type = request.form.get('internal_cv_type', 'kfold')
        session['mlr_ga_internal_cv_type'] = internal_cv_type
        print(f"DEBUG: internal_cv_type = '{internal_cv_type}'")

        # Parse sorted CV parameters
        try:
            sorted_step = int(request.form.get('sorted_step', 5))
            session['mlr_ga_sorted_step'] = sorted_step
        except (ValueError, TypeError):
            session['mlr_ga_sorted_step'] = 5

        try:
            sorted_iterations = int(request.form.get('sorted_iterations', 5))
            session['mlr_ga_sorted_iterations'] = sorted_iterations
        except (ValueError, TypeError):
            session['mlr_ga_sorted_iterations'] = 5

        session['mlr_ga_step'] = 'ga_parameters'
        print("DEBUG: Step set to 'ga_parameters'")
        flash('Preprocessing configured! Proceed to GA parameters.', 'success')

    except Exception as e:
        print(f"DEBUG: Exception in mlr_ga_step3: {e}")
        flash(f'Error configuring preprocessing: {str(e)}', 'danger')
        import traceback
        traceback.print_exc()

    return redirect(url_for('mlr_ga.mlr_ga_analysis'))


@mlr_ga_bp.route("/mlr_ga_get_progress")
def mlr_ga_get_progress():
    """Polling endpoint for GA progress updates - returns JSON"""
    session_id = request.args.get('session_id') or get_user_id()

    with ga_progress_lock:
        progress_data = ga_latest_progress.get(session_id, {'status': 'waiting'})

    # Additional safety: convert any numpy types to Python native types
    def convert_to_json_serializable(obj):
        """Recursively convert numpy types to Python native types and skip non-serializable objects"""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            value = float(obj)
            return value if math.isfinite(value) else None
        elif isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        elif isinstance(obj, np.ndarray):
            return convert_to_json_serializable(obj.tolist())
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Skip non-serializable objects (sklearn objects, etc.)
            return str(type(obj).__name__)

    progress_data = convert_to_json_serializable(progress_data)
    return jsonify(progress_data)


@mlr_ga_bp.route("/mlr_ga_start", methods=['POST'])
def mlr_ga_start():
    """Start GA in background thread and return immediately"""
    if not check_dataset():
        return jsonify({'status': 'error', 'message': 'No dataset found'}), 400

    session_id = get_user_id()

    # Check if GA is already running for this session
    with ga_progress_lock:
        if session_id in ga_threads and ga_threads[session_id].is_alive():
            return jsonify({'status': 'error', 'message': 'GA is already running'}), 400

        # Initialize progress
        ga_latest_progress[session_id] = {'status': 'starting', 'message': 'Initializing...'}

    # Get all form data
    form_data = request.form.to_dict()

    # Compute file paths in request context (before starting thread)
    csv_name = session.get('csv_name')
    if not csv_name:
        return jsonify({'status': 'error', 'message': 'No dataset found in session'}), 400

    clean_path = get_clean_path(csv_name)
    temp_path = ensure_temp_dir()  # Get temp path in request context

    # Start GA in background thread
    thread = threading.Thread(target=run_ga_background, args=(session_id, form_data, dict(session), clean_path, temp_path))
    thread.daemon = True
    thread.start()

    with ga_progress_lock:
        ga_threads[session_id] = thread

    return jsonify({'status': 'started', 'session_id': session_id})


def run_ga_background(session_id, form_data, session_data, clean_path, temp_path):
    """Background thread function to run GA"""
    try:
        # Extract parameters from form_data
        n_variables_str = form_data.get('n_variables', '').strip()
        n_variables = int(n_variables_str) if n_variables_str else None

        correlation_threshold = float(form_data.get('correlation_threshold', 0.95))
        mutation_rate = float(form_data.get('mutation_rate', 0.1))
        random_models_ratio = float(form_data.get('random_models_ratio', 0.1))
        use_validation = 'use_validation' in form_data
        population_size = int(form_data.get('population_size', 50))
        n_iterations = int(form_data.get('n_iterations', 100))
        max_retries = int(form_data.get('max_retries', 3))
        cv_folds = int(form_data.get('cv_folds', 5))
        cv_folds_validation = int(form_data.get('cv_folds_validation', 3))
        test_normality = 'test_normality' in form_data
        normality_alpha = float(form_data.get('normality_alpha', 0.05))
        n_best_models = int(form_data.get('n_best_models', 5))
        check_ad = 'check_ad' in form_data
        ad_threshold = float(form_data.get('ad_threshold', 100.0))

        # Read data (clean_path passed from main thread)
        df = read_dataset(clean_path)

        # Apply outlier removal from previous steps
        # Priority: Step 2 > Step 1 (Step 2 includes Step 1 if both were done)
        step2_clean_indices = session_data.get('mlr_ga_step2_clean_indices')
        step1_clean_indices = session_data.get('mlr_ga_step1_clean_indices')

        if step2_clean_indices:
            if step1_clean_indices:
                # Map step2 indices (post-step1) back to original indices
                mapped_indices = [
                    step1_clean_indices[i]
                    for i in step2_clean_indices
                    if i < len(step1_clean_indices)
                ]
                df = df.loc[mapped_indices].reset_index(drop=True)
            else:
                # Step 2 indices are already based on original data
                df = df.loc[step2_clean_indices].reset_index(drop=True)
        elif step1_clean_indices:
            # Only Step 1 was performed
            df = df.loc[step1_clean_indices].reset_index(drop=True)

        target_var = session_data.get('mlr_ga_target_var')
        if not target_var:
            with ga_progress_lock:
                ga_latest_progress[session_id] = {'status': 'error', 'message': 'Target variable not set'}
            return

        # Filter only numeric columns
        df_numeric = df.select_dtypes(include=[np.number])

        # Check if target variable is numeric
        if target_var not in df_numeric.columns:
            with ga_progress_lock:
                ga_latest_progress[session_id] = {'status': 'error', 'message': f'Target variable "{target_var}" is not numeric'}
            return

        # Remove columns with missing values
        df_clean = df_numeric.dropna(axis=1)

        if df_clean.shape[1] < 2:
            with ga_progress_lock:
                ga_latest_progress[session_id] = {'status': 'error', 'message': 'Not enough numeric columns without missing values'}
            return

        # Update progress
        with ga_progress_lock:
            ga_latest_progress[session_id] = {'status': 'preprocessing', 'message': 'Preprocessing data...'}

        # Preprocessing
        y_transformation = session_data.get('mlr_ga_y_transformation', 'none')
        autoscale = session_data.get('mlr_ga_autoscale', True)
        remove_zero_variance = session_data.get('mlr_ga_remove_zero_variance', True)
        remove_low_variance = session_data.get('mlr_ga_remove_low_variance', False)
        variance_threshold = session_data.get('mlr_ga_variance_threshold', 0.01)

        split_method = session_data.get('mlr_ga_split_method', 'random')
        split_params = {}
        if split_method == 'random':
            split_params = {
                'test_size': session_data.get('mlr_ga_test_size', 0.2),
                'shuffle': session_data.get('mlr_ga_shuffle', True),
                'random_state': session_data.get('mlr_ga_random_state', 42)
            }
        elif split_method == 'stratified':
            split_params = {
                'test_size': session_data.get('mlr_ga_strat_test_size', 0.2),
                'n_bins': session_data.get('mlr_ga_strat_bins', 5)
            }
        elif split_method == 'time':
            split_params = {
                'time_column': session_data.get('mlr_ga_time_column', ''),
                'test_size': session_data.get('mlr_ga_time_test_size', 0.2)
            }
        elif split_method == 'kfold':
            split_params = {
                'n_folds': session_data.get('mlr_ga_n_folds', 5),
                'shuffle': session_data.get('mlr_ga_shuffle_kfold', True)
            }
        elif split_method == 'systematic':
            split_params = {
                'step': session_data.get('mlr_ga_systematic_step', 3),
                'include_last': session_data.get('mlr_ga_include_last_point', False)
            }

        if split_method == 'time' or split_method == 'systematic':
            shuffle_cv = False
        elif split_method == 'kfold':
            shuffle_cv = session_data.get('mlr_ga_shuffle_kfold', True)
        elif split_method == 'random':
            shuffle_cv = session_data.get('mlr_ga_shuffle', True)
        else:
            shuffle_cv = True

        # Get internal CV type parameters
        internal_cv_type = session_data.get('mlr_ga_internal_cv_type', 'kfold')
        sorted_step = session_data.get('mlr_ga_sorted_step', 5)
        sorted_iterations = session_data.get('mlr_ga_sorted_iterations', 5)

        X, y, removed_features, preprocessing_info = preprocess_for_ga(
            df_clean, target_var,
            y_transformation=y_transformation,
            autoscale=autoscale,
            remove_zero_variance=remove_zero_variance,
            remove_low_variance=remove_low_variance,
            variance_threshold=variance_threshold
        )

        if X.shape[1] == 0:
            with ga_progress_lock:
                ga_latest_progress[session_id] = {'status': 'error', 'message': 'No features remaining after preprocessing'}
            return

        split_train_idx = None
        split_test_idx = None
        if split_method not in ['kfold', 'loocv']:
            split_train_idx = session_data.get('mlr_ga_step2_train_idx')
            split_test_idx = session_data.get('mlr_ga_step2_test_idx')
            if split_train_idx is not None:
                if split_test_idx is None:
                    split_test_idx = []
                if step2_clean_indices:
                    cleaned_indices = sorted(step2_clean_indices)
                    index_map = {orig_idx: new_pos for new_pos, orig_idx in enumerate(cleaned_indices)}
                    split_train_idx = [index_map[i] for i in split_train_idx if i in index_map]
                    split_test_idx = [index_map[i] for i in split_test_idx if i in index_map]
                data_len = len(y)
                split_train_idx = [int(i) for i in split_train_idx if 0 <= int(i) < data_len]
                split_test_idx = [int(i) for i in split_test_idx if 0 <= int(i) < data_len]
            else:
                split_train_idx = None
                split_test_idx = None

        # Split data for validation if requested
        X_train, X_val, y_train, y_val = None, None, None, None
        if use_validation:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, y_train = X, y

        # Create progress callback
        def progress_callback(data):
            """Callback to report GA progress"""
            # Convert all numpy types to Python native types for JSON serialization
            def convert_numpy_types(obj):
                """Recursively convert numpy types to Python native types"""
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj

            converted_data = convert_numpy_types(data)
            with ga_progress_lock:
                ga_latest_progress[session_id] = converted_data

        # Update progress
        with ga_progress_lock:
            ga_latest_progress[session_id] = {'status': 'running', 'message': 'Starting Genetic Algorithm...'}

        # Initialize GA with progress callback
        ga = GeneticAlgorithmSelector(
            n_variables=n_variables,
            correlation_threshold=correlation_threshold,
            mutation_rate=mutation_rate,
            random_models_ratio=random_models_ratio,
            population_size=population_size,
            n_iterations=n_iterations,
            max_retries=max_retries,
            cv_folds=cv_folds,
            cv_folds_validation=cv_folds_validation,
            use_validation=use_validation,
            test_normality=test_normality,
            normality_alpha=normality_alpha,
            n_best_models=n_best_models,
            check_ad=check_ad,
            ad_threshold=ad_threshold,
            split_method=split_method,
            split_params=split_params,
            split_train_idx=split_train_idx,
            split_test_idx=split_test_idx,
            metrics_X=X,
            metrics_y=y,
            progress_callback=progress_callback,
            shuffle_cv=shuffle_cv,
            internal_cv_type=internal_cv_type,
            sorted_step=sorted_step,
            sorted_iterations=sorted_iterations,
            random_state=42
        )

        # Run GA
        ga.fit(X_train, y_train, X_val, y_val)

        # Get all top models
        try:
            best_models = ga.get_best_models()
        except ValueError as e:
            message = str(e)
            if not message or message == "Model not fitted yet!":
                message = (
                    'No valid models found. Try disabling AD check, '
                    'lowering correlation threshold, or reducing CV folds.'
                )
            with ga_progress_lock:
                ga_latest_progress[session_id] = {
                    'status': 'error',
                    'message': message
                }
            return

        # Get selected features (best model)
        try:
            selected_features = ga.get_selected_features()
        except ValueError:
            with ga_progress_lock:
                ga_latest_progress[session_id] = {
                    'status': 'error',
                    'message': 'No valid models found. Try disabling AD check, lowering correlation threshold, or reducing CV folds.'
                }
            return
        best_score = float(ga.best_score_)

        # Plot fitness history (temp_path passed from main thread)
        fitness_plot = ga.plot_fitness_history(temp_path=temp_path)

        # Store results in a way that can be retrieved by main thread
        # We'll use the progress dict to store completion status and results
        with ga_progress_lock:
            ga_latest_progress[session_id] = {
                'status': 'complete',
                'message': f'GA completed! Found {len(best_models)} candidate models.',
                'results': {
                    'best_models': best_models,
                    'selected_features': selected_features,
                    'best_score': best_score,
                    'fitness_plot': fitness_plot,
                    'removed_features': removed_features,
                    'preprocessing_info': preprocessing_info,
                    # Save X_train/y_train (data used for GA fitting) for MLR to use
                    'X_data': X_train.to_json() if hasattr(X_train, 'to_json') else pd.DataFrame(X_train).to_json(),
                    'y_data': y_train.tolist() if hasattr(y_train, 'tolist') else list(y_train),
                    # Also save full data for reference
                    'X_full': X.to_json(),
                    'y_full': y.tolist(),
                    'n_variables': n_variables if n_variables else '',
                    'correlation_threshold': correlation_threshold,
                    'mutation_rate': mutation_rate,
                    'random_models_ratio': random_models_ratio,
                    'use_validation': use_validation,
                    'population_size': population_size,
                    'n_iterations': n_iterations,
                    'max_retries': max_retries,
                    'cv_folds': cv_folds,
                    'cv_folds_validation': cv_folds_validation,
                    'test_normality': test_normality,
                    'normality_alpha': normality_alpha,
                    'n_best_models': n_best_models,
                    'check_ad': check_ad,
                    'ad_threshold': ad_threshold,
                    # Store split indices used by GA for MLR to use
                    'split_method': split_method,
                    'split_train_idx': split_train_idx,
                    'split_test_idx': split_test_idx,
                    # Store internal CV type settings
                    'internal_cv_type': internal_cv_type,
                    'sorted_step': sorted_step,
                    'sorted_iterations': sorted_iterations,
                }
            }

    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        with ga_progress_lock:
            ga_latest_progress[session_id] = {
                'status': 'error',
                'message': f'Error during GA analysis: {error_msg}'
            }


@mlr_ga_bp.route("/mlr_ga_complete", methods=['POST'])
def mlr_ga_complete():
    """Retrieve GA results from background thread and store in session"""
    if not check_dataset():
        return jsonify({'status': 'error', 'message': 'No dataset found'}), 400

    session_id = get_user_id()

    try:
        # Get results from background thread
        with ga_progress_lock:
            progress_data = ga_latest_progress.get(session_id, {})

        if progress_data.get('status') != 'complete':
            return jsonify({'status': 'error', 'message': 'GA has not completed yet'}), 400

        results = progress_data.get('results', {})
        if not results:
            return jsonify({'status': 'error', 'message': 'No results found'}), 400

        # Store all results in session
        session['mlr_ga_best_models'] = results['best_models']
        session['mlr_ga_selected_features'] = results['selected_features']
        session['mlr_ga_best_score'] = results['best_score']
        session['mlr_ga_fitness_plot'] = results['fitness_plot']
        session['mlr_ga_removed_features'] = results['removed_features']
        session['mlr_ga_preprocessing_info'] = results['preprocessing_info']
        session['mlr_ga_X_data'] = results['X_data']
        session['mlr_ga_y_data'] = results['y_data']
        session['mlr_ga_X_full'] = results.get('X_full', results['X_data'])  # Full data for MLR
        session['mlr_ga_y_full'] = results.get('y_full', results['y_data'])  # Full data for MLR

        # Store GA parameters
        session['mlr_ga_n_variables'] = results['n_variables']
        session['mlr_ga_correlation_threshold'] = results['correlation_threshold']
        session['mlr_ga_mutation_rate'] = results['mutation_rate']
        session['mlr_ga_random_models_ratio'] = results['random_models_ratio']
        session['mlr_ga_use_validation'] = results['use_validation']
        session['mlr_ga_population_size'] = results['population_size']
        session['mlr_ga_n_iterations'] = results['n_iterations']
        session['mlr_ga_max_retries'] = results['max_retries']
        session['mlr_ga_cv_folds'] = results['cv_folds']
        session['mlr_ga_cv_folds_validation'] = results['cv_folds_validation']
        session['mlr_ga_test_normality'] = results['test_normality']
        session['mlr_ga_normality_alpha'] = results['normality_alpha']
        session['mlr_ga_n_best_models'] = results['n_best_models']
        session['mlr_ga_check_ad'] = results['check_ad']
        session['mlr_ga_ad_threshold'] = results['ad_threshold']

        # Store split indices used by GA (mapped to preprocessed data)
        session['mlr_ga_final_split_method'] = results.get('split_method')
        session['mlr_ga_final_train_idx'] = results.get('split_train_idx')
        session['mlr_ga_final_test_idx'] = results.get('split_test_idx')

        # Store internal CV type settings used by GA
        session['mlr_ga_final_internal_cv_type'] = results.get('internal_cv_type', 'kfold')
        session['mlr_ga_final_sorted_step'] = results.get('sorted_step', 5)
        session['mlr_ga_final_sorted_iterations'] = results.get('sorted_iterations', 5)

        session['mlr_ga_step'] = 'model_selection'

        # Cleanup
        with ga_progress_lock:
            if session_id in ga_latest_progress:
                del ga_latest_progress[session_id]
            if session_id in ga_threads:
                del ga_threads[session_id]

        return jsonify({
            'status': 'success',
            'message': f'GA completed! Found {len(results["best_models"])} candidate models.',
            'redirect': url_for('mlr_ga.mlr_ga_analysis')
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@mlr_ga_bp.route("/mlr_ga_run_mlr", methods=['POST'])
def mlr_ga_run_mlr():
    """Run MLR analysis for selected model from GA results"""
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))

    try:
        # Get selected model rank
        selected_rank = int(request.form.get('selected_model', 1))

        best_models = session.get('mlr_ga_best_models', [])
        if not best_models:
            flash('No GA results found! Please run GA first.', 'danger')
            return redirect(url_for('mlr_ga.mlr_ga_analysis'))

        # Find selected model
        selected_model = None
        for model in best_models:
            if model['rank'] == selected_rank:
                selected_model = model
                break

        if not selected_model:
            flash('Selected model not found!', 'danger')
            return redirect(url_for('mlr_ga.mlr_ga_analysis'))

        selected_features = selected_model['feature_names']
        session['mlr_ga_selected_features'] = selected_features
        session['mlr_ga_selected_model_rank'] = selected_rank

        flash(f'Running MLR for Model {selected_rank} with {len(selected_features)} features...', 'info')

        # Load full preprocessed data (X_full contains all data, X_data contains training data)
        X_full_json = session.get('mlr_ga_X_full')
        y_full_list = session.get('mlr_ga_y_full')

        # Fallback to X_data if X_full not available (backward compatibility)
        if X_full_json is None:
            X_full_json = session.get('mlr_ga_X_data')
            y_full_list = session.get('mlr_ga_y_data')

        X = pd.read_json(X_full_json)
        y = np.array(y_full_list)
        target_var = session.get('mlr_ga_target_var')

        # Get split indices that GA actually used (mapped to preprocessed data)
        ga_split_method = session.get('mlr_ga_final_split_method')
        ga_train_idx = session.get('mlr_ga_final_train_idx')
        ga_test_idx = session.get('mlr_ga_final_test_idx')

        # Prepare predefined indices - use EXACT split that GA used
        predefined_train_idx = None
        predefined_test_idx = None
        split_params = {}

        if ga_train_idx is not None and ga_split_method not in ['kfold', 'loocv']:
            # Use the exact indices GA used for train/test split
            predefined_train_idx = [int(i) for i in ga_train_idx if 0 <= int(i) < len(y)]
            predefined_test_idx = [int(i) for i in (ga_test_idx or []) if 0 <= int(i) < len(y)]
            split_method = 'predefined'
        elif ga_split_method in ['kfold', 'loocv']:
            # For K-Fold/LOOCV, use all data as train (MLR will show training metrics)
            predefined_train_idx = list(range(len(y)))
            predefined_test_idx = []
            split_method = 'predefined'
        else:
            # Fallback: check if GA used validation split
            use_validation = session.get('mlr_ga_use_validation', False)
            if use_validation:
                # GA used train_test_split(X, y, test_size=0.2, random_state=42)
                from sklearn.model_selection import train_test_split
                n_samples = len(y)
                all_indices = list(range(n_samples))
                train_indices, test_indices = train_test_split(
                    all_indices, test_size=0.2, random_state=42
                )
                predefined_train_idx = sorted(train_indices)
                predefined_test_idx = sorted(test_indices)
            else:
                # GA used all data for training (no validation split)
                predefined_train_idx = list(range(len(y)))
                predefined_test_idx = []
            split_method = 'predefined'

        # Prepare dataframe with preprocessed data and selected features
        df_for_mlr = X[selected_features].copy()
        df_for_mlr[target_var] = y

        # Get temp path
        temp_path = ensure_temp_dir()

        # Save GA preprocessed data to file for MLR to use
        ga_data_filename = 'ga_preprocessed_data.csv'
        ga_data_path = os.path.join(temp_path, ga_data_filename)
        df_for_mlr.to_csv(ga_data_path, index=False)
        session['ga_preprocessed_data_path'] = ga_data_path

        # Get coefficients from selected GA model
        ga_intercept = selected_model.get('intercept')
        ga_coefficients = selected_model.get('coefficients')

        # Run MLR with predefined indices and GA coefficients
        results = mlr.perform_mlr(
            df=df_for_mlr,
            target_var=target_var,
            selected_features=selected_features,
            include_intercept=True,
            split_method='predefined' if predefined_train_idx else split_method,
            split_params=split_params,
            scale_data=False,  # Already scaled
            check_assumptions=True,
            detect_outliers=True,
            temp_path=temp_path,
            predefined_train_idx=predefined_train_idx,
            predefined_test_idx=predefined_test_idx,
            ga_coefficients=ga_coefficients,
            ga_intercept=ga_intercept
        )

        # Store MLR results in session (compatible with standard MLR module)
        session['mlr_performed'] = True
        session['target_var'] = target_var
        session['selected_features'] = selected_features
        session['include_intercept'] = True
        session['scale_data'] = False  # Already scaled in GA preprocessing
        session['check_assumptions'] = True
        session['detect_outliers'] = True
        session['split_method'] = 'predefined' if predefined_train_idx else split_method

        # Store predefined indices for MLR to use
        if predefined_train_idx:
            session['predefined_train_idx'] = predefined_train_idx
            session['predefined_test_idx'] = predefined_test_idx

        # Add timestamp to prevent browser caching of images
        timestamp = int(time.time() * 1000)

        # Store all MLR results using standard MLR keys
        # Convert image file paths to URLs (same as standard MLR module)
        for key, value in results.items():
            # Check if the value looks like an image file path
            if isinstance(value, str) and (value.endswith('.png') or value.endswith('.jpg')):
                # Extract filename and convert to URL with timestamp
                filename = os.path.basename(value)
                session_key = key if key.startswith('mlr_') else f'mlr_{key}' if '_plot' in key else key
                session[session_key] = url_for('utils.serve_temp_image', filename=filename, t=timestamp)
            elif key in ['pred_actual_plot', 'residuals_plot', 'residuals_hist', 'qq_plot', 'williams_plot']:
                # Handle plot keys specifically
                if value and isinstance(value, str):
                    filename = os.path.basename(value)
                    session[f'mlr_{key}'] = url_for('utils.serve_temp_image', filename=filename, t=timestamp)
            else:
                # Store other values as-is
                session[key] = value

        # Calculate Q²ext and RMSE_ext on TEST set (external validation)
        # These are calculated AFTER model selection, on the external test set
        q2_ext = None
        rmse_ext = None
        if predefined_test_idx and len(predefined_test_idx) > 0:
            try:
                from sklearn.linear_model import LinearRegression
                X_train_data = X[selected_features].iloc[predefined_train_idx].values
                y_train_data = y[predefined_train_idx]
                X_test_data = X[selected_features].iloc[predefined_test_idx].values
                y_test_data = y[predefined_test_idx]

                # Fit model on TRAIN
                model_ext = LinearRegression()
                model_ext.fit(X_train_data, y_train_data)

                # Predict on TEST
                y_pred_test = model_ext.predict(X_test_data)

                # Calculate Q²ext = 1 - SSE/TSS (using train mean for TSS)
                y_train_mean = np.mean(y_train_data)
                sse_test = np.sum((y_test_data - y_pred_test) ** 2)
                tss_test = np.sum((y_test_data - y_train_mean) ** 2)
                if tss_test > 0:
                    q2_ext = float(1 - (sse_test / tss_test))

                # Calculate RMSE_ext
                rmse_ext = float(np.sqrt(np.mean((y_test_data - y_pred_test) ** 2)))
            except Exception as e:
                print(f"Error calculating Q²ext/RMSE_ext: {e}")

        # Store external validation metrics
        session['q2_ext'] = q2_ext
        session['rmse_ext'] = rmse_ext

        # Store GA-specific info for reference
        session['from_ga'] = True
        session['ga_model_rank'] = selected_rank
        session['ga_best_score'] = selected_model.get('r2cv', selected_model.get('score', 0))  # Use r2cv (new) or score (old)

        # Store GA model metrics for display
        session['ga_r2cv'] = selected_model.get('r2cv')
        session['ga_r2'] = selected_model.get('r2')
        session['ga_r2loo'] = selected_model.get('r2loo')
        session['ga_rmse_tr'] = selected_model.get('rmse_tr')
        session['ga_ad_coverage'] = selected_model.get('ad_coverage')
        session['ga_internal_cv_type'] = session.get('mlr_ga_internal_cv_type', 'kfold')

        flash(f'MLR analysis completed for GA Model #{selected_rank}! Viewing full results with visualizations...', 'success')

    except Exception as e:
        flash(f'Error during MLR analysis: {str(e)}', 'danger')
        import traceback
        traceback.print_exc()
        return redirect(url_for('mlr_ga.mlr_ga_analysis'))

    # Redirect to standard MLR module for full visualization
    return redirect(url_for('mlr.mlr_analysis'))


@mlr_ga_bp.route("/download_y_outliers")
def download_y_outliers():
    """Download CSV with removed Y outliers from preprocessing"""
    if not check_dataset():
        flash('No dataset found', 'danger')
        return redirect(url_for('mlr_ga.mlr_ga_analysis'))

    preprocessing_info = session.get('mlr_ga_preprocessing_info', {})
    removed_samples = preprocessing_info.get('removed_samples', [])

    if not removed_samples:
        flash('No outliers were removed during preprocessing', 'info')
        return redirect(url_for('mlr_ga.mlr_ga_analysis'))

    try:
        # Create DataFrame from removed samples
        outliers_df = pd.DataFrame(removed_samples)

        # Reorder columns: Sample_Index first, Y_Value, then rest
        cols = outliers_df.columns.tolist()
        priority_cols = []

        if 'Sample_Index' in cols:
            priority_cols.append('Sample_Index')
        if 'Y_Value' in cols:
            priority_cols.append('Y_Value')

        # Add ID columns if present
        id_candidates = [col for col in cols if col.lower() in ['id', 'sample_id', 'name', 'compound', 'sample_name', 'sample']]
        priority_cols.extend([col for col in id_candidates if col not in priority_cols])

        # Add remaining columns
        remaining_cols = [col for col in cols if col not in priority_cols]
        final_cols = priority_cols + remaining_cols

        outliers_df = outliers_df[final_cols]

        # Save to temp CSV
        temp_path = ensure_temp_dir()
        temp_file = os.path.join(temp_path, 'y_outliers_removed.csv')
        outliers_df.to_csv(temp_file, index=False)

        return send_file(temp_file, as_attachment=True, download_name='y_outliers_removed.csv')

    except Exception as e:
        flash(f'Error generating outliers file: {str(e)}', 'danger')
        return redirect(url_for('mlr_ga.mlr_ga_analysis'))


@mlr_ga_bp.route("/mlr_ga_reset", methods=['POST'])
def mlr_ga_reset():
    """Reset the GA MLR workflow"""
    # Clear all GA-related session variables
    keys_to_remove = [k for k in session.keys() if k.startswith('mlr_ga_')]
    for key in keys_to_remove:
        session.pop(key, None)

    flash('MLR GA workflow reset successfully!', 'info')
    return redirect(url_for('mlr_ga.mlr_ga_analysis'))


@mlr_ga_bp.route("/mlr_ga_back_to_models")
def mlr_ga_back_to_models():
    """Return to GA model selection from MLR visualization"""
    # Clear only MLR-specific session variables (keep GA results)
    mlr_keys_to_clear = [
        'mlr_performed', 'train_r2', 'test_r2', 'adj_r2', 'q2_loo', 'q2_test',
        'train_rmse', 'test_rmse', 'rmse_loo', 'train_mae', 'test_mae',
        'f_statistic', 'f_pvalue', 'aic', 'bic', 'dw_stat', 'vif_values',
        'ccc_ext', 'coefficients', 'std_errors', 't_values', 'p_values',
        'feature_names', 'mlr_pred_actual_plot', 'mlr_residuals_plot',
        'mlr_residuals_hist', 'mlr_qq_plot', 'mlr_williams_plot',
        'AD_train', 'AD_test', 'h_star', 'from_ga', 'ga_model_rank', 'ga_best_score'
    ]

    for key in mlr_keys_to_clear:
        session.pop(key, None)

    # Set step back to model_selection to show the list of models
    session['mlr_ga_step'] = 'model_selection'

    flash('Returned to GA model selection. Choose another model to analyze.', 'info')
    return redirect(url_for('mlr_ga.mlr_ga_analysis'))
