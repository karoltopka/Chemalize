"""
Data visualization routes
"""
from flask import Blueprint, render_template, request, session, flash, redirect, url_for, jsonify, send_file
import os
import json
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from app.config import get_clean_path, get_temp_path, get_upload_path, get_unified_path, TEMP_DIR, CLEAN_DIR, UPLOAD_DIR, UNIFIED_DIR
from app.utils import read_dataset, clean_temp_folder, check_dataset, get_dataset_info, ensure_temp_dir
from app.preprocessing import generic_preprocessing as gp

from app.visualization import visualize as vis
from app.visualization import coloring
from app.nocache import nocache
import matplotlib
matplotlib.use('Agg')


visualization_bp = Blueprint('visualization', __name__)

@visualization_bp.route("/visualize", methods=["GET", "POST"])
@nocache
def visualize():
    try:
        if not session.get("haha"):
            flash('Please upload a dataset first!', 'danger')
            return redirect(url_for('preprocessing.preprocess'))
            
        if request.method == "POST":
            is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
            
            # ===========================
            # UPLOAD COLOR FILE
            # ===========================
            if request.form.get('action') == 'upload_color_file':
                try:
                    if 'color_file' not in request.files:
                        return jsonify({
                            "status": "error",
                            "message": "No file uploaded"
                        })
                    
                    file = request.files['color_file']
                    
                    if file.filename == '':
                        return jsonify({
                            "status": "error",
                            "message": "No file selected"
                        })
                    
                    # Zapisz plik tymczasowo
                    temp_path = ensure_temp_dir()
                    color_file_path = os.path.join(temp_path, 'color_data_' + secure_filename(file.filename))
                    file.save(color_file_path)
                    
                    # Zapisz ścieżkę w sesji
                    session['color_file_path'] = color_file_path
                    
                    # Wczytaj plik z kolorami
                    df_color = coloring.load_coloring_file(color_file_path)
                    
                    if df_color is None:
                        return jsonify({
                            "status": "error",
                            "message": "Failed to load color file. Please check the file format."
                        })
                    
                    # Wczytaj dane PCA
                    pca_file = os.path.join(temp_path, 'pca_components.csv')
                    if not os.path.exists(pca_file):
                        return jsonify({
                            "status": "error",
                            "message": "PCA data not found. Please run PCA analysis first."
                        })
                    
                    df_pca = pd.read_csv(pca_file)
                    
                    # Wykryj kolumnę klucza
                    detection = coloring.detect_key_column(df_pca, df_color)
                    
                    # Waliduj setup
                    validation = coloring.validate_coloring_setup(
                        df_pca, 
                        df_color, 
                        detection['key_column'] if detection['found'] else None
                    )
                    
                    # Przygotuj informacje o kolumnach do kolorowania
                    color_column_info = {}
                    if validation['available_color_columns']:
                        for col in validation['available_color_columns']:
                            is_numeric = pd.api.types.is_numeric_dtype(df_color[col])
                            n_unique = df_color[col].nunique()
                            color_column_info[col] = {
                                'is_numeric': is_numeric,
                                'n_unique': n_unique
                            }
                    
                    # === NOWE: Przygotuj merged_data dla wszystkich kolumn ===
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
                    
                    # Zapisz informacje w sesji
                    session['color_file_loaded'] = True
                    session['color_detection'] = detection
                    session['color_validation'] = validation
                    session['color_column_info'] = color_column_info
                    session['color_merged_data'] = merged_data  # NOWE!
                    
                    response_data = {
                        "status": "success",
                        "auto_detected": detection['found'],
                        "key_column": detection['key_column'],
                        "common_columns": detection.get('common_columns', []),
                        "available_color_columns": validation['available_color_columns'],
                        "color_column_info": color_column_info,
                        "merged_data": merged_data,  # NOWE!
                        "message": validation['recommendations'][0] if validation['recommendations'] else 
                                  "File uploaded successfully",
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
            
            # ===========================
            # CONFIRM KEY COLUMN
            # ===========================
            elif request.form.get('action') == 'confirm_key_column':
                try:
                    key_column = request.form.get('key_column')
                    
                    if not key_column:
                        return jsonify({
                            "status": "error",
                            "message": "No key column provided"
                        })
                    
                    color_file_path = session.get('color_file_path')
                    if not color_file_path:
                        return jsonify({
                            "status": "error",
                            "message": "Color file not found. Please upload the file again."
                        })
                    
                    # Wczytaj pliki
                    df_color = coloring.load_coloring_file(color_file_path)
                    temp_path = ensure_temp_dir()
                    pca_file = os.path.join(temp_path, 'pca_components.csv')
                    df_pca = pd.read_csv(pca_file)
                    
                    # Waliduj z wybraną kolumną klucza
                    validation = coloring.validate_coloring_setup(df_pca, df_color, key_column)
                    
                    if not validation['valid']:
                        return jsonify({
                            "status": "error",
                            "message": "Invalid key column selection"
                        })
                    
                    # Przygotuj informacje o kolumnach
                    color_column_info = {}
                    for col in validation['available_color_columns']:
                        is_numeric = pd.api.types.is_numeric_dtype(df_color[col])
                        n_unique = df_color[col].nunique()
                        color_column_info[col] = {
                            'is_numeric': is_numeric,
                            'n_unique': n_unique
                        }
                    
                    # === NOWE: Przygotuj merged_data ===
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
                    
                    # Zaktualizuj sesję
                    session['color_key_column'] = key_column
                    session['color_validation'] = validation
                    session['color_column_info'] = color_column_info
                    session['color_merged_data'] = merged_data  # NOWE!
                    
                    return jsonify({
                        "status": "success",
                        "key_column": key_column,
                        "available_color_columns": validation['available_color_columns'],
                        "color_column_info": color_column_info,
                        "merged_data": merged_data,  # NOWE!
                        "message": f"Key column '{key_column}' confirmed successfully"
                    })
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return jsonify({
                        "status": "error",
                        "message": f"Error confirming key column: {str(e)}"
                    })

            # ===========================
            # PCA VISUALIZE
            # ===========================
            if "Submit" in request.form and request.form["Submit"] == "PCAVisualize":
                try:
                    if not session.get('pca_performed'):
                        error_msg = "PCA analysis has not been performed yet. Please run PCA first."
                        if is_ajax:
                            return jsonify({"status": "error", "message": error_msg})
                        flash(error_msg, "warning")
                        return redirect(url_for('visualization.visualize'))
                    
                    pc_x = int(request.form.get("pc_x", 1))
                    pc_y = int(request.form.get("pc_y", 2))
                    hover_column = request.form.get("pca_hover_column", "")

                    use_external_coloring = request.form.get("use_external_coloring") == "1"
                    external_filter_column = request.form.get("external_filter_column", "")  # POPRAWIONE!
                    
                    print(f"DEBUG: use_external_coloring={use_external_coloring}")
                    print(f"DEBUG: external_filter_column={external_filter_column}")

                    temp_path = ensure_temp_dir()
                    pca_file = os.path.join(temp_path, 'pca_components.csv')
                    if not os.path.exists(pca_file):
                        error_msg = "PCA components file not found. Please run PCA analysis again."
                        if is_ajax:
                            return jsonify({"status": "error", "message": error_msg})
                        flash(error_msg, "danger")
                        return redirect(url_for('pca.pca_analysis'))
                    
                    pc_df = pd.read_csv(pca_file)
                    pc_x_col = f'PC{pc_x}'
                    pc_y_col = f'PC{pc_y}'
                    if pc_x_col not in pc_df.columns or pc_y_col not in pc_df.columns:
                        error_msg = f"Selected principal components not found in PCA results."
                        if is_ajax:
                            return jsonify({"status": "error", "message": error_msg})
                        flash(error_msg, "danger")
                        return redirect(url_for('visualization.visualize'))
                    
                    # Hover text
                    hover_text = None
                    if hover_column and hover_column in pc_df.columns:
                        hover_text = pc_df[hover_column].tolist()
                    
                    # Dane do koloru
                    plot_data, color_data, color_categories, color_by = [], None, None, None

                    # === NOWE: słownik external_data dla filtrowania ===
                    external_data = {}

                    if use_external_coloring:
                        # Zewnętrzne kolorowanie
                        key_column = request.form.get("external_key_column")
                        color_column = request.form.get("external_color_column")
                        
                        print(f"DEBUG: key_column={key_column}, color_column={color_column}")
                        
                        if not key_column:
                            error_msg = "External key column missing"
                            if is_ajax:
                                return jsonify({"status": "error", "message": error_msg})
                            flash(error_msg, "danger")
                            return redirect(url_for('visualization.visualize'))

                        color_file_path = session.get('color_file_path')
                        if not color_file_path or not os.path.exists(color_file_path):
                            error_msg = "Color file not found. Please upload it again."
                            if is_ajax:
                                return jsonify({"status": "error", "message": error_msg})
                            flash(error_msg, "danger")
                            return redirect(url_for('visualization.visualize'))

                        df_color = coloring.load_coloring_file(color_file_path)
                        
                        # === KOLOROWANIE (jeśli wybrano color_column) ===
                        if color_column:
                            color_result = coloring.prepare_color_data(pc_df, df_color, key_column, color_column)
                            if not color_result['success']:
                                error_msg = color_result['message']
                                if is_ajax:
                                    return jsonify({"status": "error", "message": error_msg})
                                flash(error_msg, "danger")
                                return redirect(url_for('visualization.visualize'))

                            color_series = color_result['data']
                            is_numeric = color_result['is_numeric']
                            
                            if is_numeric:
                                plot_data = [{'x': pc_df[pc_x_col].tolist(), 'y': pc_df[pc_y_col].tolist()}]
                                color_data = color_series.tolist()
                                color_by = color_column
                                if hover_text:
                                    plot_data[0]['hover_text'] = hover_text
                            else:
                                unique_categories = color_series.dropna().unique()
                                n_categories = len(unique_categories)
                                color_palette = coloring.generate_distinct_colors(n_categories)
                                color_map = {cat: color_palette[i] for i, cat in enumerate(unique_categories)}
                                color_categories = unique_categories.tolist()

                                for category in unique_categories:
                                    mask = color_series == category
                                    subset = pc_df[mask]
                                    category_data = {
                                        'label': str(category),
                                        'data': [{'x': row[pc_x_col], 'y': row[pc_y_col]} for _, row in subset.iterrows()],
                                        'color': color_map[category]
                                    }
                                    if hover_text:
                                        subset_indices = subset.index.tolist()
                                        category_data['hover_text'] = [hover_text[i] for i in subset_indices]
                                    plot_data.append(category_data)

                                color_by = color_column
                        else:
                            # Brak kolorowania external - tylko podstawowy plot
                            plot_data = [{'x': pc_df[pc_x_col].tolist(), 'y': pc_df[pc_y_col].tolist()}]
                            if hover_text:
                                plot_data[0]['hover_text'] = hover_text

                        # === BUDOWA SŁOWNIKA external_data ze WSZYSTKICH kolumn dla filtrowania ===
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
                                
                        print(f"DEBUG: external_data keys: {list(external_data.keys())}")

                    else:
                        # Kolorowanie z danych PCA
                        color_by = request.form.get("pca_color_by", "")
                        if color_by and color_by in pc_df.columns:
                            if pc_df[color_by].dtype == 'object' or pd.api.types.is_categorical_dtype(pc_df[color_by]):
                                color_categories = pc_df[color_by].unique().tolist()
                                for category in color_categories:
                                    subset = pc_df[pc_df[color_by] == category]
                                    category_data = {
                                        'label': str(category),
                                        'data': [{'x': row[pc_x_col], 'y': row[pc_y_col]} for _, row in subset.iterrows()]
                                    }
                                    if hover_text:
                                        subset_indices = subset.index.tolist()
                                        category_data['hover_text'] = [hover_text[i] for i in subset_indices]
                                    plot_data.append(category_data)
                            else:
                                plot_data = [{'x': pc_df[pc_x_col].tolist(), 'y': pc_df[pc_y_col].tolist()}]
                                color_data = pc_df[color_by].tolist()
                                if hover_text:
                                    plot_data[0]['hover_text'] = hover_text
                        else:
                            plot_data = [{'x': pc_df[pc_x_col].tolist(), 'y': pc_df[pc_y_col].tolist()}]
                            if hover_text:
                                plot_data[0]['hover_text'] = hover_text

                    # Variance explained
                    pca_summary = session.get('pca_summary', [])
                    variance_x = pca_summary[pc_x-1]['explained_variance'] if pc_x <= len(pca_summary) else 0
                    variance_y = pca_summary[pc_y-1]['explained_variance'] if pc_y <= len(pca_summary) else 0
                    
                    success_msg = f"PCA scatter plot created: PC{pc_x} vs PC{pc_y}"
                    if use_external_coloring and color_by:
                        success_msg += f" with external coloring from column '{color_by}'"

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
                            "pc_x_raw": pc_x_raw,  # NOWE - surowe PC arrays w globalnej kolejności
                            "pc_y_raw": pc_y_raw   # NOWE - surowe PC arrays w globalnej kolejności
                        }

                        
                        # Dodaj external_data jeśli dostępne (dla filtrowania)
                        if external_data:
                            response_data["external_data"] = external_data
                            print(f"DEBUG: Sending external_data with {len(external_data)} columns")
                        
                        return jsonify(response_data)
                    
                    flash(success_msg, "success")
                
                except Exception as e:
                    import traceback
                    traceback_str = traceback.format_exc()
                    print(f"PCA visualization error: {str(e)}")
                    print(traceback_str)
                    
                    error_msg = f"PCA visualization error: {str(e)}"
                    if is_ajax:
                        return jsonify({"status": "error", "message": error_msg})
                    flash(error_msg, "danger")
                    return redirect(url_for('visualization.visualize'))

            # ===========================
            # HISTOGRAM VISUALIZATION
            # ===========================
            if "Submit" in request.form and request.form["Submit"] == "Visualize":
                try:
                    # Get the selected column
                    x_col = request.form["x_col"]
                    print(f"Selected column for visualization: {x_col}")
                    
                    # Get the dataframe
                    unified_path = get_clean_path(session["csv_name"])
                    if not os.path.exists(unified_path):
                        error_msg = f"Data file does not exist: {unified_path}"
                        if is_ajax:
                            return jsonify({"status": "error", "message": error_msg})
                        flash(error_msg, "danger")
                        return redirect(url_for('preprocessing.preprocess'))
                    
                    df = read_dataset(unified_path)
                    print(f"Loaded dataframe from: {unified_path}")
                    
                    # Verify column exists in dataframe
                    if x_col not in df.columns:
                        error_msg = f"Column '{x_col}' not found in the dataframe"
                        if is_ajax:
                            return jsonify({"status": "error", "message": error_msg})
                        flash(error_msg, "danger")
                        return redirect(url_for('visualization.visualize'))
                    
                    # Check for sufficient data
                    if df.empty or df[x_col].dropna().empty:
                        error_msg = f"No valid data available for column '{x_col}'"
                        if is_ajax:
                            return jsonify({"status": "error", "message": error_msg})
                        flash(error_msg, "warning")
                        return redirect(url_for('visualization.visualize'))
                    
                    # Memory usage reporting
                    memory_usage = df.memory_usage(deep=True).sum() / 1024
                    print(f"memory usage: {memory_usage:.1f} KB")
                    print(f"DataFrame info: {df.info()}")
                    
                    # Generate static plot first as a fallback
                    try:
                        plot_path = vis.static_hist_plot(df, x_col)
                        print(f"Wykres zapisany w: {os.path.join('ChemAlize', plot_path)}")
                        print(f"Ścieżka względna: {plot_path}")
                    except Exception as plot_error:
                        print(f"Static plot error: {str(plot_error)}")
                        plot_path = None
                    
                    # Generate histogram data for Plotly
                    column_data = df[x_col]
                    
                    # Prepare data structure based on data type
                    try:
                        if pd.api.types.is_numeric_dtype(column_data):
                            # For numeric data
                            histogram_data = {
                                "is_numeric": True,
                                "column_name": x_col,
                                "values": column_data.dropna().tolist()
                            }
                        else:
                            # For categorical data
                            value_counts = column_data.value_counts()
                            histogram_data = {
                                "is_numeric": False,
                                "column_name": x_col,
                                "categories": value_counts.index.tolist(),
                                "counts": value_counts.values.tolist()
                            }
                        
                        # Store histogram data in session as JSON
                        session["histogram_data"] = json.dumps(histogram_data)
                        posted = 1
                        success_msg = f"Interactive visualization created for column: {x_col}"
                        print(success_msg)
                        
                        if not is_ajax:
                            flash(success_msg, "success")
                    except Exception as data_error:
                        print(f"Error creating histogram data: {str(data_error)}")
                        histogram_data = None
                        # Define success_msg even in error case (fallback to static plot)
                        success_msg = f"Static visualization created for column: {x_col}"
                        if not plot_path:
                            error_msg = f"Failed to create visualization for '{x_col}': {str(data_error)}"
                            if is_ajax:
                                return jsonify({"status": "error", "message": error_msg})
                            flash(error_msg, "danger")
                            return redirect(url_for('visualization.visualize'))

                    # Get all columns for dropdowns
                    columns = df.columns.tolist()

                    if is_ajax:
                        return jsonify({
                            "status": "success",
                            "message": success_msg,
                            "x_col": x_col,
                            "histogram_data": histogram_data,
                            "plot_path": plot_path
                        })
                        
                    return render_template(
                        "visualize.html",
                        cols=columns,
                        src="img/pairplot1.png",
                        posted=1,
                        active="visualize",
                        title="Visualize",
                        x_col=x_col,
                        histogram_data=session.get("histogram_data"),
                        plot_path=plot_path
                    )
                        
                except Exception as e:
                    import traceback
                    traceback_str = traceback.format_exc()
                    print(f"Visualization error: {str(e)}")
                    print(traceback_str)
                    
                    error_msg = f"Visualization error: {str(e)}"
                    if is_ajax:
                        return jsonify({
                            "status": "error", 
                            "message": error_msg,
                            "traceback": traceback_str
                        })
                    flash(error_msg, "danger")
                    return redirect(url_for('visualization.visualize'))
                    
            else:
                # ===========================
                # XY PLOT VISUALIZATION
                # ===========================
                try:
                    x_col = request.form["x_col"]
                    y_col = request.form["y_col"]
                    print(f"Selected columns: {x_col}, {y_col}")
                    
                    unified_path = get_clean_path(session["csv_name"])
                    if not os.path.exists(unified_path):
                        error_msg = f"File not found: {unified_path}"
                        if is_ajax:
                            return jsonify({"status": "error", "message": error_msg})
                        flash(error_msg, "danger")
                        return redirect(url_for('preprocessing.preprocess'))
                        
                    df = read_dataset(unified_path)
                    print(f"DataFrame loaded, shape: {df.shape}")
                    
                    # Verify columns exist
                    missing_cols = []
                    if x_col not in df.columns:
                        missing_cols.append(x_col)
                    if y_col not in df.columns:
                        missing_cols.append(y_col)
                        
                    if missing_cols:
                        error_msg = f"Columns not found in dataframe: {', '.join(missing_cols)}"
                        if is_ajax:
                            return jsonify({"status": "error", "message": error_msg})
                        flash(error_msg, "danger")
                        return redirect(url_for('visualization.visualize'))
                    
                    # Save to temp directory for compatibility (optional - for backward compatibility)
                    try:
                        temp_dir = ensure_temp_dir()
                        col_csv_path = os.path.join(temp_dir, "col.csv")
                        df.to_csv(col_csv_path, index=False)
                        print(f"Cleaned DataFrame, shape: {df.shape}")

                        # Call the visualization function, passing DataFrame directly
                        print(f"Starting xy_plot with {x_col} and {y_col}")
                        result_df = vis.xy_plot(x_col, y_col, df=df)
                        
                        # Check if result_df has any rows
                        if result_df is None or result_df.empty:
                            error_msg = f"No valid data for selected columns"
                            if is_ajax:
                                return jsonify({"status": "error", "message": error_msg})
                            flash(error_msg, "warning")
                            return redirect(url_for('visualization.visualize'))
                        
                        print(f"Returning DataFrame with {len(result_df)} rows and mapped numeric values")
                        
                        # Check which columns should be used for the plot
                        x_data_col = 'x_numeric' if 'x_numeric' in result_df.columns else x_col
                        y_data_col = 'y_numeric' if 'y_numeric' in result_df.columns else y_col
                        
                        # Prepare data for the chart
                        heights = np.array(result_df[x_data_col]).tolist()
                        weights = np.array(result_df[y_data_col]).tolist()
                        newlist = []
                        for h, w in zip(heights, weights):
                            newlist.append({"x": h, "y": w})
                        
                        # Get all columns from original dataset for dropdown
                        columns = df.columns.tolist()
                        
                        success_msg = f"Scatter plot created for {x_col} vs {y_col}"
                        
                        if is_ajax:
                            return jsonify({
                                "status": "success",
                                "message": success_msg,
                                "x_col_name": str(x_col),
                                "y_col_name": str(y_col),
                                "plot_data": newlist
                            })
                        
                        # Format data for JavaScript if not AJAX
                        ugly_blob = str(newlist).replace("'", "")
                        
                        return render_template(
                            "visualize.html",
                            cols=columns,
                            src="img/pairplot1.png",
                            xy_src="img/fig.png",
                            posted=1,
                            data=ugly_blob,
                            active="visualize",
                            x_col_name=str(x_col),
                            y_col_name=str(y_col),
                            default_x=str(x_col),
                            default_y=str(y_col),
                            title="Visualize",
                        )
                    except Exception as vis_error:
                        error_msg = f"Error in visualization process: {str(vis_error)}"
                        if is_ajax:
                            return jsonify({"status": "error", "message": error_msg})
                        flash(error_msg, "danger")
                        return redirect(url_for('visualization.visualize'))
                    
                except Exception as e:
                    import traceback
                    traceback_str = traceback.format_exc()
                    print(f"XY visualization error: {str(e)}")
                    print(traceback_str)
                    
                    error_msg = f"XY visualization error: {str(e)}"
                    if is_ajax:
                        return jsonify({
                            "status": "error", 
                            "message": error_msg,
                            "traceback": traceback_str
                        })
                    flash(error_msg, "danger")
                    return redirect(url_for('visualization.visualize'))
        else:
            # ===========================
            # GET REQUEST - INITIAL PAGE LOAD
            # ===========================
            try:
                unified_path = get_clean_path(session["csv_name"])
                if not os.path.exists(unified_path):
                    flash(f"File not found: {unified_path}", "danger")
                    return redirect(url_for('preprocessing.preprocess'))
                    
                df = read_dataset(unified_path)

                # Save to temp directory for compatibility with vis.pair_plot
                temp_dir = ensure_temp_dir()
                col_csv_path = os.path.join(temp_dir, "col.csv")
                df.to_csv(col_csv_path, index=False)

                # Call pair_plot after saving col.csv file
                vis.pair_plot()
                columns = df.columns.tolist()
                
                pca_available = session.get('pca_performed', False)
                pca_n_components = 0
                pca_color_options = []
                
                if pca_available:
                    temp_path = ensure_temp_dir()
                    pca_file = os.path.join(temp_path, 'pca_components.csv')
                    if os.path.exists(pca_file):
                        pc_df_check = pd.read_csv(pca_file)
                        pca_n_components = len([col for col in pc_df_check.columns if col.startswith('PC')])
                        # Get columns that can be used for coloring
                        pca_color_options = [col for col in pc_df_check.columns if not col.startswith('PC')]
                
                return render_template(
                    "visualize.html",
                    cols=columns,
                    src="img/pairplot1.png",
                    posted=0,
                    active="visualize",
                    title="Visualize",
                    pca_available=pca_available,
                    pca_n_components=pca_n_components,
                    pca_color_options=pca_color_options
                )
            except Exception as e:
                flash(f"Error loading data: {str(e)}", "danger")
                return redirect(url_for('preprocessing.preprocess'))
    
    except Exception as global_error:
        import traceback
        traceback.print_exc()
        
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        if is_ajax:
            return jsonify({
                "status": "error",
                "message": f"An unexpected error occurred: {str(global_error)}",
                "traceback": traceback.format_exc()
            })
        
        flash(f"An unexpected error occurred: {str(global_error)}", "danger")
        return redirect(url_for('main.home'))


@visualization_bp.route("/col.csv")
@nocache
def col():
    # Serve col.csv from temp directory
    temp_dir = ensure_temp_dir()
    col_csv_path = os.path.join(temp_dir, "col.csv")
    return send_file(col_csv_path, mimetype="text/csv", as_attachment=True)


@visualization_bp.route("/pairplot1.png")
@nocache
def pairplot1():
    return send_file(
        "static/img/pairplot1.png", 
        mimetype="image/png", 
        as_attachment=True
    )


@visualization_bp.route("/tree.png")
@nocache
def tree():
    return send_file(
        "static/img/tree.png", 
        mimetype="image/png", 
        as_attachment=True
    )



