import secrets
import asyncio
import os.path
import numpy as np
import pandas as pd
from shutil import copyfile
from flask import *
from ChemAlize.preprocessing import generic_preprocessing as gp
from ChemAlize.modules import logistic as lg
from ChemAlize.modules import naive_bayes as nb
from ChemAlize.modules import linear_svc as lsvc
from ChemAlize.modules import knn
from ChemAlize.modules import decision_tree as dtree
from ChemAlize.modules import random_forest as rfc
from ChemAlize.visualization import visualize as vis
from ChemAlize.nocache import nocache
from ChemAlize import app
from ChemAlize.preprocessing.generic_preprocessing import read_dataset, get_columns, get_rows, get_dim, get_description, get_head, treat_missing_numeric
from flask import render_template, url_for, flash, redirect, request, session, jsonify
import glob


def clean_temp_folder():
    """
    Funkcja czyści folder tymczasowy, usuwając wszystkie pliki .csv.
    Obsługuje również potencjalne problemy ze ścieżkami.
    """
    # Użyj ścieżki bezwzględnej dla folderu tymczasowego
    app_root = os.path.abspath(os.path.dirname(__file__))
    temp_dir = os.path.join(app_root, "ChemAlize/temp/")
    
    print(f"Czyszczenie folderu tymczasowego: {temp_dir}")
    
    if os.path.exists(temp_dir):
        # Usuń wszystkie pliki CSV z folderu temp
        for csv_file in glob.glob(os.path.join(temp_dir, "*.csv")):
            try:
                os.remove(csv_file)
                print(f"Usunięto plik tymczasowy: {csv_file}")
            except Exception as e:
                print(f"Błąd podczas usuwania pliku {csv_file}: {str(e)}")
    else:
        print(f"Folder tymczasowy nie istnieje: {temp_dir}")
        
        # Sprawdź alternatywną ścieżkę (bez podwójnego ChemAlize)
        alt_path = temp_dir.replace('ChemAlize/ChemAlize/temp/', 'ChemAlize/temp/')
        if os.path.exists(alt_path):
            print(f"Znaleziono alternatywną ścieżkę: {alt_path}")
            for csv_file in glob.glob(os.path.join(alt_path, "*.csv")):
                try:
                    os.remove(csv_file)
                    print(f"Usunięto plik tymczasowy z alternatywnej ścieżki: {csv_file}")
                except Exception as e:
                    print(f"Błąd podczas usuwania pliku {csv_file}: {str(e)}")

global posted
save_path = "ChemAlize/uploads/"
exts = ["csv", "xlsx", "json", "yaml", "txt", "xls"]
posted = 0

def read_dataset(filepath):
    ext = filepath.split('.')[-1].lower()
    try:
        if ext == 'csv':
            return pd.read_csv(filepath)
        elif ext in ['xlsx', 'xls']:
            return pd.read_excel(filepath, engine='openpyxl')
        elif ext == 'json':
            return pd.read_json(filepath)
        elif ext == 'yaml':
            import yaml
            with open(filepath, 'r') as file:
                return pd.json_normalize(yaml.safe_load(file))
        elif ext == 'txt':
            return pd.read_csv(filepath, delimiter='\t')
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

@app.route("/")
@app.route("/preprocess", methods=["GET", "POST"])
@nocache
def preprocess():
    global posted
    if request.method == "POST":
        if request.form["Submit"] == "Upload":
            try:
                data = request.files["data"]
                ext = data.filename.split(".")[-1].lower()
                if ext in exts:
                    session["ext"] = ext
                    session["fname"] = data.filename
                    
                    # Tworzenie wymaganych folderów
                    os.makedirs("ChemAlize/uploads/", exist_ok=True)
                    os.makedirs("ChemAlize/unified/", exist_ok=True)
                    os.makedirs("ChemAlize/clean/", exist_ok=True)
                    
                    # Zapisz oryginalny plik
                    upload_path = os.path.join("ChemAlize/uploads/", data.filename)
                    data.save(upload_path)
                    
                    # Konwertuj do CSV i zapisz w obu folderach
                    df = read_dataset(upload_path)
                    csv_filename = os.path.splitext(data.filename)[0] + '.csv'
                    
                    # Zapisz w unified (oryginał)
                    unified_path = os.path.join("ChemAlize/unified/", csv_filename)
                    df.to_csv(unified_path, index=False)
                    
                    # Skopiuj do clean (do preprocessingu)
                    clean_path = os.path.join("ChemAlize/clean/", csv_filename)
                    df.to_csv(clean_path, index=False)
                    
                    session["csv_name"] = csv_filename
                    session["haha"] = True
                    flash("File uploaded and converted successfully", "success")
                else:
                    flash(f"Unsupported file format: {ext}", "danger")
            except Exception as e:
                flash(f"Upload failed: {str(e)}", "danger")


        elif request.form["Submit"] == "DeleteColumn":
            try:
                clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
                df = read_dataset(clean_path)
                df = gp.delete_column(df, request.form.getlist("check_cols"))
                df.to_csv(clean_path, index=False)
                flash("Column(s) deleted successfully", "success")
            except Exception as e:
                flash(f"Error: {str(e)}", "danger")



        elif request.form["Submit"] == "DeleteRow":
            try:
                clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
                df = read_dataset(clean_path)
                
                # Pobierz listę wierszy do usunięcia (indeksy)
                rows_to_delete = request.form.getlist("check_rows")
                rows_to_delete = [int(row) for row in rows_to_delete]
                
                # Użyj funkcji gp.deleterows()
                df = gp.delete_rows(df, rows_to_delete)
                
                # Zapisz zmiany
                df.to_csv(clean_path, index=False)
                flash("Row(s) deleted successfully", "success")
            except Exception as e:
                flash(f"Error deleting rows: {str(e)}", "danger")


                
        elif request.form["Submit"] == "Clean":
            try:
                # Operuj na pliku w clean
                clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
                df = read_dataset(clean_path)
                how = request.form.get("how")
                
                if how != "any":
                    df = gp.treat_missing_numeric(
                        df, request.form.getlist("check_cols"), how=how
                    )
                elif request.form.get("howNos"):
                    df = gp.treat_missing_numeric(
                        df,
                        request.form.getlist("check_cols"),
                        how=float(request.form["howNos"]),
                    )
                
                # Zapisz zmiany tylko w clean
                df.to_csv(clean_path, index=False)
                flash("Column(s) cleaned successfully", "success")
            except Exception as e:
                flash(f"Error: {str(e)}", "danger")

    if session.get("haha"):
        try:
            clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
            df = read_dataset(clean_path)
            description = gp.get_description(df)
            columns = df.columns.tolist()
            rows = list(range(len(df)))
            dim1, dim2 = df.shape
            head = df.head()
            
            return render_template(
                "preprocess.html",
                active="preprocess",
                title="Preprocess",
                filename=session["fname"],
                posted=0,
                no_of_rows=len(df),
                no_of_cols=len(columns),
                dim=f"{dim1} x {dim2}",
                description=description.to_html(
                    classes=[
                        "table-bordered",
                        "table-striped",
                        "table-hover",
                        "thead-light",
                    ]
                ),
                columns=columns,
                rows=rows, 
                head=head.to_html(
                    classes=[
                        "table",
                        "table-bordered",
                        "table-striped",
                        "table-hover",
                        "thead-light",
                    ]
                ),
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            flash(f"Błąd ładowania danych: {str(e)}", "danger")
            print(f"Błąd ładowania danych: {str(e)}")
    
    # Always return a valid response, even when no data is loaded
    return render_template("preprocess.html", active="preprocess", title="Preprocess")


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    acc = 0
        
    if not session.get("haha"):
        flash("Please upload a file first", "warning")
        return redirect(url_for("preprocess"))
    
    if request.method == "POST":
        target = request.form["target"]
        gp.arrange_columns(target)

        classifier = int(request.form["classifier"])
        hidden_val = int(request.form["hidden"])
        scale_val = int(request.form["scale_hidden"])
        encode_val = int(request.form["encode_hidden"])
        columns = vis.get_columns()

        if hidden_val == 0:
            data = request.files["choiceVal"]
            ext = data.filename.split(".")[1]
            if ext in exts:
                data.save("uploads/test." + ext)
            else:
                return "File type not accepted!"
            choiceVal = 0
        else:
            choiceVal = int(request.form["choiceVal"])

        if classifier == 0:
            ret_vals = lg.logisticReg(choiceVal, hidden_val, scale_val, encode_val)
            if hidden_val == 0 or hidden_val == 1:
                return render_template(
                    "analyze_page.html",
                    acc=ret_vals[0],
                    report=[
                        ret_vals[1].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    conf_matrix=[
                        ret_vals[2].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="analyze",
                    title="analyze",
                )
            elif hidden_val == 2:
                return render_template(
                    "analyze_page.html",
                    acc=ret_vals[0],
                    report=ret_vals[1],
                    conf_matrix=ret_vals[2],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="analyze",
                    title="analyze",
                    cols=columns,
                )

        elif classifier == 1:
            ret_vals = nb.naiveBayes(choiceVal, hidden_val, scale_val, encode_val)
            if hidden_val == 0 or hidden_val == 1:
                return render_template(
                    "analyze_page.html",
                    acc=ret_vals[0],
                    report=[
                        ret_vals[1].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    conf_matrix=[
                        ret_vals[2].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="analyze",
                    title="analyze",
                    cols=columns,

                )
            elif hidden_val == 2:
                return render_template(
                    "analyze_page.html",
                    acc=ret_vals[0],
                    report=ret_vals[1],
                    conf_matrix=ret_vals[2],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="analyze",
                    title="analyze",
                    cols=columns,
                )

        elif classifier == 2:
            ret_vals = lsvc.lin_svc(choiceVal, hidden_val, scale_val, encode_val)
            if hidden_val == 0 or hidden_val == 1:
                return render_template(
                    "analyze_page.html",
                    acc=ret_vals[0],
                    report=[
                        ret_vals[1].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    conf_matrix=[
                        ret_vals[2].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="analyze",
                    title="analyze",
                    cols=columns,
                )
            elif hidden_val == 2:
                return render_template(
                    "analyze_page.html",
                    acc=ret_vals[0],
                    report=ret_vals[1],
                    conf_matrix=ret_vals[2],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="analyze",
                    title="analyze",
                    cols=columns,
                )

        elif classifier == 3:

            scale_val = 1
            ret_vals = knn.KNearestNeighbours(
                choiceVal, hidden_val, scale_val, encode_val
            )
            if hidden_val == 0 or hidden_val == 1:
                return render_template(
                    "analyze_page.html",
                    acc=ret_vals[0],
                    report=[
                        ret_vals[1].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    conf_matrix=[
                        ret_vals[2].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="analyze",
                    title="analyze",
                    cols=columns,
                )
            elif hidden_val == 2:
                return render_template(
                    "analyze_page.html",
                    acc=ret_vals[0],
                    report=ret_vals[1],
                    conf_matrix=ret_vals[2],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="analyze",
                    title="analyze",
                    cols=columns,
                )

        elif classifier == 4:
            ret_vals = dtree.DecisionTree(choiceVal, hidden_val, scale_val, encode_val)
            if hidden_val == 0 or hidden_val == 1:
                return render_template(
                    "analyze_page.html",
                    acc=ret_vals[0],
                    report=[
                        ret_vals[1].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    conf_matrix=[
                        ret_vals[2].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="analyze",
                    title="analyze",
                    cols=columns,
                )
            elif hidden_val == 2:
                return render_template(
                    "analyze_page.html",
                    acc=ret_vals[0],
                    report=ret_vals[1],
                    conf_matrix=ret_vals[2],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="analyze",
                    title="analyze",
                    cols=columns,
                )
        elif classifier == 5:
            ret_vals = rfc.RandomForest(choiceVal, hidden_val, scale_val, encode_val)
            if hidden_val == 0 or hidden_val == 1:
                return render_template(
                    "analyze_page.html",
                    acc=ret_vals[0],
                    report=[
                        ret_vals[1].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    conf_matrix=[
                        ret_vals[2].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="analyze",
                    title="analyze",
                    cols=columns,
                )
            elif hidden_val == 2:
                return render_template(
                    "analyze_page.html",
                    acc=ret_vals[0],
                    report=ret_vals[1],
                    conf_matrix=ret_vals[2],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="analyze",
                    title="analyze",
                    cols=columns,
                )
    elif request.method == "GET":
        columns = vis.get_columns()  # Pobierz kolumny
        return render_template(
            "analyze_page.html", 
            active="analyze", 
            title="analyze", 
            cols=columns,
        )

@app.route("/clear", methods=["GET"])
def clear():
    session.clear()
    return redirect("/")


@app.route("/visualize", methods=["GET", "POST"])
@nocache
def visualize():
    try:
        os.makedirs("ChemAlize/visualization", exist_ok=True)
        os.makedirs("ChemAlize/unified", exist_ok=True)
        os.makedirs("ChemAlize/clean", exist_ok=True)
        
        if not session.get("haha"):
            flash("Please upload a file first", "warning")
            return redirect(url_for("preprocess"))
            
        if request.method == "POST":
            # Check if it's an AJAX request
            is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
            
            # Check which submit button was pressed
            if "Submit" in request.form and request.form["Submit"] == "Visualize":
                # HISTOGRAM VISUALIZATION
                try:
                    # Get the selected column
                    x_col = request.form["x_col"]
                    print(f"Selected column for visualization: {x_col}")
                    
                    # Get the dataframe
                    unified_path = os.path.join("ChemAlize/clean/", session["csv_name"])
                    if not os.path.exists(unified_path):
                        error_msg = f"Data file does not exist: {unified_path}"
                        if is_ajax:
                            return jsonify({"status": "error", "message": error_msg})
                        flash(error_msg, "danger")
                        return redirect(url_for("preprocess"))
                    
                    df = read_dataset(unified_path)
                    print(f"Loaded dataframe from: {unified_path}")
                    
                    # Verify column exists in dataframe
                    if x_col not in df.columns:
                        error_msg = f"Column '{x_col}' not found in the dataframe"
                        if is_ajax:
                            return jsonify({"status": "error", "message": error_msg})
                        flash(error_msg, "danger")
                        return redirect(url_for("visualize"))
                    
                    # Check for sufficient data
                    if df.empty or df[x_col].dropna().empty:
                        error_msg = f"No valid data available for column '{x_col}'"
                        if is_ajax:
                            return jsonify({"status": "error", "message": error_msg})
                        flash(error_msg, "warning")
                        return redirect(url_for("visualize"))
                    
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
                        if not plot_path:
                            error_msg = f"Failed to create visualization for '{x_col}': {str(data_error)}"
                            if is_ajax:
                                return jsonify({"status": "error", "message": error_msg})
                            flash(error_msg, "danger")
                            return redirect(url_for("visualize"))
                    
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
                    return redirect(url_for("visualize"))
                    
            else:
                # XY PLOT VISUALIZATION
                try:
                    x_col = request.form["x_col"]
                    y_col = request.form["y_col"]
                    print(f"Selected columns: {x_col}, {y_col}")
                    
                    unified_path = os.path.join("ChemAlize/clean/", session["csv_name"])
                    if not os.path.exists(unified_path):
                        error_msg = f"File not found: {unified_path}"
                        if is_ajax:
                            return jsonify({"status": "error", "message": error_msg})
                        flash(error_msg, "danger")
                        return redirect(url_for("preprocess"))
                        
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
                        return redirect(url_for("visualize"))
                    
                    # Save to ChemAlize/visualization/col.csv for compatibility with vis.xy_plot
                    try:
                        os.makedirs("ChemAlize/visualization", exist_ok=True)
                        df.to_csv("ChemAlize/visualization/col.csv", index=False)
                        print(f"Cleaned DataFrame, shape: {df.shape}")
                        
                        # Call the visualization function
                        print(f"Starting xy_plot with {x_col} and {y_col}")
                        result_df = vis.xy_plot(x_col, y_col)
                        
                        # Check if result_df has any rows
                        if result_df is None or result_df.empty:
                            error_msg = f"No valid data for selected columns"
                            if is_ajax:
                                return jsonify({"status": "error", "message": error_msg})
                            flash(error_msg, "warning")
                            return redirect(url_for("visualize"))
                        
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
                        return redirect(url_for("visualize"))
                    
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
                    return redirect(url_for("visualize"))
        else:
            # GET request - initial page load
            try:
                unified_path = os.path.join("ChemAlize/clean/", session["csv_name"])
                if not os.path.exists(unified_path):
                    flash(f"File not found: {unified_path}", "danger")
                    return redirect(url_for("preprocess"))
                    
                df = read_dataset(unified_path)
                
                # Save to ChemAlize/visualization/col.csv for compatibility with vis.pair_plot
                os.makedirs("ChemAlize/visualization", exist_ok=True)
                df.to_csv("ChemAlize/visualization/col.csv", index=False)
                
                # Call pair_plot after saving col.csv file
                vis.pair_plot()
                columns = df.columns.tolist()
                
                return render_template(
                    "visualize.html",
                    cols=columns,
                    src="img/pairplot1.png",
                    posted=0,
                    active="visualize",
                    title="Visualize",
                )
                
            except Exception as e:
                flash(f"Error loading data: {str(e)}", "danger")
                return redirect(url_for("preprocess"))
    
    except Exception as global_error:
        # Global error handler for any unexpected errors
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
        return redirect(url_for("home"))


@app.route("/col.csv")
@nocache
def col():
    return send_file("visualization/col.csv", mimetype="text/csv", as_attachment=True)

@app.route("/pairplot1.png")
@nocache
def pairplot1():
    return send_file(
        "static/img/pairplot1.png", 
        mimetype="image/png", 
        as_attachment=True
    )

@app.route("/tree.png")
@nocache
def tree():
    return send_file(
        "static/img/tree.png", 
        mimetype="image/png", 
        as_attachment=True
    )


@app.route('/manual')
def manual_mode():
    if "csv_name" not in session:
        flash("Najpierw załaduj plik danych", "warning")
        return redirect(url_for('preprocess'))
    
    # Określ, który plik wczytać - preferuj plik z temp jeśli istnieje (tak samo jak w manual_process)
    if "temp_csv_path" in session and os.path.exists(session["temp_csv_path"]):
        # Używaj pliku tymczasowego, jeśli istnieje
        file_path = session["temp_csv_path"]
        is_temp_file = True
    else:
        # W przeciwnym razie użyj pliku z clean
        file_path = os.path.join("ChemAlize/clean/", session["csv_name"])
        is_temp_file = False
    
    # Wczytaj dane
    df = read_dataset(file_path)
    
    # Uzyskaj informacje o danych
    columns = get_columns(df)
    rows = get_rows(df)
    no_of_rows, no_of_cols = get_dim(df)
    head = get_head(df).to_html(classes=['table', 'table-bordered', 'table-striped', 'table-hover'])
    description = get_description(df).to_html(classes=['table', 'table-bordered', 'table-striped', 'table-hover'])
    
    # Identyfikuj typy kolumn
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Oblicz dodatkowe informacje (tak samo jak w manual_process)
    missing_values = int(df.isna().sum().sum())
    has_low_variance = False
    max_correlation = 0.0
    
    if numeric_columns:
        variances = df[numeric_columns].var()
        has_low_variance = bool(any(variances < 0.01))
    
    if len(numeric_columns) > 1:
        import numpy as np
        corr_matrix = df[numeric_columns].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        if not upper_tri.empty and not upper_tri.values.size == 0:
            max_correlation = float(upper_tri.values.max())
    
    # Aktualizuj zmienne sesji
    session["no_of_rows"] = no_of_rows
    session["no_of_cols"] = no_of_cols
    session["dim"] = f"{no_of_rows}x{no_of_cols}"
    session["missing_values"] = missing_values
    session["has_low_variance"] = has_low_variance
    session["max_correlation"] = max_correlation
    
    # Pobierz nazwę pliku tymczasowego
    temp_filename = os.path.basename(session.get("temp_csv_path", "")) if "temp_csv_path" in session else ""
    
    # Domyślne parametry procesowania
    data_cleaned = session.get('data_cleaned', False)
    target_transformed = session.get('target_transformed', False)
    features_transformed = session.get('features_transformed', False)
    features_selected = session.get('features_selected', False)
    
    # Sprawdź czy są dostępne przetworzone dane
    processed_data = None
    if os.path.exists(file_path):
        processed_df = read_dataset(file_path)
        processed_data = processed_df.head(10).to_html(
            classes=['table', 'table-bordered', 'table-striped', 'table-hover', 'thead-light']
        )
    
    return render_template('manual.html',
                          title='Manual Preprocessing',
                          active="manual",
                          filename=session["csv_name"],
                          temp_filename=temp_filename,
                          is_temp_file=is_temp_file,
                          no_of_rows=no_of_rows,
                          no_of_cols=no_of_cols,
                          dim=f"{no_of_rows}x{no_of_cols}",
                          missing_values=missing_values,
                          has_low_variance=has_low_variance,
                          max_correlation=max_correlation,
                          columns=columns,
                          rows=rows,
                          head=head,
                          description=description,
                          numeric_columns=numeric_columns,
                          categorical_columns=categorical_columns,
                          data_cleaned=data_cleaned,
                          target_transformed=target_transformed,
                          features_transformed=features_transformed,
                          features_selected=features_selected,
                          processed_data=processed_data)


# Zmodyfikuj funkcję manual_process, aby obsługiwała czyszczenie folderu tymczasowego
@app.route("/manual_process", methods=['GET', 'POST'])
def manual_process():
    from ChemAlize.preprocessing.manual_preprocessing import process_dataset, get_dataset_stats
    from datetime import datetime
    import numpy as np
    
    if "csv_name" not in session:
        flash("Najpierw załaduj plik danych", "warning")
        return redirect(url_for('preprocess'))
    
    # Określ, który plik wczytać - preferuj plik z temp jeśli istnieje
    if "temp_csv_path" in session and os.path.exists(session["temp_csv_path"]):
        # Używaj pliku tymczasowego, jeśli istnieje
        file_path = session["temp_csv_path"]
        is_temp_file = True
    else:
        # Jeśli nie ma pliku w temp, skopiuj oryginalny plik do temp
        original_path = os.path.join("ChemAlize/clean/", session["csv_name"])
        os.makedirs("ChemAlize/temp/", exist_ok=True)  # Upewnij się, że folder temp istnieje
        
        # Generuj nazwę pliku z datą i godziną
        timestamp = datetime.now().strftime("%Y%m%d_%H")
        base_name = os.path.splitext(session["csv_name"])[0]
        temp_file_name = f"{base_name}_{timestamp}.csv"
        temp_path = os.path.join("ChemAlize/temp/", temp_file_name)
        
        # Kopiujemy plik tylko jeśli oryginał istnieje
        if os.path.exists(original_path):
            # Skopiuj plik
            original_df = read_dataset(original_path)
            original_df.to_csv(temp_path, index=False)
            
            # Aktualizuj sesję
            session["temp_csv_path"] = temp_path
            file_path = temp_path
            is_temp_file = True
        else:
            flash("Nie można znaleźć pliku danych", "danger")
            return redirect(url_for('preprocess'))
    
    # Wczytaj dane
    df = read_dataset(file_path)
    
    # Przygotuj zmienne do przekazania do szablonu
    filename = session.get("csv_name", "")
    temp_filename = os.path.basename(session.get("temp_csv_path", "")) if "temp_csv_path" in session else ""
    no_of_rows = len(df)
    no_of_cols = len(df.columns)
    dim = f"{no_of_rows} x {no_of_cols}"
    missing_values = int(df.isna().sum().sum())
    
    # Dodatkowe statystyki
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    has_low_variance = False
    max_correlation = 0.0
    
    # Sprawdź, czy istnieją kolumny o niskiej wariancji
    if numeric_columns:
        variances = df[numeric_columns].var()
        has_low_variance = bool(any(variances < 0.01))
    
    # Oblicz maksymalną korelację, jeśli jest przynajmniej 2 kolumny numeryczne
    if len(numeric_columns) > 1:
        corr_matrix = df[numeric_columns].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        if not upper_tri.empty and not upper_tri.values.size == 0:
            max_correlation = float(upper_tri.values.max())
    
    if request.method == 'POST':
        # Pobierz typ akcji
        action_type = request.form.get('action_type', '')
        
        # Przygotuj parametry w zależności od typu akcji
        params = {}
        
        if action_type == 'scale':
            params['scaling_method'] = request.form.get('scaling_method')
            
        elif action_type == 'remove_low_variance':
            params['variance_threshold'] = float(request.form.get('variance_threshold', 0.01))
            
        elif action_type == 'remove_correlated':
            params['correlation_threshold'] = float(request.form.get('correlation_threshold', 0.9))
            
        elif action_type == 'handle_missing':
            params['missing_method'] = request.form.get('missing_method')
            if params['missing_method'] == 'fill_constant':
                constant_value = request.form.get('constant_value', '0')
                params['constant_value'] = float(constant_value) if constant_value.replace('.', '', 1).isdigit() else constant_value
        # Obsługa pobierania pliku - zawsze pobieraj z temp
        elif action_type == 'download':
            # Sprawdź, czy istnieje plik tymczasowy
            if "temp_csv_path" in session and os.path.exists(session["temp_csv_path"]):
                # Ustaw domyślną nazwę pliku do pobrania
                download_filename = f"processed_{filename}"
                if not download_filename.endswith('.csv'):
                    download_filename += '.csv'
                
                # Ścieżka do tymczasowego pliku
                temp_path = session["temp_csv_path"]
                
                # Sprawdź, czy plik istnieje
                if not os.path.exists(temp_path):
                    print(f"Plik nie istnieje: {temp_path}")
                    
                    # Próbuj naprawić ścieżkę
                    alt_path = temp_path.replace('ChemAlize/ChemAlize/temp/', 'ChemAlize/temp/')
                    if os.path.exists(alt_path):
                        print(f"Znaleziono alternatywną ścieżkę: {alt_path}")
                        temp_path = alt_path
                    else:
                        # Spróbuj znaleźć plik w inny sposób, korzystając z nazwy pliku
                        base_filename = os.path.basename(temp_path)
                        app_root = os.path.abspath(os.path.dirname(__file__))
                        temp_dir = os.path.join(app_root, "ChemAlize/temp/")
                        temp_path = os.path.join(temp_dir, base_filename)
                        
                        if not os.path.exists(temp_path):
                            flash(f"Nie można znaleźć pliku do pobrania. Sprawdź ścieżkę: {temp_path}", "danger")
                            return redirect(url_for('manual_process'))
                
                # Zwróć plik do pobrania
                return send_file(temp_path, as_attachment=True, download_name=download_filename)
            else:
                flash("Brak przetworzonego pliku do pobrania", "warning")
        
        # Zapisz do clean, wyczyść folder temp i przejdź dalej
        elif action_type == 'next_step':
            # Sprawdź, czy istnieje plik tymczasowy
            if "temp_csv_path" in session and os.path.exists(session["temp_csv_path"]):
                # Pobierz nową nazwę pliku jeśli podana
                new_filename = request.form.get('new_filename', '')
                if not new_filename:
                    # Użyj prostej nazwy bez znacznika czasu
                    base_name = os.path.splitext(filename)[0]
                    new_filename = f"{base_name}.csv"
                elif not new_filename.endswith('.csv'):
                    new_filename += '.csv'
                
                # Zapisz plik w folderze clean
                clean_path = os.path.join("ChemAlize/clean/", new_filename)
                
                # Zapisz nowy plik do clean
                temp_df = read_dataset(session["temp_csv_path"])
                temp_df.to_csv(clean_path, index=False)
                
                # Zaktualizuj zmienne sesji
                session["csv_name"] = new_filename
                session["processed_csv_name"] = new_filename
                
                # Wyczyść folder tymczasowy
                clean_temp_folder()
                
                # Usuń ścieżkę pliku tymczasowego z sesji
                if "temp_csv_path" in session:
                    session.pop("temp_csv_path")
                
                flash(f"Plik '{new_filename}' został zapisany. Możesz przejść do kolejnego etapu.", "success")
                return redirect(url_for('preprocess'))  # Lub inny odpowiedni URL
            else:
                flash("Brak przetworzonego pliku do zapisania", "warning")
        
        # Dodaj obsługę anulowania i wyjścia
        elif action_type == 'cancel':
            # Wyczyść folder tymczasowy
            clean_temp_folder()
            
            # Usuń ścieżkę pliku tymczasowego z sesji
            if "temp_csv_path" in session:
                session.pop("temp_csv_path")
                
            flash("Operacja została anulowana. Zmiany nie zostały zapisane.", "info")
            return redirect(url_for('preprocess'))
        
        # Przetwórz dane, jeśli nie jest to akcja pobierania lub przejścia dalej
        if action_type not in ['download', 'next_step', 'cancel']:
            try:
                # Przetwórz dane z użyciem naszego modułu
                processed_df, result_info = process_dataset(df, action_type, params)
                
                # Zapisz przetworzone dane do pliku tymczasowego z datą i godziną
                os.makedirs("ChemAlize/temp/", exist_ok=True)  # Upewnij się, że folder temp istnieje
                
                # Generuj nazwę pliku z datą i godziną
                timestamp = datetime.now().strftime("%Y%m%d_%H")
                base_name = os.path.splitext(filename)[0]
                temp_file_name = f"{base_name}_{timestamp}.csv"
                temp_path = os.path.join("ChemAlize/temp/", temp_file_name)
                
                processed_df.to_csv(temp_path, index=False)
                
                # Usuń poprzedni plik tymczasowy, jeśli istnieje i różni się od aktualnego
                if "temp_csv_path" in session and os.path.exists(session["temp_csv_path"]) and session["temp_csv_path"] != temp_path:
                    try:
                        os.remove(session["temp_csv_path"])
                        print(f"Usunięto poprzedni plik tymczasowy: {session['temp_csv_path']}")
                    except Exception as e:
                        print(f"Błąd podczas usuwania poprzedniego pliku tymczasowego: {str(e)}")
                
                # Aktualizuj ścieżkę w sesji
                session["temp_csv_path"] = temp_path
                
                # Zaktualizuj informacje o danych
                no_of_rows = int(len(processed_df))
                no_of_cols = int(len(processed_df.columns))
                dim = f"{no_of_rows} x {no_of_cols}"
                missing_values = int(processed_df.isna().sum().sum())
                
                # Zaktualizuj dodatkowe statystyki
                numeric_columns = processed_df.select_dtypes(include=['number']).columns.tolist()
                has_low_variance = False
                max_correlation = 0.0
                
                if numeric_columns:
                    variances = processed_df[numeric_columns].var()
                    has_low_variance = bool(any(variances < 0.01))
                
                if len(numeric_columns) > 1:
                    corr_matrix = processed_df[numeric_columns].corr().abs()
                    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    if not upper_tri.empty and not upper_tri.values.size == 0:
                        max_correlation = float(upper_tri.values.max())
                
                # Zaktualizuj WSZYSTKIE zmienne w sesji
                session["no_of_rows"] = no_of_rows
                session["no_of_cols"] = no_of_cols
                session["dim"] = dim
                session["missing_values"] = missing_values
                session["has_low_variance"] = has_low_variance
                session["max_correlation"] = max_correlation
                session["data_cleaned"] = True
                
                # Aktualizuj nazwę pliku tymczasowego
                temp_filename = os.path.basename(temp_path)
                
                # Wyświetl komunikat
                flash(result_info.get('message', 'Dane zostały przetworzone'), "info")
                
                # Użyj przetworzonego dataframe
                df = processed_df
                
            except Exception as e:
                flash(f"Błąd podczas przetwarzania danych: {str(e)}", "danger")
    
    # Zaktualizuj zmienną wskazującą, czy używamy pliku tymczasowego
    is_temp_file = "temp_csv_path" in session and os.path.exists(session["temp_csv_path"])
    
    return render_template('manual.html',
                          filename=filename,
                          temp_filename=temp_filename,
                          is_temp_file=is_temp_file,
                          no_of_rows=no_of_rows,
                          no_of_cols=no_of_cols,
                          dim=dim,
                          missing_values=missing_values,
                          has_low_variance=has_low_variance,
                          max_correlation=max_correlation,
                          data_cleaned=session.get("data_cleaned", False),
                          target_transformed=session.get("target_transformed", False),
                          features_transformed=session.get("features_transformed", False),
                          features_selected=session.get("features_selected", False))


@app.route("/download_temp_file", methods=['GET'])
def download_temp_file():
    if "temp_csv_path" not in session:
        flash("Brak informacji o pliku do pobrania", "warning")
        return redirect(url_for('manual_process'))
    
    # Wyciągnij tylko nazwę pliku z ścieżki w sesji
    temp_filename = os.path.basename(session["temp_csv_path"])
    base_filename = session.get("csv_name", "processed_data")
    
    # Ustaw domyślną nazwę pliku do pobrania
    download_filename = f"processed_{base_filename}"
    if not download_filename.endswith('.csv'):
        download_filename += '.csv'
    
    # Uzyskaj ścieżkę bezwzględną do katalogu aplikacji
    app_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    print(f"Katalog aplikacji: {app_dir}")
    
    # Określ dokładną ścieżkę bezwzględną do katalogu temp
    temp_dir = os.path.join(app_dir, "temp")
    if not os.path.exists(temp_dir):
        temp_dir = os.path.join(app_dir, "ChemAlize", "temp")
    print(f"Katalog temp: {temp_dir}")
    
    # Sprawdź, czy plik istnieje w katalogu temp
    file_path = os.path.join(temp_dir, temp_filename)
    print(f"Szukam pliku: {file_path}")
    
    if os.path.exists(file_path):
        print(f"Znaleziono plik, wysyłam: {file_path}")
        return send_file(file_path, as_attachment=True, download_name=download_filename)
    
    # Jeśli nie znaleziono pliku, poszukaj wszystkich plików CSV w katalogu temp
    all_temp_files = glob.glob(os.path.join(temp_dir, "*.csv"))
    if all_temp_files:
        newest_file = max(all_temp_files, key=os.path.getmtime)
        print(f"Używam najnowszego pliku: {newest_file}")
        return send_file(newest_file, as_attachment=True, download_name=download_filename)
    
    # Ostatnia próba - szukaj w bezpośrednim katalogu "temp" (folder równoległy)
    alt_temp_dir = os.path.join(os.path.dirname(app_dir), "temp")
    if os.path.exists(alt_temp_dir):
        all_temp_files = glob.glob(os.path.join(alt_temp_dir, "*.csv"))
        if all_temp_files:
            newest_file = max(all_temp_files, key=os.path.getmtime)
            print(f"Używam najnowszego pliku z alternatywnego katalogu: {newest_file}")
            return send_file(newest_file, as_attachment=True, download_name=download_filename)
    
    # Jeśli wszystko zawiedzie
    flash("Nie można znaleźć pliku do pobrania. Spróbuj zapisać plik ponownie.", "danger")
    return redirect(url_for('manual_process'))

@app.route("/cleanup_temp", methods=['POST'])
def cleanup_temp():
    """Endpoint do czyszczenia folderu tymczasowego przy opuszczaniu strony."""
    clean_temp_folder()
    if "temp_csv_path" in session:
        session.pop("temp_csv_path")
    return jsonify({"status": "success"})