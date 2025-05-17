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
# Import new analysis modules
from ChemAlize.modules import pca
from ChemAlize.modules import pcr
from ChemAlize.modules import mlr
from ChemAlize.modules import clustering
from ChemAlize.visualization import visualize as vis
from ChemAlize.nocache import nocache
from ChemAlize import app
from ChemAlize.preprocessing.generic_preprocessing import read_dataset, get_columns, get_rows, get_dim, get_description, get_head, treat_missing_numeric
from flask import render_template, url_for, flash, redirect, request, session, jsonify, send_file
import glob
import json
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for plotting


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
temp_path = "ChemAlize/temp/"
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
                    
                    # Reset any previous analysis status
                    for key in ['pca_performed', 'pcr_performed', 'mlr_performed', 'clustering_performed',
                              'target_var', 'temp_csv_path']:
                        if key in session:
                            session.pop(key)
                            
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


# Helper function to check if dataset is loaded
def check_dataset():
    if not session.get("haha"):
        flash('Please upload a dataset first!', 'danger')
        return False
    return True

# Helper function to ensure temp directory exists
def ensure_temp_dir():
    app_root = os.path.abspath(os.path.dirname(__file__))
    temp_dir = os.path.join(app_root, "temp/")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir

# Helper function to get dataset info
def get_dataset_info(df=None):
    clean_path = os.path.join("ChemAlize/clean/", session.get("csv_name", ""))
    if df is None and session.get("haha") and os.path.exists(clean_path):
        df = read_dataset(clean_path)
    
    if df is not None:
        return {
            'filename': session.get('fname', 'Unknown'),
            'no_of_rows': len(df),
            'no_of_cols': len(df.columns),
            'dim': f"{len(df)} × {len(df.columns)}",
            'columns': df.columns.tolist(),
            'target_var': session.get('target_var', None),
            'missing_values': df.isna().sum().sum()
        }
    return {}


# Analysis Dashboard Route
@app.route("/analysis_dashboard")
def analysis_dashboard():
    if not check_dataset():
        return redirect(url_for('preprocess'))
    
    clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
    df = read_dataset(clean_path)
    info = get_dataset_info(df)
    
    return render_template('analysis_dashboard.html', 
                          title='Analysis Dashboard',
                          active="analyze",
                          **info)

# Set Target Variable Route
@app.route("/set_target_variable", methods=['POST'])
def set_target_variable():
    if not check_dataset():
        return redirect(url_for('preprocess'))
    
    target_var = request.form.get('target_variable')
    if target_var:
        session['target_var'] = target_var
        flash(f'Target variable set to {target_var}', 'success')
    
    return redirect(url_for('analysis_dashboard'))

# PCA Analysis Routes
@app.route("/pca_analysis")
def pca_analysis():
    if not check_dataset():
        return redirect(url_for('preprocess'))
    
    clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
    df = read_dataset(clean_path)
    info = get_dataset_info(df)
    
    # Add any additional PCA-specific parameters from session
    pca_params = {k: session.get(k) for k in [
        'n_components', 'scale_data', 'show_variance', 'show_scatter',
        'show_loading', 'show_biplot', 'pc_color_by', 'pca_performed'
    ]}
    
    if session.get('pca_performed'):
        # Add PCA results if analysis was performed
        pca_results = {
            'pca_summary': session.get('pca_summary', []),
            'pca_variance_plot': session.get('pca_variance_plot'),
            'pca_scatter_plot': session.get('pca_scatter_plot'),
            'pca_loadings_plot': session.get('pca_loadings_plot'),
            'pca_biplot': session.get('pca_biplot')
        }
        return render_template('pca_analysis.html', title='PCA Analysis', active="analyze", **info, **pca_params, **pca_results)
    
    return render_template('pca_analysis.html', title='PCA Analysis', active="analyze", **info, **pca_params)

@app.route("/temp_image/<filename>")
def serve_temp_image(filename):
    """Serve images from the temp directory"""
    temp_dir = ensure_temp_dir()
    return send_file(os.path.join(temp_dir, filename), mimetype='image/png')

@app.route("/perform_pca", methods=['POST'])
def perform_pca():
    if not check_dataset():
        return redirect(url_for('preprocess'))
    
    # Get form parameters
    n_components = int(request.form.get('n_components', 2))
    scale_data = 'scale_data' in request.form
    show_variance = 'show_variance' in request.form
    show_scatter = 'show_scatter' in request.form
    show_loading = 'show_loading' in request.form
    show_biplot = 'show_biplot' in request.form
    pc_color_by = request.form.get('pc_color_by', '')
    
    # Save parameters to session
    session['n_components'] = n_components
    session['scale_data'] = scale_data
    session['show_variance'] = show_variance
    session['show_scatter'] = show_scatter
    session['show_loading'] = show_loading
    session['show_biplot'] = show_biplot
    session['pc_color_by'] = pc_color_by
    
    # Perform PCA using the module
    clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
    df = read_dataset(clean_path)
    
    try:
        # Create temp directory if it doesn't exist
        os.makedirs(ensure_temp_dir(), exist_ok=True)
        
        # Call the PCA module
        results = pca.perform_pca(
            df, 
            n_components=n_components,
            scale_data=scale_data,
            show_variance=show_variance,
            show_scatter=show_scatter,
            show_loading=show_loading,
            show_biplot=show_biplot,
            color_by=pc_color_by,
            temp_path=ensure_temp_dir()
        )
        
        # Store results in session
        session['pca_performed'] = True
        session['pca_summary'] = results.get('summary', [])
        
        # Extract just the filenames from the full paths
        variance_filename = os.path.basename(results.get('variance_plot', '')) if results.get('variance_plot') else ''
        scatter_filename = os.path.basename(results.get('scatter_plot', '')) if results.get('scatter_plot') else ''
        loadings_filename = os.path.basename(results.get('loadings_plot', '')) if results.get('loadings_plot') else ''
        biplot_filename = os.path.basename(results.get('biplot', '')) if results.get('biplot') else ''
        
        # Store URLs to the serve_temp_image route
        session['pca_variance_plot'] = url_for('serve_temp_image', filename=variance_filename) if variance_filename else None
        session['pca_scatter_plot'] = url_for('serve_temp_image', filename=scatter_filename) if scatter_filename else None
        session['pca_loadings_plot'] = url_for('serve_temp_image', filename=loadings_filename) if loadings_filename else None
        session['pca_biplot'] = url_for('serve_temp_image', filename=biplot_filename) if biplot_filename else None
        
        flash('PCA analysis completed successfully!', 'success')
    except Exception as e:
        flash(f'Error performing PCA: {str(e)}', 'danger')
    
    return redirect(url_for('pca_analysis'))

@app.route("/download_pca_components")
def download_pca_components():
    if not check_dataset() or not session.get('pca_performed'):
        flash('No PCA analysis results available', 'danger')
        return redirect(url_for('pca_analysis'))
    
    # Generate and return the file
    try:
        clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
        temp_file = pca.generate_components_file(clean_path, temp_path=ensure_temp_dir())
        return send_file(temp_file, as_attachment=True, download_name='pca_components.csv')
    except Exception as e:
        flash(f'Error generating components file: {str(e)}', 'danger')
        return redirect(url_for('pca_analysis'))

@app.route("/download_pca_loadings")
def download_pca_loadings():
    if not check_dataset() or not session.get('pca_performed'):
        flash('No PCA analysis results available', 'danger')
        return redirect(url_for('pca_analysis'))
    
    try:
        clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
        temp_file = pca.generate_loadings_file(clean_path, temp_path=ensure_temp_dir())
        return send_file(temp_file, as_attachment=True, download_name='pca_loadings.csv')
    except Exception as e:
        flash(f'Error generating loadings file: {str(e)}', 'danger')
        return redirect(url_for('pca_analysis'))

@app.route("/download_pca_report")
def download_pca_report():
    if not check_dataset() or not session.get('pca_performed'):
        flash('No PCA analysis results available', 'danger')
        return redirect(url_for('pca_analysis'))
    
    try:
        clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
        temp_file = pca.generate_report(clean_path, temp_path=ensure_temp_dir())
        return send_file(temp_file, as_attachment=True, download_name='pca_report.pdf')
    except Exception as e:
        flash(f'Error generating report: {str(e)}', 'danger')
        return redirect(url_for('pca_analysis'))

# PCR Analysis Routes
@app.route("/pcr_analysis")
def pcr_analysis():
    if not check_dataset():
        return redirect(url_for('preprocess'))
    
    clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
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

@app.route("/perform_pcr", methods=['POST'])
def perform_pcr():
    if not check_dataset():
        return redirect(url_for('preprocess'))
    
    if not session.get('target_var'):
        flash('Please select a target variable first!', 'danger')
        return redirect(url_for('pcr_analysis'))
    
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
    clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
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
        
        # Convert image paths to URLs
        # PCR variance plot
        if results.get('pcr_variance_plot'):
            variance_filename = os.path.basename(results.get('pcr_variance_plot'))
            session['pcr_variance_plot'] = url_for('serve_temp_image', filename=variance_filename)
        else:
            session['pcr_variance_plot'] = None
            
        # PCR pred vs actual plot
        if results.get('pcr_pred_actual_plot'):
            pred_actual_filename = os.path.basename(results.get('pcr_pred_actual_plot'))
            session['pcr_pred_actual_plot'] = url_for('serve_temp_image', filename=pred_actual_filename)
        else:
            session['pcr_pred_actual_plot'] = None
            
        # PCR residuals plot
        if results.get('pcr_residuals_plot'):
            residuals_filename = os.path.basename(results.get('pcr_residuals_plot'))
            session['pcr_residuals_plot'] = url_for('serve_temp_image', filename=residuals_filename)
        else:
            session['pcr_residuals_plot'] = None
        
        # Optimization plot (if enabled)
        if optimize_components and results.get('optimization_plot'):
            optimization_filename = os.path.basename(results.get('optimization_plot'))
            session['pcr_optimization_plot'] = url_for('serve_temp_image', filename=optimization_filename)
        
        flash('PCR analysis completed successfully!', 'success')
    except Exception as e:
        flash(f'Error performing PCR: {str(e)}', 'danger')
    
    return redirect(url_for('pcr_analysis'))

@app.route("/download_pcr_predictions")
def download_pcr_predictions():
    if not check_dataset() or not session.get('pcr_performed'):
        flash('No PCR analysis results available', 'danger')
        return redirect(url_for('pcr_analysis'))
    
    try:
        clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
        temp_file = pcr.generate_predictions_file(clean_path, session['target_var'], temp_path=ensure_temp_dir())
        return send_file(temp_file, as_attachment=True, download_name='pcr_predictions.csv')
    except Exception as e:
        flash(f'Error generating predictions file: {str(e)}', 'danger')
        return redirect(url_for('pcr_analysis'))

@app.route("/download_pcr_model")
def download_pcr_model():
    if not check_dataset() or not session.get('pcr_performed'):
        flash('No PCR analysis results available', 'danger')
        return redirect(url_for('pcr_analysis'))
    
    try:
        clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
        temp_file = pcr.generate_model_file(clean_path, session['target_var'], temp_path=ensure_temp_dir())
        return send_file(temp_file, as_attachment=True, download_name='pcr_model_coefficients.csv')
    except Exception as e:
        flash(f'Error generating model file: {str(e)}', 'danger')
        return redirect(url_for('pcr_analysis'))

@app.route("/download_pcr_report")
def download_pcr_report():
    if not check_dataset() or not session.get('pcr_performed'):
        flash('No PCR analysis results available', 'danger')
        return redirect(url_for('pcr_analysis'))
    
    try:
        clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
        temp_file = pcr.generate_report(clean_path, session['target_var'], temp_path=ensure_temp_dir())
        return send_file(temp_file, as_attachment=True, download_name='pcr_report.pdf')
    except Exception as e:
        flash(f'Error generating report: {str(e)}', 'danger')
        return redirect(url_for('pcr_analysis'))
# MLR Analysis Routes
@app.route("/mlr_analysis")
def mlr_analysis():
    if not check_dataset():
        return redirect(url_for('preprocess'))
    
    clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
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

@app.route("/perform_mlr", methods=['POST'])
def perform_mlr():
    if not check_dataset():
        return redirect(url_for('preprocess'))
    
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
        return redirect(url_for('mlr_analysis'))
    
    if not selected_features:
        flash('Please select at least one feature!', 'danger')
        return redirect(url_for('mlr_analysis'))
    
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
    clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
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
        
        # Process results before storing in session
        for key, value in results.items():
            # Check if the value looks like an image file path
            if isinstance(value, str) and (value.endswith('.png') or value.endswith('.jpg')):
                # Extract filename and convert to URL
                filename = os.path.basename(value)
                session[key] = url_for('serve_temp_image', filename=filename)
            else:
                # Store other values as-is
                session[key] = value
        
        flash('MLR analysis completed successfully!', 'success')
    except Exception as e:
        flash(f'Error performing MLR: {str(e)}', 'danger')
    
    return redirect(url_for('mlr_analysis'))

@app.route("/download_mlr_model")
def download_mlr_model():
    if not check_dataset() or not session.get('mlr_performed'):
        flash('No MLR analysis results available', 'danger')
        return redirect(url_for('mlr_analysis'))
    
    try:
        clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
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
        return redirect(url_for('mlr_analysis'))

@app.route("/download_mlr_report")
def download_mlr_report():
    if not check_dataset() or not session.get('mlr_performed'):
        flash('No MLR analysis results available', 'danger')
        return redirect(url_for('mlr_analysis'))
    
    try:
        clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
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
        return redirect(url_for('mlr_analysis'))

@app.route("/download_mlr_predictions")
def download_mlr_predictions():
    if not check_dataset() or not session.get('mlr_performed'):
        flash('No MLR analysis results available', 'danger')
        return redirect(url_for('mlr_analysis'))
    
    try:
        clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
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
        return redirect(url_for('mlr_analysis'))

@app.route("/reset_mlr_analysis")
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
    return redirect(url_for('mlr_analysis'))

# Clustering Analysis Routes
@app.route("/clustering_analysis")
def clustering_analysis():
    if not check_dataset():
        return redirect(url_for('preprocess'))
    
    clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
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
@app.route("/perform_clustering", methods=['POST'])
def perform_clustering():
    if not check_dataset():
        return redirect(url_for('preprocess'))
    
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
    clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
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
        
        # Handle numeric/text data normally
        for key, value in results.items():
            # Check if this is potentially an image path (look for known image keys or .png extension)
            if key.endswith('_plot') or (isinstance(value, str) and value.endswith('.png')):
                # Convert filesystem path to URL
                if value:
                    filename = os.path.basename(value)
                    session[key] = url_for('serve_temp_image', filename=filename)
            else:
                # For non-image data, store directly
                session[key] = value
        
        flash('Clustering analysis completed successfully!', 'success')
    except Exception as e:
        flash(f'Error performing clustering: {str(e)}', 'danger')
    
    return redirect(url_for('clustering_analysis'))

@app.route("/download_clustering_results")
def download_clustering_results():
    if not check_dataset() or not session.get('clustering_performed'):
        flash('No clustering analysis results available', 'danger')
        return redirect(url_for('clustering_analysis'))
    
    try:
        clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
        temp_file = clustering.generate_results_file(clean_path, temp_path=ensure_temp_dir())
        return send_file(temp_file, as_attachment=True, download_name='clustering_results.csv')
    except Exception as e:
        flash(f'Error generating results file: {str(e)}', 'danger')
        return redirect(url_for('clustering_analysis'))

@app.route("/download_clustering_report")
def download_clustering_report():
    if not check_dataset() or not session.get('clustering_performed'):
        flash('No clustering analysis results available', 'danger')
        return redirect(url_for('clustering_analysis'))
    
    try:
        clean_path = os.path.join("ChemAlize/clean/", session["csv_name"])
        temp_file = clustering.generate_report(
            clean_path, 
            method=session.get('clustering_method', 'kmeans'),
            temp_path=ensure_temp_dir()
        )
        return send_file(temp_file, as_attachment=True, download_name='clustering_report.pdf')
    except Exception as e:
        flash(f'Error generating report: {str(e)}', 'danger')
        return redirect(url_for('clustering_analysis'))

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    # Redirect to new dashboard (HTTP 301 = permanent redirect)
    return redirect(url_for('analysis_dashboard'), code=301)


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
            flash('Please upload a dataset first!', 'danger')
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