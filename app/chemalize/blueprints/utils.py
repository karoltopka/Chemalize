"""
Utility routes for downloads and cleanup
"""
from flask import Blueprint, render_template, request, session, flash, redirect, url_for, jsonify, send_file
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from app.config import get_clean_path, get_temp_path, get_upload_path, get_unified_path, TEMP_DIR, CLEAN_DIR, UPLOAD_DIR, UNIFIED_DIR
from app.chemalize.utils import read_dataset, clean_temp_folder, check_dataset, get_dataset_info, ensure_temp_dir
from app.chemalize.preprocessing import generic_preprocessing as gp

from app.nocache import nocache
import glob


utils_bp = Blueprint('utils', __name__)

@utils_bp.route("/set_target_variable", methods=['POST'])
def set_target_variable():
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))
    
    target_var = request.form.get('target_variable')
    if target_var:
        session['target_var'] = target_var
        flash(f'Target variable set to {target_var}', 'success')
    
    return redirect(url_for('analysis.analysis_dashboard'))

# Enhanced PCA Analysis Routes

@utils_bp.route("/temp_image/<filename>")
@nocache
def serve_temp_image(filename):
    """Serve images from the temp directory"""
    temp_dir = ensure_temp_dir()
    return send_file(os.path.join(temp_dir, filename), mimetype='image/png')


@utils_bp.route("/download_feature_importance")
def download_feature_importance():
    if not check_dataset() or not session.get('pca_performed'):
        flash('No PCA analysis results available', 'danger')
        return redirect(url_for('pca.pca_analysis'))
    
    try:
        temp_file = pca.generate_feature_importance_file(temp_path=ensure_temp_dir())
        return send_file(temp_file, as_attachment=True, download_name='feature_importance.csv')
    except Exception as e:
        flash(f'Error generating feature importance file: {str(e)}', 'danger')
        return redirect(url_for('pca.pca_analysis'))


@utils_bp.route("/download_selected_features")
def download_selected_features():
    if not check_dataset() or not session.get('pca_performed'):
        flash('No PCA analysis results available', 'danger')
        return redirect(url_for('pca.pca_analysis'))
    
    try:
        clean_path = get_clean_path(session["csv_name"])
        temp_file = pca.generate_selected_features_file(clean_path, temp_path=ensure_temp_dir())
        return send_file(temp_file, as_attachment=True, download_name='selected_features_data.csv')
    except Exception as e:
        flash(f'Error generating selected features file: {str(e)}', 'danger')
        return redirect(url_for('pca.pca_analysis'))


@utils_bp.route("/download_temp_file", methods=['GET'])
def download_temp_file():
    """Pobierz plik po preprocessingu lub oryginalny z clean."""
    
    # Sprawdź czy istnieje informacja o pliku w sesji
    if "temp_csv_path" not in session and "csv_name" not in session:
        flash("No file information available for download", "warning")
        return redirect(url_for('preprocessing.manual_process'))
    
    # Pobierz informacje z sesji
    csv_name = session.get("csv_name", "processed_data")
    temp_csv_path = session.get("temp_csv_path")
    
    # Ustaw nazwę do pobrania
    base_filename = os.path.splitext(csv_name)[0] if csv_name.endswith('.csv') else csv_name
    download_filename = f"processed_{base_filename}.csv"
    
    # Pobierz absolutną ścieżkę do katalogu aplikacji
    app_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    print(f"Application directory: {app_dir}")
    
    # Określ katalog temp
    temp_dir = os.path.join(app_dir, "temp")
    if not os.path.exists(temp_dir):
        temp_dir = os.path.join(app_dir, "ChemAlize", "temp")
    print(f"Temp directory: {temp_dir}")
    
    file_to_download = None
    
    # NAJPIERW sprawdź czy istnieje plik po preprocessingu
    if temp_csv_path:
        # Konwertuj względną ścieżkę z sesji na absolutną
        if not os.path.isabs(temp_csv_path):
            # temp_csv_path jest względny (np. "ChemAlize/temp/Enrichment_20250614_17.csv")
            absolute_temp_path = os.path.join(app_dir, temp_csv_path)
        else:
            absolute_temp_path = temp_csv_path
            
        print(f"Checking absolute temp path: {absolute_temp_path}")
        
        if os.path.exists(absolute_temp_path):
            file_to_download = absolute_temp_path
            print(f"Found preprocessed file: {file_to_download}")
        else:
            # Spróbuj znaleźć plik tylko po nazwie w temp directory
            temp_filename = os.path.basename(temp_csv_path)
            file_path = os.path.join(temp_dir, temp_filename)
            print(f"Looking for temp file by name: {file_path}")
            
            if os.path.exists(file_path):
                file_to_download = file_path
                print(f"Found temp file: {file_to_download}")
    
    # Jeśli nadal nie znaleziono, spróbuj z oryginalnym plikiem z clean
    if not file_to_download and csv_name:
        clean_path = os.path.join(app_dir, "ChemAlize", "clean", csv_name)
        print(f"Looking for clean file: {clean_path}")
        
        if os.path.exists(clean_path):
            # Upewnij się, że katalog temp istnieje
            os.makedirs(temp_dir, exist_ok=True)
            
            # Skopiuj z clean do temp
            import shutil
            temp_copy_path = os.path.join(temp_dir, csv_name)
            try:
                shutil.copy2(clean_path, temp_copy_path)
                file_to_download = temp_copy_path
                session["temp_csv_path"] = temp_copy_path
                download_filename = f"original_{base_filename}.csv"
                print(f"Copied from clean to temp: {file_to_download}")
            except Exception as e:
                print(f"Error copying from clean: {str(e)}")
    
    # Jeśli znaleziono plik, wyślij go
    if file_to_download and os.path.exists(file_to_download):
        print(f"File found, sending: {file_to_download}")
        return send_file(file_to_download, as_attachment=True, download_name=download_filename)
    
    # Jeśli nie znaleziono, spróbuj znaleźć najnowszy plik CSV w temp
    if os.path.exists(temp_dir):
        all_temp_files = glob.glob(os.path.join(temp_dir, "*.csv"))
        if all_temp_files:
            newest_file = max(all_temp_files, key=os.path.getmtime)
            print(f"Using newest file: {newest_file}")
            return send_file(newest_file, as_attachment=True, download_name=download_filename)
    
    # Ostatnia próba - sprawdź alternatywny katalog temp
    alt_temp_dir = os.path.join(os.path.dirname(app_dir), "temp")
    if os.path.exists(alt_temp_dir):
        all_temp_files = glob.glob(os.path.join(alt_temp_dir, "*.csv"))
        if all_temp_files:
            newest_file = max(all_temp_files, key=os.path.getmtime)
            print(f"Using newest file from alternative directory: {newest_file}")
            return send_file(newest_file, as_attachment=True, download_name=download_filename)
    
    # Jeśli wszystko zawodzi
    flash("Cannot find file to download. Try saving the file again.", "danger")
    return redirect(url_for('preprocessing.manual_process'))




@utils_bp.route("/cleanup_temp", methods=['POST'])
def cleanup_temp():
    """Endpoint do czyszczenia folderu tymczasowego przy opuszczaniu strony."""
    clean_temp_folder()
    if "temp_csv_path" in session:
        session.pop("temp_csv_path")
    return jsonify({"status": "success"})
