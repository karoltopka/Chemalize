"""
Data preprocessing and upload routes
"""
from flask import Blueprint, render_template, request, session, flash, redirect, url_for, jsonify, send_file
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from app.config import get_clean_path, get_temp_path, get_upload_path, get_unified_path, TEMP_DIR, CLEAN_DIR, UPLOAD_DIR, UNIFIED_DIR
from app.chemalize.utils import read_dataset, clean_temp_folder, check_dataset, get_dataset_info, ensure_temp_dir
from app.chemalize.preprocessing import generic_preprocessing as gp
from app.chemalize.preprocessing.generic_preprocessing import get_columns, get_rows, get_dim, get_head, get_description

from app.chemalize.preprocessing.manual_preprocessing import *
from app.nocache import nocache
from app.chemalize.utils import exts, posted
import glob
import json


preprocessing_bp = Blueprint('preprocessing', __name__)

@preprocessing_bp.route("/merger", methods=["GET", "POST"])
def merger():
    """Placeholder dla Data Merger - do zaimplementowania"""
    if request.method == "POST":
        # Tutaj dodasz logikę mergera w przyszłości
        flash("Merger functionality coming soon!", "info")
    
    return render_template("merger.html", active="merger", title="Data Merger")


@preprocessing_bp.route("/preprocess", methods=["GET", "POST"])
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
                    os.makedirs(UPLOAD_DIR, exist_ok=True)
                    os.makedirs(UNIFIED_DIR, exist_ok=True)
                    os.makedirs(CLEAN_DIR, exist_ok=True)

                    # Zapisz oryginalny plik
                    upload_path = get_upload_path(data.filename)
                    data.save(upload_path)

                    # Konwertuj do CSV i zapisz w obu folderach
                    df = read_dataset(upload_path)
                    csv_filename = os.path.splitext(data.filename)[0] + '.csv'

                    # Zapisz w unified (oryginał)
                    unified_path = get_unified_path(csv_filename)
                    df.to_csv(unified_path, index=False)

                    # Skopiuj do clean (do preprocessingu)
                    clean_path = get_clean_path(csv_filename)
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
                selected_columns = request.form.getlist("check_cols")
                if not selected_columns:
                    flash("Please select at least one column to delete", "warning")
                else:
                    clean_path = get_clean_path(session["csv_name"])
                    df = read_dataset(clean_path)
                    df = gp.delete_column(df, selected_columns)
                    df.to_csv(clean_path, index=False)
                    flash(f"Column(s) {', '.join(selected_columns)} deleted successfully", "success")
            except Exception as e:
                flash(f"Error: {str(e)}", "danger")

        elif request.form["Submit"] == "DeleteRow":
            try:
                selected_rows = request.form.getlist("check_rows")
                if not selected_rows:
                    flash("Please select at least one row to delete", "warning")
                else:
                    clean_path = get_clean_path(session["csv_name"])
                    df = read_dataset(clean_path)
                    
                    # Pobierz listę wierszy do usunięcia (indeksy)
                    rows_to_delete = [int(row) for row in selected_rows]
                    
                    # Użyj funkcji gp.deleterows()
                    df = gp.delete_rows(df, rows_to_delete)
                    
                    # Zapisz zmiany
                    df.to_csv(clean_path, index=False)
                    flash(f"Row(s) {', '.join(selected_rows)} deleted successfully", "success")
            except Exception as e:
                flash(f"Error deleting rows: {str(e)}", "danger")

        elif request.form["Submit"] == "RenameMultiple":
            try:
                # Operuj na pliku w clean
                clean_path = get_clean_path(session["csv_name"])
                df = read_dataset(clean_path)
                
                # Pobierz oryginalne nazwy kolumn i nowe nazwy
                original_names = request.form.getlist("original_names")
                new_names = request.form.getlist("new_names")
                
                if not original_names:
                    flash("No columns selected for renaming", "warning")
                    return redirect(url_for('preprocessing.preprocess'))
                
                # Sprawdź czy listy mają tę samą długość
                if len(original_names) != len(new_names):
                    flash("Mismatch between original and new column names", "danger")
                    return redirect(url_for('preprocessing.preprocess'))
                
                # Przygotuj mapowanie zmian nazw
                rename_mapping = {}
                renamed_columns = []
                
                for i, (original, new) in enumerate(zip(original_names, new_names)):
                    # Sprawdź czy pole nie jest puste i czy nazwa się różni
                    if new and new.strip() and new.strip() != original:
                        new_name = new.strip()
                        
                        # Sprawdź czy nowa nazwa już istnieje w DataFrame
                        if new_name in df.columns and new_name not in original_names:
                            flash(f"Column name '{new_name}' already exists", "warning")
                            return redirect(url_for('preprocessing.preprocess'))
                        
                        # Sprawdź czy nowa nazwa nie jest duplikatem w aktualnym renamingu
                        if new_name in rename_mapping.values():
                            flash(f"Duplicate new column name '{new_name}' found", "warning")
                            return redirect(url_for('preprocessing.preprocess'))
                        
                        rename_mapping[original] = new_name
                        renamed_columns.append(f"'{original}' → '{new_name}'")
                
                # Jeśli są zmiany do wykonania
                if rename_mapping:
                    # Wykonaj rename
                    df = df.rename(columns=rename_mapping)
                    
                    # Zapisz zmiany
                    df.to_csv(clean_path, index=False)
                    
                    if len(renamed_columns) == 1:
                        flash(f"Column {renamed_columns[0]} renamed successfully", "success")
                    else:
                        flash(f"{len(renamed_columns)} columns renamed successfully: {', '.join(renamed_columns)}", "success")
                else:
                    flash("No changes were made - all new names were empty or identical to original names", "info")
                    
            except Exception as e:
                flash(f"Error renaming columns: {str(e)}", "danger")

        # Backward compatibility - keep old single rename functionality
        elif request.form["Submit"] == "Rename":
            try:
                # Operuj na pliku w clean
                clean_path = get_clean_path(session["csv_name"])
                df = read_dataset(clean_path)
                
                # Pobierz zaznaczone kolumny i nową nazwę
                selected_columns = request.form.getlist("check_cols")
                new_column_name = request.form.get("new_column_name")
                
                if len(selected_columns) != 1:
                    flash("Please select exactly one column to rename", "warning")
                elif not new_column_name:
                    flash("Please provide a new column name", "warning")
                elif new_column_name in df.columns:
                    flash("Column name already exists", "warning")
                else:
                    # Przemianuj kolumnę
                    old_name = selected_columns[0]
                    df = df.rename(columns={old_name: new_column_name})
                    
                    # Zapisz zmiany
                    df.to_csv(clean_path, index=False)
                    flash(f"Column '{old_name}' renamed to '{new_column_name}' successfully", "success")
                    
            except Exception as e:
                flash(f"Error: {str(e)}", "danger")

    if session.get("haha"):
        try:
            clean_path = get_clean_path(session["csv_name"])
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


@preprocessing_bp.route('/manual')
def manual_mode():
    if "csv_name" not in session:
        flash("First load the data", "warning")
        return redirect(url_for('preprocessing.preprocess'))
    
    # Określ, który plik wczytać - preferuj plik z temp jeśli istnieje (tak samo jak w manual_process)
    if "temp_csv_path" in session and os.path.exists(session["temp_csv_path"]):
        # Używaj pliku tymczasowego, jeśli istnieje
        file_path = session["temp_csv_path"]
        is_temp_file = True
    else:
        # W przeciwnym razie użyj pliku z clean
        file_path = get_clean_path(session["csv_name"])
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

@preprocessing_bp.route("/manual_process", methods=['GET', 'POST'])
def manual_process():
    from app.chemalize.preprocessing.manual_preprocessing import process_dataset, get_dataset_stats
    from datetime import datetime
    import numpy as np

    # Inicjalizuj statusy tylko jeśli nie istnieją w sesji
    if "data_cleaned" not in session:
        session["data_cleaned"] = False
    if "target_transformed" not in session:
        session["target_transformed"] = False
    if "features_transformed" not in session:
        session["features_transformed"] = False
    if "features_selected" not in session:
        session["features_selected"] = False

    
    if "csv_name" not in session:
        flash("First upload the data", "warning")
        return redirect(url_for('preprocessing.preprocess'))
    
    # Określ, który plik wczytać - preferuj plik z temp jeśli istnieje
    if "temp_csv_path" in session and os.path.exists(session["temp_csv_path"]):
        # Używaj pliku tymczasowego, jeśli istnieje
        file_path = session["temp_csv_path"]
        is_temp_file = True
    else:
        # Jeśli nie ma pliku w temp, skopiuj oryginalny plik do temp
        original_path = get_clean_path(session["csv_name"])
        temp_dir = ensure_temp_dir()  # Upewnij się, że folder temp istnieje
        
        # Generuj nazwę pliku z datą i godziną
        timestamp = datetime.now().strftime("%Y%m%d_%H")
        base_name = os.path.splitext(session["csv_name"])[0]
        temp_file_name = f"{base_name}_{timestamp}.csv"
        temp_path = os.path.join(temp_dir, temp_file_name)
        
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
            flash("Can not find the data", "danger")
            return redirect(url_for('preprocessing.preprocess'))
    
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
                    alt_path = temp_path.replace('data/temp/', 'data/temp/')
                    if os.path.exists(alt_path):
                        print(f"Alternative path found: {alt_path}")
                        temp_path = alt_path
                    else:
                        # Spróbuj znaleźć plik w inny sposób, korzystając z nazwy pliku
                        base_filename = os.path.basename(temp_path)
                        temp_path = get_temp_path(base_filename)
                        
                        if not os.path.exists(temp_path):
                            flash(f"Can not find file to download. Check path: {temp_path}", "danger")
                            return redirect(url_for('preprocessing.manual_process'))
                
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
                clean_path = get_clean_path(new_filename)
                
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
                
                flash(f"File '{new_filename}' has been saved. You can proceed to next step.", "success")
                return redirect(url_for('preprocessing.preprocess'))  # Lub inny odpowiedni URL
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
            return redirect(url_for('preprocessing.preprocess'))
        
        # Przetwórz dane, jeśli nie jest to akcja pobierania lub przejścia dalej
        if action_type not in ['download', 'next_step', 'cancel']:
            try:
                # Przetwórz dane z użyciem naszego modułu
                processed_df, result_info = process_dataset(df, action_type, params)
                
                # Zapisz przetworzone dane do pliku tymczasowego z datą i godziną
                temp_dir = ensure_temp_dir()  # Upewnij się, że folder temp istnieje
                
                # Generuj nazwę pliku z datą i godziną
                timestamp = datetime.now().strftime("%Y%m%d_%H")
                base_name = os.path.splitext(filename)[0]
                temp_file_name = f"{base_name}_{timestamp}.csv"
                temp_path = os.path.join(temp_dir, temp_file_name)
                
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

                # Mapuj operacje na odpowiednie statusy zgodnie z template
                if action_type == 'scale':
                    session["data_cleaned"] = True
                elif action_type == 'remove_low_variance':
                    session["target_transformed"] = True  
                elif action_type == 'remove_correlated':
                    session["features_transformed"] = True
                elif action_type == 'handle_missing':
                    session["features_selected"] = True

                # Aktualizuj nazwę pliku tymczasowego
                temp_filename = os.path.basename(temp_path)
                
                # Wyświetl komunikat
                flash(result_info.get('message', 'Data has been processed'), "info")
                
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



