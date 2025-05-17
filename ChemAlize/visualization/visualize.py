import os
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # MUST BE BEFORE pyplot import
import matplotlib.pyplot as plt
import traceback
import json
import plotly.express as px
import plotly.io as pio

def get_clean_path():
    from flask import session
    csv_filename = session.get("csv_name")
    if not csv_filename:
        raise ValueError("No file uploaded")
    return os.path.join("ChemAlize/clean/", csv_filename)

def get_rows(df):
    clean_path = get_clean_path()
    df = pd.read_csv(clean_path)
    return df.index.tolist()

def get_columns():
    clean_path = get_clean_path()
    df = pd.read_csv(clean_path)
    return df.columns

def pair_plot():
    clean_path = get_clean_path()
    df = pd.read_csv(clean_path)
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numerical columns found")
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr().abs()
    
    # For each column, find the sum of correlation values (excluding self-correlation)
    correlation_strength = {}
    for col in corr_matrix.columns:
        # Sum all correlations excluding the diagonal (correlation with itself)
        correlation_strength[col] = corr_matrix[col].sum() - 1  # Subtract 1 to exclude self-correlation
    
    # Get the 15 columns with highest correlation sum (or all if less than 15)
    top_cols = sorted(correlation_strength.items(), key=lambda x: x[1], reverse=True)
    num_cols = min(8, len(top_cols))
    most_correlated_cols = [col[0] for col in top_cols[:num_cols]]
    
    plt.figure(figsize=(20, 8))
    sns_plot = sns.pairplot(df[most_correlated_cols], height=2.5)
    
    # Create directory if needed
    os.makedirs("ChemAlize/static/img", exist_ok=True)
    
    plot_path = "ChemAlize/static/img/pairplot1.png"
    sns_plot.savefig(plot_path)
    plt.close()  # Cleanup memory
    return "/static/img/pairplot1.png"  # Zwróć ścieżkę względną

def xy_plot(feature_x, feature_y):
    """
    Generate scatter plot between two selected variables
    Returns the DataFrame containing the selected columns for Chart.js visualization
    """
    try:
        print(f"Starting xy_plot with {feature_x} and {feature_y}")
        
        # Wczytaj dane z pliku col.csv
        df = pd.read_csv("ChemAlize/visualization/col.csv")
        print(f"DataFrame loaded, shape: {df.shape}")
        
        # Sprawdź, czy kolumny istnieją
        if feature_x not in df.columns or feature_y not in df.columns:
            error_msg = f"Columns {feature_x} or {feature_y} not found in dataset"
            print(error_msg)
            raise ValueError(error_msg)
            
        # Sprawdź, czy mamy dane do wyświetlenia
        if df[feature_x].dropna().empty or df[feature_y].dropna().empty:
            error_msg = f"Columns {feature_x} or {feature_y} don't contain valid data"
            print(error_msg)
            raise ValueError(error_msg)
        
        # Usuń wiersze z brakującymi wartościami w wybranych kolumnach
        df_clean = df.dropna(subset=[feature_x, feature_y])
        print(f"Cleaned DataFrame, shape: {df_clean.shape}")
        
        # Sprawdź typy danych
        is_x_numeric = pd.api.types.is_numeric_dtype(df_clean[feature_x])
        is_y_numeric = pd.api.types.is_numeric_dtype(df_clean[feature_y])
        
        # Przygotuj kopię DataFramu do zwrócenia
        result_df = df_clean.copy()
        
        # Jeśli któraś kolumna nie jest numeryczna, ale może zawierać liczby zapisane jako tekst
        if not is_x_numeric:
            try:
                # Spróbuj konwersji, ale NIE aktualizuj oryginalnej kolumny
                numeric_x = pd.to_numeric(df_clean[feature_x], errors='coerce')
                # Sprawdź, czy konwersja poskutkowała sensownymi danymi
                if not numeric_x.dropna().empty and len(numeric_x.dropna()) > 0.5 * len(numeric_x):
                    result_df['x_numeric'] = numeric_x
                    is_x_numeric = True
                    print(f"Converted {feature_x} to numeric")
                else:
                    # Stwórz mapowanie kategorii do wartości liczbowych
                    unique_x = df_clean[feature_x].unique()
                    x_mapping = {val: i for i, val in enumerate(unique_x)}
                    result_df['x_numeric'] = df_clean[feature_x].map(x_mapping)
                    print(f"Created categorical mapping for {feature_x}")
            except:
                print(f"{feature_x} cannot be converted to numeric")
                # Stwórz mapowanie kategorii do wartości liczbowych
                unique_x = df_clean[feature_x].unique()
                x_mapping = {val: i for i, val in enumerate(unique_x)}
                result_df['x_numeric'] = df_clean[feature_x].map(x_mapping)
                print(f"Created categorical mapping for {feature_x}")
        
        if not is_y_numeric:
            try:
                # Spróbuj konwersji, ale NIE aktualizuj oryginalnej kolumny
                numeric_y = pd.to_numeric(df_clean[feature_y], errors='coerce')
                # Sprawdź, czy konwersja poskutkowała sensownymi danymi
                if not numeric_y.dropna().empty and len(numeric_y.dropna()) > 0.5 * len(numeric_y):
                    result_df['y_numeric'] = numeric_y
                    is_y_numeric = True
                    print(f"Converted {feature_y} to numeric")
                else:
                    # Stwórz mapowanie kategorii do wartości liczbowych
                    unique_y = df_clean[feature_y].unique()
                    y_mapping = {val: i for i, val in enumerate(unique_y)}
                    result_df['y_numeric'] = df_clean[feature_y].map(y_mapping)
                    print(f"Created categorical mapping for {feature_y}")
            except:
                print(f"{feature_y} cannot be converted to numeric")
                # Stwórz mapowanie kategorii do wartości liczbowych
                unique_y = df_clean[feature_y].unique()
                y_mapping = {val: i for i, val in enumerate(unique_y)}
                result_df['y_numeric'] = df_clean[feature_y].map(y_mapping)
                print(f"Created categorical mapping for {feature_y}")
        
        # Upewnij się, że katalog istnieje
        os.makedirs("ChemAlize/static/img", exist_ok=True)
        
        # Przygotuj dane do wykresu
        if 'x_numeric' in result_df.columns:
            plot_x = result_df['x_numeric']
            x_col_for_chart = 'x_numeric'
        else:
            plot_x = result_df[feature_x]
            x_col_for_chart = feature_x
            
        if 'y_numeric' in result_df.columns:
            plot_y = result_df['y_numeric']
            y_col_for_chart = 'y_numeric'
        else:
            plot_y = result_df[feature_y]
            y_col_for_chart = feature_y
        
        # Wizualizacja w zależności od typów danych
        plt.figure(figsize=(10, 6))
        
        if is_x_numeric and is_y_numeric:
            # Standardowy scatter plot dla danych liczbowych
            plt.scatter(plot_x, plot_y, alpha=0.6, color="red")
            
            # Dodaj linię trendu
            try:
                import numpy as np
                from scipy import stats
                
                x = plot_x.values
                y = plot_y.values
                
                if len(x) > 1:  # Potrzebujemy co najmniej 2 punktów
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    line_x = np.array([min(x), max(x)])
                    line_y = slope * line_x + intercept
                    
                    plt.plot(line_x, line_y, 'b-', alpha=0.7)
                    plt.title(f'Scatter plot: {feature_x} vs {feature_y}\nR²: {r_value**2:.4f}')
                else:
                    plt.title(f'Scatter plot: {feature_x} vs {feature_y}')
            except Exception as e:
                print(f"Nie udało się dodać linii trendu: {str(e)}")
                plt.title(f'Scatter plot: {feature_x} vs {feature_y}')
        
        elif not is_x_numeric and not is_y_numeric:
            # Obie kolumny kategoryczne - użyj barplot lub countplot
            plt.figure(figsize=(12, 8))
            
            # Policz kombinacje wartości
            crosstab = pd.crosstab(df_clean[feature_x], df_clean[feature_y])
            
            if crosstab.shape[0] > 10 or crosstab.shape[1] > 10:
                plt.title(f"Za dużo unikalnych wartości do wizualizacji\n{feature_x} vs {feature_y}")
                plt.text(0.5, 0.5, "Zbyt wiele unikalnych kategorii",
                         horizontalalignment='center', fontsize=14)
                plt.axis('off')
            else:
                sns.heatmap(crosstab, annot=True, cmap="YlGnBu", fmt="d")
                plt.title(f'Heatmap: {feature_x} vs {feature_y}')
                plt.tight_layout()
        
        elif is_x_numeric and not is_y_numeric:
            # X numeryczne, Y kategoryczne - boxplot
            sns.boxplot(x=feature_y, y=feature_x, data=df_clean)
            plt.title(f'Boxplot: {feature_x} by {feature_y}')
            plt.tight_layout()
        
        elif not is_x_numeric and is_y_numeric:
            # X kategoryczne, Y numeryczne - boxplot z odwróconymi osiami
            sns.boxplot(x=feature_x, y=feature_y, data=df_clean)
            plt.title(f'Boxplot: {feature_y} by {feature_x}')
            plt.tight_layout()
        
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.grid(True, alpha=0.3)
        
        # Zapisz wykres
        plot_path = "ChemAlize/static/img/fig.png"
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.close()  # Zwolnij pamięć
        
        # Przygotuj dane do zwrócenia
        if 'x_numeric' in result_df.columns or 'y_numeric' in result_df.columns:
            cols_to_return = [feature_x, feature_y]
            if 'x_numeric' in result_df.columns:
                cols_to_return.append('x_numeric')
            if 'y_numeric' in result_df.columns:
                cols_to_return.append('y_numeric')
            
            print(f"Returning DataFrame with {len(result_df)} rows and mapped numeric values")
            return result_df[cols_to_return]
        else:
            print(f"Returning DataFrame with {len(result_df)} rows")
            return result_df[[feature_x, feature_y]]
        
    except Exception as e:
        print(f"Error in xy_plot: {str(e)}")
        traceback.print_exc()
        # W przypadku błędu zwróć pusty DataFrame
        return pd.DataFrame()
    
def static_hist_plot(df, feature_x):
    try:
        print(f"Generating histogram for column: {feature_x}")
        print(f"DataFrame info: {df.info()}")
        
        # Sprawdź czy kolumna istnieje
        if feature_x not in df.columns:
            print(f"Kolumna {feature_x} nie istnieje w ramce danych")
            return None
            
        # Upewnij się że mamy dane do wyświetlenia
        if df[feature_x].dropna().empty:
            print(f"Kolumna {feature_x} nie zawiera prawidłowych danych")
            return None
            
        # WAŻNE: Folder dla obrazów w strukturze Flask
        # Zauważ, że tworzymy folder w katalogu ChemAlize/static/img
        static_folder = "ChemAlize/static/img"
        os.makedirs(static_folder, exist_ok=True)
        
        # Ścieżka do zapisania pliku
        file_name = f"hist_{feature_x}.png"
        abs_plot_path = os.path.join(static_folder, file_name)
        
        # Względna ścieżka dla URL Flask (WAŻNE!)
        rel_plot_path = f"/static/img/{file_name}"
        
        plt.figure(figsize=(10, 6))
        
        # Użyj displot zamiast histplot dla lepszej kompatybilności
        try:
            plot = sns.histplot(data=df, x=feature_x, kde=True)
        except Exception as e:
            print(f"Błąd podczas tworzenia histogramu: {str(e)}")
            try:
                # Spróbuj alternatywnej metody
                plt.hist(df[feature_x].dropna(), bins=20)
                plt.xlabel(feature_x)
                plt.ylabel('Frequency')
                plt.title(f'Histogram of {feature_x}')
            except Exception as e2:
                print(f"Również alternatywna metoda nie działa: {str(e2)}")
                return None
        
        # Zapisz wykres
        plt.tight_layout()
        plt.savefig(abs_plot_path)
        plt.close()  # Cleanup memory
        
        print(f"Wykres zapisany w: {abs_plot_path}")
        print(f"Ścieżka względna: {rel_plot_path}")
        
        return rel_plot_path
    except Exception as e:
        print(f"Błąd w hist_plot: {str(e)}")
        traceback.print_exc()
        return None

def interactive_hist_plot(df, feature_x):
    """
    Generate interactive histogram using Plotly for a specified column
    Returns JSON data for the histogram
    """
    try:
        print(f"Generating interactive histogram for column: {feature_x}")
        
        # Check if column exists
        if feature_x not in df.columns:
            print(f"Kolumna {feature_x} nie istnieje w ramce danych")
            return None
            
        # Ensure we have valid data to display
        if df[feature_x].dropna().empty:
            print(f"Kolumna {feature_x} nie zawiera prawidłowych danych")
            return None
        
        # Try to determine if column is numeric or categorical
        is_numeric = pd.api.types.is_numeric_dtype(df[feature_x])
        
        # Create appropriate histogram data
        if is_numeric:
            # For numeric data, create a more detailed histogram
            # Get the values and counts
            values = df[feature_x].dropna().tolist()
            
            # Return data as JSON
            histogram_data = {
                "values": values,
                "column_name": feature_x,
                "is_numeric": True
            }
        else:
            # For categorical data, count values
            value_counts = df[feature_x].value_counts().reset_index()
            value_counts.columns = ['category', 'count']
            
            # Return data as JSON
            histogram_data = {
                "categories": value_counts['category'].tolist(),
                "counts": value_counts['count'].tolist(),
                "column_name": feature_x,
                "is_numeric": False
            }
        
        return histogram_data
        
    except Exception as e:
        print(f"Błąd w interactive_hist_plot: {str(e)}")
        traceback.print_exc()
        return None