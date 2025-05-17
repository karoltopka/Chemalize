# ChemAlize/preprocessing/manual_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def scale_data(df, scaling_method='standard'):
    """
    Skaluje kolumny numeryczne w dataframe zgodnie z wybraną metodą.
    
    Args:
        df (pandas.DataFrame): Dataframe do przetworzenia
        scaling_method (str): Metoda skalowania ('standard', 'minmax', 'robust')
        
    Returns:
        pandas.DataFrame: Dataframe z przeskalowanymi danymi
    """
    # Pracuj na kopii dataframe
    result_df = df.copy()
    
    # Wybierz tylko kolumny numeryczne
    numeric_columns = result_df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_columns:
        return result_df  # Brak kolumn numerycznych do przeskalowania
    
    # Zastosuj odpowiednią metodę skalowania
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Nieznana metoda skalowania: {scaling_method}")
    
    # Wykonaj skalowanie na wybranych kolumnach
    result_df[numeric_columns] = scaler.fit_transform(result_df[numeric_columns])
    
    return result_df

def remove_low_variance_columns(df, threshold=0.01):
    """
    Usuwa kolumny numeryczne o wariancji mniejszej lub równej podanemu progowi.
    
    Args:
        df (pandas.DataFrame): Dataframe do przetworzenia
        threshold (float): Próg wariancji (kolumny z wariancją <= threshold zostaną usunięte)
        
    Returns:
        tuple: (pandas.DataFrame, list) - Przetworzony dataframe i lista usuniętych kolumn
    """
    # Pracuj na kopii dataframe
    result_df = df.copy()
    
    # Wybierz tylko kolumny numeryczne
    numeric_columns = result_df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_columns:
        return result_df, []  # Brak kolumn numerycznych do analizy
    
    # Oblicz wariancję dla każdej kolumny numerycznej
    variances = result_df[numeric_columns].var()
    
    # Znajdź kolumny o wariancji poniżej lub równej progowi
    low_variance_cols = variances[variances <= threshold].index.tolist()
    
    # Usuń kolumny o niskiej wariancji
    if low_variance_cols:
        result_df = result_df.drop(columns=low_variance_cols)
    
    return result_df, low_variance_cols

def remove_highly_correlated_columns(df, threshold=0.9):
    """
    Usuwa jedną z każdej pary wysoko skorelowanych kolumn.
    
    Args:
        df (pandas.DataFrame): Dataframe do przetworzenia
        threshold (float): Próg korelacji (kolumny z korelacją > threshold będą analizowane)
        
    Returns:
        tuple: (pandas.DataFrame, list) - Przetworzony dataframe i lista usuniętych kolumn
    """
    # Pracuj na kopii dataframe
    result_df = df.copy()
    
    # Wybierz tylko kolumny numeryczne
    numeric_columns = result_df.select_dtypes(include=['number']).columns.tolist()
    
    # Potrzebujemy co najmniej 2 kolumn do analizy korelacji
    if len(numeric_columns) < 2:
        return result_df, []
    
    # Oblicz macierz korelacji dla kolumn numerycznych
    corr_matrix = result_df[numeric_columns].corr().abs()
    
    # Pobierz tylko trójkąt górny macierzy korelacji
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Znajdź kolumny do usunięcia
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    # Usuń wysoko skorelowane kolumny
    if to_drop:
        result_df = result_df.drop(columns=to_drop)
    
    return result_df, to_drop

def handle_missing_values(df, method='drop_rows', constant_value=0):
    """
    Obsługuje brakujące wartości w dataframe.
    
    Args:
        df (pandas.DataFrame): Dataframe do przetworzenia
        method (str): Metoda obsługi brakujących wartości:
                     'drop_columns' - usuń kolumny z brakującymi wartościami
                     'drop_rows' - usuń wiersze z brakującymi wartościami
                     'fill_mean' - wypełnij średnią z kolumny (tylko numeryczne)
                     'fill_median' - wypełnij medianą z kolumny (tylko numeryczne)
                     'fill_mode' - wypełnij modą (najczęstszą wartością) z kolumny
                     'fill_constant' - wypełnij podaną stałą wartością
        constant_value: Wartość do użycia z metodą 'fill_constant'
        
    Returns:
        tuple: (pandas.DataFrame, dict) - Przetworzony dataframe i statystyki zmian
    """
    # Pracuj na kopii dataframe
    result_df = df.copy()
    
    # Przygotuj słownik do przechowywania statystyk zmian
    stats = {
        'rows_before': len(result_df),
        'cols_before': len(result_df.columns),
        'missing_before': result_df.isna().sum().sum()
    }
    
    # Obsłuż brakujące wartości w zależności od wybranej metody
    if method == 'drop_columns':
        # Usuń kolumny z brakującymi wartościami
        result_df = result_df.dropna(axis=1)
        
    elif method == 'drop_rows':
        # Usuń wiersze z brakującymi wartościami
        result_df = result_df.dropna(axis=0)
        
    elif method == 'fill_mean':
        # Wypełnij brakujące wartości średnią z kolumny (tylko kolumny numeryczne)
        numeric_columns = result_df.select_dtypes(include=['number']).columns.tolist()
        if numeric_columns:
            result_df[numeric_columns] = result_df[numeric_columns].fillna(result_df[numeric_columns].mean())
        
    elif method == 'fill_median':
        # Wypełnij brakujące wartości medianą z kolumny (tylko kolumny numeryczne)
        numeric_columns = result_df.select_dtypes(include=['number']).columns.tolist()
        if numeric_columns:
            result_df[numeric_columns] = result_df[numeric_columns].fillna(result_df[numeric_columns].median())
        
    elif method == 'fill_mode':
        # Wypełnij brakujące wartości modą (najczęstszą wartością) z kolumny
        for col in result_df.columns:
            # Pobierz modę dla kolumny
            mode_val = result_df[col].mode()
            # Wypełnij brakujące wartości, jeśli istnieje moda
            if not mode_val.empty:
                result_df[col] = result_df[col].fillna(mode_val[0])
        
    elif method == 'fill_constant':
        # Wypełnij brakujące wartości stałą
        result_df = result_df.fillna(constant_value)
    
    else:
        raise ValueError(f"Nieznana metoda obsługi brakujących wartości: {method}")
    
    # Uzupełnij statystyki zmian
    stats['rows_after'] = len(result_df)
    stats['cols_after'] = len(result_df.columns)
    stats['missing_after'] = result_df.isna().sum().sum()
    stats['rows_removed'] = stats['rows_before'] - stats['rows_after']
    stats['cols_removed'] = stats['cols_before'] - stats['cols_after']
    stats['missing_filled'] = stats['missing_before'] - stats['missing_after']
    
    return result_df, stats

def get_dataset_stats(df):
    """
    Zwraca podstawowe statystyki o zestawie danych.
    
    Args:
        df (pandas.DataFrame): Dataframe do analizy
        
    Returns:
        dict: Słownik zawierający statystyki zestawu danych
    """
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    stats = {
        'no_of_rows': len(df),
        'no_of_cols': len(df.columns),
        'dim': f"{len(df)} x {len(df.columns)}",
        'missing_values': df.isna().sum().sum(),
        'numeric_cols': len(numeric_columns),
        'has_low_variance': False,
        'max_correlation': 0
    }
    
    # Sprawdź, czy istnieją kolumny o niskiej wariancji
    if numeric_columns:
        variances = df[numeric_columns].var()
        stats['has_low_variance'] = any(variances < 0.01)
    
    # Oblicz maksymalną korelację, jeśli jest przynajmniej 2 kolumny numeryczne
    if len(numeric_columns) > 1:
        corr_matrix = df[numeric_columns].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        if not upper_tri.empty:
            stats['max_correlation'] = upper_tri.values.max()
    
    return stats

def process_dataset(df, action_type, params=None):
    """
    Główna funkcja przetwarzająca zestaw danych w zależności od wybranej akcji.
    
    Args:
        df (pandas.DataFrame): Dataframe do przetworzenia
        action_type (str): Typ akcji do wykonania
        params (dict): Parametry dla wybranej akcji
        
    Returns:
        tuple: (pandas.DataFrame, dict) - Przetworzony dataframe i statystyki/informacje
    """
    # Inicjalizuj parametry, jeśli nie zostały podane
    if params is None:
        params = {}
    
    # Inicjalizuj słownik na informacje wynikowe
    result_info = {}
    
    # Pracuj na kopii dataframe
    result_df = df.copy()
    
    # Wykonaj odpowiednią akcję
    if action_type == 'scale':
        scaling_method = params.get('scaling_method', 'standard')
        result_df = scale_data(result_df, scaling_method)
        result_info['message'] = f"Dane zostały przeskalowane metodą: {scaling_method}"
        
    elif action_type == 'remove_low_variance':
        threshold = params.get('variance_threshold', 0.01)
        result_df, removed_cols = remove_low_variance_columns(result_df, threshold)
        result_info['removed_cols'] = removed_cols
        result_info['message'] = f"Usunięto {len(removed_cols)} kolumn o wariancji <= {threshold}"
        
    elif action_type == 'remove_correlated':
        threshold = params.get('correlation_threshold', 0.9)
        result_df, removed_cols = remove_highly_correlated_columns(result_df, threshold)
        result_info['removed_cols'] = removed_cols
        result_info['message'] = f"Usunięto {len(removed_cols)} kolumn wysoko skorelowanych (> {threshold})"
        
    elif action_type == 'handle_missing':
        method = params.get('missing_method', 'drop_rows')
        constant_value = params.get('constant_value', 0)
        result_df, stats = handle_missing_values(result_df, method, constant_value)
        result_info.update(stats)
        
        if method == 'drop_columns':
            result_info['message'] = f"Usunięto {stats['cols_removed']} kolumn z pustymi wartościami"
        elif method == 'drop_rows':
            result_info['message'] = f"Usunięto {stats['rows_removed']} wierszy z pustymi wartościami"
        elif method == 'fill_mean':
            result_info['message'] = f"Wypełniono {stats['missing_filled']} pustych wartości średnią z kolumny"
        elif method == 'fill_median':
            result_info['message'] = f"Wypełniono {stats['missing_filled']} pustych wartości medianą z kolumny"
        elif method == 'fill_mode':
            result_info['message'] = f"Wypełniono {stats['missing_filled']} pustych wartości najczęstszą wartością"
        elif method == 'fill_constant':
            result_info['message'] = f"Wypełniono {stats['missing_filled']} pustych wartości stałą: {constant_value}"
    
    else:
        raise ValueError(f"Nieznany typ akcji: {action_type}")
    
    # Dodaj aktualne statystyki zestawu danych
    result_info['current_stats'] = get_dataset_stats(result_df)
    
    return result_df, result_info