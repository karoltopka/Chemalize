"""
Utility functions shared across blueprints
"""
import os
import glob
import pandas as pd
from flask import session, flash
from app.config import (
    get_clean_path,
    get_temp_path,
    get_upload_path,
    get_user_data_dir,
    clean_temp_folder as config_clean_temp
)

# Global variables
posted = 0
exts = ["csv", "xlsx", "json", "yaml", "txt", "xls"]
CHEMALIZE_STATE_KEYS_TO_KEEP = {
    'user_id',
    'haha',
    'csv_name',
    'fname',
    'ext',
    'processed_csv_name',
    'data_cleaned',
    'target_transformed',
    'features_transformed',
    'features_selected',
    'no_of_rows',
    'no_of_cols',
    'dim',
    'missing_values',
    'has_low_variance',
    'max_correlation',
}

def clean_temp_folder():
    """
    Funkcja czyści folder tymczasowy użytkownika, usuwając wszystkie pliki .csv.
    """
    # Use the config function which is now user-aware
    config_clean_temp()

def read_dataset(filepath):
    """Read dataset from various file formats."""
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

def check_dataset():
    """Check if a dataset is loaded in the session."""
    if not session.get("haha"):
        flash('Please upload a dataset first!', 'danger')
        return False
    return True

def ensure_temp_dir():
    """Ensure user-specific temp directory exists and return its path."""
    user_dir = get_user_data_dir()
    temp_dir = os.path.join(user_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir

def get_dataset_info(df=None):
    """Get dataset information from session or provided dataframe."""
    clean_path = get_clean_path(session.get("csv_name", ""))
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
            'missing_values': df.isna().sum().sum(),
            # Additional fields for compatibility
            'csv_name': session.get('csv_name', 'Unknown'),
            'n_samples': len(df),
            'n_features': len(df.columns)
        }
    return {}


def reset_analysis_state(preserve_temp_csv=True):
    """
    Reset derived analysis/visualization state while keeping current preprocessing context.
    """
    keep_keys = set(CHEMALIZE_STATE_KEYS_TO_KEEP)
    if preserve_temp_csv:
        keep_keys.add('temp_csv_path')

    preserved_values = {}
    for key in list(session.keys()):
        if key.startswith('_') or key in keep_keys:
            preserved_values[key] = session.get(key)

    session.clear()

    for key, value in preserved_values.items():
        session[key] = value
