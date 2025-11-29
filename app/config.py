"""
Configuration module for ChemAlize application.
Centralizes all path configurations and constants.
"""
import os
import uuid
from flask import session

# Base directory - root of the Chemalize package
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Project root - parent directory of Chemalize package (for Descriptors_group.txt, etc.)
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Data directories - centralized location for all data
DATA_DIR = os.path.join(BASE_DIR, 'data')
UPLOAD_DIR = os.path.join(DATA_DIR, 'uploads')
CLEAN_DIR = os.path.join(DATA_DIR, 'clean')
TEMP_DIR = os.path.join(DATA_DIR, 'temp')
UNIFIED_DIR = os.path.join(DATA_DIR, 'unified')

# Static files directory
STATIC_DIR = os.path.join(BASE_DIR, 'app', 'static')
STATIC_IMG_DIR = os.path.join(STATIC_DIR, 'img')

# Session directory
SESSION_DIR = os.path.join(BASE_DIR, 'flask_session')

# Application data directory (for configuration files, reference data, etc.)
APP_DATA_DIR = os.path.join(BASE_DIR, 'app', 'data')

# Alvadesk descriptor groups file
DESCRIPTOR_GROUPS_FILE = os.path.join(APP_DATA_DIR, 'Descriptors_group.txt')

# Supported file extensions
ALLOWED_EXTENSIONS = ['csv', 'xlsx', 'json', 'yaml', 'txt', 'xls']

# Flask session configuration
SESSION_CONFIG = {
    'SESSION_TYPE': 'filesystem',
    'SESSION_PERMANENT': False,
    'SESSION_USE_SIGNER': True,
    'SESSION_FILE_DIR': SESSION_DIR
}

def get_user_id():
    """Get or create unique user ID for current session."""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']

def get_user_data_dir(user_id=None):
    """Get user-specific data directory."""
    if user_id is None:
        user_id = get_user_id()
    return os.path.join(DATA_DIR, 'users', user_id)

def ensure_directories(user_id=None):
    """Create all required directories if they don't exist."""
    # Global directories
    global_dirs = [
        DATA_DIR,
        os.path.join(DATA_DIR, 'users'),
        STATIC_IMG_DIR,
        SESSION_DIR
    ]
    for directory in global_dirs:
        os.makedirs(directory, exist_ok=True)

    # User-specific directories (only if user_id explicitly provided)
    if user_id:
        user_dir = get_user_data_dir(user_id)
        user_dirs = [
            user_dir,
            os.path.join(user_dir, 'uploads'),
            os.path.join(user_dir, 'clean'),
            os.path.join(user_dir, 'temp'),
            os.path.join(user_dir, 'unified'),
        ]
        for directory in user_dirs:
            os.makedirs(directory, exist_ok=True)

def get_upload_path(filename, user_id=None):
    """Get full path for uploaded file (user-specific)."""
    user_dir = get_user_data_dir(user_id)
    return os.path.join(user_dir, 'uploads', filename)

def get_clean_path(filename, user_id=None):
    """Get full path for cleaned dataset (user-specific)."""
    user_dir = get_user_data_dir(user_id)
    return os.path.join(user_dir, 'clean', filename)

def get_temp_path(filename, user_id=None):
    """Get full path for temporary file (user-specific)."""
    user_dir = get_user_data_dir(user_id)
    return os.path.join(user_dir, 'temp', filename)

def get_unified_path(filename, user_id=None):
    """Get full path for unified dataset (user-specific)."""
    user_dir = get_user_data_dir(user_id)
    return os.path.join(user_dir, 'unified', filename)

def get_static_img_path(filename):
    """Get full path for static image."""
    return os.path.join(STATIC_IMG_DIR, filename)

def clean_temp_folder(user_id=None):
    """Clean temporary folder by removing all CSV files (user-specific)."""
    import glob
    user_dir = get_user_data_dir(user_id)
    temp_dir = os.path.join(user_dir, 'temp')

    if os.path.exists(temp_dir):
        for csv_file in glob.glob(os.path.join(temp_dir, "*.csv")):
            try:
                os.remove(csv_file)
                print(f"Removed temporary file: {csv_file}")
            except Exception as e:
                print(f"Error removing file {csv_file}: {str(e)}")

def clean_old_user_data(days=7):
    """Remove user data older than specified days."""
    import time
    import shutil

    users_dir = os.path.join(DATA_DIR, 'users')
    if not os.path.exists(users_dir):
        return

    current_time = time.time()
    cutoff_time = current_time - (days * 24 * 60 * 60)

    for user_id in os.listdir(users_dir):
        user_path = os.path.join(users_dir, user_id)
        if os.path.isdir(user_path):
            # Check last modification time
            mtime = os.path.getmtime(user_path)
            if mtime < cutoff_time:
                try:
                    shutil.rmtree(user_path)
                    print(f"Removed old user data: {user_id}")
                except Exception as e:
                    print(f"Error removing user data {user_id}: {str(e)}")
