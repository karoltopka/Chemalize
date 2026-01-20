"""
NanoTox model loading with singleton pattern for thread-safe access
"""
import os
import joblib
from threading import Lock

# Path to the model file
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'tio2_model_package_kompromisowy.pkl')

# Singleton cache
_model_cache = {}
_model_lock = Lock()


def get_tio2_predictor():
    """
    Get the TiO2 predictor instance (singleton pattern with thread safety).

    Returns:
        TiO2Predictor: The predictor instance

    Raises:
        FileNotFoundError: If the model file is not found
        RuntimeError: If model loading fails
    """
    from .tio2_predictor import TiO2Predictor

    if 'tio2' not in _model_cache:
        with _model_lock:
            # Double-check after acquiring lock
            if 'tio2' not in _model_cache:
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError(
                        f"Model file not found at {MODEL_PATH}. "
                        "Please place 'tio2_model_package_kompromisowy.pkl' in app/data/models/"
                    )
                try:
                    pkg = joblib.load(MODEL_PATH)
                    _model_cache['tio2'] = TiO2Predictor(pkg)
                except Exception as e:
                    raise RuntimeError(f"Failed to load TiO2 model: {str(e)}")

    return _model_cache['tio2']


def is_model_available():
    """Check if the model file exists."""
    return os.path.exists(MODEL_PATH)


def get_model_path():
    """Get the expected model file path."""
    return MODEL_PATH


__all__ = ['get_tio2_predictor', 'is_model_available', 'get_model_path', 'MODEL_PATH']
