"""
Blueprints package - modular route organization
"""
from app.blueprints.main import main_bp
from app.blueprints.preprocessing import preprocessing_bp
from app.blueprints.analysis import analysis_bp
from app.blueprints.pca import pca_bp
from app.blueprints.pcr import pcr_bp
from app.blueprints.mlr import mlr_bp
from app.blueprints.clustering import clustering_bp
from app.blueprints.visualization import visualization_bp
from app.blueprints.utils import utils_bp

__all__ = [
    'main_bp',
    'preprocessing_bp',
    'analysis_bp',
    'pca_bp',
    'pcr_bp',
    'mlr_bp',
    'clustering_bp',
    'visualization_bp',
    'utils_bp',
]
