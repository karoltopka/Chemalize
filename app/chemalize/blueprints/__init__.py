"""
ChemAlize Blueprints package - modular route organization
"""
from app.chemalize.blueprints.main import main_bp
from app.chemalize.blueprints.preprocessing import preprocessing_bp
from app.chemalize.blueprints.analysis import analysis_bp
from app.chemalize.blueprints.pca import pca_bp
from app.chemalize.blueprints.pcr import pcr_bp
from app.chemalize.blueprints.mlr import mlr_bp
from app.chemalize.blueprints.mlr_ga import mlr_ga_bp
from app.chemalize.blueprints.clustering import clustering_bp
from app.chemalize.blueprints.visualization import visualization_bp
from app.chemalize.blueprints.utils import utils_bp
from app.chemalize.blueprints.alvadesk_pca import alvadesk_pca_bp
from app.chemalize.blueprints.episuite import episuite_bp

__all__ = [
    'main_bp',
    'preprocessing_bp',
    'analysis_bp',
    'pca_bp',
    'pcr_bp',
    'mlr_bp',
    'mlr_ga_bp',
    'clustering_bp',
    'visualization_bp',
    'utils_bp',
    'alvadesk_pca_bp',
    'episuite_bp',
]
