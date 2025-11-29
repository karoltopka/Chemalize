"""
Routes module - registers all blueprints
Organized by application: ChemAlize and ScopeHub
"""
from app import app

# ChemAlize blueprints
from app.chemalize.blueprints import (
    main_bp,
    preprocessing_bp,
    analysis_bp,
    pca_bp,
    pcr_bp,
    mlr_bp,
    mlr_ga_bp,
    clustering_bp,
    visualization_bp,
    utils_bp,
    alvadesk_pca_bp,
    episuite_bp
)

# ScopeHub blueprints
from app.scopehub.blueprints import scopehub_main_bp

# Register blueprints with /chemalize prefix
# Main blueprint stays at root for your homepage
app.register_blueprint(main_bp)

# All ChemAlize routes under /chemalize prefix
app.register_blueprint(preprocessing_bp, url_prefix='/chemalize')
app.register_blueprint(analysis_bp, url_prefix='/chemalize')
app.register_blueprint(pca_bp, url_prefix='/chemalize')
app.register_blueprint(pcr_bp, url_prefix='/chemalize')
app.register_blueprint(mlr_bp, url_prefix='/chemalize')
app.register_blueprint(mlr_ga_bp, url_prefix='/chemalize')
app.register_blueprint(clustering_bp, url_prefix='/chemalize')
app.register_blueprint(visualization_bp, url_prefix='/chemalize')
app.register_blueprint(utils_bp, url_prefix='/chemalize')
app.register_blueprint(alvadesk_pca_bp, url_prefix='/chemalize')
app.register_blueprint(episuite_bp, url_prefix='/chemalize')

# ScopeHub routes under /scopehub prefix
app.register_blueprint(scopehub_main_bp, url_prefix='/scopehub')
