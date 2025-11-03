"""
Routes module - registers all blueprints
This file now only registers blueprints. Individual routes are in app/blueprints/
"""
from app import app
from app.blueprints import (
    main_bp,
    preprocessing_bp,
    analysis_bp,
    pca_bp,
    pcr_bp,
    mlr_bp,
    clustering_bp,
    visualization_bp,
    utils_bp
)

# Register blueprints with /chemalize prefix
# Main blueprint stays at root for your homepage
app.register_blueprint(main_bp)

# All ChemAlize routes under /chemalize prefix
app.register_blueprint(preprocessing_bp, url_prefix='/chemalize')
app.register_blueprint(analysis_bp, url_prefix='/chemalize')
app.register_blueprint(pca_bp, url_prefix='/chemalize')
app.register_blueprint(pcr_bp, url_prefix='/chemalize')
app.register_blueprint(mlr_bp, url_prefix='/chemalize')
app.register_blueprint(clustering_bp, url_prefix='/chemalize')
app.register_blueprint(visualization_bp, url_prefix='/chemalize')
app.register_blueprint(utils_bp, url_prefix='/chemalize')
