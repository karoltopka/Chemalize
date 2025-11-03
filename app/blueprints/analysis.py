"""
Analysis dashboard and classic ML models
"""
from flask import Blueprint, render_template, request, session, flash, redirect, url_for, jsonify, send_file
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from app.config import get_clean_path, get_temp_path, get_upload_path, get_unified_path, TEMP_DIR, CLEAN_DIR, UPLOAD_DIR, UNIFIED_DIR
from app.utils import read_dataset, clean_temp_folder, check_dataset, get_dataset_info, ensure_temp_dir
from app.preprocessing import generic_preprocessing as gp

from app.modules import logistic as lg
from app.modules import naive_bayes as nb
from app.modules import linear_svc as lsvc
from app.modules import knn
from app.modules import decision_tree as dtree
from app.modules import random_forest as rfc


analysis_bp = Blueprint('analysis', __name__)

@analysis_bp.route("/analysis_dashboard")
def analysis_dashboard():
    if not check_dataset():
        return redirect(url_for('preprocessing.preprocess'))
    
    clean_path = get_clean_path(session["csv_name"])
    df = read_dataset(clean_path)
    info = get_dataset_info(df)
    
    return render_template('analysis_dashboard.html', 
                          title='Analysis Dashboard',
                          active="analyze",
                          **info)

# Set Target Variable Route

@analysis_bp.route("/analyze", methods=["GET", "POST"])
def analyze():
    # Redirect to new dashboard (HTTP 301 = permanent redirect)
    return redirect(url_for('analysis.analysis_dashboard'), code=301)



