"""
Main routes - Home and basic pages
"""
from flask import Blueprint, render_template, request, session, flash, redirect, url_for, jsonify, send_file
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from app.config import get_clean_path, get_temp_path, get_upload_path, get_unified_path, TEMP_DIR, CLEAN_DIR, UPLOAD_DIR, UNIFIED_DIR
from app.utils import read_dataset, clean_temp_folder
from app.preprocessing import generic_preprocessing as gp



main_bp = Blueprint('main', __name__)

@main_bp.route("/")
def home():
    """Strona główna - Landing page"""
    return render_template("landing_page.html", active="home", title="Home")

@main_bp.route("/chemalize")
def chemalize_home():
    """ChemAlize homepage z wyborem Analysis/Merger"""
    return render_template("home.html", active="home", title="ChemAlize - Home")


@main_bp.route("/clear", methods=["GET"])
def clear():
    session.clear()
    return redirect(url_for('preprocessing.preprocess'))


