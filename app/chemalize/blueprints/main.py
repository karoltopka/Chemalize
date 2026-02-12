"""
Main routes - Home and basic pages
"""
from datetime import datetime
from flask import Blueprint, render_template, request, session, flash, redirect, url_for, jsonify, send_file, Response
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from app.config import get_clean_path, get_temp_path, get_upload_path, get_unified_path, TEMP_DIR, CLEAN_DIR, UPLOAD_DIR, UNIFIED_DIR
from app.chemalize.utils import read_dataset, clean_temp_folder
from app.chemalize.preprocessing import generic_preprocessing as gp



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


@main_bp.route("/sitemap.xml", methods=["GET"])
def sitemap():
    """
    Dynamic XML sitemap for public pages.
    """
    today = datetime.utcnow().date().isoformat()
    pages = []

    sitemap_routes = [
        ("main.home", "daily", "1.0"),
        ("main.chemalize_home", "daily", "0.9"),
        ("preprocessing.preprocess", "daily", "0.8"),
        ("analysis.analysis_dashboard", "daily", "0.8"),
        ("pca.pca_analysis", "daily", "0.8"),
        ("visualization.visualize", "daily", "0.8"),
        ("alvadesk_pca.alvadesk_pca_analysis", "weekly", "0.7"),
        ("scopehub_main.scopehub_home", "weekly", "0.7"),
        ("scopehub_main.scopehub_database", "weekly", "0.6"),
        ("scopehub_main.query_manager", "weekly", "0.6"),
        ("nanotox_main.nanotox_home", "weekly", "0.7"),
        ("nanotox_main.nanotox_predict", "weekly", "0.6"),
        ("nanotox_main.nanotox_batch", "weekly", "0.6"),
    ]

    for endpoint, changefreq, priority in sitemap_routes:
        try:
            pages.append({
                "loc": url_for(endpoint, _external=True),
                "lastmod": today,
                "changefreq": changefreq,
                "priority": priority,
            })
        except Exception:
            # Skip routes that cannot be built in current runtime context.
            continue

    xml = render_template("sitemap.xml", pages=pages)
    return Response(xml, mimetype="application/xml")


@main_bp.route("/robots.txt", methods=["GET"])
def robots():
    """
    robots.txt with sitemap pointer for search engines.
    """
    robots_content = "\n".join([
        "User-agent: *",
        "Allow: /",
        f"Sitemap: {url_for('main.sitemap', _external=True)}",
        "",
    ])
    return Response(robots_content, mimetype="text/plain")
