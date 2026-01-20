"""
NanoTox main blueprint - TiO2 nanoparticle toxicity prediction routes
"""
import csv
import io
import tempfile
import os
from flask import Blueprint, render_template, request, jsonify, send_file, session
from app.nocache import nocache
from app.nanotox.utils.feature_options import (
    get_all_options,
    validate_prediction_input,
    parse_csv_row,
    generate_concentration_range,
    NUMERIC_CONSTRAINTS
)
from app.nanotox.utils.visualization import (
    create_concentration_curve,
    create_feature_importance_plot,
    create_feature_contribution_plot,
    estimate_ic50
)
from app.nanotox.models import get_tio2_predictor, is_model_available, get_model_path

nanotox_main_bp = Blueprint('nanotox_main', __name__)


def get_nanotox_session_data():
    """Get NanoTox session data."""
    if 'nanotox_data' not in session:
        session['nanotox_data'] = {}
    return session['nanotox_data']


def set_nanotox_session_data(data):
    """Save NanoTox session data."""
    session['nanotox_data'] = data
    session.modified = True


@nanotox_main_bp.route('/')
@nocache
def nanotox_home():
    """NanoTox home page - landing page for TiO2 toxicity prediction"""
    model_available = is_model_available()
    return render_template(
        'nanotox/home.html',
        model_available=model_available,
        model_path=get_model_path()
    )


@nanotox_main_bp.route('/predict', methods=['GET', 'POST'])
@nocache
def nanotox_predict():
    """Single prediction form and results page"""
    options = get_all_options()
    model_available = is_model_available()

    if request.method == 'GET':
        return render_template(
            'nanotox/predict.html',
            options=options,
            model_available=model_available,
            result=None
        )

    # POST - handle prediction
    if not model_available:
        return render_template(
            'nanotox/predict.html',
            options=options,
            model_available=model_available,
            result=None,
            error="Model file not found. Please upload the model first."
        )

    # Extract form data
    try:
        data = {
            'Shape': request.form.get('shape'),
            'Anatase': request.form.get('anatase'),
            'Rutile': request.form.get('rutile'),
            'Diameter (nm)': float(request.form.get('diameter', 0)),
            'Cell_Type': request.form.get('cell_type'),
            'Time (hr)': float(request.form.get('time', 0)),
            'Test': request.form.get('test'),
            'Concentration (ug/ml)': float(request.form.get('concentration', 0))
        }
    except (ValueError, TypeError) as e:
        return render_template(
            'nanotox/predict.html',
            options=options,
            model_available=model_available,
            result=None,
            error=f"Invalid input values: {str(e)}"
        )

    # Validate input
    is_valid, errors = validate_prediction_input(data)
    if not is_valid:
        return render_template(
            'nanotox/predict.html',
            options=options,
            model_available=model_available,
            result=None,
            error="; ".join(errors),
            form_data=data
        )

    # Make prediction
    try:
        predictor = get_tio2_predictor()
        prediction = predictor.predict(data)

        # Get feature contributions if available
        contributions = predictor.get_feature_contributions(data)
        contribution_plot = None
        if contributions:
            contribution_plot = create_feature_contribution_plot(contributions, prediction)

        result = {
            'prediction': prediction,
            'data': data,
            'contributions': contributions,
            'contribution_plot': contribution_plot
        }

        # Store in session for potential curve generation
        session_data = get_nanotox_session_data()
        session_data['last_prediction'] = result
        session_data['last_input'] = data
        set_nanotox_session_data(session_data)

        return render_template(
            'nanotox/predict.html',
            options=options,
            model_available=model_available,
            result=result,
            form_data=data
        )

    except Exception as e:
        return render_template(
            'nanotox/predict.html',
            options=options,
            model_available=model_available,
            result=None,
            error=f"Prediction error: {str(e)}",
            form_data=data
        )


@nanotox_main_bp.route('/concentration-curve', methods=['POST'])
@nocache
def generate_concentration_curve():
    """Generate concentration-response curve for current parameters"""
    if not is_model_available():
        return jsonify({'error': 'Model not available'}), 400

    session_data = get_nanotox_session_data()
    base_data = session_data.get('last_input')

    if not base_data:
        return jsonify({'error': 'No previous prediction data. Make a prediction first.'}), 400

    # Get concentration range parameters from request
    req_data = request.get_json() or {}
    min_conc = float(req_data.get('min_conc', 0.1))
    max_conc = float(req_data.get('max_conc', 1000))
    n_points = int(req_data.get('n_points', 50))

    try:
        predictor = get_tio2_predictor()
        concentrations = generate_concentration_range(min_conc, max_conc, n_points)

        curve_data = predictor.predict_concentration_curve(base_data, concentrations)

        # Create plot
        plot_json = create_concentration_curve(
            curve_data,
            title=f"Concentration-Response Curve ({base_data.get('Cell_Type', 'Unknown')})"
        )

        # Estimate IC50
        ic50 = estimate_ic50(curve_data)

        return jsonify({
            'success': True,
            'plot': plot_json,
            'curve_data': curve_data,
            'ic50': ic50
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@nanotox_main_bp.route('/batch', methods=['GET', 'POST'])
@nocache
def nanotox_batch():
    """Batch prediction page with CSV upload"""
    options = get_all_options()
    model_available = is_model_available()

    if request.method == 'GET':
        return render_template(
            'nanotox/batch.html',
            options=options,
            model_available=model_available,
            results=None
        )

    # POST - handle batch prediction
    if not model_available:
        return render_template(
            'nanotox/batch.html',
            options=options,
            model_available=model_available,
            results=None,
            error="Model file not found."
        )

    if 'file' not in request.files:
        return render_template(
            'nanotox/batch.html',
            options=options,
            model_available=model_available,
            results=None,
            error="No file uploaded."
        )

    file = request.files['file']
    if file.filename == '':
        return render_template(
            'nanotox/batch.html',
            options=options,
            model_available=model_available,
            results=None,
            error="No file selected."
        )

    if not file.filename.endswith('.csv'):
        return render_template(
            'nanotox/batch.html',
            options=options,
            model_available=model_available,
            results=None,
            error="Please upload a CSV file."
        )

    try:
        # Read CSV
        content = file.read().decode('utf-8')
        reader = csv.DictReader(io.StringIO(content))

        rows = []
        for row in reader:
            parsed = parse_csv_row(row)
            rows.append(parsed)

        if not rows:
            return render_template(
                'nanotox/batch.html',
                options=options,
                model_available=model_available,
                results=None,
                error="CSV file is empty or has no valid data."
            )

        # Validate all rows
        all_errors = []
        for i, row in enumerate(rows):
            is_valid, errors = validate_prediction_input(row)
            if not is_valid:
                all_errors.append(f"Row {i + 1}: {'; '.join(errors)}")

        if all_errors:
            return render_template(
                'nanotox/batch.html',
                options=options,
                model_available=model_available,
                results=None,
                error="Validation errors:\n" + "\n".join(all_errors[:10])  # Show first 10 errors
            )

        # Make predictions
        predictor = get_tio2_predictor()
        predictions = predictor.predict_batch(rows)

        # Combine input with predictions
        results = []
        for i, (row, pred) in enumerate(zip(rows, predictions)):
            result = row.copy()
            result['Predicted Viability (%)'] = round(pred, 2)
            result['row_num'] = i + 1
            results.append(result)

        # Store results in session for download
        session_data = get_nanotox_session_data()
        session_data['batch_results'] = results
        set_nanotox_session_data(session_data)

        return render_template(
            'nanotox/batch.html',
            options=options,
            model_available=model_available,
            results=results,
            total_count=len(results)
        )

    except Exception as e:
        return render_template(
            'nanotox/batch.html',
            options=options,
            model_available=model_available,
            results=None,
            error=f"Error processing file: {str(e)}"
        )


@nanotox_main_bp.route('/download/template')
@nocache
def download_template():
    """Download CSV template for batch prediction"""
    # Create template CSV
    headers = [
        'Shape', 'Anatase', 'Rutile', 'Diameter (nm)',
        'Cell_Type', 'Time (hr)', 'Test', 'Concentration (ug/ml)'
    ]

    # Example data
    example_rows = [
        ['Spherical', 'Yes', 'No', '25', 'A549', '24', 'MTT', '10'],
        ['Spehical', 'Yes', 'Yes', '15,84', 'HEK293', '72', 'MTT', '200'],
        ['Irregular', 'Yes', 'Yes', '100', 'BEAS-2B', '24', 'LDH', '100']
    ]

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    for row in example_rows:
        writer.writerow(row)

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='nanotox_template.csv'
    )


@nanotox_main_bp.route('/download/results')
@nocache
def download_results():
    """Download batch prediction results as CSV"""
    session_data = get_nanotox_session_data()
    results = session_data.get('batch_results')

    if not results:
        return jsonify({'error': 'No results to download'}), 400

    # Create CSV
    output = io.StringIO()

    # Get all unique keys from results
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())

    # Define column order
    ordered_columns = [
        'row_num', 'Shape', 'Anatase', 'Rutile', 'Diameter (nm)',
        'Cell_Type', 'Time (hr)', 'Test', 'Concentration (ug/ml)',
        'Predicted Viability (%)'
    ]

    # Add any extra columns
    headers = [col for col in ordered_columns if col in all_keys]

    writer = csv.DictWriter(output, fieldnames=headers, extrasaction='ignore')
    writer.writeheader()
    for result in results:
        writer.writerow(result)

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='nanotox_predictions.csv'
    )


@nanotox_main_bp.route('/api/predict', methods=['POST'])
@nocache
def api_predict():
    """API endpoint for single prediction (JSON input/output)"""
    if not is_model_available():
        return jsonify({'error': 'Model not available'}), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    # Validate
    is_valid, errors = validate_prediction_input(data)
    if not is_valid:
        return jsonify({'error': 'Validation failed', 'details': errors}), 400

    try:
        predictor = get_tio2_predictor()
        prediction = predictor.predict(data)
        contributions = predictor.get_feature_contributions(data)

        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'unit': '%',
            'interpretation': get_viability_interpretation(prediction),
            'contributions': contributions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@nanotox_main_bp.route('/api/batch', methods=['POST'])
@nocache
def api_batch():
    """API endpoint for batch prediction (JSON input/output)"""
    if not is_model_available():
        return jsonify({'error': 'Model not available'}), 503

    data = request.get_json()
    if not data or 'samples' not in data:
        return jsonify({'error': 'No samples provided'}), 400

    samples = data['samples']
    if not isinstance(samples, list):
        return jsonify({'error': 'Samples must be a list'}), 400

    # Validate all samples
    all_errors = []
    for i, sample in enumerate(samples):
        is_valid, errors = validate_prediction_input(sample)
        if not is_valid:
            all_errors.append({'index': i, 'errors': errors})

    if all_errors:
        return jsonify({'error': 'Validation failed', 'details': all_errors}), 400

    try:
        predictor = get_tio2_predictor()
        predictions = predictor.predict_batch(samples)

        results = []
        for i, (sample, pred) in enumerate(zip(samples, predictions)):
            results.append({
                'index': i,
                'input': sample,
                'prediction': round(pred, 2),
                'interpretation': get_viability_interpretation(pred)
            })

        return jsonify({
            'success': True,
            'results': results,
            'total': len(results)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@nanotox_main_bp.route('/api/feature-importance')
@nocache
def api_feature_importance():
    """Get global feature importance from the model"""
    if not is_model_available():
        return jsonify({'error': 'Model not available'}), 503

    try:
        predictor = get_tio2_predictor()
        importance = predictor.get_global_feature_importance()

        plot_json = create_feature_importance_plot(importance)

        return jsonify({
            'success': True,
            'importance': importance,
            'plot': plot_json
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def get_viability_interpretation(viability):
    """Get human-readable interpretation of viability value."""
    if viability >= 80:
        return "Low toxicity - cells largely viable"
    elif viability >= 50:
        return "Moderate toxicity - significant cell death"
    elif viability >= 20:
        return "High toxicity - majority of cells affected"
    else:
        return "Severe toxicity - extensive cell death"
