"""
Feature options and validation for TiO2 nanoparticle toxicity prediction.

Contains dropdown options for categorical features and validation functions.
Note: SHAPE_OPTIONS and CELL_TYPE_OPTIONS should be updated with values
from the training data.
"""

# Shape options from training data
SHAPE_OPTIONS = [
    'Irregular',
    'Lentil',
    'Sphere'
]

# Cell type options from training data
CELL_TYPE_OPTIONS = [
    'A431',
    'A549',
    'AGS',
    'Caco-2',
    'HEK293',
    'HFL1',
    'HUVEC',
    'SHSY5Y',
    'THP-1',
    'WISH'
]

# Test/Assay options from training data
TEST_OPTIONS = [
    'CCK-8',
    'MTS',
    'MTT',
    'NRU'
]

# Yes/No options for Anatase and Rutile
YES_NO_OPTIONS = [
    ('Yes', 'Yes'),
    ('No', 'No')
]

# Numeric field constraints (based on model training data ranges)
NUMERIC_CONSTRAINTS = {
    'Diameter (nm)': {
        'min': 5,
        'max': 100,
        'step': 0.1,
        'default': 25,
        'label': 'Diameter (nm)',
        'description': 'Nanoparticle diameter in nanometers (model trained on 5-100nm)'
    },
    'Time (hr)': {
        'min': 3,
        'max': 72,
        'step': 1,
        'default': 24,
        'label': 'Exposure Time (hr)',
        'description': 'Exposure time in hours (model trained on 3-72 hr)'
    },
    'Concentration (ug/ml)': {
        'min': 0.01,
        'max': 1000,
        'step': 0.01,
        'default': 10,
        'label': 'Concentration (ug/ml)',
        'description': 'Nanoparticle concentration in micrograms per milliliter (log-transformed internally)'
    }
}


def validate_prediction_input(data):
    """
    Validate prediction input data.

    Args:
        data: Dictionary with feature names as keys

    Returns:
        tuple: (is_valid, errors) where errors is a list of error messages
    """
    errors = []

    # Required fields
    required_fields = [
        'Shape', 'Anatase', 'Rutile', 'Diameter (nm)',
        'Cell_Type', 'Time (hr)', 'Test', 'Concentration (ug/ml)'
    ]

    for field in required_fields:
        if field not in data or data[field] is None or data[field] == '':
            errors.append(f"Missing required field: {field}")

    if errors:
        return False, errors

    # Validate categorical fields
    if data.get('Shape') not in SHAPE_OPTIONS:
        errors.append(f"Invalid Shape value: {data.get('Shape')}")

    if data.get('Cell_Type') not in CELL_TYPE_OPTIONS:
        errors.append(f"Invalid Cell_Type value: {data.get('Cell_Type')}")

    if data.get('Test') not in TEST_OPTIONS:
        errors.append(f"Invalid Test value: {data.get('Test')}")

    if data.get('Anatase') not in ['Yes', 'No']:
        errors.append(f"Invalid Anatase value: {data.get('Anatase')} (must be Yes or No)")

    if data.get('Rutile') not in ['Yes', 'No']:
        errors.append(f"Invalid Rutile value: {data.get('Rutile')} (must be Yes or No)")

    # Validate numeric fields
    for field, constraints in NUMERIC_CONSTRAINTS.items():
        try:
            value = float(data.get(field, 0))
            if value < constraints['min'] or value > constraints['max']:
                errors.append(
                    f"{field} must be between {constraints['min']} and {constraints['max']}"
                )
        except (ValueError, TypeError):
            errors.append(f"{field} must be a valid number")

    return len(errors) == 0, errors


def get_all_options():
    """
    Get all feature options for the prediction form.

    Returns:
        dict: Dictionary with all dropdown options and numeric constraints
    """
    return {
        'shape_options': SHAPE_OPTIONS,
        'cell_type_options': CELL_TYPE_OPTIONS,
        'test_options': TEST_OPTIONS,
        'yes_no_options': YES_NO_OPTIONS,
        'numeric_constraints': NUMERIC_CONSTRAINTS
    }


def parse_csv_row(row):
    """
    Parse a CSV row into prediction input format.

    Args:
        row: Dictionary from CSV DictReader

    Returns:
        dict: Parsed data ready for prediction
    """
    # Map common column name variations
    column_mapping = {
        'shape': 'Shape',
        'anatase': 'Anatase',
        'rutile': 'Rutile',
        'diameter': 'Diameter (nm)',
        'diameter_nm': 'Diameter (nm)',
        'diameter (nm)': 'Diameter (nm)',
        'cell_type': 'Cell_Type',
        'celltype': 'Cell_Type',
        'cell type': 'Cell_Type',
        'time': 'Time (hr)',
        'time_hr': 'Time (hr)',
        'time (hr)': 'Time (hr)',
        'exposure_time': 'Time (hr)',
        'test': 'Test',
        'assay': 'Test',
        'concentration': 'Concentration (ug/ml)',
        'concentration_ug_ml': 'Concentration (ug/ml)',
        'concentration (ug/ml)': 'Concentration (ug/ml)',
        'conc': 'Concentration (ug/ml)'
    }

    parsed = {}
    for key, value in row.items():
        # Normalize key
        normalized_key = key.lower().strip()
        mapped_key = column_mapping.get(normalized_key, key)
        parsed[mapped_key] = value

    # Convert numeric fields
    for field in NUMERIC_CONSTRAINTS.keys():
        if field in parsed:
            try:
                parsed[field] = float(parsed[field])
            except (ValueError, TypeError):
                pass  # Keep original value, validation will catch it

    return parsed


def generate_concentration_range(min_conc=1, max_conc=1000, n_points=50):
    """
    Generate a logarithmic range of concentration values.

    Args:
        min_conc: Minimum concentration (default: 1 ug/ml)
        max_conc: Maximum concentration (default: 1000 ug/ml)
        n_points: Number of points to generate

    Returns:
        list: List of concentration values
    """
    import numpy as np
    return np.logspace(np.log10(min_conc), np.log10(max_conc), n_points).tolist()
