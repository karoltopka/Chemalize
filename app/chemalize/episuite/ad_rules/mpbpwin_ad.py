"""
MPBPWIN Applicability Domain Rules

MPBPWIN estimates:
- Melting Point (MP)
- Boiling Point (BP)
- Vapor Pressure (VP)

Reference:
- Original Joback methodology: 388 compounds
- Complete training sets NOT available
- Test sets available in app/episuite/reference_data/:
  * Melting_Pt_TestSet.xls
  * Boiling_Pt_TestSet.xls
  * VaporPressure_TestSet.xls

Source: EPI Suite Documentation
Note: "Currently there is no universally accepted definition of model domain"
      for MPBPWIN. Significant errors are possible.
"""

import os


# Reference data file paths
REFERENCE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'reference_data'
)

MELTING_PT_TESTSET = os.path.join(REFERENCE_DATA_DIR, 'Melting_Pt_TestSet.xls')
BOILING_PT_TESTSET = os.path.join(REFERENCE_DATA_DIR, 'Boiling_Pt_TestSet.xls')
VAPOR_PRESSURE_TESTSET = os.path.join(REFERENCE_DATA_DIR, 'VaporPressure_TestSet.xls')


# Training set information (limited data available)
JOBACK_METHOD = {
    'num_compounds': 388,
    'description': 'Original Joback methodology',
    'note': 'Complete training set data not available. '
            'Maximum fragment counts per compound not documented.'
}

GOLD_OGLE_METHOD = {
    'description': 'Gold and Ogle melting point method',
    'note': 'Training set data not available'
}


def check_applicability_domain(molecular_weight, property_type='MP',
                                estimated_value=None, fragments=None):
    """
    Check if MPBPWIN prediction is within applicability domain.

    Note: Due to limited training set information, AD assessment is based on:
    1. Molecular weight ranges (when available from test sets)
    2. Presence of unusual structural features
    3. Extreme property values

    Args:
        molecular_weight (float): Molecular weight of the compound
        property_type (str): Type of property ('MP', 'BP', or 'VP')
        estimated_value (float, optional): Estimated property value
        fragments (list, optional): List of fragment dictionaries

    Returns:
        dict: {
            'in_ad': bool - Conservative assessment (True if likely within AD)
            'status': str - Detailed status message
            'warnings': list - List of warning messages
            'details': dict - Additional details about the assessment
        }
    """
    warnings = []
    status = "Applicability domain cannot be precisely defined"
    details = {
        'property_type': property_type,
        'training_set_available': False,
        'assessment_basis': 'Conservative - based on limited training data',
        'reference_files_available': []
    }

    # Check if reference files exist
    if os.path.exists(MELTING_PT_TESTSET):
        details['reference_files_available'].append('Melting_Pt_TestSet.xls')
    if os.path.exists(BOILING_PT_TESTSET):
        details['reference_files_available'].append('Boiling_Pt_TestSet.xls')
    if os.path.exists(VAPOR_PRESSURE_TESTSET):
        details['reference_files_available'].append('VaporPressure_TestSet.xls')

    if molecular_weight is None:
        return {
            'in_ad': None,  # Cannot determine
            'status': 'Cannot assess AD - molecular weight not available',
            'warnings': ['Molecular weight required for AD assessment'],
            'details': details
        }

    # General warnings about MPBPWIN limitations
    warnings.append(
        "MPBPWIN: Complete training set data not available. "
        "Precise applicability domain cannot be defined."
    )

    warnings.append(
        "Original Joback method trained on only 388 compounds. "
        "Significant prediction errors are possible."
    )

    # Molecular weight-based assessment (very conservative)
    # Since we don't have precise MW ranges from training set,
    # we use general organic compound ranges
    details['mw_assessment'] = None

    if molecular_weight < 16.0:  # Below methane
        warnings.append(f"MW ({molecular_weight:.2f}) is extremely low - below typical organic compounds")
        status = "CAUTION: MW outside typical range"

    elif molecular_weight > 1000.0:  # Very large molecules
        warnings.append(f"MW ({molecular_weight:.2f}) is very high - may be outside training range")
        status = "CAUTION: MW may be outside training range"

    else:
        details['mw_assessment'] = f"MW ({molecular_weight:.2f}) within typical organic compound range"

    # Property-specific warnings
    if property_type == 'MP' and estimated_value is not None:
        if estimated_value < -200 or estimated_value > 400:
            warnings.append(
                f"Melting point estimate ({estimated_value:.2f}°C) is extreme - "
                f"verify result carefully"
            )

    elif property_type == 'BP' and estimated_value is not None:
        if estimated_value < -50 or estimated_value > 600:
            warnings.append(
                f"Boiling point estimate ({estimated_value:.2f}°C) is extreme - "
                f"verify result carefully"
            )

    elif property_type == 'VP' and estimated_value is not None:
        if estimated_value is not None:
            warnings.append(
                f"Vapor pressure estimates can have significant uncertainty - "
                f"verify against experimental data if available"
            )

    # Fragment-based warnings
    if fragments:
        details['num_fragments'] = len(fragments)
        if len(fragments) > 20:
            warnings.append(
                f"Compound has {len(fragments)} fragments - complex structures "
                f"may have lower prediction accuracy"
            )

    # Conservative assessment: we cannot definitively say "out of AD"
    # without training set data, but we provide extensive warnings
    in_ad = None  # Unknown/Cannot determine precisely

    # Add recommendation
    warnings.append(
        "Recommendation: Compare estimate against experimental data if available. "
        f"Test set files available in {REFERENCE_DATA_DIR}"
    )

    return {
        'in_ad': in_ad,
        'status': status,
        'warnings': warnings,
        'details': details
    }


def get_module_info():
    """
    Get information about the MPBPWIN module and its applicability domain.

    Returns:
        dict: Module information including available training/test set data
    """
    return {
        'module_name': 'MPBPWIN',
        'description': 'Melting Point, Boiling Point, and Vapor Pressure estimation',
        'methods': {
            'melting_point': ['Adapted Joback Method', 'Gold and Ogle Method'],
            'boiling_point': ['Adapted Stein and Brown Method'],
            'vapor_pressure': ['Antoine Method', 'Modified Grain Method', 'Mackay Method']
        },
        'joback_method': JOBACK_METHOD,
        'gold_ogle_method': GOLD_OGLE_METHOD,
        'reference_data': {
            'melting_pt_testset': MELTING_PT_TESTSET,
            'boiling_pt_testset': BOILING_PT_TESTSET,
            'vapor_pressure_testset': VAPOR_PRESSURE_TESTSET
        },
        'primary_ad_criteria': 'Not precisely defined - limited training data',
        'secondary_ad_criteria': ['Molecular Weight (general range)', 'Structural complexity'],
        'limitations': [
            'Complete training sets not available',
            'Original Joback method: only 388 compounds',
            'Fragment maximum counts not documented',
            'Significant prediction errors possible'
        ],
        'recommendations': [
            'Verify estimates against experimental data when available',
            'Use caution for complex structures',
            'Compare with test set compounds (>5800 compounds available)',
            'Consider prediction uncertainty in risk assessments'
        ]
    }


def load_reference_statistics():
    """
    Load statistics from reference test set files (if xlrd available).

    Note: This is optional and requires xlrd package for .xls files.
    Returns None if files cannot be read.

    Returns:
        dict or None: Statistics from test sets if available
    """
    try:
        import pandas as pd

        stats = {}

        # Try to load Melting Point test set
        if os.path.exists(MELTING_PT_TESTSET):
            try:
                df_mp = pd.read_excel(MELTING_PT_TESTSET)
                stats['melting_point'] = {
                    'num_compounds': len(df_mp),
                    'columns': list(df_mp.columns)
                }
                if 'MW' in df_mp.columns:
                    stats['melting_point']['mw_range'] = (
                        float(df_mp['MW'].min()),
                        float(df_mp['MW'].max())
                    )
                    stats['melting_point']['mw_mean'] = float(df_mp['MW'].mean())
            except Exception as e:
                stats['melting_point'] = {'error': str(e)}

        # Try to load Boiling Point test set
        if os.path.exists(BOILING_PT_TESTSET):
            try:
                df_bp = pd.read_excel(BOILING_PT_TESTSET)
                stats['boiling_point'] = {
                    'num_compounds': len(df_bp),
                    'columns': list(df_bp.columns)
                }
                if 'MW' in df_bp.columns:
                    stats['boiling_point']['mw_range'] = (
                        float(df_bp['MW'].min()),
                        float(df_bp['MW'].max())
                    )
                    stats['boiling_point']['mw_mean'] = float(df_bp['MW'].mean())
            except Exception as e:
                stats['boiling_point'] = {'error': str(e)}

        return stats if stats else None

    except ImportError:
        # pandas or xlrd not available
        return None
