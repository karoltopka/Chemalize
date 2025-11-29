"""
EPI Suite Integration
Provides tools for estimating physical/chemical properties using EPI Suite methodology
"""
from flask import Blueprint, render_template, request, session, flash, redirect, url_for, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import pandas as pd
from typing import Dict, List

from app.config import get_user_id
from app.nocache import nocache


episuite_bp = Blueprint('episuite', __name__)


def get_episuite_upload_path(filename=""):
    """Get path for EPI Suite uploads (user-specific)."""
    user_id = get_user_id()
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            "data", "users", user_id, "episuite_uploads")
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, filename) if filename else base_dir


def get_episuite_results_path(filename=""):
    """Get path for EPI Suite results (user-specific)."""
    user_id = get_user_id()
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            "data", "users", user_id, "episuite_results")
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, filename) if filename else base_dir


@episuite_bp.route("/episuite", methods=["GET"])
@nocache
def episuite_main():
    """Main EPI Suite page - standalone tool."""
    # Get results from session if available
    results = session.get('episuite_results', None)

    return render_template("episuite_main.html",
                          title="EPI Suite",
                          active="episuite",
                          results=results)


@episuite_bp.route("/episuite/upload", methods=["POST"])
@nocache
def upload_epi_file():
    """Upload EPI Suite output file for analysis."""
    if 'epi_file' not in request.files:
        flash("No file uploaded!", "danger")
        return redirect(url_for("episuite.episuite_main"))

    file = request.files['epi_file']

    if file.filename == '':
        flash("No file selected!", "danger")
        return redirect(url_for("episuite.episuite_main"))

    # Check file extension
    allowed_extensions = {'.txt', '.out'}
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        flash(f"Invalid file type! Please upload .txt or .out files.", "danger")
        return redirect(url_for("episuite.episuite_main"))

    # Save file
    filename = secure_filename(file.filename)
    upload_path = get_episuite_upload_path(filename)
    file.save(upload_path)

    # Parse the file
    try:
        results = parse_epi_suite_file(upload_path)

        # Store results in session
        session['episuite_results'] = results
        session['episuite_filename'] = filename

        flash(f"File '{filename}' uploaded and analyzed successfully!", "success")

    except Exception as e:
        flash(f"Error processing file: {str(e)}", "danger")
        return redirect(url_for("episuite.episuite_main"))

    return redirect(url_for("episuite.episuite_main"))

IDENTITY_COLUMNS = [
    'Filename',
    'Record_Index',
    'Chemical_ID',
    'SMILES',
    'Molecular_Formula',
    'Molecular_Weight',
]

BASIC_DATA_COLUMNS = IDENTITY_COLUMNS + [
    'KOWWIN_Log_Kow',
    'KOWWIN_Log_Kow_Assessment',
    'BIOWIN_Biowin3_Rating',
    'BIOWIN_Biowin3_Prediction',
    'BIOWIN_Biowin34_AD_Status',
    'BIOWIN_Biowin5_Probability',
    'BIOWIN_Biowin5_Prediction',
    'BIOWIN_Biowin6_Probability',
    'BIOWIN_Biowin6_Prediction',
    'BIOWIN_Biowin56_AD_Status',
    'BCFBAF_Log_BCF_Regression',
    'BCFBAF_BCF_Regression_AD_Status',
    'KOCWIN_LogKoc_MCI_Corrected',
    'KOCWIN_AD_Status',
]


def _build_results_dataframe(results: Dict) -> pd.DataFrame:
    """Convert parsed results to a flat DataFrame."""
    rows: List[Dict] = []
    for record in results.get('records', []):
        row_data = {
            'Filename': results.get('filename', ''),
            'Record_Index': record.get('index'),
            'Chemical_ID': record.get('chem_id', ''),
            'SMILES': record.get('smiles', ''),
            'Molecular_Formula': record.get('mol_formula', ''),
            'Molecular_Weight': record.get('mol_weight', ''),
        }

        for node_name, node_data in record.get('nodes', {}).items():
            for prop_name, prop_value in node_data.items():
                column_name = f"{node_name}_{prop_name}"
                row_data[column_name] = prop_value

        rows.append(row_data)

    return pd.DataFrame(rows)


@episuite_bp.route("/episuite/download", methods=["GET"])
@nocache
def download_results():
    """
    Download EPI Suite analysis results.

    Query parameters:
        mode: 'basic' (subset of key outputs) or 'advanced' (full dataset)
        format: 'csv' or 'xlsx'
    """
    if 'episuite_results' not in session:
        flash("No results available to download!", "danger")
        return redirect(url_for("episuite.episuite_main"))

    mode = request.args.get('mode', 'advanced').lower()
    file_format = request.args.get('format', 'csv').lower()

    if mode not in {'basic', 'advanced'}:
        flash("Invalid download mode requested.", "danger")
        return redirect(url_for("episuite.episuite_main"))
    if file_format not in {'csv', 'xlsx'}:
        flash("Invalid download format requested.", "danger")
        return redirect(url_for("episuite.episuite_main"))

    results = session['episuite_results']
    df = _build_results_dataframe(results)

    if df.empty:
        flash("No parsed results available to download!", "danger")
        return redirect(url_for("episuite.episuite_main"))

    if mode == 'basic':
        df = df.reindex(columns=BASIC_DATA_COLUMNS).fillna('N/A')

    filename_base = os.path.splitext(session.get('episuite_filename', 'episuite_output'))[0]
    download_basename = f"{filename_base}_{mode}"

    if file_format == 'csv':
        output_filename = f"{download_basename}.csv"
        output_path = get_episuite_results_path(output_filename)
        df.to_csv(output_path, index=False)
        mimetype = 'text/csv'
    else:
        output_filename = f"{download_basename}.xlsx"
        output_path = get_episuite_results_path(output_filename)
        df.to_excel(output_path, index=False)
        mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

    return send_file(
        output_path,
        as_attachment=True,
        download_name=output_filename,
        mimetype=mimetype,
    )


def parse_epi_suite_file(filepath):
    """
    Parse EPI Suite output file and extract properties with AD assessment.
    Returns data in spreadsheet format with node results and assessments.
    Currently supports: KOWWIN, BIOWIN (models 1-6), BCFBAF
    """
    from app.chemalize.episuite import (
        parse_kowwin,
        check_kowwin_ad,
        parse_biowin,
        check_biowin_ad,
        parse_bcfbaf,
        check_bcfbaf_ad,
        parse_kocwin,
        check_kocwin_ad,
    )

    # Read file content
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        file_content = f.read()

    kowwin_entries = parse_kowwin(file_content) or []
    biowin_entries = parse_biowin(file_content) or []
    bcfbaf_entries = parse_bcfbaf(file_content) or []
    kocwin_entries = parse_kocwin(file_content) or []

    compound_count = max(
        len(kowwin_entries),
        len(biowin_entries),
        len(bcfbaf_entries),
        len(kocwin_entries),
    )

    results = {
        'filename': os.path.basename(filepath),
        'records': []
    }

    def _fmt(value):
        if isinstance(value, (int, float)):
            return f"{value:.4f}"
        return 'N/A'

    def _format_corrections(corrections):
        if not corrections:
            return 'None'
        formatted = []
        for item in corrections:
            if isinstance(item, dict):
                desc = item.get('descriptor', 'Unknown')
                count = item.get('count')
                formatted.append(f"{desc} (count {count})" if count is not None else desc)
            else:
                formatted.append(str(item))
        return '; '.join(formatted)

    def _format_corrections(corrections):
        if not corrections:
            return 'None'
        parts = []
        for corr in corrections:
            if isinstance(corr, dict):
                desc = corr.get('descriptor', 'Unknown')
                count = corr.get('count')
                if count is not None:
                    parts.append(f"{desc} (count {count})")
                else:
                    parts.append(desc)
            else:
                parts.append(str(corr))
        return '; '.join(parts)

    def _miti_prediction(probability, fallback='Not reported'):
        if probability is None:
            return fallback
        try:
            value = float(probability)
        except (TypeError, ValueError):
            return fallback
        return "Readily Degradable" if value >= 0.5 else "NOT Readily Degradable"

    for idx in range(compound_count):
        record = {
            'index': idx + 1,
            'chem_id': None,
            'smiles': None,
            'mol_formula': None,
            'mol_weight': None,
            'nodes': {}
        }

        kowwin_data = kowwin_entries[idx] if idx < len(kowwin_entries) else None
        biowin_data = biowin_entries[idx] if idx < len(biowin_entries) else None
        bcfbaf_data = bcfbaf_entries[idx] if idx < len(bcfbaf_entries) else None
        kocwin_data = kocwin_entries[idx] if idx < len(kocwin_entries) else None

        metal_note = None
        for source in (bcfbaf_data, kocwin_data, biowin_data, kowwin_data):
            if source and source.get('metal_warning'):
                metal_note = source['metal_warning']
                break

        for entry in (kowwin_data, biowin_data, bcfbaf_data, kocwin_data):
            if entry:
                if record['smiles'] is None and entry.get('smiles'):
                    record['smiles'] = entry.get('smiles')
                if record['mol_formula'] is None and entry.get('mol_formula'):
                    record['mol_formula'] = entry.get('mol_formula')
                if record['mol_weight'] is None and entry.get('mol_weight'):
                    record['mol_weight'] = entry.get('mol_weight')
                if record['chem_id'] is None and entry.get('chem_id'):
                    record['chem_id'] = entry.get('chem_id')

        if kowwin_data and kowwin_data.get('log_kow') is not None:
            ad_check = check_kowwin_ad(kowwin_data)
            assessment = "IN AD" if ad_check['in_ad'] else "OUT OF AD"
            if ad_check['warnings']:
                assessment += f" ({len(ad_check['warnings'])} warnings)"
            record['nodes']['KOWWIN'] = {
                'Log_Kow': _fmt(kowwin_data['log_kow']),
                'Log_Kow_Assessment': assessment,
                'Assessment_Details': ad_check['status'],
                'Warnings': '; '.join(ad_check['warnings']) if ad_check['warnings'] else 'None'
            }

        if biowin_data and biowin_data.get('models'):
            biowin_ad = check_biowin_ad(
                biowin_data,
                biowin_data.get('metal_warning') or metal_note,
            )
            overall_ad = biowin_ad['overall']
            biowin_assessment = "IN AD" if overall_ad['in_ad'] else "OUT OF AD"
            if overall_ad['warnings']:
                biowin_assessment += f" ({len(overall_ad['warnings'])} warnings)"

            biowin1 = biowin_data['models'].get('Biowin1', {})
            biowin2 = biowin_data['models'].get('Biowin2', {})
            biowin3 = biowin_data['models'].get('Biowin3', {})
            biowin4 = biowin_data['models'].get('Biowin4', {})
            biowin5 = biowin_data['models'].get('Biowin5', {})
            biowin6 = biowin_data['models'].get('Biowin6', {})

            record['nodes']['BIOWIN'] = {
                'Biowin1_Probability': _fmt(biowin1.get('probability')),
                'Biowin1_Prediction': biowin1.get('classification', 'Not reported'),
                'Biowin2_Probability': _fmt(biowin2.get('probability')),
                'Biowin2_Prediction': biowin2.get('classification', 'Not reported'),
                'Biowin2_Logit': _fmt(biowin2.get('computed_total')),
                'Biowin3_Rating': _fmt(biowin3.get('rating')),
                'Biowin3_Prediction': biowin3.get('classification', 'Not reported'),
                'Biowin3_Time_Category': biowin3.get('time_category', 'Not reported'),
                'Biowin4_Rating': _fmt(biowin4.get('rating')),
                'Biowin4_Prediction': biowin4.get('classification', 'Not reported'),
                'Biowin4_Time_Category': biowin4.get('time_category', 'Not reported'),
                'Biowin5_Probability': _fmt(biowin5.get('probability')),
                'Biowin5_Prediction': _miti_prediction(biowin5.get('probability'), biowin5.get('classification', 'Not reported')),
                'Biowin6_Probability': _fmt(biowin6.get('probability')),
                'Biowin6_Prediction': _miti_prediction(biowin6.get('probability'), biowin6.get('classification', 'Not reported')),
                'Biowin6_Logit': _fmt(biowin6.get('computed_total')),
                'AD_Status': biowin_assessment,
                'Assessment_Details': overall_ad['status'],
                'Warnings': '; '.join(overall_ad['warnings']) if overall_ad['warnings'] else 'None',
                'Fragment_Summary': ', '.join(
                    f"{frag['description']}: {frag['count']}"
                    for frag in biowin_data.get('fragments', [])
                ) or 'Not reported',
                'Metal_Note': biowin_data.get('metal_warning') or metal_note or 'None',
            }

            group_prefix_map = {
                'Biowin1_2': 'Biowin12',
                'Biowin3_4': 'Biowin34',
                'Biowin5_6': 'Biowin56',
            }
            for group_key, group_result in biowin_ad['groups'].items():
                prefix = group_prefix_map.get(group_key, group_key.replace('_', ''))
                record['nodes']['BIOWIN'][f'{prefix}_AD_Status'] = group_result['status']
                record['nodes']['BIOWIN'][f'{prefix}_Warnings'] = (
                    '; '.join(group_result['warnings']) if group_result['warnings'] else 'None'
                )
                record['nodes']['BIOWIN'][f'{prefix}_Evaluated'] = (
                    'Yes' if group_result['evaluated'] else 'No'
                )

        if bcfbaf_data:
            bcf_ad = check_bcfbaf_ad(
                bcfbaf_data.get('mol_weight'),
                bcfbaf_data.get('log_kow_used'),
                bcfbaf_data.get('fragments'),
                bcfbaf_data.get('bcf_corrections'),
                bcfbaf_data.get('metal_warning') or metal_note,
            )

            overall = bcf_ad['overall']
            bcf_group = bcf_ad['bcf']
            baf_group = bcf_ad['baf']

            overall_assessment = "IN AD" if overall['in_ad'] else "OUT OF AD"
            if overall['warnings']:
                overall_assessment += f" ({len(overall['warnings'])} warnings)"

            bcf_assessment = "IN AD" if bcf_group['in_ad'] else "OUT OF AD"
            if bcf_group['warnings']:
                bcf_assessment += f" ({len(bcf_group['warnings'])} warnings)"

            baf_assessment = "IN AD" if baf_group['in_ad'] else "OUT OF AD"
            if baf_group['warnings']:
                baf_assessment += f" ({len(baf_group['warnings'])} warnings)"

            node_payload = {
                'Log_BCF_Regression': _fmt(bcfbaf_data.get('log_bcf_regression')),
                'BCF_Regression_L_per_kg': _fmt(bcfbaf_data.get('bcf_regression_value')),
                'Log_BAF_Upper_Trophic': _fmt(bcfbaf_data.get('log_baf_upper')),
                'BAF_Upper_Trophic_L_per_kg': _fmt(bcfbaf_data.get('baf_upper_value')),
                'Biotrans_Half_Life_Days': _fmt(bcfbaf_data.get('biotrans_half_life_days')),
                'LogKow_Used': _fmt(bcfbaf_data.get('log_kow_used')),
                'LogKow_Experimental': _fmt(bcfbaf_data.get('log_kow_experimental')),
                'BCF_Corrections': _format_corrections(bcfbaf_data.get('bcf_corrections')),
                'Fragment_Summary': ', '.join(
                    f"{frag['description']}: {frag['count']}"
                    for frag in bcfbaf_data.get('fragments', [])
                ) or 'Not reported',
                'Metal_Note': bcfbaf_data.get('metal_warning') or metal_note or 'None',
                'AD_Status': overall_assessment,
                'Assessment_Details': overall['status'],
                'Warnings': '; '.join(overall['warnings']) if overall['warnings'] else 'None',
                'BCF_Regression_AD_Status': bcf_assessment,
                'BCF_Regression_Details': bcf_group['status'],
                'BCF_Regression_Warnings': '; '.join(bcf_group['warnings']) if bcf_group['warnings'] else 'None',
                'BAF_AD_Status': baf_assessment,
                'BAF_Details': baf_group['status'],
                'BAF_Warnings': '; '.join(baf_group['warnings']) if baf_group['warnings'] else 'None',
            }

            table_results = bcfbaf_data.get('table_results', {})
            if 'LOG Bio Half-Life (days)' in table_results:
                node_payload['Log_Bio_Half_Life'] = _fmt(table_results['LOG Bio Half-Life (days)'])
            if 'Bio Half-Life (days)' in table_results:
                node_payload['Bio_Half_Life_Table_Days'] = _fmt(table_results['Bio Half-Life (days)'])

            for key, value in bcfbaf_data.get('arnot_gobas', {}).items():
                pretty_key = f"ArnotGobas_{key.replace('-', '_').replace(' ', '_').upper()}"
                node_payload[pretty_key] = _fmt(value)

            record['nodes']['BCFBAF'] = node_payload

        if kocwin_data:
            koc_ad = check_kocwin_ad(
                kocwin_data.get('mol_weight'),
                kocwin_data.get('mci_corrections'),
                kocwin_data.get('logkow_corrections'),
                kocwin_data.get('metal_warning') or metal_note,
            )

            koc_assessment = "IN AD" if koc_ad['in_ad'] else "OUT OF AD"
            if koc_ad['warnings']:
                koc_assessment += f" ({len(koc_ad['warnings'])} warnings)"

            record['nodes']['KOCWIN'] = {
                'MCI_Index': _fmt(kocwin_data.get('mci_index')),
                'LogKoc_MCI_NonCorrected': _fmt(kocwin_data.get('log_koc_mci_non_corrected')),
                'LogKoc_MCI_Corrected': _fmt(kocwin_data.get('log_koc_mci_over_correction') or kocwin_data.get('log_koc_mci_corrected')),
                'Koc_MCI_L_per_kg': _fmt(kocwin_data.get('koc_mci')),
                'LogKoc_LogKow_NonCorrected': _fmt(kocwin_data.get('log_koc_logkow_non_corrected')),
                'LogKoc_LogKow_Corrected': _fmt(kocwin_data.get('log_koc_logkow_corrected')),
                'Koc_LogKow_L_per_kg': _fmt(kocwin_data.get('koc_logkow')),
                'LogKow_Input': _fmt(kocwin_data.get('log_kow_used')),
                'MCI_Corrections': _format_corrections(kocwin_data.get('mci_corrections')),
                'LogKow_Corrections': _format_corrections(kocwin_data.get('logkow_corrections')),
                'AD_Status': koc_assessment,
                'Assessment_Details': koc_ad['status'],
                'Warnings': '; '.join(koc_ad['warnings']) if koc_ad['warnings'] else 'None',
                'Metal_Note': kocwin_data.get('metal_warning') or metal_note or 'None',
            }

        results['records'].append(record)

    return results
