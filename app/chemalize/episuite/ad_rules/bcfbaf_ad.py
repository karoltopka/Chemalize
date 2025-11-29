"""
BCFBAF Applicability Domain Rules
---------------------------------

Reference data:
 - Appendix C (BCF regression correction factors; 527-compound training set)
 - Appendix D (Biotransformation fragments; 421-compound training set)
"""
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import zipfile
from xml.etree import ElementTree as ET

# BAF (biotransformation) training ranges (Appendix D)
BAF_MW_MIN = 68.08
BAF_MW_MAX = 959.17
BAF_MW_AVG = 259.75
BAF_LOGKOW_MIN = 0.31
BAF_LOGKOW_MAX = 8.70

# BCF regression training ranges (Appendix C)
BCF_MW_MIN = 68.08
BCF_MW_MAX = 991.80  # ionic cases
BCF_MW_NONIONIC_MAX = 959.17
BCF_MW_AVG = 244.00
BCF_LOGKOW_MIN = -6.50
BCF_LOGKOW_NONIONIC_MIN = -1.37
BCF_LOGKOW_MAX = 11.26

IONIZATION_CAUTION_MW = 600.0  # limited number of high-mass/ionizing substances in calibration


def _normalize(text: str) -> str:
    return ' '.join(text.split())


def _load_shared_strings(zf: zipfile.ZipFile) -> Tuple[List[str], Dict[str, str]]:
    ns = {'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    shared_strings = ET.fromstring(zf.read('xl/sharedStrings.xml'))
    strings = [
        ''.join(node.text or '' for node in si.findall('.//a:t', ns))
        for si in shared_strings.findall('a:si', ns)
    ]
    return strings, ns


def _load_sheet_rows(zf: zipfile.ZipFile, sheet_name: str) -> List[Dict[str, str]]:
    strings, ns = _load_shared_strings(zf)
    sheet = ET.fromstring(zf.read(f'xl/worksheets/{sheet_name}.xml'))

    def _cell_value(cell):
        if cell.attrib.get('t') == 's':
            return strings[int(cell.find('a:v', ns).text)]
        value_node = cell.find('a:v', ns)
        return value_node.text if value_node is not None else None

    rows: List[Dict[str, str]] = []
    for row in sheet.findall('a:sheetData/a:row', ns):
        cells = {
            ''.join(filter(str.isalpha, cell.attrib.get('r', ''))): _cell_value(cell)
            for cell in row.findall('a:c', ns)
        }
        if cells:
            rows.append(cells)
    return rows


def _load_baf_fragment_counts() -> Dict[str, Optional[int]]:
    reference_path = (
        Path(__file__).resolve().parents[1]
        / 'reference_data'
        / 'Appendix_D.xlsx'
    )
    if not reference_path.exists():
        return {}

    try:
        with zipfile.ZipFile(reference_path) as zf:
            rows = _load_sheet_rows(zf, 'sheet1')
    except (zipfile.BadZipFile, KeyError, ET.ParseError):
        return {}

    fragment_counts: Dict[str, Optional[int]] = {}
    for cells in rows:
        descriptor = cells.get('A')
        if not descriptor or descriptor.lower().startswith('fragment description'):
            continue
        max_count = cells.get('D')
        try:
            max_value = int(float(max_count))
        except (TypeError, ValueError):
            max_value = None
        fragment_counts[_normalize(descriptor)] = max_value
    return fragment_counts


def _load_bcf_correction_counts() -> Dict[str, Optional[int]]:
    reference_path = (
        Path(__file__).resolve().parents[1]
        / 'reference_data'
        / 'Appendix_C_BCF.xlsx'
    )
    if not reference_path.exists():
        return {}

    try:
        with zipfile.ZipFile(reference_path) as zf:
            rows = _load_sheet_rows(zf, 'sheet1')
    except (zipfile.BadZipFile, KeyError, ET.ParseError):
        return {}

    correction_counts: Dict[str, Optional[int]] = {}
    for cells in rows:
        descriptor = cells.get('A')
        if not descriptor or descriptor.lower().startswith('correction factor'):
            continue
        max_count = cells.get('E')
        try:
            max_value = int(float(max_count))
        except (TypeError, ValueError):
            max_value = None
        correction_counts[_normalize(descriptor)] = max_value
    return correction_counts


FRAGMENT_MAX_COUNTS = _load_baf_fragment_counts()
BCF_CORRECTION_MAX_COUNTS = _load_bcf_correction_counts()


def _evaluate_bcf(molecular_weight: Optional[float],
                  log_kow: Optional[float],
                  corrections: Optional[List[str]],
                  metal_warning: Optional[str]) -> Dict:
    warnings: List[str] = []
    details = {
        'mw_in_training_range': False,
        'mw_in_nonionic_range': False,
        'log_kow_in_training_range': False,
        'log_kow_in_nonionic_range': False,
        'correction_counts': Counter(),
        'correction_exceedances': [],
        'correction_unknown': [],
        'metal_removed': metal_warning,
        'mw_avg': BCF_MW_AVG,
    }
    in_ad = True

    if molecular_weight is None:
        return {
            'in_ad': False,
            'status': 'Cannot determine BCF AD - molecular weight not available.',
            'warnings': ['BCF: molecular weight missing.'],
            'details': details,
        }

    if BCF_MW_MIN <= molecular_weight <= BCF_MW_MAX:
        details['mw_in_training_range'] = True
        if molecular_weight <= BCF_MW_NONIONIC_MAX:
            details['mw_in_nonionic_range'] = True
        else:
            warnings.append(
                f'BCF: Molecular weight {molecular_weight:.2f} exceeds non-ionic training maximum ({BCF_MW_NONIONIC_MAX}).'
            )
    else:
        in_ad = False
        if molecular_weight < BCF_MW_MIN:
            warnings.append(
                f'BCF: Molecular weight {molecular_weight:.2f} below training minimum ({BCF_MW_MIN}).'
            )
        else:
            warnings.append(
                f'BCF: Molecular weight {molecular_weight:.2f} exceeds training maximum ({BCF_MW_MAX}).'
            )

    if log_kow is None:
        warnings.append('BCF: Log Kow not provided; cannot confirm domain.')
        in_ad = False
    else:
        if BCF_LOGKOW_MIN <= log_kow <= BCF_LOGKOW_MAX:
            details['log_kow_in_training_range'] = True
            if log_kow >= BCF_LOGKOW_NONIONIC_MIN:
                details['log_kow_in_nonionic_range'] = True
            else:
                warnings.append(
                    f'BCF: Log Kow {log_kow:.2f} within ionic training domain only (non-ionic minimum {BCF_LOGKOW_NONIONIC_MIN}).'
                )
        else:
            in_ad = False
            if log_kow < BCF_LOGKOW_MIN:
                warnings.append(
                    f'BCF: Log Kow {log_kow:.2f} below training minimum ({BCF_LOGKOW_MIN}).'
                )
            else:
                warnings.append(
                    f'BCF: Log Kow {log_kow:.2f} exceeds training maximum ({BCF_LOGKOW_MAX}).'
                )

    if corrections:
        descriptor_counts = Counter(_normalize(entry.split(':')[0].strip()) for entry in corrections)
        details['correction_counts'] = descriptor_counts
        for descriptor, count in descriptor_counts.items():
            max_allowed = BCF_CORRECTION_MAX_COUNTS.get(descriptor)
            if max_allowed is None:
                details['correction_unknown'].append(descriptor)
                warnings.append(
                    f'BCF: Correction "{descriptor}" not documented in Appendix C; interpret with caution.'
                )
            elif count > max_allowed:
                details['correction_exceedances'].append(
                    f'{descriptor} (count {count} > max {max_allowed})'
                )
                warnings.append(
                    f'BCF: Correction "{descriptor}" exceeds documented maximum ({max_allowed}).'
                )
                in_ad = False

    if molecular_weight > IONIZATION_CAUTION_MW:
        warnings.append(
            f'BCF: Limited calibration above ~{IONIZATION_CAUTION_MW} g/mol; predictions may be less certain.'
        )

    warnings.append(
        'BCF: Median confidence factor ≈5.5 (≈1.5 orders of magnitude); MAE ≈7 (~1.7 orders).'
    )
    warnings.append(
        'BCF: Training set included few strongly ionizing species, metals/organometals, pigments/dyes, or perfluorinated substances.'
    )

    if metal_warning:
        warnings.append(metal_warning + ' Consider neutral (non-salt) form when interpreting BCF estimates.')

    if in_ad:
        status = 'IN AD: BCF regression within documented domain.'
        if warnings:
            status = 'IN AD (with caution): BCF regression – review warnings.'
    else:
        reasons = []
        for warning in warnings:
            if 'Molecular weight' in warning:
                reasons.append('molecular weight outside range')
            elif 'Log Kow' in warning:
                reasons.append('log Kow outside range')
            elif 'Correction' in warning and 'exceeds' in warning:
                reasons.append('correction exceeds maxima')
        if not reasons:
            reasons.append('see warnings')
        status = 'OUT OF AD: BCF regression – ' + '; '.join(dict.fromkeys(reasons))

    return {
        'in_ad': in_ad,
        'status': status,
        'warnings': warnings,
        'details': details,
    }


def _evaluate_baf(molecular_weight: Optional[float],
                  log_kow: Optional[float],
                  fragments: Optional[List[Dict]],
                  metal_warning: Optional[str]) -> Dict:
    warnings: List[str] = []
    details = {
        'mw_in_training_range': False,
        'log_kow_in_training_range': False,
        'fragment_exceedances': [],
        'unknown_fragments': [],
        'checked_fragments': 0,
        'metal_removed': metal_warning,
    }
    in_ad = True

    if molecular_weight is None:
        return {
            'in_ad': False,
            'status': 'Cannot determine BAF AD - molecular weight not available.',
            'warnings': ['BAF: molecular weight missing.'],
            'details': details,
        }

    if BAF_MW_MIN <= molecular_weight <= BAF_MW_MAX:
        details['mw_in_training_range'] = True
    else:
        in_ad = False
        if molecular_weight < BAF_MW_MIN:
            warnings.append(
                f'BAF: Molecular weight {molecular_weight:.2f} below training minimum ({BAF_MW_MIN}).'
            )
        else:
            warnings.append(
                f'BAF: Molecular weight {molecular_weight:.2f} exceeds training maximum ({BAF_MW_MAX}).'
            )

    if log_kow is None:
        warnings.append('BAF: Log Kow not provided; cannot confirm domain.')
        in_ad = False
    else:
        if BAF_LOGKOW_MIN <= log_kow <= BAF_LOGKOW_MAX:
            details['log_kow_in_training_range'] = True
        else:
            in_ad = False
            if log_kow < BAF_LOGKOW_MIN:
                warnings.append(
                    f'BAF: Log Kow {log_kow:.2f} below training minimum ({BAF_LOGKOW_MIN}).'
                )
            else:
                warnings.append(
                    f'BAF: Log Kow {log_kow:.2f} exceeds training maximum ({BAF_LOGKOW_MAX}).'
                )

    if fragments:
        details['checked_fragments'] = len(fragments)
        if not FRAGMENT_MAX_COUNTS:
            warnings.append('BAF: Fragment maxima unavailable; fragment checks skipped.')
        else:
            for fragment in fragments:
                description = fragment.get('normalized_description') or _normalize(
                    fragment.get('description', '')
                )
                count = fragment.get('count', 0)
                max_allowed = FRAGMENT_MAX_COUNTS.get(description)
                if max_allowed is None:
                    details['unknown_fragments'].append(f"{fragment.get('description', description)} (count {count})")
                    continue
                if count > max_allowed:
                    in_ad = False
                    info = f"{fragment.get('description', description)} (count {count} > max {max_allowed})"
                    details['fragment_exceedances'].append(info)

        if details['fragment_exceedances']:
            warnings.append(
                'BAF: Fragment counts exceed documented maxima: ' +
                '; '.join(details['fragment_exceedances'])
            )
        if details['unknown_fragments']:
            warnings.append(
                'BAF: Fragments not documented in Appendix D – interpret with caution: ' +
                '; '.join(details['unknown_fragments'])
            )

    if metal_warning:
        warnings.append(metal_warning + ' Consider neutral (non-salt) form when interpreting BAF estimates.')

    if in_ad:
        status = 'IN AD: BAF biotransformation within documented domain.'
        if warnings:
            status = 'IN AD (with caution): BAF biotransformation – review warnings.'
    else:
        reasons = []
        for warning in warnings:
            if 'Molecular weight' in warning:
                reasons.append('molecular weight outside range')
            elif 'Log Kow' in warning:
                reasons.append('log Kow outside range')
            elif 'Fragment counts exceed' in warning:
                reasons.append('fragment counts exceed maxima')
        if not reasons:
            reasons.append('see warnings')
        status = 'OUT OF AD: BAF biotransformation – ' + '; '.join(dict.fromkeys(reasons))

    return {
        'in_ad': in_ad,
        'status': status,
        'warnings': warnings,
        'details': details,
    }


def check_applicability_domain(molecular_weight: Optional[float],
                               log_kow: Optional[float],
                               fragments: Optional[List[Dict]],
                               bcf_corrections: Optional[List[str]] = None,
                               metal_warning: Optional[str] = None) -> Dict:
    """
    Evaluate applicability domains separately for BCF (regression) and BAF (Arnot-Gobas).
    """
    bcf_result = _evaluate_bcf(molecular_weight, log_kow, bcf_corrections, metal_warning)
    baf_result = _evaluate_baf(molecular_weight, log_kow, fragments, metal_warning)

    overall_in_ad = bcf_result['in_ad'] and baf_result['in_ad']
    overall_warnings: List[str] = []

    if bcf_result['warnings']:
        overall_warnings.append('BCF: ' + '; '.join(bcf_result['warnings']))
    if baf_result['warnings']:
        overall_warnings.append('BAF: ' + '; '.join(baf_result['warnings']))

    if overall_in_ad:
        overall_status = 'IN AD: BCF and BAF estimates within documented domains.'
        if overall_warnings:
            overall_status = 'IN AD (with caution): review BCF/BAF warnings.'
    else:
        out_groups = []
        if not bcf_result['in_ad']:
            out_groups.append('BCF')
        if not baf_result['in_ad']:
            out_groups.append('BAF')
        overall_status = 'OUT OF AD: ' + ', '.join(out_groups)

    return {
        'overall': {
            'in_ad': overall_in_ad,
            'status': overall_status,
            'warnings': overall_warnings,
            'details': {
                'bcf_in_ad': bcf_result['in_ad'],
                'baf_in_ad': baf_result['in_ad'],
            },
        },
        'bcf': bcf_result,
        'baf': baf_result,
    }


def get_module_info() -> Dict:
    """Provide descriptive metadata for the BCFBAF module."""
    return {
        'module_name': 'BCFBAF',
        'description': 'Bioconcentration (regression) and bioaccumulation (Arnot-Gobas) estimations.',
        'training_sets': {
            'BCF_regression': {
                'size': 527,
                'molecular_weight_range': {'min': BCF_MW_MIN, 'max': BCF_MW_MAX, 'avg': BCF_MW_AVG},
                'log_kow_range': {'min': BCF_LOGKOW_MIN, 'max': BCF_LOGKOW_MAX},
                'correction_factor_source': 'Appendix C',
            },
            'BAF_biotransformation': {
                'size': 421,
                'molecular_weight_range': {'min': BAF_MW_MIN, 'max': BAF_MW_MAX, 'avg': BAF_MW_AVG},
                'log_kow_range': {'min': BAF_LOGKOW_MIN, 'max': BAF_LOGKOW_MAX},
                'fragment_source': 'Appendix D',
            },
        },
        'notes': (
            'Median confidence factor ~5.5 (test set ~7); predictions outside documented MW/logKow ranges, with '
            'unseen fragments/corrections, or for strongly ionizing/metallic/perfluorinated substances should be '
            'interpreted cautiously.'
        ),
    }
