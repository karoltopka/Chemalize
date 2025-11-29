"""
KOWWIN Applicability Domain Rules

Reference:
- Training Set: 2447 compounds (fragment maxima sourced from Appendix A)
- Validation Set: 10946 compounds
- Source: EPI Suite Documentation, Section 2.2.3
"""

import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET


# Molecular Weight Ranges
TRAINING_SET = {
    'mw_min': 18.02,
    'mw_max': 719.92,
    'mw_avg': 199.98,
    'num_compounds': 2447
}

VALIDATION_SET = {
    'mw_min': 27.03,
    'mw_max': 991.15,
    'mw_avg': 258.98,
    'num_compounds': 10946
}


def _normalize_fragment(description: str) -> str:
    return ' '.join(description.split())


def _load_fragment_max_counts() -> Dict[str, Tuple[Optional[int], Optional[int]]]:
    """
    Load fragment maxima for KOWWIN from Appendix A (training and validation).
    Returns a mapping of normalized fragment description -> (training_max, validation_max).
    """
    reference_path = (
        Path(__file__).resolve().parents[1]
        / 'reference_data'
        / 'Appendix_A_KOWWIN.xlsx'
    )
    if not reference_path.exists():
        return {}

    try:
        with zipfile.ZipFile(reference_path) as zf:
            shared_strings = ET.fromstring(zf.read('xl/sharedStrings.xml'))
            sheet = ET.fromstring(zf.read('xl/worksheets/sheet1.xml'))
    except (KeyError, zipfile.BadZipFile, ET.ParseError):
        return {}

    ns = {'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    strings = [
        ''.join(t.text or '' for t in si.findall('.//a:t', ns))
        for si in shared_strings.findall('a:si', ns)
    ]

    fragment_counts: Dict[str, Tuple[Optional[int], Optional[int]]] = {}

    for row in sheet.findall('a:sheetData/a:row', ns):
        cells = {}
        for cell in row.findall('a:c', ns):
            col = ''.join(filter(str.isalpha, cell.attrib.get('r', '')))
            if cell.attrib.get('t') == 's':
                value = strings[int(cell.find('a:v', ns).text)]
            else:
                v = cell.find('a:v', ns)
                value = v.text if v is not None else None
            cells[col] = value

        descriptor = cells.get('A')
        if not descriptor or descriptor.lower().startswith('fragment descriptor'):
            continue

        training_max = cells.get('C')
        validation_max = cells.get('E')

        def _to_int(raw: Optional[str]) -> Optional[int]:
            if raw is None or raw.strip() == '':
                return None
            try:
                return int(float(raw))
            except ValueError:
                return None

        normalized = _normalize_fragment(descriptor)
        fragment_counts[normalized] = (_to_int(training_max), _to_int(validation_max))

    return fragment_counts


FRAGMENT_MAX_COUNTS = _load_fragment_max_counts()


def check_applicability_domain(molecular_weight, fragments=None):
    """
    Check if a compound is within the KOWWIN applicability domain.

    The applicability domain is primarily defined by molecular weight range.
    Users should consider that log P estimates may be less accurate for:
    1. Compounds outside the MW range of the training set
    2. Compounds with more instances of a fragment than the training set maximum
    3. Compounds with functional groups not represented in the training set

    Args:
        molecular_weight (float): Molecular weight of the compound
        fragments (list, optional): List of fragment dictionaries with 'type', 'count', 'description'

    Returns:
        dict: {
            'in_ad': bool - True if within applicability domain
            'status': str - Detailed status message
            'warnings': list - List of warning messages
            'details': dict - Additional details about the assessment
        }
    """
    warnings: List[str] = []
    in_ad = True
    status = "Within applicability domain"
    details = {
        'mw_in_training_range': False,
        'mw_in_validation_range': False,
        'distance_from_training_avg': None,
        'checked_fragments': 0,
        'fragment_exceedances': [],
        'unknown_fragments': [],
        'fragment_max_counts_loaded': bool(FRAGMENT_MAX_COUNTS),
    }

    if molecular_weight is None:
        return {
            'in_ad': False,
            'status': 'Cannot determine AD - molecular weight not found',
            'warnings': ['Molecular weight not available'],
            'details': details
        }

    # Calculate distance from training set average
    training_avg = TRAINING_SET['mw_avg']
    details['distance_from_training_avg'] = abs(molecular_weight - training_avg)

    # Check if MW is within training set range
    if TRAINING_SET['mw_min'] <= molecular_weight <= TRAINING_SET['mw_max']:
        details['mw_in_training_range'] = True
        details['mw_in_validation_range'] = True
        status = f"IN AD: MW ({molecular_weight:.2f}) within training set range"

    # Check if MW is within validation set but outside training set
    elif TRAINING_SET['mw_max'] < molecular_weight <= VALIDATION_SET['mw_max']:
        details['mw_in_validation_range'] = True
        warnings.append(
            f"MW ({molecular_weight:.2f}) is outside training set range "
            f"({TRAINING_SET['mw_min']}-{TRAINING_SET['mw_max']}) "
            f"but within validation set range ({VALIDATION_SET['mw_min']}-{VALIDATION_SET['mw_max']}). "
            f"Predictions may be less accurate."
        )
        status = f"IN AD (with caution): MW within validation set range only"

    # Below training set minimum
    elif molecular_weight < TRAINING_SET['mw_min']:
        in_ad = False
        status = f"OUT OF AD: MW ({molecular_weight:.2f}) below training set minimum ({TRAINING_SET['mw_min']})"
        warnings.append(f"Molecular weight {molecular_weight:.2f} is below training set range")

    # Above validation set maximum
    elif molecular_weight > VALIDATION_SET['mw_max']:
        in_ad = False
        status = f"OUT OF AD: MW ({molecular_weight:.2f}) above validation set maximum ({VALIDATION_SET['mw_max']})"
        warnings.append(f"Molecular weight {molecular_weight:.2f} exceeds validation set range")

    # Check if MW is significantly different from average (> 3x average)
    if details['distance_from_training_avg'] > (3 * training_avg):
        warnings.append(
            f"MW ({molecular_weight:.2f}) is significantly different from "
            f"training set average ({training_avg:.2f})"
        )

    # Future enhancement: Check fragment counts against training set maximums
    if fragments:
        details['checked_fragments'] = len(fragments)
        if not FRAGMENT_MAX_COUNTS:
            warnings.append('KOWWIN fragment maxima not available; fragment checks skipped.')
        else:
            for fragment in fragments:
                desc = fragment.get('description', '')
                norm_desc = _normalize_fragment(desc)
                count = fragment.get('count', 0)
                training_max, validation_max = FRAGMENT_MAX_COUNTS.get(norm_desc, (None, None))

                if training_max is None:
                    details['unknown_fragments'].append(f"{desc} (count {count})")
                    continue

                if count > training_max:
                    details['fragment_exceedances'].append(
                        f"{desc} (count {count} > training max {training_max})"
                    )
                    in_ad = False
                elif validation_max is not None and count > validation_max:
                    details['fragment_exceedances'].append(
                        f"{desc} (count {count} exceeds validation max {validation_max})"
                    )
                    in_ad = False

            if details['fragment_exceedances']:
                warnings.append(
                    'Fragment counts exceed documented maxima: ' +
                    '; '.join(details['fragment_exceedances'])
                )
            if details['unknown_fragments']:
                warnings.append(
                    'Fragments not documented in Appendix A: ' +
                    '; '.join(details['unknown_fragments'])
                )

    if in_ad:
        if warnings and not status.lower().startswith('in ad'):
            status = 'IN AD (with caution): ' + '; '.join(warnings)
        elif warnings and 'with caution' not in status:
            status = 'IN AD (with caution): see warnings.'
    else:
        reasons = []
        for warning in warnings:
            if 'Molecular weight' in warning:
                reasons.append('molecular weight outside documented range')
            elif 'Fragment counts exceed' in warning:
                reasons.append('fragment counts exceed training maxima')
            elif 'Fragments not documented' in warning:
                reasons.append('fragments outside training library')
        if reasons:
            status = 'OUT OF AD: ' + '; '.join(dict.fromkeys(reasons))
        else:
            status = 'OUT OF AD: see warnings.'

    return {
        'in_ad': in_ad,
        'status': status,
        'warnings': warnings,
        'details': details
    }


def get_module_info():
    """
    Get information about the KOWWIN module and its applicability domain.

    Returns:
        dict: Module information including training/validation set statistics
    """
    return {
        'module_name': 'KOWWIN',
        'description': 'Octanol-Water Partition Coefficient (Log Kow) estimation',
        'training_set': TRAINING_SET,
        'validation_set': VALIDATION_SET,
        'primary_ad_criteria': 'Molecular Weight',
        'secondary_ad_criteria': ['Fragment counts', 'Structural features']
    }
