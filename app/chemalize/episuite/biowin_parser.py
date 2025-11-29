"""
BIOWIN Parser
Parses BIOWIN (Biodegradation probability) output from EPI Suite.
"""
import math
import re
from collections import OrderedDict
from typing import Dict, List, Optional

from app.chemalize.episuite.utils import extract_multiline_field

# Textual mapping for Biowin3/4 rating -> timeframe words
def _biowin_time_category(rating: Optional[float]) -> Optional[str]:
    if rating is None:
        return None
    if rating > 4.75:
        return 'Hours'
    if rating > 4.25:
        return 'Hours to Days'
    if rating > 3.75:
        return 'Days'
    if rating > 3.25:
        return 'Days to Weeks'
    if rating > 2.75:
        return 'Weeks'
    if rating > 2.25:
        return 'Weeks to Months'
    if rating > 1.75:
        return 'Months'
    return 'Recalcitrant'


def _normalize_fragment_description(description: str) -> str:
    """Canonicalize fragment descriptions by collapsing whitespace."""
    return ' '.join(description.split())


def _parse_fragment_table(section: str, model_number: str) -> Optional[Dict]:
    """
    Extract fragment table for a given BIOWIN model.

    Returns dictionary with fragments, MolWt contribution, constants, and raw table text.
    """
    table_pattern = re.compile(
        rf'TYPE\s*\|\s*NUM\s*\|\s*Biowin{model_number}\s+FRAGMENT DESCRIPTION.*?'
        r'============\+.*?RESULT.*?\|\s*([-\d.]+)',
        re.DOTALL
    )

    match = table_pattern.search(section)
    if not match:
        return None

    table_text = match.group(0)
    result_value = float(match.group(1))

    fragments: List[Dict] = []
    for frag_match in re.finditer(
        r'Frag\s*\|\s*(\d+)\s*\|\s*(.*?)\s*\|\s*([-\d.]+)\s*\|\s*([-\d.]+)',
        table_text
    ):
        count = int(frag_match.group(1))
        description = frag_match.group(2).strip()
        coefficient = float(frag_match.group(3))
        value = float(frag_match.group(4))
        fragments.append({
            'count': count,
            'description': description,
            'normalized_description': _normalize_fragment_description(description),
            'coefficient': coefficient,
            'value': value,
        })

    molwt_match = re.search(
        r'MolWt\|\s*\*\s*\|\s*(.*?)\s*\|\s*[-\d.]*\s*\|\s*([-\d.]+)', table_text
    )
    molwt_value = float(molwt_match.group(2)) if molwt_match else 0.0

    const_match = re.search(
        r'Const\|\s*\*\s*\|\s*(.*?)\s*\|\s*[-\d.]*\s*\|\s*([-\d.]+)', table_text
    )
    const_value = float(const_match.group(2)) if const_match else None

    return {
        'fragments': fragments,
        'molwt_contribution': molwt_value,
        'constant': const_value,
        'table_text': table_text,
        'reported_result': result_value,
    }


def parse_biowin(file_content: str) -> List[Dict]:
    """
    Parse all BIOWIN sections from EPI Suite output file.

    Args:
        file_content: Full content of the EPI Suite output file.

    Returns:
        list[dict]: Each entry contains compound information, model outputs,
                    fragment contributions, and aggregated fragment counts.
    """
    entries: List[Dict] = []

    section_pattern = re.compile(r'BIOWIN\s*\(v[\d.]+\)\s*Program Results:', re.DOTALL)
    matches = list(section_pattern.finditer(file_content))

    for idx, section_match in enumerate(matches):
        section_start = section_match.start()
        section_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(file_content)
        section_text = file_content[section_start:section_end]
        info_block = file_content[max(0, section_start - 1000):section_start]
        entry = {
            'smiles': None,
            'mol_formula': None,
            'mol_weight': None,
            'chem_id': None,
            'models': OrderedDict(),
            'fragments': [],
            'metal_warning': None,
        }

        smiles_value = extract_multiline_field(section_text, 'SMILES')
        if smiles_value:
            entry['smiles'] = smiles_value
        else:
            smiles_match = re.search(r'SMILES\s*:\s*(.+)', section_text)
            if smiles_match:
                entry['smiles'] = smiles_match.group(1).strip()

        mol_for_value = extract_multiline_field(section_text, 'MOL FOR', joiner=' ')
        if mol_for_value:
            entry['mol_formula'] = mol_for_value
        else:
            mol_for_match = re.search(r'MOL FOR:\s*(.+)', section_text)
            if mol_for_match:
                entry['mol_formula'] = mol_for_match.group(1).strip()

        mol_wt_value = extract_multiline_field(section_text, 'MOL WT')
        if mol_wt_value:
            try:
                entry['mol_weight'] = float(mol_wt_value.replace(',', ''))
            except ValueError:
                entry['mol_weight'] = None
        else:
            mol_wt_match = re.search(r'MOL WT\s*:\s*([-\d.Ee+,]+)', section_text)
            if mol_wt_match:
                value = mol_wt_match.group(1).replace(',', '')
                try:
                    entry['mol_weight'] = float(value)
                except ValueError:
                    entry['mol_weight'] = None

        chem_value = extract_multiline_field(section_text, 'CHEM', joiner=' ')
        if chem_value:
            entry['chem_id'] = chem_value or None
        else:
            chem_match = re.search(r'CHEM\s*:\s*(.+)', section_text)
            if chem_match:
                entry['chem_id'] = chem_match.group(1).strip() or None

        metal_note = None
        for source in (info_block, section_text):
            match = re.search(
                r'NOTE:\s*METAL\s*\(.*?\)\s*HAS BEEN REMOVED[^\n]*',
                source,
                re.IGNORECASE
            )
            if match:
                metal_note = ' '.join(match.group(0).split())
                break
        entry['metal_warning'] = metal_note

        metal_note = re.search(
            r'NOTE:\s*METAL\s*\(.*?\)\s*HAS BEEN REMOVED.*?(?:\n\s*\n|$)',
            section_text,
            re.IGNORECASE
        )
        if metal_note:
            entry['metal_warning'] = ' '.join(metal_note.group(0).split())

        prediction_lines = dict(
            re.findall(r'Biowin(\d)\s*\(.*?\)\s*:\s*(.+)', section_text)
        )

        combined_fragments: Dict[str, Dict] = OrderedDict()

        for model_number in ('1', '2', '3', '4', '5', '6'):
            table_data = _parse_fragment_table(section_text, model_number)
            if not table_data:
                continue

            fragments = table_data['fragments']
            molwt_contribution = table_data['molwt_contribution']
            constant = table_data['constant']
            linear_sum = sum(f['value'] for f in fragments) + molwt_contribution

            probability: Optional[float] = None
            rating: Optional[float] = None
            logistic_total: Optional[float] = None
            total_score: Optional[float] = None

            if model_number == '1':
                constant = constant if constant is not None else 0.7475
                total_score = linear_sum + constant
                probability = (
                    table_data['reported_result']
                    if table_data['reported_result'] is not None
                    else total_score
                )

            elif model_number == '2':
                constant = 3.0087
                logistic_total = constant + linear_sum
                if logistic_total >= 0:
                    probability = 1.0 / (1.0 + math.exp(-logistic_total))
                else:
                    exp_val = math.exp(logistic_total)
                    probability = exp_val / (1.0 + exp_val)
                total_score = logistic_total

            elif model_number == '3':
                constant = constant if constant is not None else 3.1992
                rating = linear_sum + constant
                total_score = rating

            elif model_number == '4':
                constant = constant if constant is not None else 3.8477
                rating = linear_sum + constant
                total_score = rating

            elif model_number == '5':
                constant = constant if constant is not None else 0.7121
                total_score = linear_sum + constant
                probability = (
                    table_data['reported_result']
                    if table_data['reported_result'] is not None
                    else total_score
                )

            else:  # model 6
                probability = table_data['reported_result']
                if probability is not None and 0.0 < probability < 1.0:
                    logistic_total = math.log(probability / (1.0 - probability))
                    constant = logistic_total - linear_sum
                    total_score = logistic_total
                else:
                    logistic_total = None
                    total_score = linear_sum

            model_key = f'Biowin{model_number}'
            entry['models'][model_key] = {
                'probability': probability,
                'rating': rating,
                'classification': prediction_lines.get(model_number),
                'fragments': fragments,
                'molwt_contribution': molwt_contribution,
                'constant': constant,
                'computed_total': total_score,
                'logistic_total': logistic_total,
                'reported_result': table_data['reported_result'],
            }

            if model_number in ('3', '4'):
                entry['models'][model_key]['time_category'] = _biowin_time_category(rating)

            for fragment in fragments:
                norm_desc = fragment['normalized_description']
                if norm_desc not in combined_fragments:
                    combined_fragments[norm_desc] = {
                        'description': fragment['description'],
                        'normalized_description': norm_desc,
                        'count': fragment['count'],
                    }

        entry['fragments'] = list(combined_fragments.values())
        entries.append(entry)

    return entries
