"""
KOWWIN Parser
Parses KOWWIN output from EPI Suite
"""
import re
from app.chemalize.episuite.ad_rules import kowwin_ad
from app.chemalize.episuite.utils import extract_multiline_field


def parse_kowwin(file_content):
    """
    Parse all KOWWIN sections from an EPI Suite output file.

    Args:
        file_content (str): Full content of EPI Suite output file

    Returns:
        list[dict]: Each entry contains:
            - smiles: SMILES notation
            - mol_formula: Molecular formula
            - mol_weight: Molecular weight
            - log_kow: Calculated Log Kow value
            - fragments: List of fragment contributions
    """
    entries = []

    section_pattern = re.compile(
        r'KOWWIN Program.*?(?=KOWWIN Program|BIOWIN \(v|BCFBAF Program|\Z)',
        re.DOTALL
    )

    for match in section_pattern.finditer(file_content):
        section_start = match.start()
        section_text = match.group(0)

        info_block = file_content[max(0, section_start - 1000):section_start]

        def _last_match(pattern: str):
            matches = list(re.finditer(pattern, info_block))
            return matches[-1] if matches else None

        entry = {
            'smiles': None,
            'mol_formula': None,
            'mol_weight': None,
            'chem_id': None,
            'log_kow': None,
            'fragments': []
        }

        smiles_value = extract_multiline_field(info_block, 'SMILES')
        if smiles_value:
            entry['smiles'] = smiles_value
        else:
            smiles_match = _last_match(r'SMILES\s*:\s*(.+)')
            if smiles_match:
                entry['smiles'] = smiles_match.group(1).strip()

        mol_for_value = extract_multiline_field(info_block, 'MOL FOR', joiner=' ')
        if mol_for_value:
            entry['mol_formula'] = mol_for_value
        else:
            mol_for_match = _last_match(r'MOL FOR:\s*(.+)')
            if mol_for_match:
                entry['mol_formula'] = mol_for_match.group(1).strip()

        mol_wt_value = extract_multiline_field(info_block, 'MOL WT')
        if mol_wt_value:
            try:
                entry['mol_weight'] = float(mol_wt_value)
            except ValueError:
                entry['mol_weight'] = None
        else:
            mol_wt_match = _last_match(r'MOL WT\s*:\s*([-\d.Ee+]+)')
            if mol_wt_match:
                try:
                    entry['mol_weight'] = float(mol_wt_match.group(1))
                except ValueError:
                    entry['mol_weight'] = None

        chem_value = extract_multiline_field(info_block, 'CHEM', joiner=' ')
        if chem_value:
            entry['chem_id'] = chem_value or None
        else:
            chem_match = _last_match(r'CHEM\s*:\s*(.+)')
            if chem_match:
                entry['chem_id'] = chem_match.group(1).strip() or None

        if entry['smiles'] is None:
            smiles_body = extract_multiline_field(section_text, 'SMILES', last=False)
            if smiles_body:
                entry['smiles'] = smiles_body

        if entry['mol_formula'] is None:
            mol_for_body = extract_multiline_field(section_text, 'MOL FOR', joiner=' ', last=False)
            if mol_for_body:
                entry['mol_formula'] = mol_for_body

        if entry['mol_weight'] is None:
            mol_wt_body = extract_multiline_field(section_text, 'MOL WT', last=False)
            if mol_wt_body:
                try:
                    entry['mol_weight'] = float(mol_wt_body)
                except ValueError:
                    entry['mol_weight'] = None

        if entry['chem_id'] is None:
            chem_body = extract_multiline_field(section_text, 'CHEM', joiner=' ', last=False)
            if chem_body:
                entry['chem_id'] = chem_body or None
            else:
                chem_body_match = re.search(r'CHEM\s*:\s*(.+)', section_text)
                if chem_body_match:
                    entry['chem_id'] = chem_body_match.group(1).strip() or None

        log_kow_match = re.search(r'Log Kow\(version.*?estimate\):\s*([-\d.]+)', section_text)
        if log_kow_match:
            entry['log_kow'] = float(log_kow_match.group(1))
        else:
            log_kow_alt = re.search(r'Log Kow\s*=\s*([-\d.]+)', section_text)
            if log_kow_alt:
                entry['log_kow'] = float(log_kow_alt.group(1))

        fragment_table_match = re.search(
            r'TYPE\s+\|\s+NUM\s+\|.*?DESCRIPTION.*?\n[-+]+\n(.*?)\n[-+]+',
            section_text,
            re.DOTALL
        )

        if fragment_table_match:
            fragment_lines = fragment_table_match.group(1).strip().split('\n')
            for line in fragment_lines:
                if '|' in line and 'TYPE' not in line and not line.strip().startswith('-'):
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 5:
                        frag_type = parts[0]
                        try:
                            frag_num_str = parts[1].strip()
                            frag_num = int(frag_num_str) if frag_num_str else 0
                            frag_desc = parts[2]
                            frag_coeff_str = parts[3].strip()
                            frag_coeff = float(frag_coeff_str) if frag_coeff_str else 0.0
                            if frag_type in ['Frag', 'Factor'] and frag_num > 0:
                                entry['fragments'].append({
                                    'type': frag_type,
                                    'count': frag_num,
                                    'description': frag_desc,
                                    'coefficient': frag_coeff
                                })
                        except (ValueError, IndexError):
                            continue

        entries.append(entry)

    return entries


def check_kowwin_ad(kowwin_data):
    """
    Check if KOWWIN prediction is within applicability domain.
    Uses rules from app.chemalize.episuite.ad_rules.kowwin_ad module.

    Args:
        kowwin_data (dict): Parsed KOWWIN data from parse_kowwin()

    Returns:
        dict: Applicability domain assessment with:
            - in_ad (bool): True if within AD
            - status (str): Detailed status message
            - warnings (list): List of warning messages
            - details (dict): Additional details about the assessment
    """
    mol_weight = kowwin_data.get('mol_weight')
    fragments = kowwin_data.get('fragments')

    # Use the dedicated AD rules module
    return kowwin_ad.check_applicability_domain(mol_weight, fragments)
