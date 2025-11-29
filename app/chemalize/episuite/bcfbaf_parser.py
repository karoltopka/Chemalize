"""
BCFBAF Parser
Parses BCFBAF (Bioconcentration and Bioaccumulation) output from EPI Suite.
"""
import re
from collections import OrderedDict
from typing import Dict, List, Optional

from app.chemalize.episuite.utils import extract_multiline_field

BCF_SECTION_PATTERN = re.compile(
    r'BCFBAF Program\s*\(v[\d.]+\)\s*Results:',
    re.DOTALL
)


def _normalize_fragment_description(description: str) -> str:
    """Canonicalize fragment descriptions by collapsing whitespace."""
    return ' '.join(description.split())


def _parse_fragment_table(section: str) -> Optional[Dict]:
    """
    Extract the LOG BIOTRANSFORMATION fragment table.
    Returns fragments along with coefficient contributions.
    """
    header_match = re.search(
        r'TYPE\s*\|\s*NUM\s*\|\s*LOG\s+BIOTRANSFORMATION\s+FRAGMENT\s+DESCRIPTION',
        section
    )
    if not header_match:
        return None

    start = header_match.start()
    end_match = re.search(r'\n\s*Biotransformation Rate Constant', section[start:], re.IGNORECASE)
    if end_match:
        end = start + end_match.start()
    else:
        end = len(section)

    table_text = section[start:end]
    fragments: List[Dict] = []

    for frag_match in re.finditer(
        r'Frag\s*\|\s*(\d+)\s*\|\s*(.*?)\s*\|\s*([-+\d.Ee]+)\s*\|\s*([-+\d.Ee]+)',
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

    def _extract_value(label: str) -> Optional[float]:
        pattern = rf'{label}\|\s*\*\s*\|\s*(.*?)\s*\|\s*([-+\d.Ee]*)\s*\|\s*([-+\d.Ee]+)'
        m = re.search(pattern, table_text)
        if m:
            try:
                return float(m.group(3))
            except ValueError:
                return None
        return None

    log_kow_contribution = _extract_value('L Kow')
    molwt_contribution = _extract_value('MolWt')
    constant_value = _extract_value('Const')

    # Extract result rows
    result_rows = {}
    for res_match in re.finditer(
        r'RESULT\s*\|\s*(.*?)\s*\|\s*[^\|]*\|\s*([-+\d.Ee]+)',
        table_text
    ):
        label = ' '.join(res_match.group(1).split())
        value = float(res_match.group(2))
        result_rows[label] = value

    return {
        'fragments': fragments,
        'log_kow_contribution': log_kow_contribution,
        'molwt_contribution': molwt_contribution,
        'constant': constant_value,
        'results': result_rows,
    }


def parse_bcfbaf(file_content: str) -> List[Dict]:
    """
    Parse all BCFBAF sections from an EPI Suite output file.
    """
    entries: List[Dict] = []

    matches = list(BCF_SECTION_PATTERN.finditer(file_content))

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
            'log_bcf_regression': None,
            'bcf_regression_value': None,
            'biotrans_half_life_days': None,
            'log_baf_upper': None,
            'baf_upper_value': None,
            'log_kow_experimental': None,
            'log_kow_used': None,
            'arnot_gobas': {},
            'fragments': [],
            'fragment_details': [],
            'table_results': {},
            'bcf_corrections': [],
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
                entry['mol_weight'] = float(mol_wt_value)
            except ValueError:
                entry['mol_weight'] = None
        else:
            mol_wt_match = re.search(r'MOL WT\s*:\s*([-\d.Ee+]+)', section_text)
            if mol_wt_match:
                try:
                    entry['mol_weight'] = float(mol_wt_match.group(1))
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

        def _extract_metric(pattern: str) -> Optional[float]:
            match = re.search(pattern, section_text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return None
            return None

        entry['log_bcf_regression'] = _extract_metric(
            r'Log BCF \(regression-based estimate\):\s*([-+\d.Ee]+)'
        )
        entry['bcf_regression_value'] = _extract_metric(
            r'Log BCF \(regression-based estimate\):\s*[-+\d.Ee]+\s*\(BCF\s*=\s*([-+\d.Ee]+)'
        )
        entry['biotrans_half_life_days'] = _extract_metric(
            r'Biotransformation Half-Life \(days\)\s*:\s*([-+\d.Ee]+)'
        )
        entry['log_baf_upper'] = _extract_metric(
            r'Log BAF \(Arnot-Gobas upper trophic\):\s*([-+\d.Ee]+)'
        )
        entry['baf_upper_value'] = _extract_metric(
            r'Log BAF \(Arnot-Gobas upper trophic\):\s*[-+\d.Ee]+\s*\(BAF\s*=\s*([-+\d.Ee]+)'
        )

        exp_logkow_match = re.search(
            r'Log Kow \(experimental\):\s*(.+)', section_text
        )
        if exp_logkow_match:
            value = exp_logkow_match.group(1).strip()
            if value.lower().startswith('not available'):
                entry['log_kow_experimental'] = None
            else:
                try:
                    entry['log_kow_experimental'] = float(value)
                except ValueError:
                    entry['log_kow_experimental'] = None

        entry['log_kow_used'] = _extract_metric(
            r'Log Kow used by BCF estimates:\s*([-+\d.Ee]+)'
        )

        metal_note = re.search(
            r'NOTE:\s*METAL\s*\(.*?\)\s*HAS BEEN REMOVED.*?(?:\n\s*\n|$)',
            section_text,
            re.IGNORECASE
        )
        if metal_note:
            entry['metal_warning'] = ' '.join(metal_note.group(0).split())

        corrections_block = re.search(
            r'Correction\(s\):(.*?)(?:\n\s*\n|\Z)',
            section_text,
            re.DOTALL
        )
        if corrections_block:
            lines = [line.strip() for line in corrections_block.group(1).splitlines()]
            cleaned = []
            for line in lines:
                if not line or 'No Applicable' in line or '-----' in line or line == 'Value':
                    continue
                cleaned.append(re.sub(r'\s{2,}', ' ', line))
            entry['bcf_corrections'] = cleaned

        for trophic in ('upper', 'mid', 'lower'):
            pattern = (
                rf'Estimated Log BCF \({trophic} trophic\)\s*=\s*([-+\d.Ee]+)\s*'
                r'\(BCF\s*=\s*([-+\d.Ee]+)'
            )
            match = re.search(pattern, section_text)
            if match:
                try:
                    entry['arnot_gobas'][f'{trophic}_trophic_log_bcf'] = float(match.group(1))
                except ValueError:
                    entry['arnot_gobas'][f'{trophic}_trophic_log_bcf'] = None
                try:
                    entry['arnot_gobas'][f'{trophic}_trophic_bcf'] = float(match.group(2))
                except ValueError:
                    entry['arnot_gobas'][f'{trophic}_trophic_bcf'] = None

            baf_pattern = (
                rf'Estimated Log BAF \({trophic} trophic\)\s*=\s*([-+\d.Ee]+)\s*'
                r'\(BAF\s*=\s*([-+\d.Ee]+)'
            )
            match_baf = re.search(baf_pattern, section_text)
            if match_baf:
                try:
                    entry['arnot_gobas'][f'{trophic}_trophic_log_baf'] = float(match_baf.group(1))
                except ValueError:
                    entry['arnot_gobas'][f'{trophic}_trophic_log_baf'] = None
                try:
                    entry['arnot_gobas'][f'{trophic}_trophic_baf'] = float(match_baf.group(2))
                except ValueError:
                    entry['arnot_gobas'][f'{trophic}_trophic_baf'] = None

        zero_block = re.search(
            r'Arnot-Gobas BCF & BAF Methods \(assuming a biotransformation rate of zero\):'
            r'(.*?)(?:\n\n|\Z)',
            section_text,
            re.DOTALL
        )
        if zero_block:
            block_text = zero_block.group(1)
            for trophic in ('upper', 'mid', 'lower'):
                pattern = (
                    rf'Estimated Log BCF \({trophic} trophic\)\s*=\s*([-\d.Ee]+)\s*'
                    r'\(BCF\s*=\s*([-\d.Ee]+)'
                )
                match = re.search(pattern, block_text)
                if match:
                    try:
                        entry['arnot_gobas'][f'{trophic}_trophic_log_bcf_no_bio'] = float(match.group(1))
                    except ValueError:
                        entry['arnot_gobas'][f'{trophic}_trophic_log_bcf_no_bio'] = None
                    try:
                        entry['arnot_gobas'][f'{trophic}_trophic_bcf_no_bio'] = float(match.group(2))
                    except ValueError:
                        entry['arnot_gobas'][f'{trophic}_trophic_bcf_no_bio'] = None

                baf_pattern = (
                    rf'Estimated Log BAF \({trophic} trophic\)\s*=\s*([-\d.Ee]+)\s*'
                    r'\(BAF\s*=\s*([-\d.Ee]+)'
                )
                match_baf = re.search(baf_pattern, block_text)
                if match_baf:
                    try:
                        entry['arnot_gobas'][f'{trophic}_trophic_log_baf_no_bio'] = float(match_baf.group(1))
                    except ValueError:
                        entry['arnot_gobas'][f'{trophic}_trophic_log_baf_no_bio'] = None
                    try:
                        entry['arnot_gobas'][f'{trophic}_trophic_baf_no_bio'] = float(match_baf.group(2))
                    except ValueError:
                        entry['arnot_gobas'][f'{trophic}_trophic_baf_no_bio'] = None

        table_data = _parse_fragment_table(section_text)
        if table_data:
            fragment_map: Dict[str, Dict] = OrderedDict()
            for fragment in table_data['fragments']:
                norm = fragment['normalized_description']
                if norm not in fragment_map:
                    fragment_map[norm] = {
                        'description': fragment['description'],
                        'normalized_description': norm,
                        'count': fragment['count'],
                    }
            entry['fragments'] = list(fragment_map.values())
            entry['fragment_details'] = table_data['fragments']
            entry['table_results'] = table_data['results']
            if table_data['log_kow_contribution'] is not None:
                entry['table_results']['log_kow_contribution'] = table_data['log_kow_contribution']
            if table_data['molwt_contribution'] is not None:
                entry['table_results']['molwt_contribution'] = table_data['molwt_contribution']
            if table_data['constant'] is not None:
                entry['table_results']['constant'] = table_data['constant']

        entries.append(entry)

    return entries
