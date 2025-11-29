"""
KOCWIN Parser
Parses KOCWIN (soil sorption coefficient) output from EPI Suite.
"""
import re
from typing import Dict, List, Optional

from app.chemalize.episuite.utils import extract_multiline_field


SECTION_PATTERN = re.compile(
    r'KOCWIN Program\s*\(v[\d.]+\)\s*Results:',
    re.DOTALL
)
NUMERIC_PATTERN = re.compile(r'[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?')


def _safe_float(value: str) -> Optional[float]:
    cleaned = value.replace(',', '').strip()
    if cleaned in {'', '.', '-'}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_block(text: str, start: str, end: Optional[str] = None) -> Optional[str]:
    """
    Return the substring that begins immediately after ``start`` and ends
    right before ``end`` (or the end of ``text`` when ``end`` is omitted).
    """
    start_idx = text.find(start)
    if start_idx == -1:
        return None
    start_idx += len(start)
    if end:
        end_idx = text.find(end, start_idx)
        if end_idx == -1:
            end_idx = len(text)
    else:
        end_idx = len(text)
    return text[start_idx:end_idx]


def _extract_numeric(segment: str) -> Optional[float]:
    """
    Find the first numeric token within ``segment`` and convert it to float.
    """
    cleaned = segment.replace(',', ' ')
    match = NUMERIC_PATTERN.search(cleaned)
    if not match:
        return None
    return _safe_float(match.group(0))


def _extract_value(label: str, text: str) -> Optional[float]:
    """
    Locate a line that begins with ``label`` (ignoring leading whitespace and
    dotted alignments) and return the numeric value that appears after the
    first colon.
    """
    normalized_label = label.rstrip(':')
    pattern = re.compile(
        rf'^\s*{re.escape(normalized_label)}(?:\s*\.+\s*)?(?::)?\s*(?P<value>[-+\d.,Ee]+)',
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return None
    return _extract_numeric(match.group('value'))


def _parse_corrections(block: str) -> List[Dict[str, Optional[int]]]:
    """Extract correction factor entries with descriptor and count."""
    corrections: List[Dict[str, Optional[int]]] = []
    lines = block.splitlines()
    capturing = False

    for raw_line in lines:
        cleaned = raw_line.strip()
        if not capturing:
            if cleaned.startswith('Fragment Correction(s):'):
                capturing = True
            continue

        if not cleaned:
            break
        if cleaned.startswith('Corrected Log Koc'):
            break
        if 'No Applicable Correction Factors' in cleaned:
            corrections = []
            break

        match = re.match(r'(?P<count>\d+)\s+(?P<desc>.+?)\s+\.+\s*:.*', cleaned)
        if match:
            count = int(match.group('count'))
            desc = re.sub(r'\s{2,}', ' ', match.group('desc')).strip()
            corrections.append({'descriptor': desc, 'count': count})
        else:
            corrections.append({'descriptor': re.sub(r'\s{2,}', ' ', cleaned), 'count': None})

    return corrections


def parse_kocwin(file_content: str) -> List[Dict]:
    """
    Parse all KOCWIN sections from an EPI Suite output file.
    """
    entries: List[Dict] = []
    matches = list(SECTION_PATTERN.finditer(file_content))

    for idx, section_match in enumerate(matches):
        section_start = section_match.start()
        section_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(file_content)
        section_text = file_content[section_start:section_end]

        entry = {
            'smiles': None,
            'mol_formula': None,
            'mol_weight': None,
            'chem_id': None,
            'mci_index': None,
            'log_koc_mci_non_corrected': None,
            'log_koc_mci_corrected': None,
            'log_koc_mci_over_correction': None,
            'koc_mci': None,
            'mci_corrections': [],
            'log_kow_used': None,
            'log_koc_logkow_non_corrected': None,
            'log_koc_logkow_corrected': None,
            'koc_logkow': None,
            'logkow_corrections': [],
            'metal_warning': None,
        }

        # Capture compound descriptors preceding the section
        info_block = file_content[max(0, section_start - 1000):section_start]
        smiles_value = extract_multiline_field(info_block, 'SMILES')
        if smiles_value:
            entry['smiles'] = smiles_value
        else:
            smiles_match = list(re.finditer(r'SMILES\s*:\s*(.+)', info_block))
            if smiles_match:
                entry['smiles'] = smiles_match[-1].group(1).strip()

        mol_for_value = extract_multiline_field(info_block, 'MOL FOR', joiner=' ')
        if mol_for_value:
            entry['mol_formula'] = mol_for_value
        else:
            mol_for_match = list(re.finditer(r'MOL FOR:\s*(.+)', info_block))
            if mol_for_match:
                entry['mol_formula'] = mol_for_match[-1].group(1).strip()

        mol_wt_value = extract_multiline_field(info_block, 'MOL WT')
        if mol_wt_value:
            entry['mol_weight'] = _safe_float(mol_wt_value)
        else:
            mol_wt_match = list(re.finditer(r'MOL WT\s*:\s*([-\d.Ee+]+)', info_block))
            if mol_wt_match:
                entry['mol_weight'] = _safe_float(mol_wt_match[-1].group(1))

        chem_value = extract_multiline_field(info_block, 'CHEM', joiner=' ')
        if chem_value:
            entry['chem_id'] = chem_value or None
        else:
            chem_match = list(re.finditer(r'CHEM\s*:\s*(.+)', info_block))
            if chem_match:
                entry['chem_id'] = chem_match[-1].group(1).strip() or None

        # Fallback: if descriptors not found in prefix, search within section body
        if entry['smiles'] is None:
            body_smiles = extract_multiline_field(section_text, 'SMILES', last=False)
            if body_smiles:
                entry['smiles'] = body_smiles
            else:
                body_smiles_match = re.search(r'SMILES\s*:\s*(.+)', section_text)
                if body_smiles_match:
                    entry['smiles'] = body_smiles_match.group(1).strip()

        if entry['mol_formula'] is None:
            body_formula = extract_multiline_field(section_text, 'MOL FOR', joiner=' ', last=False)
            if body_formula:
                entry['mol_formula'] = body_formula
            else:
                body_formula_match = re.search(r'MOL FOR:\s*(.+)', section_text)
                if body_formula_match:
                    entry['mol_formula'] = body_formula_match.group(1).strip()

        if entry['mol_weight'] is None:
            body_mw = extract_multiline_field(section_text, 'MOL WT', last=False)
            if body_mw:
                entry['mol_weight'] = _safe_float(body_mw)
            else:
                body_mw_match = re.search(r'MOL WT\s*:\s*([-\d.Ee+]+)', section_text)
                if body_mw_match:
                    entry['mol_weight'] = _safe_float(body_mw_match.group(1))

        if entry['chem_id'] is None:
            chem_body = extract_multiline_field(section_text, 'CHEM', joiner=' ', last=False)
            if chem_body:
                entry['chem_id'] = chem_body or None
            else:
                chem_body_match = re.search(r'CHEM\s*:\s*(.+)', section_text)
                if chem_body_match:
                    entry['chem_id'] = chem_body_match.group(1).strip() or None

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

        # MCI-based block
        mci_block = _extract_block(section_text, 'Koc Estimate from MCI:', 'Koc Estimate from Log Kow:')
        if mci_block:
            entry['mci_index'] = _extract_value('First Order Molecular Connectivity Index', mci_block)
            entry['log_koc_mci_non_corrected'] = _extract_value('Non-Corrected Log Koc (0.5213 MCI + 0.60)', mci_block)
            entry['log_koc_mci_corrected'] = _extract_value('Corrected Log Koc', mci_block)
            entry['log_koc_mci_over_correction'] = _extract_value('Over Correction Adjustment to Lower Limit Log Koc', mci_block)
            entry['koc_mci'] = _extract_value('Estimated Koc', mci_block)
            entry['mci_corrections'] = _parse_corrections(mci_block)

        # Log Kow block
        logkow_block = _extract_block(section_text, 'Koc Estimate from Log Kow:', None)
        if logkow_block:
            entry['log_kow_used'] = _extract_value('Log Kow  (Kowwin estimate)', logkow_block)
            entry['log_koc_logkow_non_corrected'] = _extract_value('Non-Corrected Log Koc (0.55313 logKow + 0.9251)', logkow_block)
            entry['log_koc_logkow_corrected'] = _extract_value('Corrected Log Koc', logkow_block)
            entry['koc_logkow'] = _extract_value('Estimated Koc', logkow_block)
            entry['logkow_corrections'] = _parse_corrections(logkow_block)

        entries.append(entry)

    return entries
