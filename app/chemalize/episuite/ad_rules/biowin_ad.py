"""
BIOWIN Applicability Domain Rules
---------------------------------

Evaluates the applicability domain (AD) for the BIOWIN suite of models by
grouping the models according to their shared training sets:

* Biowin1 & Biowin2 (fast biodegradation probability; 295-compound training set)
* Biowin3 & Biowin4 (ultimate/primary timeframe survey; 200-compound training set)
* Biowin5 & Biowin6 (MITI ready biodegradation probability; 589-compound training set)

Each group is checked independently so downstream consumers can report
per-group AD status as well as an overall summary.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple


def _normalize_fragment_description(description: str) -> str:
    """Canonicalize fragment descriptions by collapsing whitespace."""
    return ' '.join(description.split())


# Fragment maxima derived from Appendix A (Biowin 1 & 2 training set).
FRAGMENT_MAX_COUNTS_LINEAR_RAW: Dict[str, int] = {
    'Nitroso [-N-N=O]': 1,
    'Linear C4 terminal chain [CCC-CH3]': 3,
    'Aliphatic alcohol [-OH]': 3,
    'Aromatic alcohol [-OH]': 3,
    'Aliphatic acid [-C(=O)-OH]': 4,
    'Aromatic acid [-C(=O)-OH]': 2,
    'Aldehyde [-CHO]': 1,
    'Ester [-C(=O)-O-C]': 3,
    'Amide [-C(=O)-N or -C(=S)-N]': 1,
    'Triazine ring (symmetric)': 1,
    'Aliphatic chloride [-CL]': 6,
    'Aromatic chloride [-CL]': 6,
    'Aliphatic bromide [-Br]': 6,
    'Aromatic bromide [-Br]': 6,
    'Aromatic iodide [-I]': 1,
    'Aromatic fluoride [-F]': 1,
    'Carbon with 4 single bonds & no hydrogens': 2,
    'Aromatic nitro [-NO2]': 2,
    'Aliphatic amine [-NH2 or -NH-]': 2,
    'Aromatic amine [-NH2 or -NH-]': 3,
    'Cyanide / Nitriles [-C#N]': 2,
    'Sulfonic acid / salt -> aromatic attach': 3,
    'Sulfonic acid / salt -> aliphatic attach': 1,
    'Polyaromatic hydrocarbon (4 or more rings)': 1,
    'Pyridine ring': 1,
    'Aromatic ether [-O-aromatic carbon]': 2,
    'Aliphatic ether [C-O-C]': 2,
    'Ketone [-C-C(=O)-C-]': 2,
    'Tertiary amine': 4,
    'Phosphate ester': 1,
    'Alkyl substituent on aromatic ring': 3,
    'Azo group [-N=N-]': 1,
    'Carbamate or Thiocarbamate': 1,
    'Trifluoromethyl group [-CF3]': 1,
    'Unsubstituted aromatic (3 or less rings)': 1,
    'Unsubstituted phenyl group (C6H5-)': 2,
}


# Fragment maxima derived from Appendix D (MITI / Biowin 5 & 6 training set).
FRAGMENT_MAX_COUNTS_MITI_RAW: Dict[str, int] = {
    'Nitroso [-N-N=O]': 1,
    'Linear C4 terminal chain [CCC-CH3]': 3,
    'Aliphatic alcohol [-OH]': 3,
    'Aromatic alcohol [-OH]': 2,
    'Aliphatic acid [-C(=O)-OH]': 2,
    'Aldehyde [-CHO]': 1,
    'Ester [-C(=O)-O-C]': 2,
    'Amide [-C(=O)-N or -C(=S)-N]': 2,
    'Triazine ring (symmetric)': 1,
    'Aliphatic chloride [-CL]': 12,
    'Aromatic chloride [-CL]': 10,
    'Aliphatic bromide [-Br]': 6,
    'Aromatic bromide [-Br]': 10,
    'Aromatic iodide [-I]': 1,
    'Carbon with 4 single bonds & no hydrogens': 10,
    'Aromatic nitro [-NO2]': 2,
    'Aliphatic amine [-NH2 or -NH-]': 1,
    'Aromatic amine [-NH2 or -NH-]': 2,
    'Cyanide / Nitriles [-C#N]': 2,
    'Sulfonic acid / salt -> aromatic attach': 1,
    'Polyaromatic hydrocarbon (4 or more rings)': 1,
    'Pyridine ring': 1,
    'Aromatic ether [-O-aromatic carbon]': 5,
    'Aliphatic ether [C-O-C]': 2,
    'Ketone [-C-C(=O)-C-]': 2,
    'Tertiary amine': 2,
    'Phosphate ester (P=O type)': 1,
    'Alkyl substituent on aromatic ring': 6,
    'Carbamate or Thiocarbamate': 1,
    'Trifluoromethyl group [-CF3]': 2,
    'Unsubstituted aromatic (3 or less rings)': 1,
    'Unsubstituted phenyl group (C6H5-)': 3,
    'Phosphate ester (P=S type)': 1,
    'Urea [N-C(=O)-N]': 1,
    'Furan or Thiofuran': 1,
    'Triazole Ring': 1,
    'Fluorine [-F]': 8,
    'Aromatic-CH3': 4,
    'Aromatic-CH2': 4,
    'Aromatic-CH': 3,
    'Aromatic-H': 15,
    'Methyl [-CH3]': 12,
    '-CH2- [linear]': 28,
    '-CH- [linear]': 2,
    '-CH2- [cyclic]': 12,
    '-CH - [cyclic]': 17,
    '-C=CH [alkenyl hydrogen]': 6,
    'Thiazole Ring': 2,
    'o-Chloro / Mono-aromatic ether': 1,
    'Number of fused acyclic rings': 5,
    'Number of fused 6-carbon aromatic rings': 5,
    'Four or more fused aromatic rings': 1,
    'Four or more fused cyclic rings': 1,
    'Benzene': 3,
    'Naphthalene': 1,
    'Indane': 1,
    'Biphenyl': 1,
}


def _normalize_fragment_dict(raw: Dict[str, int]) -> Dict[str, int]:
    """Return a dict keyed by normalized fragment descriptions."""
    return {_normalize_fragment_description(key): value for key, value in raw.items()}


FRAGMENT_MAX_COUNTS_LINEAR = _normalize_fragment_dict(FRAGMENT_MAX_COUNTS_LINEAR_RAW)
FRAGMENT_MAX_COUNTS_MITI = _normalize_fragment_dict(FRAGMENT_MAX_COUNTS_MITI_RAW)

MW_RANGE_LINEAR: Tuple[float, float] = (31.06, 697.7)
MW_RANGE_MITI: Tuple[float, float] = (31.06, 697.7)

# Rating (dependent variable) ranges for Biowin3 and Biowin4.
DEPENDENT_RANGES_SURVEY: Dict[str, Tuple[float, float]] = {
    'Biowin3': (1.44, 3.89),
    'Biowin4': (2.37, 4.57),
}


MODEL_GROUPS = {
    'Biowin1_2': {
        'label': 'Biowin 1-2',
        'models': ('Biowin1', 'Biowin2'),
        'mw_range': MW_RANGE_LINEAR,
        'fragment_max_counts': FRAGMENT_MAX_COUNTS_LINEAR,
        'dependent_ranges': {},
    },
    'Biowin3_4': {
        'label': 'Biowin 3-4',
        'models': ('Biowin3', 'Biowin4'),
        # Training set uses the same 36 fragments; documentation references the
        # same MW bounds as Biowin1/2.
        'mw_range': MW_RANGE_LINEAR,
        'fragment_max_counts': FRAGMENT_MAX_COUNTS_LINEAR,
        'dependent_ranges': DEPENDENT_RANGES_SURVEY,
    },
    'Biowin5_6': {
        'label': 'Biowin 5-6',
        'models': ('Biowin5', 'Biowin6'),
        'mw_range': MW_RANGE_MITI,
        'fragment_max_counts': FRAGMENT_MAX_COUNTS_MITI,
        'dependent_ranges': {},
    },
}


def _collect_fragments(models: Dict[str, Dict], model_names: Iterable[str]) -> List[Dict]:
    """
    Aggregate fragment occurrences across the provided models, returning a
    single list keyed by normalized description.
    """
    aggregated: Dict[str, Dict] = {}
    for model_name in model_names:
        model = models.get(model_name)
        if not model:
            continue
        for fragment in model.get('fragments', []):
            norm = fragment.get('normalized_description') or _normalize_fragment_description(
                fragment.get('description', '')
            )
            if norm not in aggregated:
                aggregated[norm] = {
                    'description': fragment.get('description', norm),
                    'normalized_description': norm,
                    'count': fragment.get('count', 0),
                }
    return list(aggregated.values())


def _evaluate_group(group_key: str,
                    config: Dict,
                    models: Dict[str, Dict],
                    molecular_weight: Optional[float],
                    metal_warning: Optional[str]) -> Dict:
    """
    Evaluate a single BIOWIN model group (e.g., Biowin1/2) and return its AD
    assessment.
    """
    warnings: List[str] = []
    models_present: List[str] = []
    models_missing: List[str] = []

    for model_name in config['models']:
        if model_name in models:
            models_present.append(model_name)
        else:
            models_missing.append(model_name)

    details = {
        'label': config['label'],
        'models_present': models_present,
        'models_missing': models_missing,
        'mw_range': config['mw_range'],
        'mw_in_range': None,
        'fragment_exceedances': [],
        'unknown_fragments': [],
        'fragments_without_max': [],
        'dependent_range_checks': {},
        'metal_removed': metal_warning,
    }

    evaluated = bool(models_present)
    if not evaluated:
        warnings.append('Model results not available for this group.')
        return {
            'label': config['label'],
            'evaluated': False,
            'in_ad': False,
            'status': 'Not evaluated: missing BIOWIN results for this group.',
            'warnings': warnings,
            'details': details,
        }

    in_ad = True

    # Molecular weight check (if bounds provided)
    if config['mw_range'] and molecular_weight is not None:
        min_mw, max_mw = config['mw_range']
        if min_mw <= molecular_weight <= max_mw:
            details['mw_in_range'] = True
        else:
            details['mw_in_range'] = False
            in_ad = False
            if molecular_weight < min_mw:
                warnings.append(
                    f'Molecular weight {molecular_weight:.2f} falls below training minimum ({min_mw}).'
                )
            else:
                warnings.append(
                    f'Molecular weight {molecular_weight:.2f} exceeds training maximum ({max_mw}).'
                )
    elif config['mw_range']:
        details['mw_in_range'] = False
        in_ad = False
        warnings.append('Cannot evaluate molecular weight bounds (value not provided).')

    # Fragment checks
    fragments = _collect_fragments(models, models_present)
    fragment_max = config.get('fragment_max_counts') or {}
    for fragment in fragments:
        desc = fragment.get('description', fragment['normalized_description'])
        norm_desc = fragment['normalized_description']
        count = fragment.get('count', 0)
        if norm_desc in fragment_max:
            max_allowed = fragment_max[norm_desc]
            if max_allowed is not None and count > max_allowed:
                in_ad = False
                detail = f"{desc} (count {count} > max {max_allowed})"
                details['fragment_exceedances'].append(detail)
        else:
            details['unknown_fragments'].append(f"{desc} (count {count})")

    if details['fragment_exceedances']:
        warnings.append(
            'Fragments exceed documented maxima: ' +
            '; '.join(details['fragment_exceedances'])
        )
    if details['unknown_fragments']:
        warnings.append(
            'Fragments not documented in training set: ' +
            '; '.join(details['unknown_fragments'])
        )

    # Dependent-variable range checks (Biowin3/4 ratings)
    dependent_ranges = config.get('dependent_ranges', {})
    for model_name, bounds in dependent_ranges.items():
        model = models.get(model_name) or {}
        rating = model.get('rating')
        if rating is None:
            details['dependent_range_checks'][model_name] = 'missing'
            warnings.append(f'{model_name} rating not reported; domain check skipped.')
            continue
        min_val, max_val = bounds
        if min_val <= rating <= max_val:
            details['dependent_range_checks'][model_name] = 'within'
        else:
            details['dependent_range_checks'][model_name] = 'outside'
            in_ad = False
            warnings.append(
                f'{model_name} rating {rating:.2f} outside training survey range ({min_val}–{max_val}).'
            )

    status: str
    if in_ad:
        status = f"IN AD for {config['label']}."
        if warnings:
            status = f"IN AD (with caution) for {config['label']}."
    else:
        status = f"OUT OF AD for {config['label']}."

    if metal_warning:
        warnings.append(
            metal_warning + ' Consider evaluating the neutral (non-salt) form.'
        )

    return {
        'label': config['label'],
        'evaluated': True,
        'in_ad': in_ad,
        'status': status,
        'warnings': warnings,
        'details': details,
    }


def check_applicability_domain(biowin_entry: Dict,
                               override_metal_warning: Optional[str] = None) -> Dict:
    """
    Compute applicability-domain assessments for BIOWIN model groups.

    Args:
        biowin_entry: Parsed BIOWIN payload (the object returned by
            ``parse_biowin`` for a single compound).
        override_metal_warning: Optional metal removal note to override the
            value stored in ``biowin_entry``.

    Returns:
        dict with keys:
            - overall: aggregated status across evaluated groups
            - groups: per-group assessment dictionaries
    """
    models = biowin_entry.get('models', {}) if biowin_entry else {}
    molecular_weight = biowin_entry.get('mol_weight') if biowin_entry else None
    metal_warning = override_metal_warning or (biowin_entry or {}).get('metal_warning')

    group_results: Dict[str, Dict] = {}
    for group_key, config in MODEL_GROUPS.items():
        group_results[group_key] = _evaluate_group(
            group_key=group_key,
            config=config,
            models=models,
            molecular_weight=molecular_weight,
            metal_warning=metal_warning,
        )

    evaluated_groups = [res for res in group_results.values() if res['evaluated']]
    if not evaluated_groups:
        overall_status = 'Not evaluated: no BIOWIN results available.'
        overall_in_ad = False
        overall_warnings: List[str] = ['BIOWIN output missing for all model groups.']
    else:
        overall_in_ad = all(res['in_ad'] for res in evaluated_groups)
        out_of_ad_groups = [res['label'] for res in evaluated_groups if not res['in_ad']]
        overall_warnings = []
        for res in evaluated_groups:
            overall_warnings.extend(res['warnings'])

        if overall_in_ad:
            overall_status = 'IN AD across evaluated BIOWIN groups.'
            if overall_warnings:
                overall_status = 'IN AD (with caution) across evaluated BIOWIN groups.'
        else:
            overall_status = (
                'OUT OF AD for: ' + ', '.join(out_of_ad_groups)
            )

    return {
        'overall': {
            'in_ad': overall_in_ad,
            'status': overall_status,
            'warnings': overall_warnings,
            'evaluated_groups': [res['label'] for res in evaluated_groups],
        },
        'groups': group_results,
    }


def get_module_info() -> Dict:
    """Provide descriptive metadata for the BIOWIN AD module."""
    return {
        'module_name': 'BIOWIN',
        'description': (
            'Biodegradation probability and timeframe models (Biowin1–Biowin6) '
            'with per-group applicability-domain evaluation.'
        ),
        'training_sets': {
            'Biowin1_2': {
                'size': 295,
                'molecular_weight_range': {
                    'min': MW_RANGE_LINEAR[0],
                    'max': MW_RANGE_LINEAR[1],
                },
                'fragment_count_source': 'Appendix A (Biowin 1 & 2)',
            },
            'Biowin3_4': {
                'size': 200,
                'molecular_weight_range': {
                    'min': MW_RANGE_LINEAR[0],
                    'max': MW_RANGE_LINEAR[1],
                },
                'dependent_variable_ranges': DEPENDENT_RANGES_SURVEY,
                'fragment_count_source': 'Appendix A (shared fragment library)',
            },
            'Biowin5_6': {
                'size': 589,
                'fragment_count_source': 'Appendix D (MITI fragment library)',
            },
        },
        'notes': (
            'Fragment maxima sourced from EPI Suite appendices. MITI (Biowin5/6) '
            'does not publish molecular-weight limits; those checks are recorded '
            'for reference but not enforced.'
        ),
    }
