"""
KOCWIN Applicability Domain Rules

Reference: EPI Suite documentation (KOCWIN v2.00) for training/validation
weight ranges and fragment/correction factor guidance.
"""
from typing import Dict, List, Optional

TRAINING_SET = {
    'mw_min': 32.04,
    'mw_max': 665.02,
    'mw_avg': 224.4,
}

VALIDATION_SET = {
    'mw_min': 73.14,
    'mw_max': 504.12,
    'mw_avg': 277.8,
}

CORRECTION_MAX_COUNTS = {
    'Nitrogen to non-fused aromatic ring': 2,
    'Ether, aromatic (-C-O-C)': 2,
    'Nitro (-NO2)': 2,
    'N-CO-C (aliphatic carbon)': 1,
    'Urea (N-CO-N)': 1,
    'Nitrogen to Carbon (aliphatic) (-N-C)': 5,
    'Carbamate (N-CO-O) or (N-CO-S)': 2,
    'Triazine ring': 1,
    'Nitrogen-to-Cycloalkane (aliphatic)': 1,
    'Uracil (-N-CO-N-CO-C=C- ring)': 1,
    'Organic Acid (-CO-OH)': 1,
    'Ketone (-C-CO-C-)': 1,
    'Aliphatic Alcohol (-C-OH)': 1,
    'Nitrile/Cyanide (-C#N)': 2,
    'Thiocarbonyl (C=S)': 1,
    'OrganoPhosphorus [P=S]': 2,
    'OrganoPhosphorus [P=O], aliphatic': 1,
    'N-CO-O-Phenyl Carbamate': 1,
    'Ether, aliphatic (-C-O-C-)': 2,
    'Ester (-C-CO-O-C-) or (HCO-O-C)': 2,
    'Sulfone (-C-SO2-C-)': 2,
    'Azo (-N=N-)': 1,
    'N-CO-O-N=': 1,
    'Aromatic ring with 2 nitrogens': 1,
    'OrganoPhosphorus [P=O], aromatic': 1,
    'Misc (C=O) Group (aliphatic attach)': 1,
    'Pyridine ring': 1,
    'Sulfoxide (-C-SO-C-)': 1,
    'Miscellaneous S(=O) group': 1,
    'Multi-Nitrogen aromatic': 1,
    'Poly-Chlorinated Aromatic': 1,
    'Dithiocarbonyl (-S-C(=S)-N)': 1,
    'Carbonyl Hydrazide (-C(=O)-N-NH2)': 1,
    'Quinone (diketone) ring': 1,
    'NH2-Phenyl/Di-ortho halogens': 1,
    'Aromatic Hydroxy (aromatic-OH)': 1,
}


def check_applicability_domain(molecular_weight: Optional[float],
                               corrections_mci: Optional[List[Dict[str, Optional[int]]]] = None,
                               corrections_logkow: Optional[List[Dict[str, Optional[int]]]] = None,
                               metal_warning: Optional[str] = None) -> Dict:
    """Assess KOCWIN applicability using molecular-weight ranges."""
    warnings: List[str] = []
    details = {
        'mw_in_training_range': False,
        'mw_in_validation_range': False,
        'distance_from_training_avg': None,
        'mci_corrections': corrections_mci or [],
        'logkow_corrections': corrections_logkow or [],
        'correction_exceedances': [],
        'correction_unknown': [],
        'metal_removed': metal_warning,
    }

    def process_corrections(corrections: Optional[List[Dict]], label: str):
        nonlocal in_ad
        if not corrections:
            return
        for corr in corrections:
            desc = corr.get('descriptor', 'Unknown') if isinstance(corr, dict) else str(corr)
            count = corr.get('count') if isinstance(corr, dict) else None
            max_allowed = CORRECTION_MAX_COUNTS.get(desc)
            if max_allowed is None:
                details['correction_unknown'].append(f"{label}: {desc}")
                warnings.append(f"{label} correction '{desc}' not documented in training set; interpret with caution.")
                continue
            if count is not None and count > max_allowed:
                info = (
                    f"{label} correction '{desc}' count {count} exceeds training maximum {max_allowed}."
                )
                details['correction_exceedances'].append(info)
                warnings.append(info)
                in_ad = False

    if molecular_weight is None:
        return {
            'in_ad': False,
            'status': 'Cannot determine AD - molecular weight not provided.',
            'warnings': ['Molecular weight not available in KOCWIN section.'],
            'details': details,
        }

    training_min = TRAINING_SET['mw_min']
    training_max = TRAINING_SET['mw_max']
    validation_min = VALIDATION_SET['mw_min']
    validation_max = VALIDATION_SET['mw_max']
    training_avg = TRAINING_SET['mw_avg']

    details['distance_from_training_avg'] = abs(molecular_weight - training_avg)

    in_ad = True

    if validation_min <= molecular_weight <= validation_max:
        details['mw_in_validation_range'] = True

    if training_min <= molecular_weight <= training_max:
        details['mw_in_training_range'] = True
        status = f"IN AD: MW ({molecular_weight:.2f}) within training range."
    elif molecular_weight < training_min:
        in_ad = False
        status = f"OUT OF AD: MW ({molecular_weight:.2f}) below training minimum ({training_min})."
        warnings.append(status)
    else:  # molecular_weight above training maximum
        in_ad = False
        status = f"OUT OF AD: MW ({molecular_weight:.2f}) exceeds training maximum ({training_max})."
        warnings.append(status)

    if details['distance_from_training_avg'] and details['distance_from_training_avg'] > 3 * training_avg:
        warnings.append(
            f"MW deviates significantly from training average ({training_avg:.2f})."
        )

    process_corrections(corrections_mci, 'MCI')
    process_corrections(corrections_logkow, 'LogKow')

    corrections_present = bool(corrections_mci) or bool(corrections_logkow)
    if in_ad and corrections_present and 'with caution' not in status:
        status = 'IN AD (with caution): MW within domain; correction factors applied.'

    if metal_warning:
        warnings.append(metal_warning + ' Consider neutral (non-salt) form for interpretation.')
        if 'with caution' not in status and in_ad:
            status = 'IN AD (with caution): metal counter-ion removed for estimation.'

    if in_ad and warnings and 'with caution' not in status:
        status = 'IN AD (with caution): see warnings.'

    return {
        'in_ad': in_ad,
        'status': status,
        'warnings': warnings,
        'details': details,
    }


def get_module_info() -> Dict:
    """Provide descriptive metadata for the KOCWIN module."""
    return {
        'module_name': 'KOCWIN',
        'description': 'Soil sorption coefficient estimation (MCI and logKow correlations).',
        'training_set': TRAINING_SET,
        'validation_set': VALIDATION_SET,
        'primary_ad_criteria': [
            'Molecular weight within training/validation ranges',
            'Correction factors consistent with Appendix A counts',
        ],
        'notes': 'Molecular weight outside validation bounds or novel correction factors may reduce reliability.',
    }
