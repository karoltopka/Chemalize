# EPI Suite Applicability Domain Rules

This directory contains individual AD (Applicability Domain) rules for each EPI Suite module.

## Structure

Each module has its own Python file defining:
- Domain parameters (MW ranges, training set statistics, etc.)
- `check_applicability_domain()` function
- `get_module_info()` function for module metadata

## Current Modules

### ✅ KOWWIN (kowwin_ad.py)
- **Property**: Octanol-Water Partition Coefficient (Log Kow)
- **Training Set**: 2,447 compounds (MW: 18.02 - 719.92)
- **Validation Set**: 10,946 compounds (MW: 27.03 - 991.15)
- **Primary AD Criteria**: Molecular Weight

### ✅ BIOWIN (biowin_ad.py)
- **Property**: Biodegradation Probability (Biowin1–Biowin6, including MITI models)
- **Training Set**: 295 compounds (MW: 31.06 - 697.7)
- **Primary AD Criteria**: Molecular Weight, fragment counts ≤ training maxima
- **Notes**: Fragment maxima sourced from Appendix A; MITI-specific fragments are tracked for presence only (no published maxima)

### ✅ BCFBAF (bcfbaf_ad.py)
- **Property**: Bioconcentration & Bioaccumulation (regression BCF/BAF) plus whole-body biotransformation rates
- **Training Set**: BCF (527 compounds; MW: 68.08 - 991.80, Log Kow: -6.50 - 11.26), Biotransformation (421 compounds; MW: 68.08 - 959.17, Log Kow: 0.31 - 8.70)
- **Primary AD Criteria**: Molecular Weight & Log Kow within respective training ranges, correction factors documented, fragment counts ≤ Appendix D maxima
- **Notes**: Ionic/non-ionic thresholds flagged (e.g., Log Kow < -1.37 or MW > 959.17 g/mol). Fragment maxima sourced from Appendix D; caution for MW >600 g/mol or ionizable/metallic/perfluorinated substances

### ✅ KOCWIN (kocwin_ad.py)
- **Property**: Soil sorption coefficient (Koc) via MCI and log Kow correlations
- **Training Set**: 447 compounds (MW: 32.04 - 665.02)
- **Validation Set**: MW: 73.14 - 504.12
- **Primary AD Criteria**: Molecular weight within training/validation bounds; review correction factors vs. Appendix A maxima
- **Notes**: Out-of-range MW or novel correction factors may degrade accuracy; validation-only MW flagged with caution

### 🔄 Pending Modules

#### MPBPWIN (mpbpwin_ad.py)
- Melting Point
- Boiling Point
- Vapor Pressure

#### AOPWIN (aopwin_ad.py)
- Atmospheric Oxidation (OH radicals, Ozone)

#### HYDROWIN (hydrowin_ad.py)
- Hydrolysis Rate

#### HENRY (henry_ad.py)
- Henry's Law Constant

#### WSKOW (wskow_ad.py)
- Water Solubility

## Adding New Module Rules

To add a new module:

1. Create `{module_name}_ad.py` in this directory
2. Define domain parameters (constants)
3. Implement `check_applicability_domain()` function:
   ```python
   def check_applicability_domain(data_dict):
       return {
           'in_ad': bool,
           'status': str,
           'warnings': list,
           'details': dict
       }
   ```
4. Implement `get_module_info()` function
5. Update module's parser to use the AD rules

## Usage Example

```python
from app.episuite.ad_rules import kowwin_ad

# Check applicability domain
result = kowwin_ad.check_applicability_domain(
    molecular_weight=408.58,
    fragments=fragment_list
)

if result['in_ad']:
    print(f"✅ {result['status']}")
else:
    print(f"❌ {result['status']}")
    for warning in result['warnings']:
        print(f"  ⚠️  {warning}")
```
