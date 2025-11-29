# EPI Suite Reference Data Files

This directory contains reference/test set files from EPI Suite for validation and applicability domain assessment.

## MPBPWIN Reference Files

### Melting_Pt_TestSet.xls
- **Property**: Melting Point
- **Size**: ~3.5 MB
- **Description**: Test set for melting point predictions
- **Contains**: Experimental melting points and MPBPWIN estimates
- **Note**: Contains >15× more compounds than the original Joback training set (388 compounds)

### Boiling_Pt_TestSet.xls
- **Property**: Boiling Point
- **Size**: ~1.6 MB
- **Description**: Test set for boiling point predictions
- **Contains**: Experimental boiling points and MPBPWIN estimates

### VaporPressure_TestSet.xls
- **Property**: Vapor Pressure
- **Size**: ~1.0 MB
- **Description**: Test set for vapor pressure predictions
- **Contains**: Experimental vapor pressures and MPBPWIN estimates

## Usage

These files are referenced by the AD rules modules to provide context about:
- Test set size and coverage
- Molecular weight distributions
- Prediction accuracy statistics

### Accessing in Code

```python
from app.episuite.ad_rules import mpbpwin_ad

# Get module info with reference file paths
info = mpbpwin_ad.get_module_info()
print(info['reference_data'])

# Load statistics from test sets (optional, requires xlrd)
stats = mpbpwin_ad.load_reference_statistics()
if stats:
    print(f"Melting Point test set: {stats['melting_point']['num_compounds']} compounds")
```

## Important Notes

1. **Training vs Test Sets**:
   - Original training sets for MPBPWIN are NOT available
   - These are TEST sets used for validation
   - Test sets are typically larger than training sets

2. **File Format**:
   - Files are in .xls format (old Excel format)
   - Requires `xlrd` package to read: `pip install xlrd`
   - AD assessment works without reading files (uses conservative approach)

3. **Applicability Domain**:
   - MPBPWIN AD cannot be precisely defined due to lack of training data
   - These test sets provide context for expected property ranges
   - Significant prediction errors are possible per EPI Suite documentation

## Source

Downloaded from EPI Suite website as part of the complete methodology documentation.
Per EPA documentation: "The current applicability of the MPBPWIN methodology is best
described by its accuracy in predicting [properties]."
