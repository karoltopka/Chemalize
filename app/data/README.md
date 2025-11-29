# Application Data Directory

This directory contains static reference data and configuration files used by the ChemAlize application.

## Contents

### `Descriptors_group.txt`
**Purpose:** Alvadesk molecular descriptor group definitions

This file contains the complete list of Alvadesk molecular descriptors organized into 34 groups:
1. Constitutional indices
2. Ring descriptors
3. Topological indices
4. Walk and path counts
5. Connectivity indices
6. Information indices
7. 2D matrix-based descriptors
8. 2D autocorrelations
9. Burden eigenvalues
10. P_VSA-like descriptors
11. ETA indices
12. Edge adjacency indices
13. Geometrical descriptors
14. 3D matrix-based descriptors
15. 3D autocorrelations
16. RDF descriptors
17. 3D-MoRSE descriptors
18. WHIM descriptors
19. GETAWAY descriptors
20. Randic molecular profiles
21. Functional group counts
22. Atom-centred fragments
23. Atom-type E-state indices
24. Pharmacophore descriptors
25. 2D Atom Pairs
26. 3D Atom Pairs
27. Charge descriptors
28. Molecular properties
29. Drug-like indices
30. CATS 3D descriptors
31. WHALES descriptors
32. MDE descriptors
33. Chirality descriptors
34. SASA descriptors

**Used by:**
- `app/blueprints/alvadesk_pca.py` - Alvadesk PCA Analysis module
- `app/utils/descriptor_groups.py` - Descriptor group parser

**Format:** Plain text file with group headers and descriptor names

**Configuration:** Path defined in `app/config.py` as `DESCRIPTOR_GROUPS_FILE`

## Directory Purpose

This directory (`app/data/`) is for **static application data** that is:
- Part of the application codebase (version controlled)
- Read-only at runtime
- Shared across all users
- Not user-generated or temporary

**Do NOT confuse with:**
- `data/` (project root level) - User-specific runtime data, uploads, processing results
- `data/users/<user-id>/` - Per-user isolated data directories
- `flask_session/` - Session storage

## Adding New Reference Data

To add new static reference files to this directory:

1. Place the file in `app/data/`
2. Add a path constant in `app/config.py`:
   ```python
   MY_REFERENCE_FILE = os.path.join(APP_DATA_DIR, 'my_file.txt')
   ```
3. Import and use in your module:
   ```python
   from app.config import MY_REFERENCE_FILE
   ```
4. Document the file in this README

## Notes

- Files in this directory should be committed to version control
- Keep file sizes reasonable (< 10 MB preferred)
- For large datasets, consider external storage or lazy loading
- This directory is created automatically if it doesn't exist
