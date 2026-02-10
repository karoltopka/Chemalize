# coding: utf-8
"""
coloring.py — przygotowanie danych do kolorowania wykresów (PCA i nie tylko)

Funkcje publiczne:
- load_coloring_file(path)
- detect_key_column(df_main, df_color)
- validate_coloring_setup(df_main, df_color, key_column=None)
- prepare_color_data(df_main, df_color, key_column, color_column)
- generate_distinct_colors(n_colors)
- get_color_scheme_for_numeric(colormap_name='Viridis')
"""

from __future__ import annotations

import os
import re
import random
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap


# ===========================
#  Helpers: parsing/liczby
# ===========================

def _coerce_numeric_series(s: pd.Series, min_numeric_ratio: float = 0.3) -> Tuple[pd.Series, bool]:
    """
    Spróbuj zamienić serię na numeric z obsługą:
      - przecinka dziesiętnego (PL),
      - separatorów tysięcy (spacje, NBSP, wąskie spacje, apostrofy, kropki),
      - sufiksów % (przeliczenie na ułamek),
      - nawiasów oznaczających liczby ujemne: (123) -> -123.
    Zwraca (nowa_seria, is_numeric).
    """
    if pd.api.types.is_numeric_dtype(s):
        return s, True
    if s.dropna().empty:
        return s, False

    # CRITICAL: Check for decimal comma FIRST, on original non-null values
    # This detects Polish format (1,98) regardless of how many N/A there are
    original_non_null = s.dropna().astype(str)
    has_decimal_comma = original_non_null.str.contains(r'\d+,\d+', na=False, regex=True).any()

    raw = s.astype(str)

    # ochrona przed ID z wiodącym zerem
    lead_zero_ratio = (raw.dropna().str.match(r"^0\d+$")).mean()
    if lead_zero_ratio >= 0.5:
        return s, False

    cleaned = raw.str.replace("\u00A0", " ", regex=False)\
                 .str.replace("\u202F", " ", regex=False)\
                 .str.strip()

    # nawiasy ujemne
    cleaned = cleaned.str.replace(r"^\(\s*(.*)\s*\)$", r"-\1", regex=True)
    # separatory tysięcy: spacje, apostrofy
    cleaned = cleaned.str.replace(r"[ '\u2019]", "", regex=True)
    # kropka jako tysiące (1.234 lub 1.234,56)
    cleaned = cleaned.str.replace(r"(?<=\d)\.(?=\d{3}(\D|$))", "", regex=True)

    has_percent = cleaned.str.endswith("%")
    cleaned_no_pct = cleaned.str.replace("%", "", regex=False)

    looks_pl = cleaned_no_pct.str.contains(r"^-?\d+(?:\.\d{3})*(?:,\d+)?$", na=False, regex=True)
    cleaned_pl = cleaned_no_pct.where(~looks_pl, cleaned_no_pct.str.replace(",", ".", regex=False))
    num_pl = pd.to_numeric(cleaned_pl, errors="coerce")
    num_pl = num_pl.where(~has_percent, num_pl / 100.0)
    ratio_pl = num_pl.notna().mean()

    # If decimal comma detected, force numeric treatment (ignore ratio threshold)
    if has_decimal_comma:
        if ratio_pl > 0:
            print(f"DEBUG: Detected decimal comma, forcing numeric (ratio={ratio_pl:.2%})")
            return num_pl.reindex(s.index), True
        else:
            print(f"DEBUG: Detected decimal comma but ratio_pl=0, trying fallback")
            # Fallback: try simple comma-to-dot replacement
            simple_replace = s.dropna().astype(str).str.replace(',', '.', regex=False)
            simple_numeric = pd.to_numeric(simple_replace, errors='coerce')
            if simple_numeric.notna().any():
                print(f"DEBUG: Fallback succeeded, ratio={simple_numeric.notna().mean():.2%}")
                return simple_numeric.reindex(s.index), True

    if ratio_pl >= min_numeric_ratio:
        out = num_pl
    else:
        num_en = pd.to_numeric(cleaned_no_pct, errors="coerce")
        num_en = num_en.where(~has_percent, num_en / 100.0)
        ratio_en = num_en.notna().mean()
        if ratio_en >= min_numeric_ratio:
            out = num_en
        else:
            return s, False

    return out.reindex(s.index), True


def _normalize_key_series(s: pd.Series) -> pd.Series:
    """
    Ujednolica wartości klucza do porównania/łączenia.
    Ważne:
    - zachowuje braki danych jako NA (nie zamienia ich na string "nan"),
    - czyści białe znaki (w tym NBSP),
    - normalizuje liczby (np. 1, 1.0, "1,0" -> "1"),
      ale zostawia identyfikatory z wiodącym zerem ("0012") bez zmian.
    """
    if s is None:
        return s

    norm = s.astype("string")\
            .str.replace("\u00A0", " ", regex=False)\
            .str.replace("\u202F", " ", regex=False)\
            .str.strip()

    lower = norm.str.lower()
    missing_tokens = {"", "nan", "none", "null", "nat", "n/a", "na"}
    missing_mask = norm.isna() | lower.isin(missing_tokens)
    norm = norm.mask(missing_mask)

    if not norm.notna().any():
        return norm.astype(object)

    numeric_candidate = norm.str.replace(",", ".", regex=False)
    numeric_values = pd.to_numeric(numeric_candidate, errors="coerce")
    lead_zero_mask = norm.str.match(r"^0\d+$", na=False)
    numeric_mask = numeric_values.notna() & norm.notna() & ~lead_zero_mask

    out = norm.astype(object)
    if numeric_mask.any():
        numeric_subset = numeric_values.loc[numeric_mask]
        rounded = np.round(numeric_subset.values)
        is_int = np.isclose(numeric_subset.values, rounded, atol=1e-12)

        int_idx = numeric_subset.index[is_int]
        float_idx = numeric_subset.index[~is_int]

        if len(int_idx) > 0:
            out.loc[int_idx] = (
                np.round(numeric_values.loc[int_idx]).astype(np.int64).astype(str)
            )
        if len(float_idx) > 0:
            out.loc[float_idx] = numeric_values.loc[float_idx].map(lambda x: format(float(x), ".15g"))

    return pd.Series(out, index=s.index, dtype=object)


# ===========================
#  Helpers: JSON sanitization
# ===========================

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1F\x7F]")

def _sanitize_json_value(v, max_str_len: int = 200):
    """
    Zwraca wartość bezpieczną dla JSON:
    - liczby: float/int, niefinity -> None
    - NaN/None -> None
    - inne typy -> string bez znaków sterujących, przycięty do max_str_len
    """
    if v is None:
        return None
    # pandas / numpy NaN
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    # liczby
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not np.isfinite(f):
            return None
        return f

    # wszystko inne jako string
    s = str(v)
    # usuń znaki sterujące (mogą rozjechać JSON/HTML)
    s = _CONTROL_CHARS_RE.sub("", s)
    if len(s) > max_str_len:
        s = s[: max_str_len - 1] + "…"
    return s


def _sanitize_series_for_json(s: pd.Series, max_str_len: int = 200) -> pd.Series:
    """
    Zwraca serię dtype=object z wartościami w pełni serializowalnymi do JSON.
    """
    return s.apply(lambda v: _sanitize_json_value(v, max_str_len))


# ===========================
#  Dopasowanie klucza
# ===========================

def check_column_overlap(df_main: pd.DataFrame, df_color: pd.DataFrame, key_column: str) -> float:
    if key_column not in df_main.columns or key_column not in df_color.columns:
        return 0.0
    left = _normalize_key_series(df_main[key_column]).dropna().unique()
    right = _normalize_key_series(df_color[key_column]).dropna().unique()
    if len(left) == 0 or len(right) == 0:
        return 0.0
    set_left = set(left)
    set_right = set(right)
    inter = set_left & set_right
    denom = max(len(set_left), 1)
    return len(inter) / denom


def detect_key_column(df_main: pd.DataFrame, df_color: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect the best key column for joining, prioritizing uniqueness.
    Returns the column with highest uniqueness score among all PCA columns.
    """
    # Get ALL non-PC columns from PCA data (not just common ones)
    pca_candidates = [c for c in df_main.columns if not str(c).startswith("PC")]
    common_columns = [c for c in (set(df_main.columns) & set(df_color.columns)) if not str(c).startswith("PC")]

    if not pca_candidates:
        return {'found': False, 'key_column': None, 'pca_candidates': [], 'common_columns': [], 'confidence': 'none'}

    # Score only common columns by overlap + uniqueness.
    # Non-common columns cannot be used as join keys.
    candidates_info = []
    for col in common_columns:
        try:
            left_norm = _normalize_key_series(df_main[col])
            right_norm = _normalize_key_series(df_color[col])

            uniq_left = left_norm.nunique(dropna=True) / max(left_norm.notna().sum(), 1)
            uniq_right = right_norm.nunique(dropna=True) / max(right_norm.notna().sum(), 1)
            uniqueness = min(uniq_left, uniq_right)

            overlap = check_column_overlap(df_main, df_color, col)

            # Overlap is primary signal; uniqueness helps break ties.
            score = overlap * 0.75 + uniqueness * 0.25

            candidates_info.append({
                'column': col,
                'uniqueness': uniqueness,
                'overlap': overlap,
                'score': score,
                'is_common': True
            })
        except Exception:
            continue

    # Sort by score (highest first)
    candidates_info.sort(key=lambda x: x['score'], reverse=True)

    if not candidates_info:
        return {'found': False, 'key_column': None, 'pca_candidates': pca_candidates, 'common_columns': common_columns, 'confidence': 'none'}

    # Best candidate
    best = candidates_info[0]

    # Auto-select only when overlap suggests reliable mapping.
    auto_select = (
        best['overlap'] >= 0.60 or
        (best['overlap'] >= 0.40 and best['uniqueness'] >= 0.80)
    )

    return {
        'found': auto_select,
        'key_column': best['column'] if auto_select else None,
        'pca_candidates': pca_candidates,
        'common_columns': common_columns,
        'candidates_info': candidates_info,  # All candidates with scores
        'confidence': 'high' if best['overlap'] >= 0.75 else ('medium' if best['overlap'] >= 0.50 else 'low'),
        'uniqueness': best['uniqueness'],
        'overlap_ratio': best['overlap']
    }


# ===========================
#  Wczytywanie pliku kolorów
# ===========================

def load_coloring_file(path: str) -> Optional[pd.DataFrame]:
    """
    Wczytuje CSV/XLSX i od razu poprawia dtype:
    - „tekstowe liczby" -> numeric (jeśli ≥ min_numeric_ratio da się sparsować),
    - reszta zostaje, ale później i tak czyścimy pod JSON.
    """
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            print(f"Nieobsługiwany format pliku: {ext}")
            return None

        # Normalize column names: remove newlines, tabs, and extra whitespace
        df.columns = df.columns.str.replace(r'[\n\r\t]+', ' ', regex=True).str.strip()

        # próba konwersji każdej kolumny
        for col in df.columns:
            new_s, is_num = _coerce_numeric_series(df[col])
            if is_num:
                df[col] = new_s

        print(f"Załadowano plik: {df.shape[0]} wierszy, {df.shape[1]} kolumn")
        return df
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku: {e}")
        return None


# ===========================
#  Przygotowanie danych do kolorowania
# ===========================

def prepare_color_data(
    df_main: pd.DataFrame,
    df_color: pd.DataFrame,
    key_column: str,
    color_column: str
) -> Dict[str, Any]:
    """
    Łączy df_main i df_color po key_column i zwraca serię color_column
    ustawioną w kolejności df_main.
    - „dobija” kolumnę do numeric, jeśli to ma sens
    - ZAWSZE zwraca serię oczyszczoną pod JSON (None zamiast NaN/Inf, bez znaków sterujących)
    """
    try:
        if key_column not in df_main.columns:
            return {'success': False, 'data': None, 'message': f"Kolumna klucza '{key_column}' nie istnieje w danych PCA."}
        if key_column not in df_color.columns:
            return {'success': False, 'data': None, 'message': f"Kolumna klucza '{key_column}' nie istnieje w pliku kolorów."}
        if color_column not in df_color.columns:
            return {'success': False, 'data': None, 'message': f"Kolumna kolorowania '{color_column}' nie istnieje w pliku kolorów."}

        left = df_main.copy()
        right = df_color.copy()
        left['_key_norm_'] = _normalize_key_series(left[key_column])
        right['_key_norm_'] = _normalize_key_series(right[key_column])

        # Never join on missing keys; this avoids accidental NA<->NA matches.
        right = right[right['_key_norm_'].notna()].copy()

        right_col_coerced, right_col_is_num = _coerce_numeric_series(right[color_column])
        if right_col_is_num:
            right[color_column] = right_col_coerced

        # If the color file has duplicated keys:
        # - numeric endpoint: average duplicates,
        # - categorical endpoint: mode (fallback to first non-null).
        if right['_key_norm_'].duplicated().any():
            if right_col_is_num:
                right = right.groupby('_key_norm_', as_index=False)[color_column].mean()
            else:
                def _mode_or_first(x: pd.Series):
                    non_null = x.dropna()
                    if non_null.empty:
                        return np.nan
                    mode_vals = non_null.mode()
                    if not mode_vals.empty:
                        return mode_vals.iloc[0]
                    return non_null.iloc[0]
                right = right.groupby('_key_norm_', as_index=False)[color_column].agg(_mode_or_first)
        else:
            right = right.drop_duplicates(subset=['_key_norm_'], keep='last')

        merged = pd.merge(
            left,
            right[['_key_norm_', color_column]],
            on='_key_norm_',
            how='left',
            suffixes=('', '_color')
        )

        color_data = merged[color_column]
        matched = color_data.notna().sum()
        total = len(merged)
        match_ratio = matched / max(total, 1)

        # Heurystyczna próba konwersji na numeric
        coerced, is_num = _coerce_numeric_series(color_data)
        if is_num:
            color_data = coerced

        # --- JSON sanitization ---
        if is_num or pd.api.types.is_numeric_dtype(color_data):
            # liczby: usuń niefinity, NaN -> None; rzut na object, by None przeszło do listy
            cd = pd.to_numeric(color_data, errors="coerce")
            cd = cd.where(np.isfinite(cd), np.nan)
            color_data_json = cd.astype(object).where(~cd.isna(), None)
            is_numeric = True
            n_categories = 0
        else:
            # kategorie: string bez znaków sterujących, docięty; NaN -> None
            color_data_json = _sanitize_series_for_json(color_data.astype(object))
            # None nie liczymy jako unikalną kategorię
            n_categories = pd.Series([v for v in color_data_json.tolist() if v is not None]).nunique()
            is_numeric = False

        message = f"Pomyślnie dopasowano {matched}/{total} próbek"
        if not is_numeric and n_categories > 150:
            message += f". Uwaga: {n_categories} kategorii — legenda może być nieczytelna."

        return {
            'success': True,
            # UWAGA: 'data' jest już JSON-bezpieczna (dtype=object),
            # więc .tolist() w backendzie nie zwróci NaN/Inf ani znaków sterujących.
            'data': color_data_json,
            'is_numeric': is_numeric,
            'n_categories': n_categories,
            'matched': matched,
            'total': total,
            'match_ratio': match_ratio,
            'message': message
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'data': None, 'message': f"Błąd podczas przygotowania danych: {e}"}


# ===========================
#  Palety kolorów
# ===========================

def generate_distinct_colors(n_colors: int) -> List[str]:
    """
    Generate N visually distinct colors for categorical data.
    Uses qualitative colormaps to maximize color distinction.
    """
    colors: List[str] = []
    if n_colors <= 0:
        return colors

    # Best qualitative colormaps with distinct colors
    qualitative = [
        ('tab20', 20),      # 20 distinct colors
        ('tab20b', 20),     # 20 more distinct colors
        ('tab20c', 20),     # 20 more distinct colors
        ('Set3', 12),       # 12 pastel colors
        ('Paired', 12),     # 12 paired colors
        ('Set1', 9),        # 9 bright colors
        ('Dark2', 8),       # 8 dark colors
        ('Set2', 8),        # 8 medium colors
        ('Accent', 8),      # 8 accent colors
    ]

    idx = 0
    while len(colors) < n_colors and idx < len(qualitative):
        cmap_name, cmap_size = qualitative[idx]
        try:
            cmap: ListedColormap = cm.get_cmap(cmap_name)  # type: ignore
            # Take ALL colors from this colormap (evenly spaced)
            for i in range(cmap_size):
                if len(colors) >= n_colors:
                    break
                # Sample evenly across the colormap
                rgba = cmap(i / max(cmap_size - 1, 1))
                hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in rgba[:3])
                colors.append(hex_color)
        except Exception:
            pass
        idx += 1

    # If still need more colors, generate random distinct ones
    while len(colors) < n_colors:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        colors.append('#%02x%02x%02x' % (r, g, b))

    return colors[:n_colors]


def get_color_scheme_for_numeric(colormap_name: str = 'viridis') -> List[str]:
    try:
        cmap = cm.get_cmap(colormap_name)
    except Exception:
        cmap = cm.get_cmap('viridis')
    steps = 256
    out = []
    for i in range(steps):
        rgba = cmap(i / (steps - 1))
        out.append('#%02x%02x%02x' % tuple(int(c * 255) for c in rgba[:3]))
    return out


# ===========================
#  Walidacja setupu
# ===========================

def validate_coloring_setup(
    df_main: pd.DataFrame,
    df_color: pd.DataFrame,
    key_column: Optional[str] = None
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        'valid': False,
        'available_color_columns': [],
        'warnings': [],
        'recommendations': []
    }

    if df_main is None or df_color is None:
        result['warnings'].append("Brak danych wejściowych.")
        return result

    # Auto-detekcja klucza (gdy nie podany)
    if not key_column:
        detection = detect_key_column(df_main, df_color)
        result['auto_detected'] = detection.get('found', False)
        result['key_column'] = detection.get('key_column')
        result['common_columns'] = detection.get('common_columns', [])
        if detection['found']:
            if detection.get('confidence') == 'high':
                result['recommendations'].append(
                    f"Automatycznie wykryto kolumnę klucza: '{detection['key_column']}' "
                    f"(pokrycie ≈ {detection.get('overlap_ratio', 0)*100:.1f}%)."
                )
            else:
                result['warnings'].append(
                    f"Potencjalny klucz: '{detection['key_column']}' (pewność {detection.get('confidence')}). "
                    f"Zweryfikuj, czy to właściwa kolumna."
                )
            key_column = detection['key_column']
        else:
            if detection.get('common_columns'):
                result['warnings'].append(
                    "Znaleziono wspólne kolumny: " + ", ".join(detection['common_columns']) +
                    ". Wybierz ręcznie kolumnę klucza."
                )
            else:
                result['warnings'].append(
                    "Brak wspólnych kolumn między plikami. "
                    "Upewnij się, że oba pliki mają kolumnę z identyfikatorami próbek."
                )
            return result
    else:
        result['key_column'] = key_column

    # Walidacja kolumny klucza
    if key_column not in df_main.columns:
        result['warnings'].append(f"Kolumna klucza '{key_column}' nie istnieje w danych PCA.")
        return result
    if key_column not in df_color.columns:
        result['warnings'].append(f"Kolumna klucza '{key_column}' nie istnieje w pliku kolorów.")
        return result

    # Overlap klucza
    overlap = check_column_overlap(df_main, df_color, key_column)
    if overlap < 0.3:
        result['warnings'].append(
            f"Niskie pokrycie wartości klucza między plikami (≈ {overlap*100:.1f}%). "
            "Wizualizacja może mieć dużo braków."
        )
    elif overlap < 0.6:
        result['recommendations'].append(
            f"Średnie pokrycie klucza (≈ {overlap*100:.1f}%). Rozważ oczyszczenie identyfikatorów."
        )
    else:
        result['recommendations'].append(
            f"Dobre pokrycie klucza (≈ {overlap*100:.1f}%)."
        )

    # Lista kolumn do kolorowania + typy
    available: List[str] = []
    column_type_map: Dict[str, str] = {}
    for col in df_color.columns:
        if col == key_column:
            continue
        _, is_num = _coerce_numeric_series(df_color[col])
        available.append(col)
        column_type_map[col] = "numeric" if is_num else "categorical"

        if not is_num:
            n_unique = df_color[col].nunique(dropna=True)
            if n_unique > 150:
                result['warnings'].append(
                    f"Kolumna '{col}' ma {n_unique} unikalnych wartości — legenda może być nieczytelna."
                )
            elif n_unique == 1:
                result['warnings'].append(
                    f"Kolumna '{col}' ma tylko jedną unikalną wartość — mało użyteczna do kolorowania."
                )

    result['available_color_columns'] = available
    result['column_type_map'] = column_type_map
    result['valid'] = True
    return result
