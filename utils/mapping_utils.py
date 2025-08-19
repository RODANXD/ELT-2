import re
from typing import List, Optional
from fuzzywuzzy import process

def normalize_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    v = str(value).lower().strip()
    # remove extra punctuation
    v = re.sub(r"[^a-z0-9 ]+", " ", v)
    v = re.sub(r"\s+", " ", v)
    return v

def fuzzy_match_value_to_list(value: str, candidates: List[str], threshold: int = 70) -> Optional[str]:
    """
    Return best candidate from list for a given value using fuzzy matching.
    If best score < threshold return None.
    """
    if not value or not candidates:
        return None
    # Normalize candidates to preserve original label mapping
    normalized_map = {normalize_text(c): c for c in candidates}
    norm_value = normalize_text(value)
    best = process.extractOne(norm_value, list(normalized_map.keys()))
    if not best:
        return None
    match_key, score = best[0], best[1]
    if score >= threshold:
        return normalized_map.get(match_key)
    return None

def normalize_unit(unit):
    """
    Normalize unit strings to canonical forms as in DE1_Unit (e.g., 'm3' -> 'm³').
    """
    mapping = {
        'm3': 'm³',
        'm^3': 'm³',
        'M3': 'm³',
        'kwh': 'kWh',
        'kw': 'kW',
        # Add more as needed
    }
    u = str(unit).strip().lower().replace('^3', '3')
    return mapping.get(u, unit)

import re
def extract_unit_from_column(col_name):
    """
    Extract unit from column names like 'consumption_m3', 'per_m3', 'Volume (m3)', etc.
    """
    match = re.search(r'(?:per[_-]?|_|\()([a-zA-Z0-9³^]+)(?:\)|$)', col_name)
    if match:
        return normalize_unit(match.group(1))
    return None

