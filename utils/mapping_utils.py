import re
from typing import List, Optional
import logging
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
    Extract unit from column names like 'consumption_m3', 'per_m3', 'Volume (m3)', 'kWh_Used', 'kWh_Used'
    Tries multiple heuristics: bracketed units, 'per_' patterns, suffixes like '_m3', and prefixes like 'kWh_...'.
    """
    if not isinstance(col_name, str):
        return None

    # 1) Bracketed units: 'Volume (m3)'
    br = re.search(r'\(([^)]+)\)', col_name)
    if br:
        logging.info(f"extract_unit_from_column: bracket match '{br.group(1)}' from '{col_name}'")
        return normalize_unit(br.group(1))

    # 2) 'per_' style: 'EmissionFactor_kgCO2e_per_m3'
    per = re.search(r'(?i)per[_-]?([a-z0-9³^]+)', col_name)
    if per:
        logging.info(f"extract_unit_from_column: per-match '{per.group(1)}' from '{col_name}'")
        return normalize_unit(per.group(1))

    # 3) Decide between prefix and suffix when both present
    suf_match = re.search(r'(?i)[_-]([a-z0-9³^]+)$', col_name)
    pre_match = re.search(r'(?i)^([a-z0-9³^]+)[_-]', col_name)

    # Heuristic: consider a token a likely unit if it is short (<=4), contains digits (e.g., m3) or is a known unit keyword
    unit_keywords = {"kwh","mwh","wh","kw","mw","m3","m^3","m³","kg","t","ton","tonne","l","lt","liter","litre","km","mi"}

    def is_unit_token(tok: str) -> bool:
        if not tok:
            return False
        t = tok.lower()
        if t in unit_keywords:
            return True
        if re.search(r"\d", t):
            return True
        if len(t) <= 4 and re.match(r'^[a-z]{1,4}$', t):
            # short tokens like 'kwh','kw','m3' — treat as possible units
            return True
        return False

    # Prefer bracketed/per patterns already handled above. Now choose prefix vs suffix.
    if pre_match and suf_match:
        pre_tok = pre_match.group(1)
        suf_tok = suf_match.group(1)
        logging.info(f"extract_unit_from_column: pre_tok='{pre_tok}', suf_tok='{suf_tok}' for '{col_name}'")
        pre_is_unit = is_unit_token(pre_tok)
        suf_is_unit = is_unit_token(suf_tok)
        logging.info(f"extract_unit_from_column: is_unit_token -> pre:{pre_is_unit}, suf:{suf_is_unit}")
        if pre_is_unit:
            selected = pre_tok
            logging.info(f"extract_unit_from_column: selected prefix '{selected}'")
            return normalize_unit(selected)
        if suf_is_unit:
            selected = suf_tok
            logging.info(f"extract_unit_from_column: selected suffix '{selected}'")
            return normalize_unit(selected)
        # fallback to suffix
        logging.info(f"extract_unit_from_column: fallback to suffix '{suf_tok}'")
        return normalize_unit(suf_tok)

    # 4) Suffix style: 'consumption_m3' or '_m3' at end
    if suf_match:
        logging.info(f"extract_unit_from_column: suffix-only match '{suf_match.group(1)}' from '{col_name}'")
        return normalize_unit(suf_match.group(1))

    # 5) Prefix style: 'kWh_Used' -> capture 'kWh'
    if pre_match:
        logging.info(f"extract_unit_from_column: prefix-only match '{pre_match.group(1)}' from '{col_name}'")
        return normalize_unit(pre_match.group(1))

    return None

def extract_currency_from_column(col_name):
    """
    Extract currency code from column names like 'amount_USD', 'price_EUR', 'Total (GBP)', etc.
    """
    # Common currency patterns in column names
    currency_patterns = [
        r'(?:amount|price|cost|total|paid|expense|spend|value|fee|charge)[_-]?([A-Z]{3})',  # amount_USD, price-EUR
        r'\(([A-Z]{3})\)',  # (USD), (EUR)
        r'_([A-Z]{3})$',    # _USD, _EUR at end
        r'_([A-Z]{3})_',    # _USD_ in middle
        r'([A-Z]{3})_',     # USD_ at start
        r'([A-Z]{3})$',     # USD at end
    ]
    
    for pattern in currency_patterns:
        match = re.search(pattern, col_name, re.IGNORECASE)
        if match:
            currency_code = match.group(1).upper()
            # Validate that it's a 3-letter currency code
            if len(currency_code) == 3 and currency_code.isalpha():
                return currency_code
    # Additional pattern: look for parentheses with currency and optional surrounding text, e.g., 'TotalPaid(EUR)'
    m = re.search(r'\(([A-Z]{3})\)', col_name)
    if m:
        return m.group(1).upper()
    return None


def extract_unit_from_value(value):
    """
    Extract unit token from a data value string (e.g., '1484.8 kWh' -> 'kWh').
    Returns normalized unit via `normalize_unit` or None if not found.
    """
    if value is None:
        return None
    s = str(value)
    # Common unit tokens (case-insensitive). Add more as needed.
    unit_tokens = [
        'kwh','mwh','wh','kw','mw',
        'm3','m^3','m³',
        'kg','t','ton','tonne',
        'l','lt','liter','litre',
        'km','mi'
    ]
    # try word-boundary search first (handles '1484.8 kWh')
    pat = re.compile(r'(?i)\b(' + '|'.join([re.escape(t) for t in unit_tokens]) + r')\b')
    m = pat.search(s)
    if m:
        return normalize_unit(m.group(1))

    # try suffix/prefix without space, e.g., '1484.8kWh'
    pat2 = re.compile(r'(?i)(' + '|'.join([re.escape(t) for t in unit_tokens]) + r')\b')
    m2 = pat2.search(s)
    if m2:
        return normalize_unit(m2.group(1))

    return None

