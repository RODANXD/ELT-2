import re
import pandas as pd
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


# Currency symbol -> ISO code mapping (used to detect currencies like 'amount_€' or '€100')
_CURRENCY_SYMBOL_MAP = {
    '€': 'EUR',
    '$': 'USD',
    '£': 'GBP',
    '¥': 'JPY',
    '₹': 'INR',
    '₩': 'KRW',
    '₽': 'RUB',
    '₺': 'TRY',
    '₪': 'ILS',
    '₫': 'VND'
}

def build_currency_symbol_map(currency_df) -> dict:
    """Build a symbol -> ISO code map from a currency dimension DataFrame.

    Tries to find a symbol column (e.g., 'CurrencySymbol' or containing 'symbol'/'char') and a
    currency code column (e.g., 'CurrencyCode', 'Code'). Returns a mapping of symbol->ISO code.
    """
    if currency_df is None:
        return {}
    try:
        # pick currency code column
        code_col = None
        for c in currency_df.columns:
            if re.search(r'currencycode|currency_code|code|iso3|iso', c, re.I):
                code_col = c
                break
        # pick symbol column
        sym_col = None
        for c in currency_df.columns:
            if re.search(r'symbol|char|sign', c, re.I):
                sym_col = c
                break

        mapping = {}
        if code_col and sym_col:
            for _, row in currency_df.iterrows():
                sym = row.get(sym_col)
                code = row.get(code_col)
                if pd_notna := (sym is not None and str(sym).strip() != ''):
                    try:
                        sym_str = str(sym).strip()
                        code_str = str(code).strip().upper() if code is not None else None
                        if code_str and len(code_str) == 3:
                            mapping[sym_str] = code_str
                    except Exception:
                        continue
        return mapping
    except Exception:
        return {}


def _currency_code_from_symbol_in_text(text: str, currency_df=None) -> Optional[str]:
    """Return ISO currency code if any known currency symbol is found in text.

    If `currency_df` is provided, build the symbol map from it first (dynamic). Otherwise
    fall back to the built-in `_CURRENCY_SYMBOL_MAP`.
    """
    if not text or not isinstance(text, str):
        return None
    # Try dynamic map first
    try:
        if currency_df is not None:
            dyn_map = build_currency_symbol_map(currency_df)
            for sym, code in dyn_map.items():
                if sym and sym in text:
                    logging.info(f"_currency_code_from_symbol_in_text: Found dynamic symbol '{sym}' -> '{code}' in '{text}'")
                    return code
    except Exception:
        pass

    # Fallback to static map
    for sym, code in _CURRENCY_SYMBOL_MAP.items():
        if sym in text:
            logging.info(f"_currency_code_from_symbol_in_text: Found symbol '{sym}' -> '{code}' in '{text}'")
            return code
    return None

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
    Extract currency code from column names like 'amount_USD', 'price_EUR', 'Total (GBP)', 'TotalPaid(EUR)', etc.
    """
    if not isinstance(col_name, str):
        return None
    
    # Priority 1: Look for parentheses with currency codes first (most reliable)
    # This handles: TotalPaid(EUR), Amount(USD), (GBP), etc.
    parentheses_pattern = r'\(([A-Z]{3})\)'
    match = re.search(parentheses_pattern, col_name, re.IGNORECASE)
    if match:
        currency_code = match.group(1).upper()
        if len(currency_code) == 3 and currency_code.isalpha():
            logging.info(f"extract_currency_from_column: Found currency '{currency_code}' in column '{col_name}' using parentheses pattern")
            return currency_code
    
    # Priority 2: Look for currency codes at the end of column names
    # This handles: amount_USD, price_EUR, cost_GBP, etc.
    end_pattern = r'[_-]([A-Z]{3})$'
    match = re.search(end_pattern, col_name, re.IGNORECASE)
    if match:
        currency_code = match.group(1).upper()
        if len(currency_code) == 3 and currency_code.isalpha():
            logging.info(f"extract_currency_from_column: Found currency '{currency_code}' in column '{col_name}' using end pattern")
            return currency_code
    
    # Priority 3: Look for currency codes at the beginning of column names
    # This handles: USD_amount, EUR_price, etc.
    start_pattern = r'^([A-Z]{3})[_-]'
    match = re.search(start_pattern, col_name, re.IGNORECASE)
    if match:
        currency_code = match.group(1).upper()
        if len(currency_code) == 3 and currency_code.isalpha():
            logging.info(f"extract_currency_from_column: Found currency '{currency_code}' in column '{col_name}' using start pattern")
            return currency_code
    
    # Priority 4: Look for currency codes in the middle with underscores
    # This handles: amount_USD_total, price_EUR_net, etc.
    middle_pattern = r'[_-]([A-Z]{3})[_-]'
    match = re.search(middle_pattern, col_name, re.IGNORECASE)
    if match:
        currency_code = match.group(1).upper()
        if len(currency_code) == 3 and currency_code.isalpha():
            logging.info(f"extract_currency_from_column: Found currency '{currency_code}' in column '{col_name}' using middle pattern")
            return currency_code
    
    # Priority 5: Look for standalone currency codes at the very end
    # This handles: amountUSD, priceEUR, etc.
    standalone_pattern = r'([A-Z]{3})$'
    match = re.search(standalone_pattern, col_name, re.IGNORECASE)
    if match:
        currency_code = match.group(1).upper()
        if len(currency_code) == 3 and currency_code.isalpha():
            logging.info(f"extract_currency_from_column: Found currency '{currency_code}' in column '{col_name}' using standalone pattern")
            return currency_code

    # Priority 6: Look for currency symbols in the column name (e.g., amount_€, amount_$)
    try:
        sym_code = _currency_code_from_symbol_in_text(col_name)
        if sym_code:
            logging.info(f"extract_currency_from_column: Found currency symbol in column '{col_name}' -> '{sym_code}'")
            return sym_code
    except Exception:
        pass
    
    logging.info(f"extract_currency_from_column: No currency found in column '{col_name}'")
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


def extract_currency_from_value(value):
    """
    Extract currency code from data values like '100 EUR', '200USD', '150.50 GBP', etc.
    Returns the currency code if found, None otherwise.
    """
    if value is None:
        return None
    
    value_str = str(value).strip()
    
    # Pattern 1: "100 EUR", "200 USD" (number + space + currency)
    pattern1 = re.compile(r'\b(\d+(?:\.\d+)?)\s+([A-Z]{3})\b', re.IGNORECASE)
    match1 = pattern1.search(value_str)
    if match1:
        currency_code = match1.group(2).upper()
        if len(currency_code) == 3 and currency_code.isalpha():
            logging.info(f"extract_currency_from_value: Found currency '{currency_code}' in value '{value_str}' using space pattern")
            return currency_code
    
    # Pattern 2: "100EUR", "200USD" (number + currency without space)
    pattern2 = re.compile(r'\b(\d+(?:\.\d+)?)([A-Z]{3})\b', re.IGNORECASE)
    match2 = pattern2.search(value_str)
    if match2:
        currency_code = match2.group(2).upper()
        if len(currency_code) == 3 and currency_code.isalpha():
            logging.info(f"extract_currency_from_value: Found currency '{currency_code}' in value '{value_str}' using no-space pattern")
            return currency_code
    
    # Pattern 3: "EUR 100", "USD 200" (currency + space + number)
    pattern3 = re.compile(r'\b([A-Z]{3})\s+(\d+(?:\.\d+)?)\b', re.IGNORECASE)
    match3 = pattern3.search(value_str)
    if match3:
        currency_code = match3.group(1).upper()
        if len(currency_code) == 3 and currency_code.isalpha():
            logging.info(f"extract_currency_from_value: Found currency '{currency_code}' in value '{value_str}' using currency-first pattern")
            return currency_code
    
    # Pattern 4: Just the currency code itself (e.g., "EUR", "USD")
    pattern4 = re.compile(r'\b([A-Z]{3})\b', re.IGNORECASE)
    match4 = pattern4.search(value_str)
    if match4:
        currency_code = match4.group(1).upper()
        if len(currency_code) == 3 and currency_code.isalpha():
            logging.info(f"extract_currency_from_value: Found currency '{currency_code}' in value '{value_str}' using standalone pattern")
            return currency_code

    # Pattern 5: Look for currency symbols in the value (e.g., '€100', 'A$ 50')
    try:
        sym_code = _currency_code_from_symbol_in_text(value_str)
        if sym_code:
            logging.info(f"extract_currency_from_value: Found currency symbol in value '{value_str}' -> '{sym_code}'")
            return sym_code
    except Exception:
        pass
    
    logging.info(f"extract_currency_from_value: No currency found in value '{value_str}'")
    return None


def clean_provider_name(value: Optional[str]) -> Optional[str]:
    """
    Clean provider/supplier strings by removing common prefixes/suffixes and identifiers.

    Examples:
    - 'Acct:Vienna Power Co.#56775' -> 'Vienna Power Co.'
    - 'Acct:GreenWatt GmbH#82567' -> 'GreenWatt GmbH'
    - 'Provider - My Supplier (ID: 1234)' -> 'My Supplier'
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    # Normalize whitespace and common unicode dashes
    s = re.sub(r'[\u2012\u2013\u2014\u2015]', '-', s)
    s = re.sub(r'\s+', ' ', s).strip()

    # Remove common leading labels like 'Acct:', 'Account', 'Provider', 'Supplier'
    s = re.sub(r'^(?:acct(?:ount)?|account|supplier|provider)\s*[:\-–—]*\s*', '', s, flags=re.I)

    # Remove common trailing identifiers (e.g., '#12345', '(#12345)', 'ID:1234', '- 5678')
    trailing_patterns = [
        r'\s*#\s*\d+\s*$',
        r'\s*\(\s*#?\s*\d+\s*\)\s*$',
        r'\s*\bID[:\s]*\d+\s*$',
        r'\s*[-–—]\s*\d+\s*$',
        r'\s*ref[:\s]*\d+\s*$',
        r'\s*:\s*\d+\s*$',
        r'\s*\b\d{3,}\s*$'  # long trailing numbers
    ]

    for pat in trailing_patterns:
        s = re.sub(pat, '', s, flags=re.I).strip()

    # Remove surrounding noise characters
    s = s.strip(' -_\n\t\r:;,./\\()')

    # Collapse multiple spaces
    s = re.sub(r'\s{2,}', ' ', s)

    # Final fallback: if result is empty, return None
    return s if s else None

