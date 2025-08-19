import pandas as pd
from typing import Dict, List, Any
import re

def infer_column_type(series: pd.Series) -> str:
    """Infer the semantic type of a column based on its content"""
    
    # Get sample non-null values
    sample = series.dropna().head(10)
    if sample.empty:
        return 'unknown'
        
    # Check for date patterns
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',
        r'\d{2}/\d{2}/\d{4}',
        r'\d{2}-\d{2}-\d{4}'
    ]
    
    if any(sample.astype(str).str.match(pat).any() for pat in date_patterns):
        return 'date'
        
    # Check for monetary values
    if series.dtype in ['float64', 'int64'] and any(col_name.lower() in series.name.lower() 
            for col_name in ['amount', 'cost', 'price', 'value', 'paid']):
        return 'monetary'
        
    # Check for quantity/measurement
    if series.dtype in ['float64', 'int64'] and any(col_name.lower() in series.name.lower() 
            for col_name in ['quantity', 'volume', 'weight', 'consumption']):
        return 'measurement'
        
    # Check for categorical
    if series.dtype == 'object' and series.nunique() < len(series) * 0.5:
        return 'categorical'
        
    return str(series.dtype)

def analyze_source_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze source data schema and suggest mappings"""
    
    analysis = {
        'column_types': {},
        'suggested_mappings': {},
        'potential_keys': []
    }
    
    # Analyze each column
    for col in df.columns:
        col_type = infer_column_type(df[col])
        analysis['column_types'][col] = col_type
        
        # Suggest potential mappings based on column characteristics
        if col_type == 'date':
            analysis['suggested_mappings'][col] = 'DateKey'
        elif col_type == 'monetary':
            analysis['suggested_mappings'][col] = 'PaidAmount'
        elif col_type == 'measurement':
            analysis['suggested_mappings'][col] = 'ConsumptionAmount'
        
        # Identify potential key columns
        if df[col].nunique() == len(df):
            analysis['potential_keys'].append(col)
            
    return analysis

import re

def _infer_currency_hints_from_headers(columns):
    """Find 3-letter ISO currency codes in column names (Paid_EUR, TotalPaidEUR, Amount-USD, etc.)."""
    hints = set()
    if not columns:
        return hints
    iso = {
        "USD","EUR","GBP","INR","JPY","CAD","AUD","CHF","CNY","SEK","NOK","DKK","PLN",
        "CZK","HUF","RUB","BRL","ZAR","MXN","NZD","SGD","HKD","AED","SAR","TRY","KRW",
        "TWD","IDR","THB","MYR"
    }
    pat = re.compile(r'(?i)(?:^|[_-])([A-Z]{3})(?:$|[_-])')
    for col in columns:
        upper = col.upper().replace(" ", "_")
        for code in iso:
            if (upper.endswith(code) or upper.endswith("_"+code) or
                upper.startswith(code+"_") or ("_"+code+"_") in upper):
                hints.add(code)
        m = pat.search(upper)
        if m and m.group(1).upper() in iso:
            hints.add(m.group(1).upper())
    return hints

def _infer_unit_hints_from_headers(columns):
    """Find consumption units from headers (Consumption_kWh, EmissionFactor_*_per_m3, Energy (kWh))."""
    hints = set()
    if not columns:
        return hints
    synonyms = {
        'kwh':'kwh','mwh':'mwh','wh':'wh','kw':'kw','mw':'mw','m3':'m3','m^3':'m3',
        'kg':'kg','t':'t','ton':'t','tonne':'t','tonnes':'t','l':'l','lt':'l',
        'liter':'l','litre':'l','liters':'l','litres':'l','km':'km','mi':'mi'
    }
    per_pat = re.compile(r'(?i)per[_-]([a-z0-9^]+)')
    suf_pat = re.compile(r'(?i)[_-]([a-z0-9^]+)$')
    br_pat  = re.compile(r'\(([^)]+)\)')  # Volume (m3)

    for col in columns:
        if not isinstance(col, str):
            continue
        low = col.lower().replace(" ", "_")

        # ... per_<unit>
        for m in per_pat.finditer(low):
            key = synonyms.get(m.group(1), m.group(1))
            hints.add(key)

        # ... suffix _unit or short suffix like _m3
        m = suf_pat.search(low)
        if m:
            key = synonyms.get(m.group(1), m.group(1))
            hints.add(key)

        # ... bracketed units
        for m in br_pat.finditer(col):
            key = m.group(1).lower()
            if ' per ' in key:
                last = key.split(' per ')[-1].strip()
                last = synonyms.get(last, last)
                hints.add(last)
            else:
                key = synonyms.get(key, key)
                hints.add(key)

        # extra heuristics: look for patterns like '_per_m3' or '_m3' anywhere
        extra = re.findall(r'per[_-]?([a-z0-9^]+)', low)
        for ex in extra:
            hints.add(synonyms.get(ex, ex))

        extra2 = re.findall(r'_([a-z0-9^]{1,4})($|_)', low)
        for ex, _ in extra2:
            hints.add(synonyms.get(ex, ex))

    return {h for h in hints if h and len(h) <= 5}
