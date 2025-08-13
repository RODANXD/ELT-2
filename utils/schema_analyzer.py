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