import pandas as pd
from typing import Dict, List
from fuzzywuzzy import fuzz
import logger
from .schema_analyzer import _infer_currency_hints_from_headers, _infer_unit_hints_from_headers

def get_next_incremental_id(df: pd.DataFrame, column_name: str):
    """Get next incremental ID for auto-increment columns"""
    if df.empty:
        return 1
    
    if column_name in df.columns:
        max_id = df[column_name].max()
        return max_id + 1 if pd.notna(max_id) else 1
    return None

def create_empty_dimension_structure(dest_df: pd.DataFrame) -> pd.DataFrame:
    """Create an empty dataframe with the same structure as the destination dimension table"""
    empty_df = pd.DataFrame(columns=dest_df.columns)
    
    # Preserve the column data types
    for column in dest_df.columns:
        empty_df[column] = empty_df[column].astype(dest_df[column].dtype)
    
    return empty_df

def format_timestamp_as_varchar(timestamp):
    """Convert timestamp to varchar format as expected by schema"""
    return timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Remove last 3 digits of microseconds

def transform_dimension(source_value: str, dest_df: pd.DataFrame, 
                       value_column: str, id_column: str) -> pd.DataFrame:

    # BUG FIX 1: Create empty dataframe instead of copying destination data with mock data  
    df = create_empty_dimension_structure(dest_df)
    
    # Add the new user record with proper data types
    new_record = pd.DataFrame({
        id_column: [1],  # Start with ID 1 for new table
        value_column: [source_value],
        'created_at': [format_timestamp_as_varchar(pd.Timestamp.now())],
        'updated_at': [format_timestamp_as_varchar(pd.Timestamp.now())]
    })
    
    # Ensure ID column is int type
    new_record[id_column] = new_record[id_column].astype('int')
    
    # Handle additional columns if they exist in the destination schema
    for col in dest_df.columns:
        if col not in new_record.columns:
            if col in ['CountryID', 'Industry']:  # Fill known nullable columns
                new_record[col] = None
    
    df = pd.concat([df, new_record], ignore_index=True)
    
    return df

def transform_dimension_dynamic(source_df: pd.DataFrame, 
                             schema_analysis: Dict,
                             dest_schema: Dict,
                             dest_table: pd.DataFrame) -> pd.DataFrame:
    """
    Dynamically transform dimension data based on schema analysis
    """
    # Create empty result with destination structure
    result_df = create_empty_dimension_structure(dest_table)
    
    # Get column mappings based on schema analysis
    suggested_mappings = schema_analysis['suggested_mappings']
    
    # Map source columns to destination columns based on type and name similarity
    for dest_col in dest_schema['columns']:
        matched_source_col = find_matching_source_column(
            dest_col,
            source_df.columns,
            schema_analysis['column_types']
        )
        
        if matched_source_col:
            # Transform the data according to the destination schema
            result_df[dest_col] = transform_column_values(
                source_df[matched_source_col],
                dest_schema['datatypes'][dest_schema['columns'].index(dest_col)]
            )
    
    return result_df

def find_matching_source_column(dest_col: str, 
                              source_cols: List[str],
                              column_types: Dict[str, str]) -> str:
    """Find the best matching source column for a destination column"""
    
    # Define similarity metrics
    def calculate_similarity(source_col):
        name_similarity = fuzz.ratio(dest_col.lower(), source_col.lower()) / 100.0
        type_compatibility = 0.0
        if source_col in column_types:
            type_compatibility = is_type_compatible(
                column_types[source_col],
                dest_col
            )
        
        return (name_similarity + type_compatibility) / 2
    
    # Find best match
    matches = [(col, calculate_similarity(col)) for col in source_cols]
    best_match = max(matches, key=lambda x: x[1] if matches else (None, 0))
    
    return best_match[0] if best_match[1] > 0.5 else None

def transform_D_Date(mapping, source_df, ReportingYear) -> pd.DataFrame:
    """Transform date dimension with fallback to reporting year"""
    
    # Check if DateKey mapping exists and has valid source column
    has_date_mapping = (
        isinstance(mapping, dict) and 
        'DateKey' in mapping and 
        isinstance(mapping['DateKey'], dict) and
        'source_column' in mapping['DateKey'] and
        mapping['DateKey']['source_column'] is not None and 
        mapping['DateKey']['source_column'] in source_df.columns
    )

    if has_date_mapping:
        # Get the date column and convert to datetime
        date_col = pd.to_datetime(source_df[mapping['DateKey']['source_column']], errors='coerce')
        
        # Calculate quarter start dates (first day of the quarter)
        quarter_start = date_col.dt.to_period('Q').dt.start_time
        
        # Calculate quarter end dates (last day of the quarter)  
        quarter_end = date_col.dt.to_period('Q').dt.end_time
        
        date_df = pd.DataFrame({
            'DateKey': date_col.dt.strftime('%Y%m%d').astype('int'),
            'StartDate': quarter_start.dt.date,
            'EndDate': quarter_end.dt.date,
            'Description': date_col.dt.year.astype(str) + ' Quarter ' + date_col.dt.quarter.astype(str) + ' Report',
            'Year': date_col.dt.year.astype('int'),
            'Quarter': date_col.dt.quarter.astype('int'),
            'Month': date_col.dt.month.astype('int'),
            'Day': date_col.dt.day.astype('int'),
            'created_at': format_timestamp_as_varchar(pd.Timestamp.now()),
            'updated_at': format_timestamp_as_varchar(pd.Timestamp.now())
        })
   
    else:
        # Create dates for all days in the reporting year
        dates = pd.date_range(start=f'{ReportingYear}-01-01', end=f'{ReportingYear}-12-31', freq='D')
        
        # Create a DataFrame with all dates in the reporting year
        date_df = pd.DataFrame({
            'DateKey': dates.strftime('%Y%m%d').astype('int'),
            'StartDate': dates.date,
            'EndDate': dates.date,
            'Description': dates.strftime('%Y-%m-%d'),
            'Year': dates.year,
            'Quarter': dates.quarter,
            'Month': dates.month,
            'Day': dates.day,
            'created_at': format_timestamp_as_varchar(pd.Timestamp.now()),
            'updated_at': format_timestamp_as_varchar(pd.Timestamp.now())
        })

    # Ensure DateKey is unique
    date_df = date_df.drop_duplicates(subset='DateKey').reset_index(drop=True)

    return date_df
def relate_country_company(country: str, company: str, company_df: pd.DataFrame, country_df: pd.DataFrame) -> pd.DataFrame:
    """
    Establish relationship between company and country
    """
    if not country_df[country_df['CountryName'] == country].empty:
        country_id = int(country_df[country_df['CountryName'] == country]['CountryID'].values[0])
        company_df.loc[company_df['CompanyName'] == company, 'CountryID'] = country_id
        company_df.loc[company_df['CompanyName'] == company, 'updated_at'] = format_timestamp_as_varchar(pd.Timestamp.now())
        
        # Ensure CountryID is int type
        company_df['CountryID'] = company_df['CountryID'].astype('int')

    return company_df

def transform_D_Currency(mapping, source_df, dest_df: pd.DataFrame) -> pd.DataFrame:
    """Return only currencies present in the data. If no currency column, infer from headers."""
    # Existing logic
    currency_col = None
    if isinstance(mapping, dict) and 'CurrencyID' in mapping and isinstance(mapping['CurrencyID'], dict):
        currency_col = mapping['CurrencyID'].get('source_column')

    if not currency_col or currency_col not in source_df.columns:
        for col in source_df.columns:
            if col.lower() in ['unit_price_currency', 'currency', 'currency_code']:
                currency_col = col
                break

    df = create_empty_dimension_structure(dest_df)
    candidate_codes = set()

    # from the explicit column (if any)
    if currency_col and currency_col in source_df.columns:
        cand = source_df[currency_col].dropna().astype(str).str.upper().str.strip().unique().tolist()
        candidate_codes.update([c[:3] for c in cand])  # normalize to 3 letters if a longer string appears

    # from headers (Paid_EUR, TotalPaidEUR, etc.)
    candidate_codes.update(_infer_currency_hints_from_headers(list(source_df.columns)))

    if candidate_codes:
        mask = dest_df['CurrencyCode'].astype(str).str.upper().isin(candidate_codes)
        filtered = dest_df[mask].copy().reset_index(drop=True)
        df = pd.concat([df, filtered], ignore_index=True)

    return df


def transform_emission_source_provider(mapping: dict, source_df: pd.DataFrame, dest_df: pd.DataFrame) -> pd.DataFrame:
    """Transform emission source provider data"""

    # BUG FIX 1: Create empty dataframe instead of copying destination data with mock data
    result_df = create_empty_dimension_structure(dest_df)
    
    # Get the source column from mapping
    provider_mapping = next((v for k, v in mapping.items()
                           if k == 'ActivityEmissionSourceProviderID'), None)

    if not provider_mapping or 'source_column' not in provider_mapping:
        return result_df
        
    source_column = provider_mapping['source_column']
    
    if source_column not in source_df.columns:
        return result_df
        
    # Get unique providers from source
    providers = source_df[source_column].dropna().unique()

    for idx, provider in enumerate(providers):
        new_row = {
            'ActivityEmissionSourceProviderID': idx + 1,  # Start with ID 1 for new table
            'ProviderName': provider
        }
        # Ensure ID is int type
        new_row['ActivityEmissionSourceProviderID'] = int(new_row['ActivityEmissionSourceProviderID'])
        result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
    
    return result_df

def transform_unit(mapping, source_df: pd.DataFrame, dest_df: pd.DataFrame, calc_method) -> pd.DataFrame:
    """Return only units needed; infer from headers when no explicit Unit column exists."""
    unit_col = None
    if isinstance(mapping, dict) and 'UnitID' in mapping and isinstance(mapping['UnitID'], dict):
        unit_col = mapping['UnitID'].get('source_column')

    if not unit_col or unit_col not in source_df.columns:
        for col in source_df.columns:
            if col.lower() in ['consumption_unit', 'unit', 'unitname']:
                unit_col = col
                break

    df = create_empty_dimension_structure(dest_df)
    candidate_units = set()

    if unit_col and unit_col in source_df.columns:
        cand = source_df[unit_col].dropna().astype(str).str.strip().unique().tolist()
        candidate_units.update([c.lower() for c in cand])

    # infer from headers (Consumption_kWh, EmissionFactor_..._per_m3, Volume (m3), etc.)
    candidate_units.update(_infer_unit_hints_from_headers(list(source_df.columns)))

    if candidate_units:
        dest_lower = dest_df.assign(_key=dest_df['UnitName'].astype(str).str.lower())
        mask = dest_lower['_key'].isin(candidate_units)
        filtered = dest_lower[mask].drop(columns=['_key']).copy().reset_index(drop=True)
        df = pd.concat([df, filtered], ignore_index=True)

    return df

def transform_organizational_unit(org_unit_name: str, company_df: pd.DataFrame, dest_df: pd.DataFrame) -> pd.DataFrame:
    """Transform organizational unit data"""
    
    # Create empty dataframe with destination structure
    df = create_empty_dimension_structure(dest_df)
    
    if org_unit_name:
        # Get company ID for the relationship
        company_id = company_df['CompanyID'].iloc[0] if not company_df.empty else None
        
        # Check if the organizational unit already exists in the destination table
        existing_org_unit = dest_df[dest_df['OrganizationalUnitName'].str.lower() == org_unit_name.lower()]
        
        if not existing_org_unit.empty:
            # If the organizational unit exists, use the existing record
            org_unit_id = existing_org_unit['OrganizationalUnitID'].iloc[0]
            org_unit_record = existing_org_unit.copy()
            df = pd.concat([df, org_unit_record], ignore_index=True)
        else:
            # If the organizational unit doesn't exist, create a new record
            # Get the next available ID
            org_unit_id = dest_df['OrganizationalUnitID'].max() + 1 if not dest_df.empty else 1
            
            # Create new record
            new_record = pd.DataFrame({
                'OrganizationalUnitID': [org_unit_id],
                'OrganizationalUnitName': [org_unit_name],
                'CompanyID': [company_id],
                'created_at': [format_timestamp_as_varchar(pd.Timestamp.now())],
                'updated_at': [format_timestamp_as_varchar(pd.Timestamp.now())]
            })
            
            # Ensure ID columns are int type
            new_record['OrganizationalUnitID'] = new_record['OrganizationalUnitID'].astype('int')
            if company_id is not None:
                new_record['CompanyID'] = new_record['CompanyID'].astype('int')
            
            df = pd.concat([df, new_record], ignore_index=True)
            
            # Also append the new record to the destination table for future reference
            dest_df = pd.concat([dest_df, new_record], ignore_index=True)
    
    return df

def is_type_compatible(source_type: str, dest_col: str) -> float:
    """
    Check compatibility between source and destination column types
    Returns a compatibility score between 0 and 1
    """
    # Normalize types for comparison
    source_type = str(source_type).lower()
    dest_col = dest_col.lower()
    
    # Define type compatibility scores
    type_scores = {
        ('date', 'datekey'): 1.0,
        ('date', 'startdate'): 1.0,
        ('date', 'enddate'): 1.0,
        ('object', 'name'): 0.8,
        ('object', 'code'): 0.8,
        ('object', 'description'): 0.8,
        ('int64', 'id'): 1.0,
        ('float64', 'amount'): 1.0,
        ('float64', 'quantity'): 1.0
    }
    
    # Check for exact type matches
    for (src, dst), score in type_scores.items():
        if source_type.startswith(src) and dest_col.endswith(dst):
            return score
            
    # Default compatibility scores
    if source_type in ['int64', 'float64'] and any(x in dest_col for x in ['id', 'amount', 'quantity']):
        return 0.7
    elif source_type == 'object' and any(x in dest_col for x in ['name', 'description', 'code']):
        return 0.6
    elif source_type == 'datetime64' and 'date' in dest_col:
        return 0.9
        
    return 0.1  # Low default compatibility

def transform_column_values(source_series: pd.Series, dest_type: str) -> pd.Series:
    """
    Transform source column values to match destination column type
    """
    try:
        # Handle different destination types
        if dest_type.lower() in ['int', 'integer', 'bigint']:
            return pd.to_numeric(source_series, errors='coerce').fillna(0).astype(int)
            
        elif dest_type.lower() in ['float', 'double', 'decimal']:
            return pd.to_numeric(source_series, errors='coerce').fillna(0.0)
            
        elif dest_type.lower() in ['date', 'datetime']:
            return pd.to_datetime(source_series, errors='coerce')
            
        elif dest_type.lower() in ['varchar', 'string', 'text']:
            return source_series.astype(str).replace('nan', '')
            
        elif dest_type.lower() == 'boolean':
            return source_series.map({'true': True, 'false': False, 
                                    '1': True, '0': False,
                                    'yes': True, 'no': False}).fillna(False)
        
        # Default to string conversion for unknown types
        return source_series.astype(str)
        
    except Exception as e:
        logger.error(f"Error transforming column values: {str(e)}")
        # Return original series if transformation fails
        return source_series