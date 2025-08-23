import pandas as pd
from typing import Dict, List
from fuzzywuzzy import fuzz
import logger
from .schema_analyzer import _infer_currency_hints_from_headers, _infer_unit_hints_from_headers
from .mapping_utils import extract_unit_from_column, normalize_unit
from . import mapping_utils
import re 
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
        raw = source_df[mapping['DateKey']['source_column']].astype(str).str.strip()

        # Detect 8-digit numeric dates which may be in YYYYDDMM format (e.g., 20231201 vs 20230112 ambiguity)
        # If a value matches 8 digits, try parsing both YYYYMMDD and YYYYDDMM and pick the one that yields a valid month (1-12)
        def _parse_ambiguous_8digit(s):
            s = str(s).strip()
            if not re.fullmatch(r"\d{8}", s):
                return pd.NaT
            # Try YYYYMMDD
            try:
                dt1 = pd.to_datetime(s, format="%Y%m%d", errors='coerce')
            except Exception:
                dt1 = pd.NaT
            # Try YYYYDDMM (swap day and month)
            try:
                y = s[0:4]
                d = s[4:6]
                m = s[6:8]
                swapped = f"{y}{m}{d}"
                dt2 = pd.to_datetime(swapped, format="%Y%m%d", errors='coerce')
            except Exception:
                dt2 = pd.NaT

            # Prefer the one with a valid month/day (month between 1-12)
            if pd.isna(dt1) and pd.isna(dt2):
                return pd.NaT
            if pd.isna(dt1):
                return dt2
            if pd.isna(dt2):
                return dt1
            # If both valid, choose dt1 (YYYYMMDD) as default
            return dt1

        # First pass: try normal parsing with pandas
        date_col = pd.to_datetime(raw, errors='coerce', dayfirst=False)

        # For any still-NaT entries, attempt ambiguous 8-digit handling
        mask_na = date_col.isna()
        if mask_na.any():
            parsed = raw.loc[mask_na].apply(_parse_ambiguous_8digit)
            # parsed may be dtype object with Timestamps/NaT; convert to DatetimeIndex
            parsed_dt = pd.to_datetime(parsed, errors='coerce')
            date_col.loc[mask_na] = parsed_dt

        # 2nd attempt: European DD/MM/YYYY if still NaT
        mask = date_col.isna()
        if mask.any():
            date_col.loc[mask] = pd.to_datetime(raw.loc[mask], errors='coerce', dayfirst=True)
        
        # Calculate quarter start dates (first day of the quarter)
        quarter_start = date_col.dt.to_period('Q').dt.start_time
        
        # Calculate quarter end dates (last day of the quarter)  
        quarter_end = date_col.dt.to_period('Q').dt.end_time
        
        date_df = pd.DataFrame({
            "DateKey": date_col.dt.strftime("%Y%m%d").astype("Int64"),
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

    # from headers (Paid_EUR, TotalPaidEUR, TotalPaid(EUR), etc.) - Enhanced with new extraction function
    from .mapping_utils import extract_currency_from_column
    for col in source_df.columns:
        extracted_currency = extract_currency_from_column(col)
        if extracted_currency:
            candidate_codes.add(extracted_currency)
            print(f"transform_D_Currency: extracted currency '{extracted_currency}' from column header '{col}'")
        else:
            # Try dynamic symbol lookup using destination currency table (preferred over static map)
            try:
                # show dynamic mapping for debugging
                try:
                    dyn_map = mapping_utils.build_currency_symbol_map(dest_df)
                    if dyn_map:
                        print(f"transform_D_Currency: dynamic currency symbol map from dest table: {dyn_map}")
                except Exception:
                    dyn_map = {}

                sym_code = mapping_utils._currency_code_from_symbol_in_text(col, dest_df)
                if sym_code:
                    candidate_codes.add(sym_code)
                    print(f"transform_D_Currency: extracted currency '{sym_code}' from column header '{col}' via symbol using dest table")
            except Exception:
                # fallback ignored
                pass

    # If still empty, scan sample values in the source data for currency tokens like 'EUR'
    if not candidate_codes:
        try:
            import re
            iso_pattern = re.compile(r'\b([A-Z]{3})\b')
            sample_vals = []
            for col in source_df.columns:
                try:
                    vals = source_df[col].dropna().astype(str).head(50).tolist()
                    sample_vals.extend(vals)
                except Exception:
                    continue
            for v in sample_vals:
                m = iso_pattern.search(v)
                if m:
                    candidate_codes.add(m.group(1).upper())
            if candidate_codes:
                print(f"transform_D_Currency: inferred currency codes from data values: {candidate_codes}")
        except Exception:
            # ignore scanning errors
            sample_vals = []

        # Also scan for currency symbols (€, $, £, ₹, etc.) and map them to ISO codes
        try:
            # Prefer dynamic symbol map from destination currency table if available
            from . import mapping_utils
            for v in sample_vals:
                try:
                    code = mapping_utils._currency_code_from_symbol_in_text(str(v), dest_df)
                    if code:
                        candidate_codes.add(code)
                except Exception:
                    # fallback to static detection inside mapping_utils
                    try:
                        code = mapping_utils._currency_code_from_symbol_in_text(str(v))
                        if code:
                            candidate_codes.add(code)
                    except Exception:
                        pass

            if candidate_codes:
                print(f"transform_D_Currency: inferred currency codes from symbols in data values: {candidate_codes}")
        except Exception:
            pass
    
    # Also keep the old header inference as fallback
    candidate_codes.update(_infer_currency_hints_from_headers(list(source_df.columns)))

    print(f"transform_D_Currency: candidate_codes={candidate_codes}")
    try:
        print(f"transform_D_Currency: dest_df columns={list(dest_df.columns)}; sample={dest_df.head(6).to_dict(orient='list')}")
    except Exception:
        pass

    # Find currency code column in destination table (support variations)
    currency_code_col = None
    try:
        norm_map = {re.sub(r'[^a-z0-9]', '', c.lower()): c for c in dest_df.columns}
        for key, orig in norm_map.items():
            if key in ('currencycode', 'currency_code', 'currency', 'code'):
                currency_code_col = orig
                break
        if currency_code_col is None and 'CurrencyCode' in dest_df.columns:
            currency_code_col = 'CurrencyCode'
    except Exception:
        currency_code_col = 'CurrencyCode' if 'CurrencyCode' in dest_df.columns else None

    if currency_code_col is None:
        print(f"transform_D_Currency: destination currency column not found in {list(dest_df.columns)}")
        return df

    # Also attempt to detect currency symbols in the source data
    detected_symbols = set()
    try:
        # collect sample values if not already collected
        sample_vals_local = []
        for col in source_df.columns:
            try:
                vals = source_df[col].dropna().astype(str).head(50).tolist()
                sample_vals_local.extend(vals)
            except Exception:
                continue

        # dynamic symbol map from destination table
        dyn_map = mapping_utils.build_currency_symbol_map(dest_df) if dest_df is not None else {}
        # check both dynamic and static symbol sets
        symbol_candidates = list(dyn_map.keys()) + list(mapping_utils._CURRENCY_SYMBOL_MAP.keys())
        # longer tokens first
        symbol_candidates = sorted(set(symbol_candidates), key=len, reverse=True)
        for v in sample_vals_local:
            for sym in symbol_candidates:
                if sym and sym in v:
                    detected_symbols.add(sym)
        if detected_symbols:
            print(f"transform_D_Currency: detected symbols in data values: {detected_symbols}")
    except Exception:
        detected_symbols = set()

    # Build final match mask using either currency code or symbol (if available)
    mask_code = pd.Series([False] * len(dest_df))
    if candidate_codes:
        mask_code = dest_df[currency_code_col].astype(str).str.upper().isin(candidate_codes)
        print(f"transform_D_Currency: using dest column '{currency_code_col}' to match candidate codes; mask sum={mask_code.sum()} out of {len(dest_df)} rows")

    mask_symbol = pd.Series([False] * len(dest_df))
    # try to find a symbol column in destination table
    symbol_col = None
    try:
        for c in dest_df.columns:
            if re.search(r'symbol|char|sign', c, re.I):
                symbol_col = c
                break
    except Exception:
        symbol_col = None

    if symbol_col and detected_symbols:
        try:
            mask_symbol = dest_df[symbol_col].astype(str).isin(detected_symbols)
            print(f"transform_D_Currency: using dest symbol column '{symbol_col}' to match detected symbols; mask sum={mask_symbol.sum()} out of {len(dest_df)} rows")
        except Exception:
            mask_symbol = pd.Series([False] * len(dest_df))

    mask = mask_code | mask_symbol
    filtered = dest_df[mask].copy().reset_index(drop=True)
    df = pd.concat([df, filtered], ignore_index=True)

    if df.empty:
        print("transform_D_Currency: No candidate currency codes found; returning empty currency df")

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
        cleaned = mapping_utils.clean_provider_name(provider) if provider is not None else None
        new_row = {
            'ActivityEmissionSourceProviderID': idx + 1,  # Start with ID 1 for new table
            'ProviderName': cleaned if cleaned else provider
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

    import logging
    # Visible debug prints to ensure output appears in user logs
    print(f"transform_unit: dest_df columns={list(dest_df.columns)} empty={dest_df.empty}")
    try:
        print(f"transform_unit: dest_df sample: {dest_df.head(3).to_dict(orient='list')}")
    except Exception:
        pass

    df = create_empty_dimension_structure(dest_df)
    candidate_units = set()

    if unit_col and unit_col in source_df.columns:
        cand = source_df[unit_col].dropna().astype(str).str.strip().unique().tolist()
        candidate_units.update([c.lower() for c in cand])

    # infer from headers (Consumption_kWh, EmissionFactor_..._per_m3, Volume (m3), etc.)
    candidate_units.update(_infer_unit_hints_from_headers(list(source_df.columns)))

    # Also use the improved extractor to parse units from complex column names (e.g., 'kWh_Used' -> 'kWh')
    for col in source_df.columns:
        try:
            ext = extract_unit_from_column(col)
            if ext:
                candidate_units.add(str(ext).lower())
        except Exception:
            # ignore extractor failures per-column
            pass

    # Debug: show inferred candidate units (small sample)
    try:
        import logging
        logging.info(f"Detected candidate units from headers/columns: {list(candidate_units)[:10]}")
    except Exception:
        pass

    if candidate_units:
        # Try to find the unit name column in destination table (support variations)
        # Normalize destination column names by removing non-alphanumerics for robust matching
        norm_map = {re.sub(r'[^a-z0-9]', '', c.lower()): c for c in dest_df.columns}
        unit_name_col = None
        for key, orig in norm_map.items():
            if key in ('unitname', 'unit', 'name'):
                unit_name_col = orig
                break
        if unit_name_col is None:
            # Fallback to 'UnitName' if present, else log and return empty
            if 'UnitName' in dest_df.columns:
                unit_name_col = 'UnitName'
            else:
                import logging
                logging.warning(f"transform_unit: destination unit column not found in {list(dest_df.columns)}")
                return df

        # Debug: show dest table sample and inferred candidate units
        try:
            print(f"transform_unit: dest unit column detected as '{unit_name_col}'. dest sample: {dest_df[unit_name_col].astype(str).dropna().head(5).tolist()}")
            print(f"transform_unit: candidate_units: {list(candidate_units)}")
        except Exception:
            pass

        dest_lower = dest_df.assign(_key=dest_df[unit_name_col].astype(str).str.lower())
        mask = dest_lower['_key'].isin(candidate_units)
        filtered = dest_lower[mask].drop(columns=['_key']).copy().reset_index(drop=True)
        print(f"transform_unit: matched units count={len(filtered)}; matched sample={filtered.head(5).to_dict(orient='list')}")
        if filtered.empty:
            # Fallback: if no candidate units matched, include entire destination unit table so lookups later can fuzzy-match
            print("transform_unit: no matches found — falling back to full DE1_Unit table to allow fuzzy matching")
            df = pd.concat([df, dest_df.copy().reset_index(drop=True)], ignore_index=True)
        else:
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