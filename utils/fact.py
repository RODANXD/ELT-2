import pandas as pd
import json
from datetime import datetime, timedelta
import logging
from tqdm.auto import tqdm  # Changed to tqdm.auto for better notebook/UI support
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console
from rich import print as rprint
import streamlit as st
from .progress_state import update_progress
from .airport_distance import calculate_airport_distance, calculate_consumption_amount_for_air_travel
from .mapping_utils import normalize_text, normalize_unit, extract_unit_from_column, extract_unit_from_value
from . import mapping_utils
from .mapping_utils import clean_provider_name
# Optional AI-based classification (will be used as a last-resort fallback)
try:
    from .gpt_mapper import classify_energy_value
except Exception:
    classify_energy_value = None
from fuzzywuzzy import process as _fuzzy_process

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def safe_int_conversion(value):
    """
    Safely convert a value to integer, handling NaN and None values.
    Returns None if conversion is not possible.
    """
    if pd.isna(value) or value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

import re
def safe_float_conversion(value):
    """
    Safely convert a value to float, handling NaN and None values.
    Returns None if conversion is not possible.
    """
    if pd.isna(value) or value is None:
        return None
    try:
        # print("????????????????????", value, type(value))  # Debugging line
        if isinstance(value, str):
            # Remove commas and other non-numeric characters
            value = re.sub(r'[^\d.-]', '', value)
        return float(value)
    except (ValueError, TypeError):
        return None

def find_emission_ids(mappings, activity_subcat, activity_subcat_df, activity_emission_source_df, country_df, country, calc_method):
    """Returns ActivityEmissionSourceID, UnitID, and EmissionFactorID based on mapping conditions."""
    
    logging.info(f"🔍 [find_emission_ids] Starting emission ID resolution for {activity_subcat}")
    logging.info(f"🔍 [find_emission_ids] Calculation method: {calc_method}")
    logging.info(f"🔍 [find_emission_ids] Country: {country}")
    
    # Get basic lookup values
    activity_sub_cat_id = lookup_value(activity_subcat_df, 'ActivitySubcategoryName', activity_subcat, 'ActivitySubcategoryID')
    logging.info(f"🔍 [find_emission_ids] ActivitySubcategoryID: {activity_sub_cat_id}")
    
    iso2_code = lookup_value(country_df, 'CountryName', country, 'ISO2Code')
    logging.info(f"🔍 [find_emission_ids] Country ISO2Code: {iso2_code}")
    
    # Special handling for Electricity (ActivitySubcategoryID 21)
    
    
    # Define transformation suffixes based on calc_method
    valid_transformations = ['Distance', 'Fuel', 'Electricity', 'Heating', 'Days'] if calc_method == 'Consumption-based' else ['Currency']
    logging.info(f"🔍 [find_emission_ids] Valid transformations for {calc_method}: {valid_transformations}")
    
    # Find the first matching transformation
    transformation = None
    amount_key = 'ConsumptionAmount' 
    for key, mapping_info in mappings.items():
        if amount_key in key:
            trans = mapping_info.get('consumption_type', '').lower()
            logging.info(f"🔍 [find_emission_ids] Found consumption_type '{trans}' in mapping key '{key}'")
            print(trans, '=', [x.lower() for x in valid_transformations])
            if trans in [x.lower() for x in valid_transformations]:
                transformation = trans
                logging.info(f"🔍 [find_emission_ids] Selected transformation: '{transformation}'")
                break

    if not transformation:
        logging.warning(f"⚠️ [find_emission_ids] No valid transformation found for {activity_subcat}")
        return None, None, None
    
    # Get emission source ID by suffix
    # Try to match emission source intelligently: by suffix first, else fuzzy match on names
    logging.info(f"🔍 [find_emission_ids] Looking for emission source with suffix '{transformation}' and subcategory ID {activity_sub_cat_id}")
    
    emission_source_id = get_emission_source_id_by_suffix(
        activity_emission_source_df, activity_sub_cat_id, transformation
    )
    logging.info(f"🔍 [find_emission_ids] Emission source ID by suffix: {emission_source_id}")
    
    if not emission_source_id:
        # fuzzy match transformation to ActivityEmissionSourceName
        logging.info(f"🔍 [find_emission_ids] No suffix match, trying fuzzy matching...")
        candidates = activity_emission_source_df['ActivityEmissionSourceName'].astype(str).tolist() if not activity_emission_source_df.empty else []
        logging.info(f"🔍 [find_emission_ids] Available emission source names: {candidates[:10]}...")
        
        mapped = mapping_utils.fuzzy_match_value_to_list(transformation, candidates, threshold=60) if transformation else None
        logging.info(f"🔍 [find_emission_ids] Fuzzy match result: {mapped}")
        
        if mapped:
            row = activity_emission_source_df[activity_emission_source_df['ActivityEmissionSourceName'].astype(str) == mapped]
            if not row.empty:
                emission_source_id = int(row['ActivityEmissionSourceID'].iloc[0])
                logging.info(f"🔍 [find_emission_ids] Fuzzy matched '{transformation}' to '{mapped}' with ID {emission_source_id}")
    
    if not emission_source_id:
        logging.warning(f"❌ [find_emission_ids] No emission source ID found for {activity_subcat}")
        return None, None, None
    
    # Get unit ID and emission factor ID
    unit_id = lookup_value(activity_emission_source_df, 'ActivityEmissionSourceID', emission_source_id, 'UnitID')
    logging.info(f"🔍 [find_emission_ids] Unit ID: {unit_id}")

    # Get emission source name for EmissionFactorID generation
    emission_source_name = lookup_value(activity_emission_source_df, 'ActivityEmissionSourceID', emission_source_id, 'ActivityEmissionSourceName')
    logging.info(f"🔍 [find_emission_ids] Emission source name: '{emission_source_name}'")
    
    # Generate EmissionFactorID
    emission_factor_id = f"{iso2_code}_{emission_source_name.replace(' ', '_')}"
    logging.info(f"🔍 [find_emission_ids] Generated EmissionFactorID: '{emission_factor_id}'")
    
    # WARNING: This EmissionFactorID may be incorrect if the emission source is later overridden by value-based detection
    logging.warning(f"⚠️ [find_emission_ids] WARNING: This EmissionFactorID '{emission_factor_id}' may be overridden by value-based detection later")
    
    return emission_source_id, unit_id, emission_factor_id


def get_emission_source_id_by_suffix(df, subcategory_id, suffix):
    """Get emission source ID for sources ending with suffix (case-insensitive) and matching subcategory ID."""
    logging.info(f"🔍 [get_emission_source_id_by_suffix] Looking for suffix '{suffix}' with subcategory ID {subcategory_id}")
    
    # Show available emission sources for this subcategory
    available_sources = df[df['ActivitySubcategoryID'] == subcategory_id]
    logging.info(f"🔍 [get_emission_source_id_by_suffix] Available sources for subcategory {subcategory_id}: {available_sources[['ActivityEmissionSourceID', 'ActivityEmissionSourceName']].to_dict(orient='list')}")
    
    filtered = df[
        (df['ActivitySubcategoryID'] == subcategory_id) &
        (df['ActivityEmissionSourceName'].str.lower().str.contains(suffix.lower()))
    ]
    
    logging.info(f"🔍 [get_emission_source_id_by_suffix] Sources containing suffix '{suffix}': {filtered[['ActivityEmissionSourceID', 'ActivityEmissionSourceName']].to_dict(orient='list')}")
    
    if not filtered.empty:
        result = filtered.iloc[0]['ActivityEmissionSourceID']
        emission_name = filtered.iloc[0]['ActivityEmissionSourceName']
        logging.info(f"✅ [get_emission_source_id_by_suffix] Selected '{emission_name}' (ID: {result}) for suffix '{suffix}'")
        return result
    else:
        logging.warning(f"⚠️ [get_emission_source_id_by_suffix] No sources found containing suffix '{suffix}' for subcategory {subcategory_id}")
        return None



def lookup_value(df: pd.DataFrame, lookup_column: str, lookup_value: str, return_column: str):
    """Generic lookup function to find values in dimension tables"""
    if df.empty or lookup_value is None:
        return None
    
    # Convert lookup column to string before comparing
    mask = df[lookup_column].astype(str).str.lower() == str(lookup_value).lower()
    result = df.loc[mask, return_column]
    
    if not result.empty:
        return result.iloc[0]
    return None



def get_date_key(date_df: pd.DataFrame, source_column: str, year: int, date_value=None):
    """
    Returns DateKey using the existing lookup_value function
    """
    import pandas as pd
    
    # if source column is None or empty, use reporting year to get DateKey
    if source_column is None or source_column == 'null' or date_value is None:
        # Get DateKey based on the reporting year
        return lookup_value(date_df, 'Year', year, 'DateKey')
    
    # Convert date to datetime
    import re

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

        if pd.isna(dt1) and pd.isna(dt2):
            return pd.NaT
        if pd.isna(dt1):
            return dt2
        if pd.isna(dt2):
            return dt1
        # If both are valid, prefer dt1 (assumed correct YYYYMMDD)
        return dt1

    source_date = pd.to_datetime(date_value, errors='coerce')

    # If parsing failed and value is an 8-digit numeric string, try ambiguous parsing
    if pd.isna(source_date) and isinstance(date_value, (str, int)):
        parsed = _parse_ambiguous_8digit(date_value)
        source_date = parsed

    if pd.isna(source_date):
        return lookup_value(date_df, 'Year', year, 'DateKey')  # fallback to year

    # Convert date to DateKey format (YYYYMMDD)
    date_key = source_date.strftime('%Y%m%d')

    # Use the existing lookup function to find the DateKey
    return lookup_value(date_df, 'DateKey', date_key, 'DateKey')
    
def get_next_incremental_id(df: pd.DataFrame, column_name: str):
    """Get next incremental ID for auto-increment columns"""
    if df.empty:
        return 1
    
    if column_name in df.columns:
        max_id = df[column_name].max()
        return max_id + 1 if pd.notna(max_id) else 1
    return None

def create_empty_fact_table_structure(dest_schema) -> pd.DataFrame:
    """
    Creates an empty DataFrame with the same structure as the destination fact table.
    
    Args:
        dest_schema (pd.DataFrame): Destination fact table DataFrame
        
    Returns:
        pd.DataFrame: Empty dataframe with fact table structure
    """
    try:
        # Create empty DataFrame with same columns as destination schema
        empty_df = pd.DataFrame(columns=dest_schema.columns)
        
        # Preserve the column data types from source DataFrame
        for column in dest_schema.columns:
            try:
                empty_df[column] = empty_df[column].astype(dest_schema[column].dtype)
            except:
                # If type conversion fails, keep as object type
                empty_df[column] = empty_df[column].astype('object')
        
        logging.info(f"Created empty fact table structure with columns: {list(empty_df.columns)}")
        return empty_df
        
    except Exception as e:
        logging.error(f"Error creating empty fact table structure: {str(e)}")
        raise Exception(f"Failed to create empty fact table structure: {str(e)}")




def generate_fact(
    mappings: dict, source_df: pd.DataFrame, dest_df: pd.DataFrame,
    activity_cat_df: pd.DataFrame, activity_subcat_df: pd.DataFrame,
    scope_df: pd.DataFrame, activity_emission_source_df: pd.DataFrame,
    activity_emmission_source_provider_df: pd.DataFrame, unit_df: pd.DataFrame,
    currency_df: pd.DataFrame, date_df: pd.DataFrame, country_df: pd.DataFrame,
    company_df: pd.DataFrame, company: str, country: str, activity_cat: str,
    activity_subcat: str, reporting_year: int, calc_method: str,
    org_unit_df: pd.DataFrame = None, dest_tables: dict = None,
    *, is_multiple_units: bool = False, unit_column: str = None   # << NEW
) -> pd.DataFrame:
    print("\n\n*** USING MODIFIED FACT.PY WITH ORGANIZATIONAL UNIT ID FIX ***\n\n")

    # Record start time
    start_time = datetime.now()

    # BUG FIX 1: Create empty dataframe instead of copying destination data with mock data
    # This ensures we only get user data, not appended to mock data
    result_df = create_empty_fact_table_structure(dest_df)
    
    total_records = len(source_df)
    
    # Update progress state
    update_progress(
        table_name=f"{activity_cat} - {activity_subcat}",
        total=total_records,
        processed=0,
        status="Processing"
    )
    
    # Display progress information in Streamlit
    with st.expander("Processing Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Company:** " + str(company))
            st.markdown("**Country:** " + str(country))
            st.markdown("**Reporting Year:** " + str(reporting_year))
        with col2:
            st.markdown("**Category:** " + str(activity_cat))
            st.markdown("**Subcategory:** " + str(activity_subcat))
            st.markdown("**Calculation Method:** " + str(calc_method))

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Get initial IDs
    emission_source_id, unit_id, emission_factor_id = find_emission_ids(
        mappings, activity_subcat, activity_subcat_df, activity_emission_source_df, 
        country_df, country, calc_method
    )
    
    # NOTE: The emission_factor_id generated above may be INCORRECT and will be overridden
    # by the value-based energy type detection that happens later in the loop.
    # This is why we ALWAYS regenerate the EmissionFactorID based on the final resolved ActivityEmissionSourceID.
    logging.info(f"🔍 [Main Loop] Initial emission_factor_id: {emission_factor_id} (may be overridden)")
    logging.info(f"🔍 [Main Loop] Initial emission_source_id: {emission_source_id} (may be overridden)")
    logging.info(f"🔍 [Main Loop] Initial unit_id: {unit_id} (may be overridden)")

    # BUG FIX 2: Detect columns for airport distance calculation
    origin_column = None
    destination_column = None
    
    # Debug: Log source data info
    logging.info(f"Source DataFrame shape: {source_df.shape}")
    logging.info(f"Source columns: {list(source_df.columns)}")
    if len(source_df) > 0:
        logging.info(f"Sample source data (first row): {dict(source_df.iloc[0])}")
    
    # Look for origin and destination columns in the source data
    for column in source_df.columns:
        column_lower = column.lower().strip()
        # More flexible detection patterns
        origin_patterns = ['origin', 'departure', 'from', 'start', 'source', 'depart']
        destination_patterns = ['destination', 'arrival', 'to', 'end', 'dest', 'arrive', 'target']
        
        # Check if column name contains any origin patterns
        if any(pattern in column_lower for pattern in origin_patterns):
            origin_column = column
        # Check if column name contains any destination patterns  
        elif any(pattern in column_lower for pattern in destination_patterns):
            destination_column = column
    
    # Check if we have air travel consumption calculation
    is_air_travel_consumption = (
        calc_method == 'Consumption-based' and 
        activity_cat.lower() == 'business travel' and
        activity_subcat.lower() == 'air travel' and
        origin_column and destination_column
    )
    
    # Debug: Log detection results
    logging.info(f"Air travel consumption detection:")
    logging.info(f"  calc_method: {calc_method}")
    logging.info(f"  activity_cat: {activity_cat}")
    logging.info(f"  activity_subcat: {activity_subcat}")
    logging.info(f"  origin_column: {origin_column}")
    logging.info(f"  destination_column: {destination_column}")
    logging.info(f"  is_air_travel_consumption: {is_air_travel_consumption}")
    
    # Check for missing airport columns in air travel scenario
    if (calc_method == 'Consumption-based' and 
        activity_cat.lower() == 'business travel' and
        activity_subcat.lower() == 'air travel'):
        
        if not origin_column or not destination_column:
            missing_cols = []
            if not origin_column:
                missing_cols.append("origin/departure airport codes")
            if not destination_column:
                missing_cols.append("destination/arrival airport codes")
            
            warning_msg = f"⚠️ Missing required columns for air travel distance calculation: {', '.join(missing_cols)}"
            st.warning(warning_msg)
            st.info("📋 Your source Excel file needs columns with airport codes. See SOURCE_DATA_FORMAT_GUIDE.md for details.")
            logging.warning(f"Missing airport columns for air travel: {missing_cols}")
            logging.warning("Available columns: " + str(list(source_df.columns)))
    
    if is_air_travel_consumption:
        st.info(f"🛫 Air travel consumption calculation enabled using columns: {origin_column} -> {destination_column}")
    else:
        # Even if we don't have perfect column detection, check if ConsumptionAmount mapping indicates distance calculation
        consumption_mapping = mappings.get('ConsumptionAmount', {})
        consumption_type = consumption_mapping.get('consumption_type', '').lower()
        if consumption_type == 'distance' and calc_method == 'Consumption-based':
            st.info(f"🛫 Air travel distance calculation enabled (consumption_type: {consumption_type})")
            logging.info(f"Air travel distance calculation enabled via mapping consumption_type: {consumption_type}")
        elif calc_method == 'Consumption-based' and activity_cat.lower() == 'business travel' and activity_subcat.lower() == 'air travel':
            st.warning("⚠️ Air travel consumption selected but distance calculation not enabled. Check your source data format.")
            logging.warning("Air travel consumption scenario but distance calculation not enabled")

    for index, (_, source_row) in enumerate(source_df.iterrows()):
        new_row = {}
        
        # Map the fixed fact columns with proper data types
        new_row['EmissionActivityID'] = int(get_next_incremental_id(result_df, 'EmissionActivityID'))
        
        # Get IDs and ensure they are integers with proper NaN handling
        company_id = lookup_value(company_df, 'CompanyName', company, 'CompanyID')
        new_row['CompanyID'] = int(company_id) if company_id is not None else None
        
        # Handle OrganizationalUnitID
        # ----------------------------------------------------------
        # OrganisationalUnitID assignment
        # ----------------------------------------------------------
        if is_multiple_units and unit_column and unit_column in source_df.columns:
            unit_name = source_row.get(unit_column)
            if unit_name and not org_unit_df.empty:
                matched = org_unit_df[
                    org_unit_df['OrganizationalUnitName'].str.lower() == str(unit_name).lower()
                ]
                new_row['OrganizationalUnitID'] = (
                    int(matched['OrganizationalUnitID'].iloc[0]) if not matched.empty else None
                )
            else:
                new_row['OrganizationalUnitID'] = None
        else:
            # single-unit mode – use the one and only unit
            new_row['OrganizationalUnitID'] = (
                int(org_unit_df['OrganizationalUnitID'].iloc[0]) if not org_unit_df.empty else None
            )
        # ----------------------------------------------------------
        
        
        country_id = lookup_value(country_df, 'CountryName', country, 'CountryID')
        new_row['CountryID'] = int(country_id) if country_id is not None else None
        
        activity_cat_id = lookup_value(activity_cat_df, 'ActivityCategory', activity_cat, 'ActivityCategoryID')
        new_row['ActivityCategoryID'] = int(activity_cat_id) if activity_cat_id is not None else None
        
        activity_subcat_id = lookup_value(activity_subcat_df, 'ActivitySubcategoryName', activity_subcat, 'ActivitySubcategoryID')
        new_row['ActivitySubcategoryID'] = int(activity_subcat_id) if activity_subcat_id is not None else None
        
        # Remove duplicate assignment
        # new_row['OrganizationalUnitID'] = int(org_unit_df.loc[0, 'OrganizationalUnitID']) if not org_unit_df.empty else None
        
        
        scope_id = lookup_value(activity_cat_df, 'ActivityCategory', activity_cat, 'ScopeID')
        new_row['ScopeID'] = int(scope_id) if scope_id is not None else None
        
        # Handle ActivityEmissionSourceID
        # --- Resolve ActivityEmissionSourceID & ActivitySubcategoryID from source energy type ---
        # Detect energy type column robustly
        energy_col = None
                    # broaden detection to many possible energy/ source columns
        energy_candidates = ['energy', 'energy origin', 'energyorigin', 'supply', 'supplycategory', 'EnergyOrigin', 'subcategory', 'fuel', 'source', 'energy_type', 'activityemissionsourcename']
        for col in source_df.columns:
            if not isinstance(col, str):
                continue
            norm = col.strip().lower().replace('_', ' ').replace('-', ' ')
            for cand in energy_candidates:
                if cand in norm:
                    energy_col = col
                    break
            if energy_col:
                break
        # Initialize raw_energy from detected energy column (if any)
        raw_energy = None
        if energy_col and energy_col in source_row.index:
            raw_energy = source_row.get(energy_col)
            if pd.notna(raw_energy):
                raw_energy = str(raw_energy).strip()
            
        resolved_emission_source_id = None
        resolved_activity_subcat_from_energy = None

        # Define common energy type mappings to standardize source values
        energy_type_mappings = {
            # Green electricity mappings
            'green': 'Green Electricity',
            'renewable': 'Green Electricity',
            'solar': 'Green Electricity',
            'wind': 'Green Electricity',
            'hydro': 'Green Electricity',
            'clean': 'Green Electricity',
            # Conventional electricity mappings
            'conventional': 'Conventional Electricity',
            'fossil': 'Conventional Electricity',
            'coal': 'Conventional Electricity',
            'gas fired': 'Conventional Electricity',
            'non-renewable': 'Conventional Electricity',
            'grid': 'Conventional Electricity',
            # Biomass electricity mappings
            'organic': 'Biomass Electricity',
            'biofuel': 'Biomass Electricity',
            'waste': 'Biomass Electricity',
            'bioelectricity': 'Biomass Electricity',
            'waste-to-energy': 'Biomass Electricity',
            # Biomass heating mappings
            'biogas': 'Biomass Heating',  # <-- Fix: biogas should map to Biomass Heating
            'heating': 'Biomass Heating',
            'wood': 'Biomass Heating',
            'pellet': 'Biomass Heating',
            'bio gas': 'Biomass Heating',  # <-- Fix: was missing 'Bio'
            'biomass heating': 'Biomass Heating',
            # District heating mappings
            'district': 'District Heating',
            'district heating': 'District Heating',
            'central heating': 'District Heating',
            # Natural gas mappings
            'natural gas': 'Natural Gas',
            'gas': 'Natural Gas',  # <-- FIX: Add "gas" mapping to Natural Gas
            'naturalgas': 'Natural Gas',
            'ng': 'Natural Gas',
            'lng': 'Natural Gas',
            'methane': 'Natural Gas',
            'cng': 'Natural Gas',  # Compressed Natural Gas
            'piped gas': 'Natural Gas',
            'utility gas': 'Natural Gas',
            # Nuclear mappings -> treat as conventional electricity
            'nuclear': 'Conventional Electricity',
            'nuclear power': 'Conventional Electricity',
            'nuclear energy': 'Conventional Electricity',
            'atomic': 'Conventional Electricity',
        }

        # --- VALUE-BASED ENERGY TYPE DETECTION (NEW) ---
        # Instead of relying only on column names, scan all columns for known energy type values
        value_based_energy_type = None
        value_based_col = None
        value_based_val = None
        
        # Only do value-based detection if we haven't already resolved the energy type
        if resolved_emission_source_id is None:
            logging.info(f"🔍 [Value-Based Detection] Starting value-based energy type detection for row {index+1}")
            logging.info(f"🔍 [Value-Based Detection] Available energy type mappings: {list(energy_type_mappings.keys())}")
            
            # iterate patterns longest-first to avoid shorter substrings (e.g., 'gas') matching before 'biogas'
            patterns_sorted = sorted(list(energy_type_mappings.keys()), key=len, reverse=True)
            for col in source_row.index:
                val = source_row[col]
                if pd.isna(val) or val is None:
                    continue
                val_str = str(val).strip().lower()
                logging.info(f"🔍 [Value-Based Detection] Checking column '{col}' with value '{val_str}'")

                for pattern in patterns_sorted:
                    canonical = energy_type_mappings[pattern]
                    if pattern in val_str:
                        value_based_energy_type = canonical
                        value_based_col = col
                        value_based_val = val_str
                        logging.info(f"✅ [Value-Based Detection] MATCH FOUND: pattern='{pattern}' -> canonical='{canonical}' on value='{val_str}' in column='{col}'")
                        break
                    else:
                        logging.debug(f"🔍 [Value-Based Detection] No match: pattern='{pattern}' not found in value='{val_str}'")
                
                if value_based_energy_type:
                    logging.info(f"✅ [Value-Based Detection] Energy type resolved via value-based detection: '{value_based_energy_type}'")
                    break
                else:
                    logging.debug(f"🔍 [Value-Based Detection] No energy type found in column '{col}' with value '{val_str}'")
            
            if not value_based_energy_type:
                logging.warning(f"⚠️ [Value-Based Detection] No energy type could be detected from any column values")
                # Log all column values for debugging
                all_values = {col: str(source_row[col]).strip() for col in source_row.index if pd.notna(source_row[col])}
                logging.info(f"🔍 [Value-Based Detection] All column values in this row: {all_values}")
                
                # FALLBACK: Try fuzzy matching for energy types when exact matching fails
                logging.info(f"🔍 [Value-Based Detection] Attempting fuzzy matching fallback...")
                for col in source_row.index:
                    val = source_row[col]
                    if pd.isna(val) or val is None:
                        continue
                    val_str = str(val).strip().lower()
                    
                    # Skip if the value is too short or looks like a number
                    if len(val_str) < 2 or val_str.replace('.', '').replace('-', '').isdigit():
                        continue
                    
                    logging.info(f"🔍 [Value-Based Detection] Fuzzy matching column '{col}' with value '{val_str}'")
                    
                    # Try fuzzy matching against all available energy type patterns
                    best_match = None
                    best_score = 0
                    threshold = 70  # Minimum similarity score
                    
                    for pattern in patterns_sorted:
                        try:
                            from fuzzywuzzy import fuzz
                            score = fuzz.ratio(val_str, pattern)
                            if score > best_score and score >= threshold:
                                best_score = score
                                best_match = (pattern, energy_type_mappings[pattern], score)
                                logging.info(f"🔍 [Value-Based Detection] Fuzzy match candidate: '{val_str}' -> '{pattern}' (score: {score})")
                        except Exception as e:
                            logging.debug(f"🔍 [Value-Based Detection] Error in fuzzy matching: {e}")
                    
                    if best_match:
                        pattern, canonical, score = best_match
                        value_based_energy_type = canonical
                        value_based_col = col
                        value_based_val = val_str
                        logging.info(f"✅ [Value-Based Detection] FUZZY MATCH FOUND: '{val_str}' -> '{pattern}' -> '{canonical}' (score: {score})")
                        break
                    else:
                        logging.debug(f"🔍 [Value-Based Detection] No fuzzy match found for '{val_str}' (best score below threshold {threshold})")
            
            # If fuzzy matching didn't resolve, try AI classification against destination ActivityEmissionSourceName
            if not value_based_energy_type:
                try:
                    if classify_energy_value is not None and not activity_emission_source_df.empty:
                        logger_msg = f"🤖 [Value-Based Detection] Attempting AI classification for row {index+1} using available ActivityEmissionSourceName candidates"
                        logging.info(logger_msg)
                        candidate_names = activity_emission_source_df['ActivityEmissionSourceName'].astype(str).tolist()
                        # Try AI classification on non-numeric textual values in the row
                        for col in source_row.index:
                            val = source_row[col]
                            if pd.isna(val) or val is None:
                                continue
                            val_str = str(val).strip()
                            if len(val_str) < 2 or val_str.replace('.', '').replace('-', '').isdigit():
                                continue
                            try:
                                choice = classify_energy_value(val_str, candidate_names)
                                logging.info(f"🤖 [Value-Based Detection] AI classification result for '{val_str}': {choice}")
                                if choice:
                                    value_based_energy_type = choice
                                    value_based_col = col
                                    value_based_val = val_str
                                    break
                            except Exception as e:
                                logging.debug(f"🤖 [Value-Based Detection] AI classification error for '{val_str}': {e}")
                except Exception as e:
                    logging.debug(f"🤖 [Value-Based Detection] AI classification overall error: {e}")

            if not value_based_energy_type:
                logging.warning(f"⚠️ [Value-Based Detection] No energy type could be detected from any column values")
                # Log all column values for debugging
                all_values = {col: str(source_row[col]).strip() for col in source_row.index if pd.notna(source_row[col])}
                logging.info(f"🔍 [Value-Based Detection] All column values in this row: {all_values}")
            
            # If found via value-based detection, override the previous mapping logic
            if value_based_energy_type:
                mapped_energy_type = value_based_energy_type
                logging.info(f"🎯 [Value-Based Detection] SUMMARY: Successfully detected '{value_based_energy_type}' from column '{value_based_col}' with value '{value_based_val}'")
                # Lookup the corresponding ID
                row = activity_emission_source_df[activity_emission_source_df['ActivityEmissionSourceName'] == mapped_energy_type]
                if not row.empty:
                    resolved_emission_source_id = int(row['ActivityEmissionSourceID'].iloc[0])
                    if 'ActivitySubcategoryID' in row.columns and not pd.isna(row['ActivitySubcategoryID'].iloc[0]):
                        resolved_activity_subcat_from_energy = int(row['ActivitySubcategoryID'].iloc[0])
                    logging.info(f"Found ID {resolved_emission_source_id} for '{mapped_energy_type}' (value-based)")
                else:
                    logging.warning(f"⚠️ Value-based mapping to '{mapped_energy_type}' found no row in DE1_ActivityEmissionSource.")
                    # Debug: show available ActivityEmissionSourceName values
                    try:
                        samples = activity_emission_source_df['ActivityEmissionSourceName'].astype(str).tolist()
                        logging.info(f"📋 ActivityEmissionSourceName candidates (sample): {samples[:30]}")
                    except Exception:
                        logging.info("📋 Could not read activity_emission_source_df names for debug")
        # --- END VALUE-BASED ENERGY TYPE DETECTION ---
        
        # Summary of value-based detection results
        if value_based_energy_type:
            logging.info(f"🎯 [Value-Based Detection] SUMMARY: Successfully detected '{value_based_energy_type}' from column '{value_based_col}' with value '{value_based_val}'")
        else:
            logging.warning(f"⚠️ [Value-Based Detection] SUMMARY: No energy type could be detected from any column values")

        # If no direct mapping found, try fuzzy matching to ActivityEmissionSourceName
        if resolved_emission_source_id is None:
            # Use raw_energy if available, otherwise skip fuzzy matching
            if 'raw_energy' in locals() and raw_energy is not None:
                candidates = activity_emission_source_df['ActivityEmissionSourceName'].astype(str).tolist() if not activity_emission_source_df.empty else []
                logging.info(f"🔍 Attempting fuzzy match for raw_energy='{raw_energy}' against {len(candidates)} candidates")
                try:
                    top = _fuzzy_process.extract(str(raw_energy), candidates, limit=5)
                    logging.info(f"🔢 Top fuzzy candidates: {top}")
                except Exception as e:
                    logging.info(f"🔢 Could not compute top fuzzy candidates: {e}")

                mapped = mapping_utils.fuzzy_match_value_to_list(str(raw_energy), candidates, threshold=60)
                logging.info(f"🔎 Fuzzy match result: {mapped}")
                if mapped:
                    row = activity_emission_source_df[activity_emission_source_df['ActivityEmissionSourceName'].astype(str) == mapped]
                    if not row.empty:
                        resolved_emission_source_id = int(row['ActivityEmissionSourceID'].iloc[0])
                        if 'ActivitySubcategoryID' in row.columns and not pd.isna(row['ActivitySubcategoryID'].iloc[0]):
                            resolved_activity_subcat_from_energy = int(row['ActivitySubcategoryID'].iloc[0])
                        logging.info(f"✅ Fuzzy matched '{raw_energy}' to '{mapped}' with ID {resolved_emission_source_id}")
                else:
                    # fallback: try normalize and direct match
                    key = str(raw_energy).strip().lower().replace('_', ' ').replace('-', ' ')
                    key = " ".join(key.split())
                    aes_df_norm = activity_emission_source_df.copy()
                    aes_df_norm['_norm'] = (
                        aes_df_norm['ActivityEmissionSourceName'].astype(str)
                        .str.lower().str.replace('_', ' ')
                        .str.replace('-', ' ')
                        .str.replace(r'\s+', ' ', regex=True).str.strip()
                    )

                    match = aes_df_norm[aes_df_norm['_norm'] == key]
                    if not match.empty:
                        resolved_emission_source_id = int(match['ActivityEmissionSourceID'].iloc[0])
                        if 'ActivitySubcategoryID' in match.columns and not pd.isna(match['ActivitySubcategoryID'].iloc[0]):
                            resolved_activity_subcat_from_energy = int(match['ActivitySubcategoryID'].iloc[0])
                        logging.info(f"Direct matched normalized '{key}' to ID {resolved_emission_source_id}")
                
                # If still unresolved, try AI classification as a last resort
                if resolved_emission_source_id is None and 'raw_energy' in locals() and raw_energy is not None and classify_energy_value is not None:
                    try:
                        candidates = activity_emission_source_df['ActivityEmissionSourceName'].astype(str).tolist() if not activity_emission_source_df.empty else []
                        logging.info(f"🤖 [GPT] Attempting AI classification for raw_energy='{raw_energy}' using {len(candidates)} candidates")
                        choice = classify_energy_value(raw_energy, candidates)
                        logging.info(f"🤖 [GPT] Classification result: {choice}")
                        if choice:
                            row = activity_emission_source_df[activity_emission_source_df['ActivityEmissionSourceName'].astype(str).str.lower() == choice.strip().lower()]
                            if not row.empty:
                                resolved_emission_source_id = int(row['ActivityEmissionSourceID'].iloc[0])
                                if 'ActivitySubcategoryID' in row.columns and not pd.isna(row['ActivitySubcategoryID'].iloc[0]):
                                    resolved_activity_subcat_from_energy = int(row['ActivitySubcategoryID'].iloc[0])
                                logging.info(f"✅ GPT classified '{raw_energy}' to '{choice}' with ID {resolved_emission_source_id}")
                    except Exception as e:
                        logging.warning(f"⚠️ GPT classification failed: {e}")
                
                # If we still couldn't find a match, log a warning
                if resolved_emission_source_id is None:
                    logging.warning(f"❌ Could not map energy type '{raw_energy}' to any ActivityEmissionSourceName")
            else:
                logging.warning(f"Could not map energy type to any ActivityEmissionSourceName (no raw_energy available)")


        # --- DEBUG: provide detailed context about ActivityEmissionSourceID resolution ---
        try:
            # Basic summary
            logging.info(f"🔔 ActivityEmissionSourceID resolved: {resolved_emission_source_id}")

            # Show which energy columns/values were considered
            logging.info(f"🔧 energy_col={energy_col}, value_based_col={value_based_col}, value_based_val={value_based_val}")

            # Show raw_energy if present
            if 'raw_energy' in locals():
                logging.info(f"🔎 raw_energy: {raw_energy}")

            # Show a sample of the activity_emission_source_df for debugging
            try:
                logging.info(f"📋 ActivityEmissionSource table sample: {activity_emission_source_df.head(10).to_dict(orient='list')}")
            except Exception:
                logging.info("📋 Could not render activity_emission_source_df sample")

            # If not resolved, compute and log top fuzzy candidates for extra insight
            if resolved_emission_source_id is None:
                try:
                    candidates = activity_emission_source_df['ActivityEmissionSourceName'].astype(str).tolist() if not activity_emission_source_df.empty else []
                    if 'raw_energy' in locals() and raw_energy is not None and candidates:
                        top5 = _fuzzy_process.extract(str(raw_energy), candidates, limit=5)
                        logging.info(f"🔢 Top fuzzy candidates for '{raw_energy}': {top5}")
                    else:
                        logging.info("🔢 No raw_energy available or no candidates to fuzzy-match against")
                except Exception as e:
                    logging.warning(f"⚠️ Error computing fuzzy candidates for ActivityEmissionSource: {e}")

            # Show the current source row for context (limited fields)
            try:
                sr = source_row.to_dict()
                # limit to 20 keys for readability
                keys = list(sr.keys())[:20]
                summary = {k: sr[k] for k in keys}
                logging.info(f"🧾 Source row sample (first 20 cols): {summary}")
            except Exception:
                logging.info("🧾 Could not serialize source_row for debug")
        except Exception as e:
            logging.warning(f"⚠️ Error while logging ActivityEmissionSource debug info: {e}")

        # Assign resolved IDs
        new_row['ActivityEmissionSourceID'] = resolved_emission_source_id if resolved_emission_source_id is not None else None

        # If we got subcategory from the energy type, override the earlier generic subcategory
        if resolved_activity_subcat_from_energy is not None:
            new_row['ActivitySubcategoryID'] = resolved_activity_subcat_from_energy

        # Map UnitID based on the unit name in the raw data
        unit_id = None
        unit_col = None
        # Try to find the unit column in mapping or auto-detect
        if isinstance(mappings, dict) and 'UnitID' in mappings and isinstance(mappings['UnitID'], dict):
            unit_col = mappings['UnitID'].get('source_column')
        if not unit_col or unit_col not in source_df.columns:
            for col in source_df.columns:
                if col.lower() in ['consumption_unit', 'unit', 'unitname']:
                    unit_col = col
                    break
        # Try to get unit from column, else infer from headers
        unit_name = None
        if unit_col and unit_col in source_df.columns:
            raw_val = source_row[unit_col]
            logging.info(f"🔍 Inspecting unit column value for '{unit_col}': {raw_val!r} ({type(raw_val)})")
            # If the column value is numeric (e.g., consumption amount) but the column name contains a unit
            # (like 'Consumption_m3' or 'Volume (m3)'), infer the unit from the column name instead.
            inferred_from_header = None
            try:
                # If the value looks numeric, prefer header-based inference
                if pd.isna(raw_val):
                    logging.info("ℹ️ Unit column value is NaN — will try to infer from header")
                    inferred_from_header = extract_unit_from_column(unit_col)
                elif isinstance(raw_val, (int, float)):
                    logging.info("ℹ️ Unit column contains numeric value — inferring unit from header")
                    inferred_from_header = extract_unit_from_column(unit_col)
                else:
                    # If it's a string, but looks like a number, also treat as numeric
                    s = str(raw_val).strip()
                    if re.match(r'^-?\d+(?:\.\d+)?$', s):
                        logging.info("ℹ️ Unit column string value looks numeric — inferring unit from header")
                        inferred_from_header = extract_unit_from_column(unit_col)
            except Exception as e:
                logging.warning(f"⚠️ Error while inferring from header: {e}")

            if inferred_from_header:
                logging.info(f"📌 Inferred unit from header '{unit_col}': {inferred_from_header}")
                unit_name = inferred_from_header
            else:
                unit_name = raw_val
        else:
            # Try to infer from column names (e.g., 'consumption_m3', 'EmissionFactor_kgCO2e_per_m3')
            for col in source_df.columns:
                inferred_unit = extract_unit_from_column(col)
                if inferred_unit:
                    unit_name = inferred_unit
                    break
                
        if unit_name:
            # Debug: show where the unit was picked from
            logging.info(f"🔎 Unit column used: {unit_col}")
            logging.info(f"🧾 Raw unit value: {unit_name!r}")

            # If unit_name looks like a value containing a unit (e.g., '1484.8 kWh'), try extracting the unit token
            try:
                if isinstance(unit_name, str):
                    extracted = extract_unit_from_value(unit_name)
                    if extracted:
                        logging.info(f"🔧 Extracted unit from value: {extracted!r} (from '{unit_name}')")
                        unit_name = extracted
            except Exception as e:
                logging.warning(f"⚠️ Error extracting unit from value: {e}")

            # Normalize and coerce to string, then perform case-insensitive, trimmed comparison
            normalized_unit_name = normalize_unit(unit_name)
            logging.info(f"⚙️ Normalized unit (mapping applied): {normalized_unit_name!r}")

            norm_name = str(normalized_unit_name).strip().lower()
            logging.info(f"🧼 Comparison key (trim+lower): {norm_name!r}")

            # Ensure unit_df is populated; if empty, try pulling from dest_tables (fallback)
            try:
                if (unit_df is None or (hasattr(unit_df, 'empty') and unit_df.empty)) and dest_tables and 'DE1_Unit' in dest_tables:
                    unit_df = dest_tables['DE1_Unit']
                    print("fact.generate_fact: fallback loaded DE1_Unit from dest_tables into unit_df")
            except Exception:
                pass

            # Show what unit_df contains (columns + sample)
            try:
                print(f"fact.generate_fact: unit_df columns={list(unit_df.columns)}; empty={unit_df.empty}")
                print(f"fact.generate_fact: unit_df sample={unit_df.head(6).to_dict(orient='list')}")
            except Exception:
                pass

            # Robustly determine unit name column in unit_df
            unit_name_col = None
            if unit_df is not None and not unit_df.empty:
                norm_map = {re.sub(r'[^a-z0-9]', '', c.lower()): c for c in unit_df.columns}
                for key, orig in norm_map.items():
                    if key in ('unitname', 'unit'):
                        unit_name_col = orig
                        break
                if unit_name_col is None and 'UnitName' in unit_df.columns:
                    unit_name_col = 'UnitName'

            logging.info(f"🔧 Using unit name column: {unit_name_col}")

            unit_id = None
            if unit_name_col and unit_name_col in unit_df.columns:
                try:
                    available = unit_df[unit_name_col].astype(str).str.strip().str.lower().unique().tolist()
                except Exception:
                    available = []
                logging.info(f"🔎 Available destination unit names (sample): {available[:20]}")
                # Extra low-level debug: show reprs and lengths to surface hidden characters
                try:
                    debug_reprs = [(u, repr(u), len(u)) for u in available[:20]]
                    print(f"fact.generate_fact: available unit reprs (val, repr, len): {debug_reprs}")
                except Exception:
                    pass

                # Exact match
                unit_id_lookup = unit_df[unit_df[unit_name_col].astype(str).str.strip().str.lower() == norm_name]
                if not unit_id_lookup.empty:
                    unit_id = int(unit_id_lookup['UnitID'].iloc[0]) if 'UnitID' in unit_id_lookup.columns else int(unit_id_lookup.iloc[0,0])
                    logging.info(f"✅ Matched UnitID: {unit_id} for unit '{normalized_unit_name}' using column '{unit_name_col}'")
                else:
                    # Try fuzzy match
                    try:
                        best = mapping_utils.fuzzy_match_value_to_list(normalized_unit_name, available, threshold=60)
                        logging.info(f"🔍 Fuzzy best match for '{normalized_unit_name}' -> {best}")
                        if best:
                            unit_id_lookup = unit_df[unit_df[unit_name_col].astype(str).str.strip().lower() == str(best).strip().lower()]
                            if not unit_id_lookup.empty:
                                unit_id = int(unit_id_lookup['UnitID'].iloc[0]) if 'UnitID' in unit_id_lookup.columns else int(unit_id_lookup.iloc[0,0])
                                logging.info(f"✅ Fuzzy Matched UnitID: {unit_id} for unit '{normalized_unit_name}'")
                    except Exception as e:
                        logging.warning(f"⚠️ Fuzzy match error: {e}")
            else:
                logging.warning(f"⚠️ Could not find a unit name column in DE1_Unit to match '{normalized_unit_name}'")

            if unit_id is None:
                # Extra diagnostics: show unicode codepoints for normalized unit and available units
                try:
                    import unicodedata
                    norm_ascii = unicodedata.normalize('NFKD', str(normalized_unit_name)).encode('ascii', 'ignore').decode().strip().lower()
                    available_ascii = [unicodedata.normalize('NFKD', str(a)).encode('ascii','ignore').decode().strip().lower() for a in available]
                except Exception:
                    norm_ascii = None
                    available_ascii = []

                try:
                    # show codepoints for the searched unit
                    cp_norm = [hex(ord(ch)) for ch in str(normalized_unit_name)]
                except Exception:
                    cp_norm = []

                try:
                    cp_available = {a: [hex(ord(ch)) for ch in a] for a in available[:20]}
                except Exception:
                    cp_available = {}

                logging.warning(f"❌ No UnitID match for '{normalized_unit_name}' (looked for {norm_name})")
                logging.info(f"diagnostics: normalized ascii='{norm_ascii}', available_ascii sample={available_ascii[:20]}")
                logging.info(f"diagnostics: normalized codepoints={cp_norm}")
                logging.info(f"diagnostics: available codepoints sample={cp_available}")
        
        new_row['UnitID'] = int(unit_id) if unit_id is not None else None
        
        # Handle EmissionFactorID - ALWAYS generate based on final resolved ActivityEmissionSourceID
        # DO NOT use pre-generated emission_factor_id as it may be based on incorrect assumptions
        logging.info(f"🔍 [EmissionFactorID] Starting generation process...")
        logging.info(f"🔍 [EmissionFactorID] Pre-generated emission_factor_id: {emission_factor_id}")
        logging.info(f"🔍 [EmissionFactorID] Final resolved ActivityEmissionSourceID: {new_row['ActivityEmissionSourceID']}")

        # Always generate EmissionFactorID based on country ISO2Code and final ActivityEmissionSourceName
        country_iso2 = lookup_value(country_df, 'CountryName', country, 'ISO2Code')
        logging.info(f"🔍 [EmissionFactorID] Country: {country}, ISO2Code: {country_iso2}")

        # Get ActivityEmissionSourceName if we have ActivityEmissionSourceID
        emission_source_name = None
        if new_row['ActivityEmissionSourceID'] is not None:
            emission_source_name_lookup = activity_emission_source_df[
                activity_emission_source_df['ActivityEmissionSourceID'] == new_row['ActivityEmissionSourceID']
            ]
            if not emission_source_name_lookup.empty:
                emission_source_name = emission_source_name_lookup['ActivityEmissionSourceName'].iloc[0]
            logging.info(f"🔍 [EmissionFactorID] Found emission source name: '{emission_source_name}' for ID {new_row['ActivityEmissionSourceID']}")
        else:
            logging.warning(f"⚠️ [EmissionFactorID] No emission source found for ID {new_row['ActivityEmissionSourceID']}")

        if emission_source_name and country_iso2:
            # Debug: show what emission_source_name lookup returned
            try:
                logging.info(f"🔍 [EmissionFactorID] Emission source lookup details: {emission_source_name_lookup.to_dict(orient='list')}")
            except Exception as e:
                logging.warning(f"⚠️ [EmissionFactorID] Error showing emission source lookup details: {e}")

            # Format the EmissionFactorID as ISO2Code_ActivityEmissionSourceName with spaces replaced by underscores
            new_row['EmissionFactorID'] = f"{country_iso2}_{str(emission_source_name).strip().replace(' ', '_')}"
            logging.info(f"✅ [EmissionFactorID] Generated new EmissionFactorID: '{new_row['EmissionFactorID']}' (country_iso2='{country_iso2}', emission_source_name='{emission_source_name}')")

            # Log the difference if pre-generated was different
            if emission_factor_id and emission_factor_id != new_row['EmissionFactorID']:
                logging.warning(f"⚠️ [EmissionFactorID] Pre-generated was '{emission_factor_id}', but corrected to '{new_row['EmissionFactorID']}' based on actual data")
        else:
            if not country_iso2:
                logging.error(f"❌ [EmissionFactorID] Missing country_iso2 for EmissionFactorID generation")
            if not emission_source_name:
                logging.error(f"❌ [EmissionFactorID] Missing emission_source_name for EmissionFactorID generation")

            # Fallback: try to use pre-generated if available, otherwise use unknown
            if emission_factor_id:
                logging.warning(f"⚠️ [EmissionFactorID] Using pre-generated as fallback: {emission_factor_id}")
                new_row['EmissionFactorID'] = emission_factor_id
            else:
                new_row['EmissionFactorID'] = "Unknown_EmissionFactor"
                logging.error(f"❌ [EmissionFactorID] No fallback available, using 'Unknown_EmissionFactor'")

        date_key = get_date_key(date_df, mappings.get('DateKey', {}).get('source_column'), reporting_year, source_row.get(mappings.get('DateKey', {}).get('source_column')))
        new_row['DateKey'] = int(date_key) if date_key is not None else None

        # Map direct values from source data
        for field_name, mapping_config in mappings.items():
            source_column = mapping_config.get("source_column")
            
            # BUG FIX 2: Handle ConsumptionAmount calculation for air travel
            if field_name == 'ConsumptionAmount':
                # Check for air travel consumption calculation
                consumption_type = mapping_config.get("consumption_type", "").lower()
                should_calculate_distance = (
                    consumption_type == "distance" and 
                    calc_method == 'Consumption-based' and 
                    activity_cat.lower() == 'business travel' and
                    activity_subcat.lower() == 'air travel'
                )
                
                if should_calculate_distance:
                    # For air travel, try to calculate distance
                    distance = None
                    
                    # Method 1: If we have detected origin/destination columns, use them
                    if origin_column and destination_column:
                        origin_code = source_row.get(origin_column)
                        dest_code = source_row.get(destination_column)
                        # Handle NaN values in origin/destination codes
                        if pd.isna(origin_code) or pd.isna(dest_code):
                            distance = None
                        else:
                            distance = calculate_airport_distance(origin_code, dest_code)
                            if distance:
                                logging.info(f"Calculated air travel distance: {origin_code} -> {dest_code} = {distance} km")
                    
                    # Method 2: Try to find airport codes in any available source columns
                    if not distance:
                        airport_codes = []
                        for col in source_df.columns:
                            value = source_row.get(col)
                            if value and not pd.isna(value) and isinstance(value, str) and len(value.strip()) == 3:
                                code = value.strip().upper()
                                # Check if this looks like an airport code (exists in our database)
                                from .airport_distance import get_airport_coordinates
                                if get_airport_coordinates(code):
                                    airport_codes.append(code)
                        
                        # If we found exactly 2 airport codes, calculate distance
                        if len(airport_codes) >= 2:
                            distance = calculate_airport_distance(airport_codes[0], airport_codes[1])
                            if distance:
                                logging.info(f"Calculated air travel distance from detected codes: {airport_codes[0]} -> {airport_codes[1]} = {distance} km")
                    
                    new_row[field_name] = float(distance) if distance is not None else None
                    if distance:
                        logging.info(f"ConsumptionAmount set to: {distance} km")
                    else:
                        logging.warning(f"Could not calculate distance for air travel - no valid airport codes found")
                elif source_column and source_column in source_df.columns:
                    value = source_row[source_column]
                    # print("###############", value)
                    # Use safe float conversion function
                    new_row[field_name] = safe_float_conversion(value)
                    if new_row[field_name] is None:
                        logging.warning(f"Could not convert ConsumptionAmount value '{value}' to float")
                else:
                    # Default value if no mapping exists
                    new_row[field_name] = 1.0  # Default consumption amount
                    logging.info(f"Using default ConsumptionAmount: 1.0")
                    
            elif field_name == 'PaidAmount':
                if source_column and source_column in source_df.columns:
                    value = source_row[source_column]
                    # Use safe float conversion function
                    new_row[field_name] = safe_float_conversion(value)
                    if new_row[field_name] is None:
                        logging.warning(f"Could not convert PaidAmount value '{value}' to float")
                else:
                    # Default value if no mapping exists
                    new_row[field_name] = 0.0  # Default paid amount
                    logging.info(f"Using default PaidAmount: 0.0")
            
            # Handle provider and currency if present in mappings
            if field_name == 'ActivityEmissionSourceProviderID':
                if source_column and source_column in source_df.columns:
                    provider_name = source_row[source_column]
                    # Handle NaN values in provider name
                    if pd.isna(provider_name) or provider_name is None:
                        new_row[field_name] = None
                    else:
                        # First try to find in the activity_emmission_source_provider_df
                        provider_id = lookup_value(activity_emmission_source_provider_df, 
                                                 'ProviderName', provider_name, 'ActivityEmissionSourceProviderID')
                        
                        if provider_id is not None:
                            new_row[field_name] = safe_int_conversion(provider_id)
                        else:
                            # If not found in the activity_emmission_source_provider_df, check if it exists in the destination tables
                            if dest_tables and 'DE1_ActivityEmissionSourceProvi' in dest_tables:
                                provider_df = dest_tables['DE1_ActivityEmissionSourceProvi']
                                provider_row = provider_df[provider_df['ProviderName'].str.lower() == provider_name.lower() if provider_name else False]
                                if not provider_row.empty:
                                    new_row[field_name] = safe_int_conversion(provider_row['ActivityEmissionSourceProviderID'].iloc[0])
                                else:
                                    # If not found in destination tables, create a new provider entry
                                    if not provider_df.empty:
                                        new_provider_id = provider_df['ActivityEmissionSourceProviderID'].max() + 1
                                    else:
                                        new_provider_id = 1
                                    
                                    # Add to activity_emmission_source_provider_df for future lookups
                                    new_provider = pd.DataFrame({
                                        'ActivityEmissionSourceProviderID': [new_provider_id],
                                        'ProviderName': [provider_name]
                                    })
                                    activity_emmission_source_provider_df = pd.concat([activity_emmission_source_provider_df, new_provider], ignore_index=True)
                                    
                                    new_row[field_name] = safe_int_conversion(new_provider_id)
                            else:
                                new_row[field_name] = None  # No destination tables provided
                else:
                    # If no mapping exists, try to find a default provider
                    if not activity_emmission_source_provider_df.empty:
                        new_row[field_name] = safe_int_conversion(activity_emmission_source_provider_df['ActivityEmissionSourceProviderID'].iloc[0])
                    elif dest_tables and 'DE1_ActivityEmissionSourceProvi' in dest_tables and not dest_tables['DE1_ActivityEmissionSourceProvi'].empty:
                        # Use the first provider from destination tables
                        new_row[field_name] = safe_int_conversion(dest_tables['DE1_ActivityEmissionSourceProvi']['ActivityEmissionSourceProviderID'].iloc[0])
                    else:
                        new_row[field_name] = None  # No providers available
            
            # Handle OrganizationalUnitID (note: there might be a space in the column name)
            if field_name.strip() == 'OrganizationalUnitID':
                # Use the correct field name with space for the new_row
                actual_field_name = 'OrganizationalUnitID ' if 'OrganizationalUnitID ' in dest_df.columns else 'OrganizationalUnitID'
                
                # Always set the organizational unit ID to 1 since we know there's only one unit
                            
                        
                    
            # Handle EmissionFactorID - This is now handled above in the main logic
            # No need to duplicate the logic here
            
            if field_name == 'CurrencyID':
                if source_column and source_column in source_df.columns:
                    currency_code = source_row[source_column]

                    # If value is not provided or NaN
                    if pd.isna(currency_code) or currency_code is None:
                        new_row[field_name] = None
                        logging.info(f"💸 [CurrencyID] Source column '{source_column}' is NaN or None for row {index+1}")
                        continue

                    # If the cell contains a numeric value (e.g., '150.25'), try to infer currency from header or nearby values/symbols
                    try:
                        float(currency_code)
                        logging.info(f"💸 [CurrencyID] Source column '{source_column}' contains numeric value '{currency_code}' - extracting currency from column name/value")
                        from .mapping_utils import extract_currency_from_column, _currency_code_from_symbol_in_text, extract_currency_from_value

                        # Try header-based extraction first
                        extracted_currency = extract_currency_from_column(source_column)

                        # Try dynamic symbol lookup using destination currency table
                        currency_df_for_map = None
                        try:
                            if dest_tables and 'DE1_Currency' in dest_tables:
                                currency_df_for_map = dest_tables['DE1_Currency']
                        except Exception:
                            currency_df_for_map = None

                        if not extracted_currency:
                            sym_code = _currency_code_from_symbol_in_text(source_column, currency_df_for_map)
                            if sym_code:
                                extracted_currency = sym_code

                        # If still not found, try to detect symbol in the value (e.g., '€100') or value-based codes
                        if not extracted_currency:
                            extracted_currency = extract_currency_from_value(currency_code)
                            if not extracted_currency and currency_df_for_map is not None:
                                try:
                                    extracted_currency = _currency_code_from_symbol_in_text(str(currency_code), currency_df_for_map)
                                except Exception:
                                    pass

                        logging.info(f"🔍💱 [CurrencyID] Column name: '{source_column}', Extracted currency: '{extracted_currency}'")

                        if extracted_currency:
                            currency_id = lookup_value(currency_df, 'CurrencyCode', extracted_currency, 'CurrencyID')
                            if currency_id is not None:
                                new_row[field_name] = safe_int_conversion(currency_id)
                                logging.info(f"💱 [CurrencyID] Extracted currency '{extracted_currency}' from column '{source_column}' and mapped to CurrencyID '{currency_id}' for row {index+1}")
                            else:
                                new_row[field_name] = None
                                logging.warning(f"⚠️💱 [CurrencyID] Extracted currency '{extracted_currency}' from column '{source_column}' but not found in currency dimension table")
                        else:
                            new_row[field_name] = None
                            logging.warning(f"⚠️💱 [CurrencyID] Could not extract currency from column name '{source_column}' for numeric value '{currency_code}'")

                    except (ValueError, TypeError):
                        # Not numeric, treat as regular currency code/value
                        currency_id = lookup_value(currency_df, 'CurrencyCode', currency_code, 'CurrencyID')
                        new_row[field_name] = safe_int_conversion(currency_id)
                        if currency_id is not None:
                            logging.info(f"💱 [CurrencyID] Mapped currency code '{currency_code}' to CurrencyID '{currency_id}' for row {index+1}")
                        else:
                            logging.warning(f"⚠️💱 [CurrencyID] Currency code '{currency_code}' not found in currency dimension table")
                else:
                    # Enhanced currency detection from column names if no explicit mapping
                    detected_currency = None

                    # Debug: Show what we're working with
                    logging.info(f"🔍💱 [CurrencyID] Row {index+1}: Starting enhanced currency detection...")
                    logging.info(f"🔍💱 [CurrencyID] Available columns: {list(source_df.columns)}")
                    logging.info(f"🔍💱 [CurrencyID] Current row values: {dict(source_row)}")

                    # Step 1: Check column names for currency patterns (e.g., TotalPaid(EUR), amount_USD, amount_€)
                    logging.info(f"🔍💱 [CurrencyID] No explicit mapping - checking column names for currency patterns...")

                    try:
                        if not currency_df.empty:
                            available_currencies = currency_df['CurrencyCode'].astype(str).tolist() if 'CurrencyCode' in currency_df.columns else []
                            logging.info(f"🔍💱 [CurrencyID] Available currencies in dimension table: {available_currencies}")
                        else:
                            logging.warning(f"⚠️💱 [CurrencyID] Currency dimension table is empty!")
                    except Exception as e:
                        logging.warning(f"⚠️💱 [CurrencyID] Error checking currency dimension table: {e}")

                    for col in source_df.columns:
                        from .mapping_utils import extract_currency_from_column
                        extracted_currency = extract_currency_from_column(col)
                        print("#######################", extracted_currency)
                        if extracted_currency:
                            # Check if this currency exists in our currency dimension table
                            currency_id = lookup_value(currency_df, 'CurrencyCode', extracted_currency, 'CurrencyID')
                            if currency_id is not None:
                                detected_currency = currency_id
                                logging.info(f"🔍💱 [CurrencyID] Auto-detected currency '{extracted_currency}' from column '{col}' and mapped to CurrencyID '{currency_id}' for row {index+1}")
                                break

                    # Step 2: If no currency found in column names, check data values for currency codes or symbols
                    if detected_currency is None:
                        logging.info(f"🔍💱 [CurrencyID] No currency found in column names - checking data values for currency codes/symbols...")
                        try:
                            from .mapping_utils import extract_currency_from_value, _currency_code_from_symbol_in_text

                            sample_rows = source_df.head(10)
                            found_currencies = set()

                            for _, sample_row in sample_rows.iterrows():
                                for col_name, value in sample_row.items():
                                    if pd.notna(value):
                                        extracted_currency = extract_currency_from_value(value)
                                        if extracted_currency:
                                            found_currencies.add(extracted_currency.upper())
                                        else:
                                            # Try symbol detection in the value
                                            sym_code = _currency_code_from_symbol_in_text(str(value), currency_df if not currency_df.empty else None)
                                            if sym_code:
                                                found_currencies.add(sym_code.upper())

                            logging.info(f"🔍💱 [CurrencyID] Found currency codes in data values: {found_currencies}")

                            for found_currency in found_currencies:
                                currency_id = lookup_value(currency_df, 'CurrencyCode', found_currency, 'CurrencyID')
                                if currency_id is not None:
                                    detected_currency = currency_id
                                    logging.info(f"🔍💱 [CurrencyID] Auto-detected currency '{found_currency}' from data values and mapped to CurrencyID '{currency_id}' for row {index+1}")
                                    break

                        except Exception as e:
                            logging.warning(f"⚠️💱 [CurrencyID] Error while checking data values for currency: {e}")

                    # Step 3: Apply the detected currency or fallback
                    if detected_currency is not None:
                        new_row[field_name] = safe_int_conversion(detected_currency)
                        logging.info(f"✅💱 [CurrencyID] Successfully mapped currency for row {index+1}")
                    elif not currency_df.empty:
                        new_row[field_name] = safe_int_conversion(currency_df['CurrencyID'].iloc[0])
                        logging.info(f"⚠️💱 [CurrencyID] No explicit or detected currency, using default CurrencyID '{currency_df['CurrencyID'].iloc[0]}' for row {index+1}")
                    else:
                        new_row[field_name] = None
                        logging.warning(f"❌💱 [CurrencyID] No currency could be mapped for row {index+1}")
        
        # Update progress
        progress = (index + 1) / total_records
        progress_bar.progress(progress)
        status_text.text(f"Processing record {index + 1} of {total_records}")
        update_progress(processed=index + 1)
        
        # Add the new row to the result DataFrame
        # print(f"Debug - Final new_row before adding to result_df: {new_row}")
        result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
        # print(f"Debug - result_df after adding new row: {result_df.tail(1)}")

    # Record end time and calculate duration
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Update final status with timing information
    status_text.text("Processing Complete!")
    update_progress(status="Complete")
    
    # Show summary with timing information  
    st.success(f"Processed {len(result_df)} new records (no mock data included)")
    if is_air_travel_consumption:
        st.info("Air travel distances calculated and included in ConsumptionAmount")
    st.write(f"Total Records in Fact Table: {len(result_df)}")
    st.write(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"Duration: {str(duration).split('.')[0]}")  # Format duration without microseconds

    return result_df