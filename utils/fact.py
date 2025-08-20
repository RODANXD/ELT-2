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
from .mapping_utils import normalize_text, fuzzy_match_value_to_list, normalize_unit, extract_unit_from_column

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

def safe_float_conversion(value):
    """
    Safely convert a value to float, handling NaN and None values.
    Returns None if conversion is not possible.
    """
    if pd.isna(value) or value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def find_emission_ids(mappings, activity_subcat, activity_subcat_df, activity_emission_source_df, country_df, country, calc_method):
    """Returns ActivityEmissionSourceID, UnitID, and EmissionFactorID based on mapping conditions."""
    
    # Get basic lookup values
    activity_sub_cat_id = lookup_value(activity_subcat_df, 'ActivitySubcategoryName', activity_subcat, 'ActivitySubcategoryID')
    iso2_code = lookup_value(country_df, 'CountryName', country, 'ISO2Code')
    
    # Special handling for Electricity (ActivitySubcategoryID 21)
    
    
    # Define transformation suffixes based on calc_method
    valid_transformations = ['Distance', 'Fuel', 'Electricity', 'Heating', 'Days'] if calc_method == 'Consumption-based' else ['Currency']
    
    # Find the first matching transformation
    transformation = None
    amount_key = 'ConsumptionAmount' 
    for key, mapping_info in mappings.items():
        if amount_key in key:
            trans = mapping_info.get('consumption_type', '').lower()
            print(trans, '=', [x.lower() for x in valid_transformations])
            if trans in [x.lower() for x in valid_transformations]:
                transformation = trans
                break

    if not transformation:
        logging.warning(f"No valid transformation found for {activity_subcat}")
        return None, None, None
    
    # Get emission source ID by suffix
    # Try to match emission source intelligently: by suffix first, else fuzzy match on names
    emission_source_id = get_emission_source_id_by_suffix(
        activity_emission_source_df, activity_sub_cat_id, transformation
    )
    if not emission_source_id:
        # fuzzy match transformation to ActivityEmissionSourceName
        candidates = activity_emission_source_df['ActivityEmissionSourceName'].astype(str).tolist() if not activity_emission_source_df.empty else []
        mapped = fuzzy_match_value_to_list(transformation, candidates, threshold=60) if transformation else None
        if mapped:
            row = activity_emission_source_df[activity_emission_source_df['ActivityEmissionSourceName'].astype(str) == mapped]
            if not row.empty:
                emission_source_id = int(row['ActivityEmissionSourceID'].iloc[0])
    
    if not emission_source_id:
        logging.warning(f"No emission source ID found for {activity_subcat}")
        return None, None, None
    
    # Get unit ID and emission factor ID
    unit_id = lookup_value(activity_emission_source_df, 'ActivityEmissionSourceID', emission_source_id, 'UnitID')
    emission_source_name = lookup_value(activity_emission_source_df, 'ActivityEmissionSourceID', emission_source_id, 'ActivityEmissionSourceName')
    emission_factor_id = f"{iso2_code}_{emission_source_name.replace(' ', '_')}"
    
    return emission_source_id, unit_id, emission_factor_id


def get_emission_source_id_by_suffix(df, subcategory_id, suffix):
    """Get emission source ID for sources ending with suffix (case-insensitive) and matching subcategory ID."""
    filtered = df[
        (df['ActivitySubcategoryID'] == subcategory_id) &
        (df['ActivityEmissionSourceName'].str.lower().str.contains(suffix.lower()))
    ]
    return filtered.iloc[0]['ActivityEmissionSourceID'] if not filtered.empty else None



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
    source_date = pd.to_datetime(date_value, errors='coerce')
    
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
            'lng': 'Natural Gas',
            'methane': 'Natural Gas',
        }

        # --- VALUE-BASED ENERGY TYPE DETECTION (NEW) ---
        # Instead of relying only on column names, scan all columns for known energy type values
        value_based_energy_type = None
        value_based_col = None
        value_based_val = None
        
        # Only do value-based detection if we haven't already resolved the energy type
        if resolved_emission_source_id is None:
            for col in source_row.index:
                val = source_row[col]
                if pd.isna(val) or val is None:
                    continue
                val_str = str(val).strip().lower()
                for pattern, canonical in energy_type_mappings.items():
                    if pattern in val_str:
                        value_based_energy_type = canonical
                        value_based_col = col
                        value_based_val = val_str
                        break
                if value_based_energy_type:
                    break
            
            # If found via value-based detection, override the previous mapping logic
            if value_based_energy_type:
                mapped_energy_type = value_based_energy_type
                logging.info(f"Value-based energy type detected: '{value_based_energy_type}' from column '{value_based_col}' and value '{value_based_val}'")
                # Lookup the corresponding ID
                row = activity_emission_source_df[activity_emission_source_df['ActivityEmissionSourceName'] == mapped_energy_type]
                if not row.empty:
                    resolved_emission_source_id = int(row['ActivityEmissionSourceID'].iloc[0])
                    if 'ActivitySubcategoryID' in row.columns and not pd.isna(row['ActivitySubcategoryID'].iloc[0]):
                        resolved_activity_subcat_from_energy = int(row['ActivitySubcategoryID'].iloc[0])
                    logging.info(f"Found ID {resolved_emission_source_id} for '{mapped_energy_type}' (value-based)")
        # --- END VALUE-BASED ENERGY TYPE DETECTION ---

        # If no direct mapping found, try fuzzy matching to ActivityEmissionSourceName
        if resolved_emission_source_id is None:
            # Use raw_energy if available, otherwise skip fuzzy matching
            if 'raw_energy' in locals() and raw_energy is not None:
                candidates = activity_emission_source_df['ActivityEmissionSourceName'].astype(str).tolist() if not activity_emission_source_df.empty else []
                mapped = fuzzy_match_value_to_list(str(raw_energy), candidates, threshold=60)
                if mapped:
                    row = activity_emission_source_df[activity_emission_source_df['ActivityEmissionSourceName'].astype(str) == mapped]
                    if not row.empty:
                        resolved_emission_source_id = int(row['ActivityEmissionSourceID'].iloc[0])
                        if 'ActivitySubcategoryID' in row.columns and not pd.isna(row['ActivitySubcategoryID'].iloc[0]):
                            resolved_activity_subcat_from_energy = int(row['ActivitySubcategoryID'].iloc[0])
                        logging.info(f"Fuzzy matched '{raw_energy}' to '{mapped}' with ID {resolved_emission_source_id}")
                else:
                    # fallback: try normalize and direct match
                    key = str(raw_energy).strip().lower().replace('_', ' ', regex=False).replace('-', ' ', regex=False)
                    key = " ".join(key.split())
                    aes_df_norm = activity_emission_source_df.copy()
                    aes_df_norm['_norm'] = (
                        aes_df_norm['ActivityEmissionSourceName'].astype(str)
                        .str.lower().str.replace('_', ' ', regex=False)
                        .str.replace('-', ' ', regex=False)
                        .str.replace(r'\s+', ' ', regex=True).str.strip()
                    )

                    match = aes_df_norm[aes_df_norm['_norm'] == key]
                    if not match.empty:
                        resolved_emission_source_id = int(match['ActivityEmissionSourceID'].iloc[0])
                        if 'ActivitySubcategoryID' in match.columns and not pd.isna(match['ActivitySubcategoryID'].iloc[0]):
                            resolved_activity_subcat_from_energy = int(match['ActivitySubcategoryID'].iloc[0])
                        logging.info(f"Direct matched normalized '{key}' to ID {resolved_emission_source_id}")
                
                # If we still couldn't find a match, log a warning
                if resolved_emission_source_id is None:
                    logging.warning(f"Could not map energy type '{raw_energy}' to any ActivityEmissionSourceName")
            else:
                logging.warning(f"Could not map energy type to any ActivityEmissionSourceName (no raw_energy available)")


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
            unit_name = source_row[unit_col]
        else:
            # Try to infer from column names (e.g., 'consumption_m3', 'EmissionFactor_kgCO2e_per_m3')
            for col in source_df.columns:
                inferred_unit = extract_unit_from_column(col)
                if inferred_unit:
                    unit_name = inferred_unit
                    break
                
        if unit_name:
            normalized_unit_name = normalize_unit(unit_name)
            unit_id_lookup = unit_df[unit_df['UnitName'].astype(str) == normalized_unit_name]
            if not unit_id_lookup.empty:
                unit_id = int(unit_id_lookup['UnitID'].iloc[0])
            else:
                unit_id = None
        else:
            unit_id = None
        
        new_row['UnitID'] = int(unit_id) if unit_id is not None else None
        
        # Handle EmissionFactorID - Use pre-generated value if available, otherwise generate new one
        if emission_factor_id is not None:
            new_row['EmissionFactorID'] = emission_factor_id
            logging.info(f"Using pre-generated EmissionFactorID: {emission_factor_id}")
        else:
            # Generate EmissionFactorID based on country ISO2Code and ActivityEmissionSourceName
            country_iso2 = lookup_value(country_df, 'CountryName', country, 'ISO2Code')
            
            # Get ActivityEmissionSourceName if we have ActivityEmissionSourceID
            emission_source_name = None
            if new_row['ActivityEmissionSourceID'] is not None:
                emission_source_name_lookup = activity_emission_source_df[
                    activity_emission_source_df['ActivityEmissionSourceID'] == new_row['ActivityEmissionSourceID']
                ]
                if not emission_source_name_lookup.empty:
                    emission_source_name = emission_source_name_lookup['ActivityEmissionSourceName'].iloc[0]
            
            if country_iso2 and emission_source_name:
                # Format the EmissionFactorID as ISO2Code_ActivityEmissionSourceName with spaces replaced by underscores
                new_row['EmissionFactorID'] = f"{country_iso2}_{str(emission_source_name).strip().replace(' ', '_')}"
                logging.info(f"Generated new EmissionFactorID: {new_row['EmissionFactorID']}")
            else:
                new_row['EmissionFactorID'] = "Unknown_EmissionFactor"
                if not country_iso2:
                    logging.warning(f"Missing country_iso2 for EmissionFactorID generation")
                if not emission_source_name:
                    logging.warning(f"Missing emission_source_name for EmissionFactorID generation")

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
                    # Handle NaN values in currency code
                    if pd.isna(currency_code) or currency_code is None:
                        new_row[field_name] = None
                    else:
                        currency_id = lookup_value(currency_df, 'CurrencyCode', currency_code, 'CurrencyID')
                        new_row[field_name] = safe_int_conversion(currency_id)
                else:
                    # Enhanced currency detection from column names if no explicit mapping
                    detected_currency = None
                    for col in source_df.columns:
                        from .mapping_utils import extract_currency_from_column
                        extracted_currency = extract_currency_from_column(col)
                        if extracted_currency:
                            # Check if this currency exists in our currency dimension table
                            currency_id = lookup_value(currency_df, 'CurrencyCode', extracted_currency, 'CurrencyID')
                            if currency_id is not None:
                                detected_currency = currency_id
                                logging.info(f"Auto-detected currency '{extracted_currency}' from column '{col}'")
                                break
                    
                    if detected_currency is not None:
                        new_row[field_name] = safe_int_conversion(detected_currency)
                    elif not currency_df.empty:
                        # Fallback to default currency
                        new_row[field_name] = safe_int_conversion(currency_df['CurrencyID'].iloc[0])
                        logging.info(f"Using default currency ID: {currency_df['CurrencyID'].iloc[0]}")
                    else:
                        new_row[field_name] = None
        
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