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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def find_emission_ids(mappings, activity_subcat, activity_subcat_df, activity_emission_source_df, country_df, country, calc_method):
    """Returns ActivityEmissionSourceID, UnitID, and EmissionFactorID based on mapping conditions."""
    
    # Get basic lookup values
    activity_sub_cat_id = lookup_value(activity_subcat_df, 'ActivitySubcategoryName', activity_subcat, 'ActivitySubcategoryID')
    iso2_code = lookup_value(country_df, 'CountryName', country, 'ISO2Code')
    
    # Special handling for Electricity (ActivitySubcategoryID 21)
    if activity_sub_cat_id == 21:
        # Try to find Green Electricity first
        green_electricity = activity_emission_source_df[
            (activity_emission_source_df['ActivitySubcategoryID'] == activity_sub_cat_id) &
            (activity_emission_source_df['ActivityEmissionSourceName'] == 'Green Electricity')
        ]
        
        if not green_electricity.empty:
            emission_source_id = green_electricity.iloc[0]['ActivityEmissionSourceID']
            unit_id = green_electricity.iloc[0]['UnitID']
            emission_factor_id = f"{iso2_code}_Green_Electricity"
            return emission_source_id, unit_id, emission_factor_id
    
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
    emission_source_id = get_emission_source_id_by_suffix(
        activity_emission_source_df, activity_sub_cat_id, transformation
    )
    
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

def create_empty_fact_table_structure(dest_df: pd.DataFrame) -> pd.DataFrame:
    """Create an empty dataframe with the same structure as the destination fact table"""
    # Create empty dataframe with same columns and dtypes
    empty_df = pd.DataFrame(columns=dest_df.columns)
    
    # Preserve the column data types
    for column in dest_df.columns:
        try:
            empty_df[column] = empty_df[column].astype(dest_df[column].dtype)
        except:
            # If dtype conversion fails, keep as object
            pass
    
    # Explicitly ensure it's empty
    empty_df = empty_df.iloc[0:0].copy()
    
    logging.info(f"Created empty fact table structure with {len(empty_df)} rows and columns: {list(empty_df.columns)}")
    return empty_df





def generate_fact(mappings: dict, source_df: pd.DataFrame, dest_df: pd.DataFrame,
                  activity_cat_df: pd.DataFrame, activity_subcat_df: pd.DataFrame,
                  scope_df: pd.DataFrame, activity_emission_source_df: pd.DataFrame,
                  activity_emmission_source_provider_df: pd.DataFrame, unit_df: pd.DataFrame,
                  currency_df: pd.DataFrame, date_df: pd.DataFrame, country_df: pd.DataFrame,
                  company_df: pd.DataFrame, company: str, country: str, activity_cat: str, 
                  activity_subcat: str, reporting_year: int, calc_method: str, 
                  org_unit_df: pd.DataFrame = None, dest_tables: dict = None) -> pd.DataFrame:    

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
        
        # Get IDs and ensure they are integers
        company_id = lookup_value(company_df, 'CompanyName', company, 'CompanyID')
        new_row['CompanyID'] = int(company_id) if company_id is not None else None
        
        country_id = lookup_value(country_df, 'CountryName', country, 'CountryID')
        new_row['CountryID'] = int(country_id) if country_id is not None else None
        
        activity_cat_id = lookup_value(activity_cat_df, 'ActivityCategory', activity_cat, 'ActivityCategoryID')
        new_row['ActivityCategoryID'] = int(activity_cat_id) if activity_cat_id is not None else None
        
        activity_subcat_id = lookup_value(activity_subcat_df, 'ActivitySubcategoryName', activity_subcat, 'ActivitySubcategoryID')
        new_row['ActivitySubcategoryID'] = int(activity_subcat_id) if activity_subcat_id is not None else None
        
        scope_id = lookup_value(activity_cat_df, 'ActivityCategory', activity_cat, 'ScopeID')
        new_row['ScopeID'] = int(scope_id) if scope_id is not None else None
        
        # Handle ActivityEmissionSourceID
        if emission_source_id is not None:
            new_row['ActivityEmissionSourceID'] = int(emission_source_id)
        else:
            # Try to find ActivityEmissionSourceID from source data if available
            source_col = None
            if isinstance(mappings, dict) and 'ActivityEmissionSourceID' in mappings and isinstance(mappings['ActivityEmissionSourceID'], dict):
                source_col = mappings['ActivityEmissionSourceID'].get('source_column')
            
            if source_col and source_col in source_df.columns and not pd.isna(source_row[source_col]):
                new_row['ActivityEmissionSourceID'] = int(source_row[source_col])
            else:
                # Check for Green Electricity source for ActivitySubcategoryID 21
                activity_subcategory_id = new_row.get('ActivitySubcategoryID')
                if activity_subcategory_id == 21:
                    # Filter for Green Electricity sources
                    green_electricity_sources = activity_emission_source_df[
                        (activity_emission_source_df['ActivitySubcategoryID'] == 21) &
                        (activity_emission_source_df['ActivityEmissionSourceName'] == 'Green Electricity')
                    ]
                    if not green_electricity_sources.empty:
                        new_row['ActivityEmissionSourceID'] = int(green_electricity_sources.iloc[0]['ActivityEmissionSourceID'])
                    else:
                        # Default to first emission source ID if available
                        if not activity_emission_source_df.empty:
                            new_row['ActivityEmissionSourceID'] = int(activity_emission_source_df['ActivityEmissionSourceID'].iloc[0])
                        else:
                            new_row['ActivityEmissionSourceID'] = None
                else:
                    # Default to first emission source ID if available
                    if not activity_emission_source_df.empty:
                        new_row['ActivityEmissionSourceID'] = int(activity_emission_source_df['ActivityEmissionSourceID'].iloc[0])
                    else:
                        new_row['ActivityEmissionSourceID'] = None
        
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
        if unit_col and unit_col in source_df.columns:
            unit_name = source_row[unit_col]
            # Lookup UnitID from unit_df
            unit_id_lookup = unit_df[unit_df['UnitName'].astype(str) == str(unit_name)]
            if not unit_id_lookup.empty:
                unit_id = int(unit_id_lookup['UnitID'].iloc[0])
        new_row['UnitID'] = int(unit_id) if unit_id is not None else None
        
        # Handle EmissionFactorID
        if emission_factor_id is not None:
            new_row['EmissionFactorID'] = emission_factor_id
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
                new_row['EmissionFactorID'] = f"{country_iso2}_{emission_source_name}"
            else:
                new_row['EmissionFactorID'] = None

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
                        distance = calculate_airport_distance(origin_code, dest_code)
                        if distance:
                            logging.info(f"Calculated air travel distance: {origin_code} -> {dest_code} = {distance} km")
                    
                    # Method 2: Try to find airport codes in any available source columns
                    if not distance:
                        airport_codes = []
                        for col in source_df.columns:
                            value = source_row.get(col)
                            if value and isinstance(value, str) and len(value.strip()) == 3:
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
                    new_row[field_name] = float(value) if value is not None else None
                else:
                    # Default value if no mapping exists
                    new_row[field_name] = 1.0  # Default consumption amount
                    logging.info(f"Using default ConsumptionAmount: 1.0")
                    
            elif field_name == 'PaidAmount':
                if source_column and source_column in source_df.columns:
                    value = source_row[source_column]
                    new_row[field_name] = float(value) if value is not None else None
                else:
                    # Default value if no mapping exists
                    new_row[field_name] = 0.0  # Default paid amount
                    logging.info(f"Using default PaidAmount: 0.0")
            
            # Handle provider and currency if present in mappings
            if field_name == 'ActivityEmissionSourceProviderID':
                if source_column and source_column in source_df.columns:
                    provider_name = source_row[source_column]
                    # First try to find in the activity_emmission_source_provider_df
                    provider_id = lookup_value(activity_emmission_source_provider_df, 
                                             'ProviderName', provider_name, 'ActivityEmissionSourceProviderID')
                    
                    if provider_id is not None:
                        new_row[field_name] = int(provider_id)
                    else:
                        # If not found in the activity_emmission_source_provider_df, check if it exists in the destination tables
                        if dest_tables and 'DE1_ActivityEmissionSourceProvi' in dest_tables:
                            provider_df = dest_tables['DE1_ActivityEmissionSourceProvi']
                            provider_row = provider_df[provider_df['ProviderName'].str.lower() == provider_name.lower() if provider_name else False]
                            if not provider_row.empty:
                                new_row[field_name] = int(provider_row['ActivityEmissionSourceProviderID'].iloc[0])
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
                                
                                new_row[field_name] = int(new_provider_id)
                        else:
                            new_row[field_name] = None  # No destination tables provided
                else:
                    # If no mapping exists, try to find a default provider
                    if not activity_emmission_source_provider_df.empty:
                        new_row[field_name] = int(activity_emmission_source_provider_df['ActivityEmissionSourceProviderID'].iloc[0])
                    elif dest_tables and 'DE1_ActivityEmissionSourceProvi' in dest_tables and not dest_tables['DE1_ActivityEmissionSourceProvi'].empty:
                        # Use the first provider from destination tables
                        new_row[field_name] = int(dest_tables['DE1_ActivityEmissionSourceProvi']['ActivityEmissionSourceProviderID'].iloc[0])
                    else:
                        new_row[field_name] = None  # No providers available
            
            # Handle OrganizationalUnitID (note: there might be a space in the column name)
            if field_name == 'OrganizationalUnitID' or field_name == 'OrganizationalUnitID ':
                # Use the correct field name with space for the new_row
                actual_field_name = 'OrganizationalUnitID ' if 'OrganizationalUnitID ' in dest_df.columns else 'OrganizationalUnitID'
                
                if source_column and source_column in source_df.columns:
                    org_unit_name = source_row[source_column]
                    # Try to find the organizational unit in the provided org_unit_df
                    if org_unit_df is not None and not org_unit_df.empty:
                        org_unit_id = lookup_value(org_unit_df, 'OrganizationalUnitName', org_unit_name, 'OrganizationalUnitID')
                        new_row[actual_field_name] = int(org_unit_id) if org_unit_id is not None else None  # Use None if not found
                    else:
                        # If org_unit_df is empty, try to find the organizational unit in the destination tables
                        if dest_tables and 'D_OrganizationalUnit' in dest_tables:
                            org_unit_df_dest = dest_tables['D_OrganizationalUnit']
                            # Try to find by company name if available
                            if company and not org_unit_df_dest.empty:
                                # First try to find by exact org unit name
                                org_unit_row = org_unit_df_dest[org_unit_df_dest['OrganizationalUnitName'].str.lower() == org_unit_name.lower() if org_unit_name else False]
                                if not org_unit_row.empty:
                                    new_row[actual_field_name] = int(org_unit_row['OrganizationalUnitID'].iloc[0])
                                else:
                                    # Try to find by company name
                                    company_id = None
                                    if 'D_Company' in dest_tables:
                                        company_df = dest_tables['D_Company']
                                        company_row = company_df[company_df['CompanyName'].str.lower() == company.lower()]
                                        if not company_row.empty:
                                            company_id = int(company_row['CompanyID'].iloc[0])
                                    
                                    if company_id is not None:
                                        # Find organizational units for this company
                                        company_org_units = org_unit_df_dest[org_unit_df_dest['CompanyID'] == company_id]
                                        if not company_org_units.empty:
                                            # Use the first organizational unit for this company
                                            new_row[actual_field_name] = int(company_org_units['OrganizationalUnitID'].iloc[0])
                                        else:
                                            new_row[actual_field_name] = None  # No org units found for this company
                                    else:
                                        new_row[actual_field_name] = None  # Company not found
                            else:
                                new_row[actual_field_name] = None  # No company name provided
                        else:
                            new_row[actual_field_name] = None  # No destination tables provided
                else:
                    # If no mapping exists, try to find the organizational unit by company name
                    if dest_tables and 'D_OrganizationalUnit' in dest_tables and company:
                        org_unit_df_dest = dest_tables['D_OrganizationalUnit']
                        company_id = None
                        if 'D_Company' in dest_tables:
                            company_df = dest_tables['D_Company']
                            company_row = company_df[company_df['CompanyName'].str.lower() == company.lower()]
                            if not company_row.empty:
                                company_id = int(company_row['CompanyID'].iloc[0])
                        
                        if company_id is not None and not org_unit_df_dest.empty:
                            # Find organizational units for this company
                            company_org_units = org_unit_df_dest[org_unit_df_dest['CompanyID'] == company_id]
                            if not company_org_units.empty:
                                # Use the first organizational unit for this company
                                new_row[actual_field_name] = int(company_org_units['OrganizationalUnitID'].iloc[0])
                            else:
                                new_row[actual_field_name] = None  # No org units found for this company
                        else:
                            new_row[actual_field_name] = None  # Company not found
                    else:
                        new_row[actual_field_name] = None  # No destination tables or company provided
                    
            # Handle EmissionFactorID
            if field_name == 'EmissionFactorID':
                # Check if we have a direct mapping for EmissionFactorID
                if source_column and source_column in source_df.columns:
                    emission_factor_id = source_row[source_column]
                    new_row[field_name] = emission_factor_id
                else:
                    # Generate EmissionFactorID based on country ISO2Code and ActivityEmissionSourceName
                    # Get the country ISO2Code
                    country_id = new_row.get('CountryID')
                    if country_id and not pd.isna(country_id) and not country_df.empty:
                        country_iso2code = lookup_value(country_df, 'CountryID', country_id, 'ISO2Code')
                        
                        # Get the ActivityEmissionSourceName
                        activity_emission_source_id = new_row.get('ActivityEmissionSourceID')
                        if activity_emission_source_id and not pd.isna(activity_emission_source_id) and not activity_emission_source_df.empty:
                            activity_subcategory_id = new_row.get('ActivitySubcategoryID')
                            
                            # Filter for Green Electricity (ActivitySubcategoryID = 21)
                            if activity_subcategory_id == 21:
                                # Find the emission source with the given ID
                                emission_source_row = activity_emission_source_df[
                                    (activity_emission_source_df['ActivityEmissionSourceID'] == activity_emission_source_id) &
                                    (activity_emission_source_df['ActivitySubcategoryID'] == activity_subcategory_id)
                                ]
                                
                                if not emission_source_row.empty:
                                    activity_emission_source_name = emission_source_row.iloc[0]['ActivityEmissionSourceName']
                                    
                                    # Concatenate ISO2Code and ActivityEmissionSourceName
                                    if country_iso2code and activity_emission_source_name:
                                        emission_factor_id = f"{country_iso2code}_{activity_emission_source_name.replace(' ', '_')}"
                                        new_row[field_name] = emission_factor_id
                                    else:
                                        # Default if we can't generate a proper ID
                                        new_row[field_name] = "Unknown_EmissionFactor"
                                else:
                                    new_row[field_name] = "Unknown_EmissionFactor"
                            else:
                                new_row[field_name] = "Unknown_EmissionFactor"
                        else:
                            new_row[field_name] = "Unknown_EmissionFactor"
                    else:
                        new_row[field_name] = "Unknown_EmissionFactor"
            
            if field_name == 'CurrencyID':
                if source_column and source_column in source_df.columns:
                    currency_code = source_row[source_column]
                    currency_id = lookup_value(currency_df, 'CurrencyCode', currency_code, 'CurrencyID')
                    new_row[field_name] = int(currency_id) if currency_id is not None else None
                else:
                    # If no mapping exists, try to find a default currency
                    if not currency_df.empty:
                        new_row[field_name] = int(currency_df['CurrencyID'].iloc[0])
        
        # Update progress
        progress = (index + 1) / total_records
        progress_bar.progress(progress)
        status_text.text(f"Processing record {index + 1} of {total_records}")
        update_progress(processed=index + 1)
        
        # Add the new row to the result DataFrame
        result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)

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