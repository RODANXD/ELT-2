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
    emission_factor_id = f"{iso2_code}_{emission_source_name}"
    
    return emission_source_id, unit_id, emission_factor_id


def get_emission_source_id_by_suffix(df, subcategory_id, suffix):
    """Get emission source ID for sources ending with suffix (case-insensitive) and matching subcategory ID."""
    filtered = df[
        (df['ActivitySubcategoryID'] == subcategory_id) &
        (df['ActivityEmissionSourceName'].str.lower().str.endswith(suffix.lower()))
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
                  activity_subcat: str, reporting_year: int, calc_method: str) -> pd.DataFrame:    

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
        
        new_row['ActivityEmissionSourceID'] = int(emission_source_id) if emission_source_id is not None else None
        new_row['UnitID'] = int(unit_id) if unit_id is not None else None
        new_row['EmissionFactorID'] = emission_factor_id

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
                    new_row[field_name] = None
                    
            elif field_name == 'PaidAmount' and source_column in source_df.columns:
                value = source_row[source_column]
                new_row[field_name] = float(value) if value is not None else None
            
            # Handle provider and currency if present in mappings
            if field_name == 'ActivityEmissionSourceProviderID' and source_column in source_df.columns:
                provider_name = source_row[source_column]
                provider_id = lookup_value(activity_emmission_source_provider_df, 
                                         'ProviderName', provider_name, 'ActivityEmissionSourceProviderID')
                new_row[field_name] = int(provider_id) if provider_id is not None else None
            
            if field_name == 'CurrencyID' and source_column in source_df.columns:
                currency_code = source_row[source_column]
                currency_id = lookup_value(currency_df, 'CurrencyCode', currency_code, 'CurrencyID')
                new_row[field_name] = int(currency_id) if currency_id is not None else None
        
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