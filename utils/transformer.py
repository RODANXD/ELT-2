import pandas as pd
import os
from utils.dimesions import transform_dimension, transform_D_Date, transform_D_Currency, \
    transform_emission_source_provider, transform_unit, relate_country_company, transform_organizational_unit, \
    create_empty_dimension_structure
from utils.fact import generate_fact, create_empty_fact_table_structure
from utils.fact import generate_fact
from utils.duplicates import drop_and_log_duplicates
from utils.unresolved import flag_unresolved
from logger import setup_logger
from utils.dimesions import transform_dimension_dynamic
from utils.schema_analyzer import analyze_source_schema

logger = setup_logger("transformer")


def ensure_company_and_country(dest_tables, company_name, country_name):
    """
    Ensure company and country exist in destination tables.
    If not, append them and return updated tables and IDs.
    """
    d_company = dest_tables['D_Company']
    d_country = dest_tables['D_Country']

    # Check if country exists
    country_row = d_country[d_country['CountryName'].str.lower() == country_name.lower()]
    if not country_row.empty:
        country_id = int(country_row['CountryID'].iloc[0])
    else:
        # Assign new CountryID
        country_id = d_country['CountryID'].max() + 1 if not d_country.empty else 1
        new_country = {
            'CountryID': country_id,
            'CountryName': country_name,
            'ISO2Code': '',  # Fill as needed
            'ISO3Code': '',  # Fill as needed
            # Add other columns as needed
        }
        d_country = pd.concat([d_country, pd.DataFrame([new_country])], ignore_index=True)

    # Check if company exists
    company_row = d_company[d_company['CompanyName'].str.lower() == company_name.lower()]
    if not company_row.empty:
        company_id = int(company_row['CompanyID'].iloc[0])
    else:
        # Assign new CompanyID
        company_id = d_company['CompanyID'].max() + 1 if not d_company.empty else 1
        new_company = {
            'CompanyID': company_id,
            'CompanyName': company_name,
            'CountryID': country_id,
            'Industry': None  # Fill as needed
        }
        d_company = pd.concat([d_company, pd.DataFrame([new_company])], ignore_index=True)

    # Update dest_tables
    dest_tables['D_Company'] = d_company
    dest_tables['D_Country'] = d_country

    return dest_tables, company_id, country_id



def ensure_organizational_unit(dest_tables, org_unit_name, company_id):
    """
    Ensure organizational unit exists in destination tables.
    If not, append it and return updated tables and ID.
    """
    d_org_unit = dest_tables['D_OrganizationalUnit']

    if org_unit_name:
        # Check if org unit exists
        org_unit_row = d_org_unit[d_org_unit['OrganizationalUnitName'].str.lower() == org_unit_name.lower()]
        if not org_unit_row.empty:
            # Use existing org unit
            org_unit_id = int(org_unit_row['OrganizationalUnitID'].iloc[0])
        else:
            # Create new org unit
            org_unit_id = d_org_unit['OrganizationalUnitID'].max() + 1 if not d_org_unit.empty else 1
            new_org_unit = {
                'OrganizationalUnitID': org_unit_id,
                'OrganizationalUnitName': org_unit_name,
                'CompanyID': company_id,
                'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'updated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            }
            d_org_unit = pd.concat([d_org_unit, pd.DataFrame([new_org_unit])], ignore_index=True)
            
            # Update dest_tables with new org unit
            dest_tables['D_OrganizationalUnit'] = d_org_unit
            
            # Save updated D_OrganizationalUnit back to DestinationTables.xlsx
            output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'DestinationTables.xlsx')
            with pd.ExcelWriter(output_path, mode='a', if_sheet_exists='replace') as writer:
                d_org_unit.to_excel(writer, sheet_name='D_OrganizationalUnit', index=False)

        return dest_tables, org_unit_id
    else:
        return dest_tables, None

def transform_data(source_df: pd.DataFrame, mapping: pd.DataFrame,
                      company: str, country: str, calc_method: str,
                      activity_cat: str, activity_subcat: str,
                      dest_schema: dict, dest_tables: dict,
                      ReportingYear: int, org_unit_name: str = None,
                      company_mode: str = "A single company / unit",
                      unit_mode: str = "Single unit"):

    
    # Ensure company and country first
    dest_tables, company_id, country_id = ensure_company_and_country(dest_tables, company, country)
    
    # Handle organizational unit(s) depending on mode
    d_company_df, d_org_unit_df = handle_company_and_units(
        dest_tables,
        company_mode,
        company,
        org_unit_name,
        source_df,
        country_id,
        unit_mode
    )
    
    # Update dest_tables with new values
    dest_tables['D_Company'] = d_company_df
    dest_tables['D_OrganizationalUnit'] = d_org_unit_df
    
    # Handle organizational unit based on user selection
    is_multiple_units = (
    company_mode == "Multiple companies or organizational units"
)

    unit_column = None
    
    # Determine the column containing organizational unit names for multiple units scenario
    if is_multiple_units:
        # Try to find the organizational unit column in the source data
        potential_unit_columns = ['organizational_unit', 'org_unit', 'unit_name', 'supplier_name', 'provider']
        for col in potential_unit_columns:
            if col in source_df.columns:
                unit_column = col
                break
    
    # Use transform_organizational_unit for fact generation (keeps schema consistent)
    org_unit_df = d_org_unit_df.copy()  # Use the actual organizational unit dataframe instead of creating a new one
    logger.info(f"Using organizational unit dataframe with {len(org_unit_df)} entries for fact generation")
    
    
    
    
    schema_analysis = analyze_source_schema(source_df)
    logger.info(f"Source schema analysis: {schema_analysis}")

    energy_type_map = {
        'green_electricity': 1,
        'grey_electricity': 2,
        'natural_gas': 3,
        'diesel': 4,
        'gasoline': 5
    }
    
    for col in source_df.columns:
        if col.lower() in ['energy_type', 'source_type', 'emission_source', 'type']:
            source_df['ActivityEmissionSourceID'] = source_df[col].str.lower().map(energy_type_map)
            break

    source_df = drop_and_log_duplicates(source_df)
    flag_unresolved(mapping, source_df)
    transformed_data = {}

    for table_name, table_schema in dest_schema.items():
        if table_name.startswith('D_'):  # Handle dimension tables
            transformed_data[table_name] = transform_dimension_dynamic(
                source_df,
                schema_analysis,
                table_schema,
                dest_tables.get(table_name)
            )
    # Transform country and company dimensions - create empty dataframes and only add specific records
    # Create empty country dataframe with the same structure
    country_df = create_empty_dimension_structure(dest_tables['D_Country'])
    # Add only the specific country record
    country_row = dest_tables['D_Country'][dest_tables['D_Country']['CountryName'].str.lower() == country.lower()]
    if not country_row.empty:
        country_df = pd.concat([country_df, country_row], ignore_index=True)
    
    # Create empty company dataframe with the same structure
    company_df = create_empty_dimension_structure(dest_tables['D_Company'])
    # Add only the specific company record
    company_row = dest_tables['D_Company'][dest_tables['D_Company']['CompanyName'].str.lower() == company.lower()]
    if not company_row.empty:
        company_df = pd.concat([company_df, company_row], ignore_index=True)
    
    # Transform organizational unit using empty structure
    org_unit_df = transform_organizational_unit(org_unit_name, company_df, dest_tables['D_OrganizationalUnit'])
    # Establish relationship between company and country
    company_df = relate_country_company(country, company, company_df, country_df)

    # Tranform date dimension

    date_df = transform_D_Date(mapping, source_df, ReportingYear)
    
    # Transform currency dimension - preserve functionality as it was working correctly
    currency_df = transform_D_Currency(mapping, source_df, dest_tables['D_Currency'])

    # Tranform ActivityEmissionSourceProvider dimension - preserve functionality
    activity_emmission_source_provider_df = transform_emission_source_provider(mapping, source_df, dest_tables['DE1_ActivityEmissionSourceProvi'])

    # Transform Unit dimension - preserve functionality as it was working correctly
    unit_df = transform_unit(mapping, source_df, dest_tables['DE1_Unit'], calc_method)

    # Fixed Destination tables - filter to include only relevant categories/subcategories
    activity_cat_df = dest_tables['DE1_ActivityCategory'][dest_tables['DE1_ActivityCategory']['ActivityCategory'].str.lower() == activity_cat.lower()].copy()
    activity_subcat_df = dest_tables['DE1_ActivitySubcategory'][dest_tables['DE1_ActivitySubcategory']['ActivitySubcategoryName'].str.lower() == activity_subcat.lower()].copy()
    
    # Keep all scopes and emission sources for now, will filter them later based on fact table
    scope_df = dest_tables['DE1_Scopes'].copy()
    activity_emmission_source_df = dest_tables['DE1_ActivityEmissionSource'].copy()

    if 'ActivityEmissionSourceID' in source_df.columns:
        schema_analysis['column_types']['ActivityEmissionSourceID'] = 'int64'
        schema_analysis['suggested_mappings']['ActivityEmissionSourceID'] = 'ActivityEmissionSourceID'
    

    # fact table generation - use empty fact table structure
    empty_fact_df = create_empty_fact_table_structure(dest_tables['FE1_EmissionActivityData'])
    # Generate fact table    
    # Ensure mapping contains necessary fields for fact table columns
    required_fields = ['ConsumptionAmount', 'CurrencyID', 'PaidAmount', 'ActivityEmissionSourceProviderID']
    for field in required_fields:
        if field not in mapping:
            # Add default mapping if missing
            mapping[field] = {}
            # Try to find appropriate source column
            for col in source_df.columns:
                if field.lower() in col.lower() or any(term in col.lower() for term in field.lower().split('ID')[0].split('Amount')):
                    mapping[field]['source_column'] = col
                    break
    
    # Log the mapping for debugging
    print(f"Mapping for fact table: {mapping}")
    
    # Ensure org_unit_df has the correct column names
    # Normalize column names for organizational unit
    if 'OrganizationalUnitID ' in org_unit_df.columns:
        org_unit_df = org_unit_df.rename(columns={'OrganizationalUnitID ': 'OrganizationalUnitID'})
    if 'OrganizationalUnitID ' in empty_fact_df.columns:
        empty_fact_df = empty_fact_df.rename(columns={'OrganizationalUnitID ': 'OrganizationalUnitID'})

    
    emmission_activity_data_df = generate_fact(
    mapping, source_df, empty_fact_df,
    activity_cat_df, activity_subcat_df, scope_df,
    activity_emmission_source_df, activity_emmission_source_provider_df,
    unit_df, currency_df, date_df, country_df, company_df,
    company, country, activity_cat, activity_subcat,
    ReportingYear, calc_method, org_unit_df, dest_tables,
    is_multiple_units=is_multiple_units,
    unit_column=unit_column
)



    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # Filter scope and emission source tables based on fact table data
    if not emmission_activity_data_df.empty:
        # Filter scopes based on ScopeID in fact table
        if 'ScopeID' in emmission_activity_data_df.columns:
            scope_ids = emmission_activity_data_df['ScopeID'].dropna().unique()
            if len(scope_ids) > 0:
                scope_df = scope_df[scope_df['ScopeID'].isin(scope_ids)]
        
        # Filter emission sources based on ActivityEmissionSourceID in fact table
        if 'ActivityEmissionSourceID' in emmission_activity_data_df.columns:
            emission_source_ids = emmission_activity_data_df['ActivityEmissionSourceID'].dropna().unique()
            if len(emission_source_ids) > 0:
                activity_emmission_source_df = activity_emmission_source_df[activity_emmission_source_df['ActivityEmissionSourceID'].isin(emission_source_ids)]
    
    # Write each DataFrame to a separate sheet in an Excel file
    with pd.ExcelWriter(os.path.join(output_dir, 'transformed_data.xlsx')) as writer:
        company_df.to_excel(writer, sheet_name='D_Company', index=False)
        country_df.to_excel(writer, sheet_name='D_Country', index=False)
        org_unit_df.to_excel(writer, sheet_name='D_OrganizationalUnit', index=False)
        activity_cat_df.to_excel(writer, sheet_name='DE1_ActivityCategory', index=False)
        activity_subcat_df.to_excel(writer, sheet_name='DE1_ActivitySubcategory', index=False)
        scope_df.to_excel(writer, sheet_name='DE1_Scopes', index=False)
        activity_emmission_source_df.to_excel(writer, sheet_name='DE1_ActivityEmissionSource', index=False)
        date_df.to_excel(writer, sheet_name='D_Date', index=False)
        unit_df.to_excel(writer, sheet_name='DE1_Unit', index=False)
        currency_df.to_excel(writer, sheet_name='D_Currency', index=False)
        activity_emmission_source_provider_df.to_excel(writer, sheet_name='DE1_ActivityEmissionSourceProvider', index=False)
        emmission_activity_data_df.to_excel(writer, sheet_name='FE1_EmissionActivityData', index=False)

    print(f"Transformed data written to {os.path.join(output_dir, 'transformed_data.xlsx')}")

    # Save updated D_Company, D_Country, D_OrganizationalUnit, and DE1_ActivityEmissionSourceProvi back to DestinationTables.xlsx
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'DestinationTables.xlsx')
    with pd.ExcelWriter(output_path, mode='a', if_sheet_exists='replace') as writer:
        dest_tables['D_Company'].to_excel(writer, sheet_name='D_Company', index=False)
        dest_tables['D_Country'].to_excel(writer, sheet_name='D_Country', index=False)
        dest_tables['D_OrganizationalUnit'].to_excel(writer, sheet_name='D_OrganizationalUnit', index=False)
        
        # Also save the provider data back to DestinationTables.xlsx
        if 'DE1_ActivityEmissionSourceProvi' in dest_tables:
            dest_tables['DE1_ActivityEmissionSourceProvi'].to_excel(writer, sheet_name='DE1_ActivityEmissionSourceProvi', index=False)

    return {
        "D_Company": company_df,
        "D_Country": country_df,
        "D_OrganizationalUnit": org_unit_df,
        "DE1_ActivityCategory": activity_cat_df,
        "DE1_ActivitySubcategory": activity_subcat_df,
        "DE1_Scopes": scope_df,  # Now filtered based on fact table
        "DE1_ActivityEmissionSource": activity_emmission_source_df,  # Now filtered based on fact table
        "D_Date": date_df,
        "DE1_Unit": unit_df,  # Preserved as it was working correctly
        "D_Currency": currency_df,  # Preserved as it was working correctly
        "DE1_ActivityEmissionSourceProvi": activity_emmission_source_provider_df,
        "FE1_EmissionActivityData": emmission_activity_data_df
    }



def handle_company_and_units(dest_tables, company_mode, company_name, org_unit_name, source_df, country_id, unit_mode="Single unit"):
    """Handle company and organizational unit creation based on user selection.
    
    Args:
        dest_tables: Dictionary of destination tables
        company_mode: "A single company / unit" or "Multiple companies or organizational units"
        company_name: Name of the company (for single company mode)
        org_unit_name: Name of the organizational unit (for single unit mode)
        source_df: Source dataframe
        country_id: Country ID for the company
        unit_mode: "Single unit" or "Multiple units" (for single company mode)
    """
    d_company_df = dest_tables['D_Company']
    d_org_unit_df = dest_tables['D_OrganizationalUnit']
    logger.info(f"Handling companies and units: mode={company_mode}, unit_mode={unit_mode}")

    # Potential columns that might contain organizational unit information
    potential_unit_columns = ['organizational_unit', 'org_unit', 'unit_name', 'supplier_name', 'provider', 'department', 'division']
    unit_column = None
    
    # Find the first column that exists in the source data
    for col in potential_unit_columns:
        if col in source_df.columns:
            unit_column = col
            logger.info(f"Found organizational unit column: {unit_column}")
            break

    if company_mode == "A single company / unit":
        # Handle single company
        existing_company = d_company_df[d_company_df['CompanyName'].str.lower() == company_name.lower()]
        if not existing_company.empty:
            company_id = existing_company['CompanyID'].iloc[0]
            logger.info(f"Using existing company: {company_name} (ID: {company_id})")
        else:
            company_id = d_company_df['CompanyID'].max() + 1 if not d_company_df.empty else 1
            new_company = pd.DataFrame({
                'CompanyID': [company_id],
                'CompanyName': [company_name],
                'CountryID': [country_id],
                'Industry': [None]
            })
            d_company_df = pd.concat([d_company_df, new_company], ignore_index=True)
            logger.info(f"Created new company: {company_name} (ID: {company_id})")

        if unit_mode == "Single unit":
            # Single organizational unit
            if org_unit_name:
                # Check if the unit already exists
                existing_unit = d_org_unit_df[d_org_unit_df['OrganizationalUnitName'].str.lower() == org_unit_name.lower()]
                if not existing_unit.empty:
                    unit_id = existing_unit['OrganizationalUnitID'].iloc[0]
                    logger.info(f"Using existing organizational unit: {org_unit_name} (ID: {unit_id})")
                else:
                    unit_id = d_org_unit_df['OrganizationalUnitID'].max() + 1 if not d_org_unit_df.empty else 1
                    new_unit = pd.DataFrame({
                        'OrganizationalUnitID': [unit_id],
                        'OrganizationalUnitName': [org_unit_name],
                        'CompanyID': [company_id],
                        'created_at': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]],
                        'updated_at': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]]
                    })
                    d_org_unit_df = pd.concat([d_org_unit_df, new_unit], ignore_index=True)
                    logger.info(f"Created new organizational unit: {org_unit_name} (ID: {unit_id})")
            else:
                # If no org_unit_name provided, use company name as the organizational unit name
                unit_id = d_org_unit_df['OrganizationalUnitID'].max() + 1 if not d_org_unit_df.empty else 1
                new_unit = pd.DataFrame({
                    'OrganizationalUnitID': [unit_id],
                    'OrganizationalUnitName': [company_name],  # Use company name as unit name
                    'CompanyID': [company_id],
                    'created_at': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]],
                    'updated_at': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]]
                })
                d_org_unit_df = pd.concat([d_org_unit_df, new_unit], ignore_index=True)
                logger.info(f"Created default organizational unit using company name: {company_name} (ID: {unit_id})")
        else:  # Multiple units
            # Detect units from source data
            if unit_column:
                unique_units = source_df[unit_column].dropna().unique()
                logger.info(f"Found {len(unique_units)} unique organizational units in column '{unit_column}'")
                
                for unit in unique_units:
                    # Check if the unit already exists
                    existing_unit = d_org_unit_df[d_org_unit_df['OrganizationalUnitName'].str.lower() == unit.lower()]
                    if not existing_unit.empty:
                        unit_id = existing_unit['OrganizationalUnitID'].iloc[0]
                        logger.info(f"Using existing organizational unit: {unit} (ID: {unit_id})")
                    else:
                        unit_id = d_org_unit_df['OrganizationalUnitID'].max() + 1 if not d_org_unit_df.empty else 1
                        new_unit = pd.DataFrame({
                            'OrganizationalUnitID': [unit_id],
                            'OrganizationalUnitName': [unit],
                            'CompanyID': [company_id],
                            'created_at': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]],
                            'updated_at': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]]
                        })
                        d_org_unit_df = pd.concat([d_org_unit_df, new_unit], ignore_index=True)
                        logger.info(f"Created new organizational unit: {unit} (ID: {unit_id})")
            else:
                logger.warning("No organizational unit column found in source data for multiple units mode")
                # Create a default organizational unit using the company name
                unit_id = d_org_unit_df['OrganizationalUnitID'].max() + 1 if not d_org_unit_df.empty else 1
                new_unit = pd.DataFrame({
                    'OrganizationalUnitID': [unit_id],
                    'OrganizationalUnitName': [company_name],  # Use company name as unit name
                    'CompanyID': [company_id],
                    'created_at': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]],
                    'updated_at': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]]
                })
                d_org_unit_df = pd.concat([d_org_unit_df, new_unit], ignore_index=True)
                logger.info(f"Created default organizational unit using company name: {company_name} (ID: {unit_id})")

    else:  # Multiple companies or organizational units
        # Try to find company column
        company_columns = ['company_name', 'company', 'organization', 'org_name']
        company_column = None
        for col in company_columns:
            if col in source_df.columns:
                company_column = col
                logger.info(f"Found company column: {company_column}")
                break
        
        if company_column:
            # Process each unique company
            for comp in source_df[company_column].dropna().unique():
                # Check if company already exists
                existing_company = d_company_df[d_company_df['CompanyName'].str.lower() == comp.lower()]
                if not existing_company.empty:
                    company_id = existing_company['CompanyID'].iloc[0]
                    logger.info(f"Using existing company: {comp} (ID: {company_id})")
                else:
                    company_id = d_company_df['CompanyID'].max() + 1 if not d_company_df.empty else 1
                    new_company = pd.DataFrame({
                        'CompanyID': [company_id],
                        'CompanyName': [comp],
                        'CountryID': [country_id],
                        'Industry': [None]
                    })
                    d_company_df = pd.concat([d_company_df, new_company], ignore_index=True)
                    logger.info(f"Created new company: {comp} (ID: {company_id})")
                
                # Process organizational units for this company
                if unit_column:
                    # Get units for this company only
                    company_units = source_df[source_df[company_column] == comp][unit_column].dropna().unique()
                    logger.info(f"Found {len(company_units)} unique organizational units for company '{comp}'")
                    
                    for unit in company_units:
                        # Check if unit already exists
                        existing_unit = d_org_unit_df[d_org_unit_df['OrganizationalUnitName'].str.lower() == unit.lower()]
                        if not existing_unit.empty:
                            unit_id = existing_unit['OrganizationalUnitID'].iloc[0]
                            logger.info(f"Using existing organizational unit: {unit} (ID: {unit_id})")
                        else:
                            unit_id = d_org_unit_df['OrganizationalUnitID'].max() + 1 if not d_org_unit_df.empty else 1
                            new_unit = pd.DataFrame({
                                'OrganizationalUnitID': [unit_id],
                                'OrganizationalUnitName': [unit],
                                'CompanyID': [company_id],
                                'created_at': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]],
                                'updated_at': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]]
                            })
                            d_org_unit_df = pd.concat([d_org_unit_df, new_unit], ignore_index=True)
                            logger.info(f"Created new organizational unit: {unit} (ID: {unit_id})")
                else:
                    # Create a default organizational unit using the company name
                    unit_id = d_org_unit_df['OrganizationalUnitID'].max() + 1 if not d_org_unit_df.empty else 1
                    new_unit = pd.DataFrame({
                        'OrganizationalUnitID': [unit_id],
                        'OrganizationalUnitName': [comp],  # Use company name as unit name
                        'CompanyID': [company_id],
                        'created_at': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]],
                        'updated_at': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]]
                    })
                    d_org_unit_df = pd.concat([d_org_unit_df, new_unit], ignore_index=True)
                    logger.info(f"Created default organizational unit using company name: {comp} (ID: {unit_id})")
        else:
            # No company column found, use the provided company name
            logger.warning("No company column found in source data for multiple companies mode")
            company_id = d_company_df['CompanyID'].max() + 1 if not d_company_df.empty else 1
            new_company = pd.DataFrame({
                'CompanyID': [company_id],
                'CompanyName': [company_name],
                'CountryID': [country_id],
                'Industry': [None]
            })
            d_company_df = pd.concat([d_company_df, new_company], ignore_index=True)
            logger.info(f"Created default company: {company_name} (ID: {company_id})")
            
            # Process organizational units
            if unit_column:
                unique_units = source_df[unit_column].dropna().unique()
                logger.info(f"Found {len(unique_units)} unique organizational units")
                
                for unit in unique_units:
                    unit_id = d_org_unit_df['OrganizationalUnitID'].max() + 1 if not d_org_unit_df.empty else 1
                    new_unit = pd.DataFrame({
                        'OrganizationalUnitID': [unit_id],
                        'OrganizationalUnitName': [unit],
                        'CompanyID': [company_id],
                        'created_at': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]],
                        'updated_at': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]]
                    })
                    d_org_unit_df = pd.concat([d_org_unit_df, new_unit], ignore_index=True)
                    logger.info(f"Created new organizational unit: {unit} (ID: {unit_id})")
            else:
                # Create a default organizational unit using the company name
                unit_id = d_org_unit_df['OrganizationalUnitID'].max() + 1 if not d_org_unit_df.empty else 1
                new_unit = pd.DataFrame({
                    'OrganizationalUnitID': [unit_id],
                    'OrganizationalUnitName': [company_name],  # Use company name as unit name
                    'CompanyID': [company_id],
                    'created_at': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]],
                    'updated_at': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]]
                })
                d_org_unit_df = pd.concat([d_org_unit_df, new_unit], ignore_index=True)
                logger.info(f"Created default organizational unit using company name: {company_name} (ID: {unit_id})")

    return d_company_df, d_org_unit_df


