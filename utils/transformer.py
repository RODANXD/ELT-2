import pandas as pd
import os
from utils.dimesions import transform_dimension, transform_D_Date, transform_D_Currency, \
    transform_emission_source_provider, transform_unit, relate_country_company, transform_organizational_unit
from utils.fact import generate_fact
from utils.duplicates import drop_and_log_duplicates
from utils.unresolved import flag_unresolved
from logger import setup_logger
from utils.dimesions import transform_dimension_dynamic
from utils.schema_analyzer import analyze_source_schema

logger = setup_logger("transformer")

def transform_data(source_df: pd.DataFrame, mapping: pd.DataFrame,
                      company: str, country: str, calc_method: str,
                      activity_cat: str, activity_subcat: str,
                      dest_schema: dict, dest_tables: dict,
                      ReportingYear: int, org_unit_name: str = None):
        
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
        # Transform country and company dimensions
        country_df = transform_dimension(country, dest_tables['D_Country'], 'CountryName', 'CountryID')
        company_df = transform_dimension(company, dest_tables['D_Company'], 'CompanyName', 'CompanyID')
        org_unit_df = transform_organizational_unit(org_unit_name, company_df, dest_tables['D_OrganizationalUnit'])
        # Establish relationship between company and country
        company_df = relate_country_company(country, company, company_df, country_df)

        # Tranform date dimension

        date_df = transform_D_Date(mapping, source_df, ReportingYear)
        
        # Transform currency dimension
        currency_df = transform_D_Currency(mapping,source_df,dest_tables['D_Currency'])

        # Tranform ActivityEmissionSourceProvider dimension
        activity_emmission_source_provider_df = transform_emission_source_provider(mapping,source_df,dest_tables['DE1_ActivityEmissionSourceProvi'])

        # Transform Unit dimension
        unit_df = transform_unit(mapping, source_df, dest_tables['DE1_Unit'], calc_method)

        # Fixed Destination tables
        activity_cat_df = dest_tables['DE1_ActivityCategory'].copy()
        activity_subcat_df = dest_tables['DE1_ActivitySubcategory'].copy()
        scope_df = dest_tables['DE1_Scopes'].copy()
        activity_emmission_source_df = dest_tables['DE1_ActivityEmissionSource'].copy()

        if 'ActivityEmissionSourceID' in source_df.columns:
            schema_analysis['column_types']['ActivityEmissionSourceID'] = 'int64'
            schema_analysis['suggested_mappings']['ActivityEmissionSourceID'] = 'ActivityEmissionSourceID'
        

        # fact table generation
        emmission_activity_data_df = generate_fact(mapping, source_df, dest_tables['FE1_EmissionActivityData'],
                                                  activity_cat_df, activity_subcat_df, scope_df,
                                                  activity_emmission_source_df, activity_emmission_source_provider_df,
                                                  unit_df, currency_df, date_df, country_df , company_df,
                                                  company, country, activity_cat, activity_subcat,
                                                  ReportingYear, calc_method)



        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), 'outputs')
        os.makedirs(output_dir, exist_ok=True)

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

        return {
            "D_Company": company_df,
            "D_Country": country_df,
            "D_OrganizationalUnit": org_unit_df,
            "DE1_ActivityCategory": activity_cat_df,
            "DE1_ActivitySubcategory": activity_subcat_df,
            "DE1_Scopes": scope_df,
            "DE1_ActivityEmissionSource": activity_emmission_source_df,
            "D_Date": date_df,
            "DE1_Unit": unit_df,
            "D_Currency": currency_df,
            "DE1_ActivityEmissionSourceProvi": activity_emmission_source_provider_df,
            "FE1_EmissionActivityData": emmission_activity_data_df
        }

        print(f"Transformed data written to {os.path.join(output_dir, 'transformed_data.xlsx')}")


 
