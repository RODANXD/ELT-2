
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gpt_prompt")

def build_prompt(source_columns, dest_schema, source_table_name,calc_method, activity_cat , activity_sub_cat):
    """
    Build a GPT prompt for environmental ETL data mapping, specifically for expense-based consumption.
    
    Parameters:
    - source_columns: list of column names (strings) from the source sheet
    - dest_schema: dict of table metadata:
        - "columns": list of column names
        - "datatypes": list of datatypes
        - "primary_key": primary key string
    - source_table_name: name of the source sheet/table
    """
    logger.info(f"Building prompt for source table: {source_table_name}")

    # Step 1: Build Destination Schema Description
    schema_lines = []
    for table, metadata in dest_schema.items():
        cols = metadata.get("columns", [])
        dtypes = metadata.get("datatypes", [])
        pk = metadata.get("primary_key", None)

        if dtypes and len(dtypes) == len(cols):
            cols_typed = [f"{col} ({dtype})" for col, dtype in zip(cols, dtypes)]
        else:
            cols_typed = cols

        cols_str = ", ".join(cols_typed)
        pk_str = f"PK: {pk}" if pk else "PK: <none>"

        if table.lower() != "sysdiagrams":
            schema_lines.append(f"{table} ({cols_str})\n{pk_str}")

    schema_block = "\n\n".join(schema_lines)

    # Step 2: Source Columns
    source_cols_str = ", ".join(source_columns)

    # Step 3: Hardcoded PK & User Input Info
    pk_note = """
                      a) PK is autoincrement value so no mapping

                      add them in output like
                      {
                        "PK": {
                          "source_column": "null",
                          "transformation": auto incremental value,
                          "relation": "null"
                        },
                      }
                      """

    user_input_note = """
                      b) 4 direct user input from UI : (fixed input -> below fact column)
                        companyID, countryID, activitycategoryID, activitysubcategoryID

                      Note: above columns are fixed mapping, no need to map from src columns again

                      add them in output like
                      {
                        "companyID": {
                          "source_column": "user_input",
                          "transformation": find company id from userinput,
                          "relation": "<DimTable.PK->FactTable_FK>"
                        },
                      }
                      """

    mapping_note = """
                    c) company, country, activitycategory, activitysubcategory these are mapped now
                      for other fact columns try to map rest best suitable source column

                    condition: we have 2 calculation types: consumption or expense based
                    this time it is = {}

                    
                                        
                   NOTE: Incase of other ids INTELLINGENT SEMANTIC MAPPING BASED ON BUSINESS UNDERSTANDING IS REQUIRED HERE
                   for example  sourceproviderid can match to expenseaccountname  
                                unit can match to unitid etc
                  
                    In both cases we have to direct map 'PaidAmount' 

                    Incase of expense based example:
                      "ConsumptionAmount": {{
                          "source_column": "<TickectPrice> or null",
                          "consumption_type": "Currency",
                          "transformation": 'null',          
                          "relation": "null"
                      }}

                    Incase of consumption based - fact column  'ConsumptionAmount' need to be mapped properly
                    add type among these ['Distance','Energy','Fuel','Heating','Electricity','Days']'   
                    Condition : here activity category = {} and activity sub category = {}
                     

                    example 1:
                      "ConsumptionAmount": {{
                          "source_column": "HotelStays/ null",
                          "consumption_type": "Days",
                          "transformation": '<logic if there is no  src column to find date `from` and `to`> ',          
                          "relation": "null"
                      }}
                      example 2 :
                       "ConsumptionAmount": {{
                          "source_column": "MileageDistance",
                          "consumption_type": "Distance",
                          "transformation": '<logic - if there is no src column to find distance `departure column` and `arrival column`>',          
                          "relation": "null"
                      }}

                  similarly understand and map energy , fuel , heating , electricity  
                    """.format(calc_method , activity_cat , activity_sub_cat)

    # Add air travel specific instructions
    air_travel_note = ""
    if calc_method == 'Consumption-based' and activity_cat.lower() == 'business travel' and activity_sub_cat.lower() == 'air travel':
        air_travel_note = """
                    
                    SPECIAL CASE FOR AIR TRAVEL CONSUMPTION:
                    Since this is air travel consumption-based calculation, the system will automatically 
                    calculate distances between airports using origin and destination airport codes.
                    
                    For Air Travel ConsumptionAmount mapping:
                    - If you find columns containing origin/departure airport codes and destination/arrival airport codes,
                      map ConsumptionAmount as follows:
                      
                      "ConsumptionAmount": {{
                          "source_column": "null",
                          "consumption_type": "Distance", 
                          "transformation": "calculate distance between origin and destination airports using IATA codes",
                          "relation": "null"
                      }}
                    
                    - The system will automatically detect columns with names like: 
                      Origin, Departure, From (for origin airport)
                      Destination, Arrival, To (for destination airport)
                    
                    - Airport codes should be IATA codes (e.g., CDG, HND, LHR, JFK)
                    - Distance will be calculated in kilometers
                    """

    null_mapping_note = """
                        d) if no matching mapping, map fact columns to null and all other values to null
                        """

    output_instruction = """
                            Output Instructions:

                            FE1_EmissionActivityData (EmissionActivityID, DateKey, CountryID, CompanyID, ActivityCategoryID, ActivitySubcategoryID, ActivityEmissionSourceID, ActivityEmissionSourceProviderID, EmissionFactorID, PaidAmount, CurrencyID, ConsumptionAmount, UnitID, ScopeID)
                            PK: EmissionActivityID

                            output JSON should consist
                            - PK should be incremental value
                            - user input mappings
                            - rest of the src columns mapped to fact column
                            - columns with NULL mapping

                            Return only the required JSON FACT SCHEMA, structured as follows:
                            missing mapping columns should be populated with null

                            ```json
                            {
                              "fact_column": {
                                "source_column": "<mapped_column_from_source> or <null>",
                                "transformation": <transformation required> or <null>,
                                "relation": "<DimTable.PK->FactTable_FK> or <null>"
                              }
                            }


                            """
    # Step 4: Final Prompt
    prompt = f"""System: You are an expert data-mapping assistant for environmental ETL. You will be given the destination schema and a source sheet. Your job is to output only the required JSON mapping between the source sheet and destination schema fields.


    STEP 1
    Understand and establish relationships between the following dimension tables, based on their primary keys (PK):

    Destination Schema (Dimensions + Fact Table):
    {schema_block}

    STEP 2
    Establish the same PK–FK relationship between dimension tables and the fact table FE1\_EmissionActivityData. Example:
    FE1\_EmissionActivityData.DateKey → D\_Date.DateKey

    STEP 3: Mapping

    Source Sheet: {source_table_name}
    Columns include:
    Columns: {source_cols_str}

    {pk_note}
    {user_input_note}
    {mapping_note}
    {air_travel_note}
    {null_mapping_note}
    {output_instruction}

    """
    logger.info("Prompt built successfully.")
    logger.info(f"PROMPT:\n{prompt}")
    return prompt
