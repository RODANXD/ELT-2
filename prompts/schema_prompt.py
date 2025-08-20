import pandas as pd
import logging
from typing import Dict, List, Any
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gpt_prompt")

def analyze_source_columns(source_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze source columns to understand data patterns and provide context to AI
    """
    analysis = {
        'column_patterns': {},
        'data_samples': {},
        'data_types': {},
        'potential_mappings': {},
        'column_statistics': {}
    }
    
    for col in source_df.columns:
        col_data = source_df[col].dropna()
        if len(col_data) == 0:
            continue
            
        # Store data type
        analysis['data_types'][col] = str(col_data.dtype)
        
        # Store sample values (first 3 non-null values)
        analysis['data_samples'][col] = col_data.head(3).tolist()
        
        # Basic statistics
        if col_data.dtype in ['int64', 'float64']:
            analysis['column_statistics'][col] = {
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'mean': float(col_data.mean()),
                'count': int(len(col_data))
            }
        else:
            analysis['column_statistics'][col] = {
                'unique_count': int(col_data.nunique()),
                'most_common': str(col_data.mode().iloc[0]) if len(col_data.mode()) > 0 else 'N/A',
                'count': int(len(col_data))
            }
        
        # Identify potential patterns
        col_lower = col.lower().strip()
        patterns = []
        
        # Financial patterns
        if any(term in col_lower for term in ['amount', 'cost', 'price', 'paid', 'expense', 'spend', 'total', 'value']):
            patterns.append('financial')
        
        # Date patterns
        if any(term in col_lower for term in ['date', 'time', 'year', 'month', 'day', 'period']):
            patterns.append('temporal')
        
        # Location patterns
        if any(term in col_lower for term in ['country', 'city', 'location', 'place', 'origin', 'destination', 'from', 'to']):
            patterns.append('location')
        
        # Company/Organization patterns
        if any(term in col_lower for term in ['company', 'organization', 'supplier', 'provider', 'vendor', 'unit', 'department']):
            patterns.append('organization')
        
        # Energy/Consumption patterns
        if any(term in col_lower for term in ['energy', 'fuel', 'electricity', 'gas', 'consumption', 'usage', 'quantity']):
            patterns.append('consumption')
        
        # Currency patterns
        if any(term in col_lower for term in ['currency', 'curr']):
            patterns.append('currency')
        
        # Unit patterns
        if any(term in col_lower for term in ['unit', 'measure', 'uom']):
            patterns.append('unit')
        
        # Travel patterns
        if any(term in col_lower for term in ['travel', 'trip', 'journey', 'flight', 'hotel', 'distance', 'miles', 'km']):
            patterns.append('travel')
        
        # Check data patterns for additional clues
        if col_data.dtype == 'object' and len(col_data) > 0:
            sample_values = col_data.astype(str).str.upper().head(10).tolist()
            
            # Check for airport codes (3-letter codes)
            if all(len(str(v).strip()) == 3 and str(v).isalpha() for v in sample_values[:3] if str(v) != 'nan'):
                patterns.append('airport_code')
            
            # Check for currency codes
            if all(len(str(v).strip()) == 3 and str(v).isalpha() for v in sample_values[:3] if str(v) != 'nan'):
                common_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'CNY', 'INR']
                if any(curr in sample_values for curr in common_currencies):
                    patterns.append('currency_code')
        
        analysis['column_patterns'][col] = patterns
    
    return analysis

def build_enhanced_mapping_rules(calc_method: str, activity_cat: str, activity_sub_cat: str) -> str:
    """
    Build dynamic mapping rules based on calculation method and activity category
    """
    base_rules = """

       GENERALIZABILITY & DYNAMIC INFERENCE RULES:
    - Do NOT assume any specific column names. Always analyze both column names and sample values.
    - If a required field (e.g., Unit, Currency, Date) is not present as a column, infer it from:
        • Suffixes or prefixes in column names (e.g., 'consumption_m3' → unit is 'm³')
        • Patterns like 'per_<unit>' in column names (e.g., 'EmissionFactor_kgCO2e_per_m3' → unit is 'm³')
        • Bracketed units (e.g., 'Volume (m3)' → unit is 'm³')
        • Common abbreviations and synonyms (e.g., 'm3', 'm³', 'M3' all mean 'm³')
    - Use fuzzy matching and normalization for units and currencies (e.g., 'm3' ≈ 'm³', 'eur' ≈ 'EUR').
    - If multiple possible mappings exist, choose the most semantically relevant based on data patterns and sample values.
    - If no mapping is possible, set the field to null and provide a reason in the unresolved report.
    - Always adapt to the actual source schema and data, not to a fixed template.


    INTELLIGENT MAPPING RULES:
    
    1. FINANCIAL MAPPING:
       - Look for columns with patterns: amount, cost, price, paid, expense, spend, total, value
       - Map to 'PaidAmount' for expense-based calculations
       - Consider currency-related columns for CurrencyID mapping
    
    2. CONSUMPTION MAPPING:
       - For consumption-based: Look for quantity, consumption, usage, distance, energy columns
       - Map to 'ConsumptionAmount' with appropriate consumption_type
    
    3. TEMPORAL MAPPING:
       - Look for date, time, year, month, day, period columns
       - Map to DateKey with proper date transformation
    
    4. ORGANIZATIONAL MAPPING:
       - Look for company, organization, supplier, provider, vendor, unit, department columns
       - Map to appropriate organizational fields
    
    5. LOCATION MAPPING:
       - Look for country, city, location, place, origin, destination columns
       - Consider for geographic relationships
       
    6. UNIT MAPPING:
       - Look for unit, measurement, metric, quantity columns 
       - Map to UnitID with appropriate unit conversion
       - Consider unit suffixes or prefixes for unit identification
       - If a source column name contains a unit (e.g., 'consumption_m3', 'EmissionFactor_kgCO2e_per_m3'), infer the unit (e.g., 'm3') and map to the correct UnitID in the destination schema.
       - If a source column contains energy type information (e.g., 'SupplyCategory', 'EnergyOrigin'), map its values to the canonical names in the destination schema using semantic/fuzzy matching.
       - For energy origin/type mapping, use these standard mappings:
           * Green Electricity: green, renewable, solar, wind, hydro, clean energy, solar PPA
           * Conventional Electricity: conventional, fossil, coal, gas fired, non-renewable, grid
           * Biomass Electricity: solar PPA, organic, biofuel, waste
           * Natural Gas: natural gas, lng, methane
           * Biomass Heating: biogas, heating, wood, pellet
           * District Heating: district, district heating, central heating
       - Use the provided list of ActivityEmissionSourceName from the destination schema for mapping.
       
    ACTIVITY EMISSION SOURCE DESCRIPTIONS:
         - Green Electricity: Renewable electricity (solar, wind, hydro, etc.)
         - Conventional Electricity: Grid or fossil-based electricity
         - Biomass Electricity: Organic material used for electricity production, often sourced from plants or waste.
         - Biomass Heating: Organic material used for heating production, often sourced from plants or waste.
         - District Heating: Centralized heating systems.
         - Natural Gas: Methane-based fuel, piped or liquefied.
    """
    
    # Add specific rules based on calculation method
    if calc_method == 'Consumption-based':
        consumption_rules = f"""
    
    CONSUMPTION-BASED SPECIFIC RULES:
    - Primary focus: Find columns representing actual consumption/usage
    - Consumption types to consider: {get_consumption_types(activity_cat, activity_sub_cat)}
    - Look for quantity measurements, distances, energy usage, fuel consumption
    """
    else:
        consumption_rules = """
    
    EXPENSE-BASED SPECIFIC RULES:
    - Primary focus: Financial amounts and costs
    - ConsumptionAmount should default to 0 or derive from financial data
    - Focus on mapping PaidAmount accurately
    """
    
    # Add activity-specific rules
    activity_rules = get_activity_specific_rules(activity_cat, activity_sub_cat)
    
    return base_rules + consumption_rules + activity_rules

def get_consumption_types(activity_cat: str, activity_sub_cat: str) -> str:
    """
    Get relevant consumption types based on activity category and subcategory
    """
    if activity_cat.lower() == 'business travel':
        if 'air' in activity_sub_cat.lower():
            return "Distance (for flights), Days (for accommodation)"
        elif 'hotel' in activity_sub_cat.lower():
            return "Days (for stays)"
        elif 'car' in activity_sub_cat.lower() or 'vehicle' in activity_sub_cat.lower():
            return "Distance (for mileage), Fuel (for fuel consumption)"
    elif activity_cat.lower() == 'energy':
        return "Energy (kWh, MWh), Electricity, Heating, Fuel"
    
    return "Distance, Energy, Fuel, Heating, Electricity, Days"

def get_activity_specific_rules(activity_cat: str, activity_sub_cat: str) -> str:
    """
    Get specific mapping rules based on activity category and subcategory
    """
    rules = f"""
    
    ACTIVITY-SPECIFIC RULES FOR {activity_cat.upper()} - {activity_sub_cat.upper()}:
    """
    
    if activity_cat.lower() == 'business travel' and 'air' in activity_sub_cat.lower():
        rules += """
    - Look for origin/departure airport columns (origin, departure, from, start_airport, dep_airport)
    - Look for destination/arrival airport columns (destination, arrival, to, end_airport, arr_airport)
    - Airport codes should be 3-letter IATA codes
    - Distance calculation will be automatic between airports
    - ConsumptionAmount: "Distance" type with airport-based calculation
    """
    elif activity_cat.lower() == 'business travel' and 'hotel' in activity_sub_cat.lower():
        rules += """
    - Look for duration/nights columns (nights, days, duration, stay_duration)
    - ConsumptionAmount: "Days" type
    - Look for accommodation-related costs
    """
    elif 'energy' in activity_cat.lower() or 'electricity' in activity_sub_cat.lower():
        rules += """
    - Look for energy consumption columns (kwh, mwh, energy_usage, consumption)
    - Look for energy type columns (electricity, gas, renewable, grid, energy_origin, energyorigin, energy_type, supply, supplycategory, SupplyCategory, HeatSource, EnergySource,)
    - ConsumptionAmount: "Energy" or "Electricity" type
    - For energy origin/type mapping, use these standard mappings:
        * Green Electricity: green, renewable, solar, wind, hydro, clean energy, solar PPA
        * Conventional Electricity: conventional, fossil, coal, gas fired, non-renewable, grid
        * Biomass Electricity: solar PPA, organic, biofuel, waste
        * Natural Gas: natural gas, lng, methane
        * Biomass Heating: biogas, heating, wood, pellet
        * District Heating: district, district heating, central heating
    - Map source energy type values to destination ActivityEmissionSourceName using semantic matching
    """
    
    return rules

def build_prompt(source_columns: List[str], dest_schema: Dict, source_table_name: str, 
                calc_method: str, activity_cat: str, activity_sub_cat: str, 
                source_df: pd.DataFrame = None) -> str:
    """
    Build an enhanced, dynamic GPT prompt for environmental ETL data mapping
    """
    logger.info(f"Building enhanced prompt for source table: {source_table_name}")

    # Analyze source data if provided
    source_analysis = {}
    if source_df is not None:
        source_analysis = analyze_source_columns(source_df)
    
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

    # Step 2: Enhanced Source Analysis
    source_analysis_block = f"Source Columns: {', '.join(source_columns)}"
    
    if source_analysis:
        source_analysis_block += "\n\nSOURCE DATA ANALYSIS:"
        for col, patterns in source_analysis['column_patterns'].items():
            samples = source_analysis['data_samples'].get(col, [])
            data_type = source_analysis['data_types'].get(col, 'unknown')
            stats = source_analysis['column_statistics'].get(col, {})
            
            source_analysis_block += f"""
    Column: {col}
    - Data Type: {data_type}
    - Patterns: {', '.join(patterns) if patterns else 'generic'}
    - Sample Values: {samples}
    - Statistics: {stats}"""

    # Step 3: Dynamic Mapping Rules
    mapping_rules = build_enhanced_mapping_rules(calc_method, activity_cat, activity_sub_cat)

    # Step 4: Enhanced Instructions
    enhanced_instructions = f"""
    ENHANCED MAPPING INSTRUCTIONS:
    
    IMPORTANT: You must analyze both column names and the actual data values in each column. If a column name is ambiguous, use the sample values to infer the correct mapping. For example, if a column contains values like 'biogas', 'district', or 'gas', map these to the appropriate energy source type in the destination schema, even if the column name is not descriptive.
    
    You are an expert AI data mapper. Your task is to intelligently map source columns to destination schema fields based on:
    1. Column name patterns and semantics
    2. Data content analysis
    3. Business context ({calc_method} calculation for {activity_cat} - {activity_sub_cat})
    4. Data types and sample values
    
    MAPPING APPROACH:
    1. Analyze each source column's name, data type, and sample values
    2. Use semantic understanding to find the best destination field match
    3. Apply business logic based on calculation method and activity type
    4. Handle missing mappings gracefully with appropriate defaults
    
    CORE MAPPING REQUIREMENTS:
    
    a) PRIMARY KEY: Auto-increment (no source mapping needed)
    {{
        "EmissionActivityID": {{
            "source_column": null,
            "transformation": "auto_increment",
            "relation": "primary_key"
        }}
    }}
    
    b) USER INPUT FIELDS (from UI - already handled):
    {{
        "CompanyID": {{
            "source_column": "user_input",
            "transformation": "lookup_from_user_input",
            "relation": "D_Company.CompanyID->FE1_EmissionActivityData.CompanyID"
        }},
        "CountryID": {{
            "source_column": "user_input", 
            "transformation": "lookup_from_user_input",
            "relation": "D_Country.CountryID->FE1_EmissionActivityData.CountryID"
        }},
        "ActivityCategoryID": {{
            "source_column": "user_input",
            "transformation": "lookup_from_user_input", 
            "relation": "DE1_ActivityCategory.ActivityCategoryID->FE1_EmissionActivityData.ActivityCategoryID"
        }},
        "ActivitySubcategoryID": {{
            "source_column": "user_input",
            "transformation": "lookup_from_user_input",
            "relation": "DE1_ActivitySubcategory.ActivitySubcategoryID->FE1_EmissionActivityData.ActivitySubcategoryID"
        }}
    }}
    
    c) INTELLIGENT SEMANTIC MAPPING:
    For remaining fields, use intelligent pattern matching:
    
    - PaidAmount: Look for financial columns (amount, cost, price, paid, expense, total, value)
    - ConsumptionAmount: Based on calculation method:
      * Consumption-based: Look for quantity/usage columns, set appropriate consumption_type
      * Expense-based: Default to financial amount or 1.0
    - DateKey: Look for date/time columns
    - CurrencyID: Look for currency code columns  
    - UnitID: Look for unit/measure columns
    - OrganizationalUnitID: Look for organization/department/unit columns
    - ActivityEmissionSourceProviderID: Look for supplier/provider columns
    
    CONSUMPTION TYPE MAPPING (for Consumption-based only):
    Current context: {calc_method} calculation for {activity_cat} - {activity_sub_cat}
    {get_consumption_types(activity_cat, activity_sub_cat)}
    
    d) DEFAULT HANDLING:
    If no suitable source column found, map to null and provide appropriate default logic:
    {{
        "field_name": {{
            "source_column": null,
            "transformation": "default_value_or_logic",
            "relation": null
        }}
    }}
    """

    # Step 5: Output format
    output_format = """
    OUTPUT FORMAT:
    Return ONLY a valid JSON mapping for the FE1_EmissionActivityData table:
    
    {
        "EmissionActivityID": {
            "source_column": null,
            "transformation": "auto_increment", 
            "relation": "primary_key"
        },
        "DateKey": {
            "source_column": "<best_matching_date_column_or_null>",
            "transformation": "<transformation_logic_or_null>",
            "relation": "D_Date.DateKey->FE1_EmissionActivityData.DateKey"
        },
        "CountryID": {
            "source_column": "user_input",
            "transformation": "lookup_from_user_input",
            "relation": "D_Country.CountryID->FE1_EmissionActivityData.CountryID"
        },
        "CompanyID": {
            "source_column": "user_input", 
            "transformation": "lookup_from_user_input",
            "relation": "D_Company.CompanyID->FE1_EmissionActivityData.CompanyID"
        },
        "OrganizationalUnitID": {
            "source_column": "<best_matching_org_column_or_null>",
            "transformation": "<transformation_logic_or_null>", 
            "relation": "D_OrganizationalUnit.OrganizationalUnitID->FE1_EmissionActivityData.OrganizationalUnitID"
        },
        "ActivityCategoryID": {
            "source_column": "user_input",
            "transformation": "lookup_from_user_input",
            "relation": "DE1_ActivityCategory.ActivityCategoryID->FE1_EmissionActivityData.ActivityCategoryID"
        },
        "ActivitySubcategoryID": {
            "source_column": "user_input",
            "transformation": "lookup_from_user_input", 
            "relation": "DE1_ActivitySubcategory.ActivitySubcategoryID->FE1_EmissionActivityData.ActivitySubcategoryID"
        },
        "ActivityEmissionSourceID": {
            "source_column": "<energy_type_column_or_null>",
            "transformation": "map_energy_type_to_emission_source",
            "mapping": {
                "green": "Green Electricity",
                "renewable": "Green Electricity",
                "solar": "Green Electricity",
                "solar PPA": "Green Electricity",
                "wind": "Green Electricity",
                "hydro": "Green Electricity",
                "clean": "Green Electricity",
                "conventional": "Conventional Electricity",
                "fossil": "Conventional Electricity",
                "coal": "Conventional Electricity",
                "gas fired": "Conventional Electricity",
                "non-renewable": "Conventional Electricity",
                "grid": "Conventional Electricity",
                
                "organic": "Biomass Electricity",
                "biofuel": "Biomass Electricity",
                "waste": "Biomass Electricity",
                "biogas": "Biomass Heating",
                "heating": "Biomass Heating",
                "wood": "Biomass Heating",
                "pellet": "Biomass Heating",
                "district": "District Heating",
                "district heating": "District Heating",
                "central heating": "District Heating",
                "natural gas": "Natural Gas",
                "lng": "Natural Gas",
                "methane": "Natural Gas"
                
            },
            "relation": "DE1_ActivityEmissionSource.ActivityEmissionSourceID->FE1_EmissionActivityData.ActivityEmissionSourceID"
        },
        "ActivityEmissionSourceProviderID": {
            "source_column": "<provider_supplier_column_or_null>",
            "transformation": "<lookup_or_create_logic>",
            "relation": "DE1_ActivityEmissionSourceProvider.ActivityEmissionSourceProviderID->FE1_EmissionActivityData.ActivityEmissionSourceProviderID"
        },
        "EmissionFactorID": {
            "source_column": "<emission_factor_column_or_null>",
            "transformation": "generate_from_country_and_source",
            "relation": null
        },
        "PaidAmount": {
            "source_column": "<best_matching_amount_column_or_null>",
            "transformation": "<conversion_logic_or_null>",
            "relation": null
        },
        "CurrencyID": {
            "source_column": "<currency_column_or_null>", 
            "transformation": "<lookup_logic_or_default>",
            "relation": "D_Currency.CurrencyID->FE1_EmissionActivityData.CurrencyID"
        },
        "ConsumptionAmount": {
            "source_column": "<consumption_quantity_column_or_null>",
            "consumption_type": "<Distance|Energy|Fuel|Heating|Electricity|Days|Currency>",
            "transformation": "<calculation_logic_based_on_type>",
            "relation": null
        },
        "UnitID": {
            "source_column": "<unit_column_or_null>",
            "transformation": "<unit_lookup_logic>", 
            "relation": "DE1_Unit.UnitID->FE1_EmissionActivityData.UnitID"
        },
        "ScopeID": {
            "source_column": "derived_from_activity_category",
            "transformation": "lookup_from_activity_category",
            "relation": "DE1_Scopes.ScopeID->FE1_EmissionActivityData.ScopeID" 
        }
    }
    """

    # Final prompt assembly
    prompt = f"""System: You are an expert AI data mapping assistant for environmental ETL systems. You excel at understanding data semantics and creating intelligent mappings between diverse source schemas and standardized destination schemas.

DESTINATION SCHEMA:
{schema_block}

SOURCE DATA INFORMATION:
Table: {source_table_name}
{source_analysis_block}

{mapping_rules}

{enhanced_instructions}

{output_format}

Remember: Be intelligent about semantic matching. Column names don't need to match exactly - use your understanding of the data meaning and business context to create the best possible mappings.
"""

    logger.info("Enhanced dynamic prompt built successfully")
    return prompt
