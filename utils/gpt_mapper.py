from openai import AzureOpenAI
import json
import re
import os
import pandas as pd
from config import API_KEY, AZURE_ENDPOINT, API_VERSION
from logger import setup_logger

logger = setup_logger("gpt_mapper")

# Enhanced prompt builder - you'll need to update prompts/schema_prompt.py with the enhanced version
from prompts.schema_prompt import build_prompt
from . import mapping_utils

client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=API_VERSION
)

def _extract_json_from_markdown(raw_text: str) -> str:
    """
    If raw_text contains a fenced code block (```json ...``` or ``` ...```),
    extract just the JSON inside. Otherwise, return raw_text unchanged.
    """
    fenced_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(fenced_pattern, raw_text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw_text.strip()

def _save_mapping_to_csv(mapping_json: dict, output_path: str = "outputs/mappings.csv"):
    """
    Save the mapping JSON to a CSV file with enhanced information
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert mapping_json to DataFrame with additional metadata
    rows = []
    for fact_column, mapping_info in mapping_json.items():
        row = {
            'fact_column': fact_column,
            'source_column': mapping_info.get('source_column'),
            'transformation': mapping_info.get('transformation'),
            'relation': mapping_info.get('relation'),
            'consumption_type': mapping_info.get('consumption_type', ''),
            'confidence': mapping_info.get('confidence', 'gpt_generated'),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Enhanced mapping CSV saved to '{output_path}'")

def _save_mapping_to_json(mapping_json: dict, output_path: str = "outputs/mappings.json"):
    """
    Save the raw mapping JSON to a file with metadata
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Add metadata to the mapping
    enhanced_mapping = {
        'metadata': {
            'generated_by': 'enhanced_gpt_mapper',
            'version': '2.0',
            'timestamp': pd.Timestamp.now().isoformat()
        },
        'mappings': mapping_json
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enhanced_mapping, f, indent=2)
    logger.info(f"Enhanced mapping JSON saved to '{output_path}'")

def create_dynamic_fallback_mapping(source_df: pd.DataFrame, calc_method: str, 
                                  activity_cat: str, activity_sub_cat: str) -> dict:
    """
    Create a dynamic fallback mapping when GPT fails, using intelligent column detection
    """
    logger.info("Creating dynamic fallback mapping...")
    
    source_columns = source_df.columns.tolist()
    mapping = {}
    
    # Define column patterns for intelligent detection
    patterns = {
        'financial': ['amount', 'cost', 'price', 'paid', 'expense', 'spend', 'total', 'value', 'fee', 'charge'],
        'temporal': ['date', 'time', 'year', 'month', 'day', 'period', 'timestamp', 'created'],
        'currency': ['currency', 'curr', 'ccy'],
        'unit': ['unit', 'measure', 'uom'],
        'organization': ['company', 'supplier', 'provider', 'vendor', 'department', 'unit', 'org'],
        'consumption': ['quantity', 'usage', 'consumption', 'distance', 'energy', 'fuel', 'kwh', 'mwh'],
        'location': ['origin', 'destination', 'from', 'to', 'departure', 'arrival', 'country', 'city'],
        'travel': ['miles', 'km', 'mileage', 'distance', 'travel', 'trip'],
        'duration': ['nights', 'days', 'duration', 'stay', 'period']
    }
    
    # Helper function to find best column match
    def find_column_match(target_patterns, field_name=""):
        best_match = None
        best_score = 0
        
        for pattern in target_patterns:
            for col in source_columns:
                if pattern.lower() in col.lower():
                    score = len(pattern) / len(col)  # Prefer more specific matches
                    if score > best_score:
                        best_score = score
                        best_match = col
        
        if best_match:
            logger.info(f"Found match for {field_name}: {best_match}")
        return best_match
    
    # Required base mappings
    mapping.update({
        'EmissionActivityID': {
            'source_column': None,
            'transformation': 'auto_increment',
            'relation': 'primary_key'
        },
        'CompanyID': {
            'source_column': 'user_input',
            'transformation': 'lookup_from_user_input',
            'relation': 'D_Company.CompanyID->FE1_EmissionActivityData.CompanyID'
        },
        'CountryID': {
            'source_column': 'user_input',
            'transformation': 'lookup_from_user_input',
            'relation': 'D_Country.CountryID->FE1_EmissionActivityData.CountryID'
        },
        'ActivityCategoryID': {
            'source_column': 'user_input',
            'transformation': 'lookup_from_user_input',
            'relation': 'DE1_ActivityCategory.ActivityCategoryID->FE1_EmissionActivityData.ActivityCategoryID'
        },
        'ActivitySubcategoryID': {
            'source_column': 'user_input',
            'transformation': 'lookup_from_user_input',
            'relation': 'DE1_ActivitySubcategory.ActivitySubcategoryID->FE1_EmissionActivityData.ActivitySubcategoryID'
        }
    })
    
    # Dynamic field mappings
    field_mappings = {
        'PaidAmount': patterns['financial'],
        'DateKey': patterns['temporal'],
        'CurrencyID': patterns['currency'],
        'UnitID': patterns['unit'],
        'OrganizationalUnitID': patterns['organization'],
        'ActivityEmissionSourceProviderID': patterns['organization']
    }
    
    # Apply dynamic mappings
    for field, field_patterns in field_mappings.items():
        match = find_column_match(field_patterns, field)
        if match:
            mapping[field] = {
                'source_column': match,
                'transformation': None,
                'relation': f'Auto-detected from {match}'
            }
        else:
            mapping[field] = {
                'source_column': None,
                'transformation': f'default_for_{field}',
                'relation': None
            }
    
    # Special handling for ConsumptionAmount based on context
    consumption_match = None
    consumption_type = 'Currency'  # Default
    
    if calc_method == 'Consumption-based':
        if activity_cat.lower() == 'business travel':
            if 'air' in activity_sub_cat.lower():
                # Look for distance or airport columns
                consumption_match = find_column_match(patterns['travel'] + patterns['location'], 'ConsumptionAmount')
                consumption_type = 'Distance'
            elif 'hotel' in activity_sub_cat.lower():
                # Look for duration columns
                consumption_match = find_column_match(patterns['duration'], 'ConsumptionAmount')
                consumption_type = 'Days'
        elif 'energy' in activity_cat.lower():
            consumption_match = find_column_match(patterns['consumption'], 'ConsumptionAmount')
            consumption_type = 'Energy'
    
    if not consumption_match:
        # Fallback to financial for expense-based or when no consumption column found
        consumption_match = find_column_match(patterns['financial'], 'ConsumptionAmount')
    
    mapping['ConsumptionAmount'] = {
        'source_column': consumption_match,
        'consumption_type': consumption_type,
        'transformation': 'auto_detect_based_on_activity' if not consumption_match else None,
        'relation': None
    }
    
    # Add remaining required fields with defaults
    required_fields = [
        'ActivityEmissionSourceID', 'EmissionFactorID', 'ScopeID'
    ]
    
    for field in required_fields:
        if field not in mapping:
            mapping[field] = {
                'source_column': None,
                'transformation': f'auto_generate_for_{field}',
                'relation': None
            }
    
    logger.info(f"Dynamic fallback mapping created with {len(mapping)} fields")
    
    # Log detected mappings
    detected = sum(1 for config in mapping.values() if config.get('source_column') and config.get('source_column') not in ['user_input', None])
    logger.info(f"Detected {detected} source column mappings dynamically")
    
    return mapping


def classify_energy_value(value: str, candidate_names: list, max_tokens: int = 200) -> str:
    """
    Use GPT to choose the best ActivityEmissionSourceName for a given raw energy value.
    Returns the chosen name (exact string from candidate_names) or None.
    """
    if not value or not candidate_names:
        return None

    # Build a concise prompt asking the model to pick the best match from the provided list
    choices_block = "\n".join([f"- {c}" for c in candidate_names])
    user_prompt = f"""
You are an expert mapper. Given a raw source value: "{value}", choose the single best matching canonical ActivityEmissionSourceName from the list below. Return only the exact name from the list if confident, otherwise return "NONE".

Candidates:
{choices_block}
"""

    try:
        # Try direct GPT classification first
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a precise mapping assistant. Only output one of the candidate names or NONE."},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=max_tokens
        )
        raw = response.choices[0].message.content.strip()
        raw = _extract_json_from_markdown(raw)
        for c in candidate_names:
            if raw.lower() == c.strip().lower():
                return c
        if raw.strip().upper() == 'NONE':
            return None
        best = mapping_utils.fuzzy_match_value_to_list(raw, candidate_names, threshold=60)
        if best:
            return best

        # If GPT returned something unexpected, fallback to local fuzzy match
        local_best = mapping_utils.fuzzy_match_value_to_list(value, candidate_names, threshold=60)
        if local_best:
            return local_best

        return None
    except Exception as e:
        logger.error(f"Energy classification GPT call failed: {e}")
        # On error, still attempt local fuzzy matching to be resilient
        try:
            return mapping_utils.fuzzy_match_value_to_list(value, candidate_names, threshold=60)
        except Exception:
            return None

def map_schema_with_gpt(source_columns: list, dest_schema: dict, source_table_name: str, 
                       calc_method: str, activity_cat: str, activity_sub_cat: str, 
                       source_df: pd.DataFrame = None) -> dict:
    """
    Enhanced GPT-based schema mapping with dynamic fallback
    """
    logger.info("Starting enhanced GPT-based schema mapping...")
    logger.info(f"Source: {source_table_name} with {len(source_columns)} columns")
    logger.info(f"Context: {calc_method} calculation for {activity_cat} - {activity_sub_cat}")
    
    # Build the prompt (enhanced version should be in prompts/schema_prompt.py)
    prompt = build_prompt(
        source_columns, dest_schema, source_table_name, 
        calc_method, activity_cat, activity_sub_cat,
        source_df  # Pass source_df if your enhanced prompt supports it
    )

    try:
        logger.info("Calling GPT-4 for schema mapping...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert AI data mapping assistant specialized in environmental ETL systems. You excel at understanding data semantics and creating intelligent mappings between diverse source schemas and standardized destination schemas."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )

        raw_text = response.choices[0].message.content.strip()
        logger.info("Received GPT response, extracting JSON...")

        json_text = _extract_json_from_markdown(raw_text)
        mapping_json = json.loads(json_text)
        logger.info(f"Successfully parsed GPT mapping with {len(mapping_json)} fields")

        # Save the GPT mapping
        _save_mapping_to_csv(mapping_json, output_path="outputs/mappings.csv")
        _save_mapping_to_json(mapping_json, output_path="outputs/mappings.json")
        
        # Log success metrics
        source_mapped = sum(1 for config in mapping_json.values() 
                          if config.get('source_column') and 
                          config.get('source_column') not in ['user_input', None])
        
        logger.info(f"GPT mapping completed successfully!")
        logger.info(f"- Total fields: {len(mapping_json)}")
        logger.info(f"- Source-mapped fields: {source_mapped}")

        return mapping_json

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse GPT response as JSON: {e}")
        logger.error(f"Raw response:\n{raw_text}")
        
        # Fallback to dynamic mapping
        if source_df is not None:
            logger.info("GPT failed - attempting dynamic fallback mapping...")
            fallback_mapping = create_dynamic_fallback_mapping(
                source_df, calc_method, activity_cat, activity_sub_cat
            )
            
            _save_mapping_to_csv(fallback_mapping, output_path="outputs/mappings.csv")
            _save_mapping_to_json(fallback_mapping, output_path="outputs/mappings.json")
            
            logger.info("Dynamic fallback mapping created successfully")
            return fallback_mapping
        else:
            raise Exception(f"GPT JSON parsing failed and no source data for fallback: {e}")
            
    except Exception as e:
        logger.error(f"Error during GPT call: {e}")
        
        # Final fallback
        if source_df is not None:
            logger.info("Creating emergency dynamic mapping...")
            emergency_mapping = create_dynamic_fallback_mapping(
                source_df, calc_method, activity_cat, activity_sub_cat
            )
            
            _save_mapping_to_csv(emergency_mapping, output_path="outputs/mappings.csv")
            _save_mapping_to_json(emergency_mapping, output_path="outputs/mappings.json")
            
            return emergency_mapping
        else:
            raise Exception(f"Complete mapping failure: {e}")