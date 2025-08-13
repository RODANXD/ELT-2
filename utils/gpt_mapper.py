from openai import AzureOpenAI
import json
import re
import os
import pandas as pd
from config import API_KEY, AZURE_ENDPOINT, API_VERSION
from logger import setup_logger
logger = setup_logger("gpt_mapper")

from prompts.schema_prompt import build_prompt


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
    Save the mapping JSON to a CSV file. Each row will have:
      fact_column, source_column, transformation, relation
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert mapping_json to DataFrame
    df = pd.DataFrame.from_dict(mapping_json, orient="index")
    df.index.name = "fact_column"
    df.reset_index(inplace=True)

    # Ensure columns exist
    for col in ["source_column", "transformation", "relation"]:
        if col not in df.columns:
            df[col] = None

    df.to_csv(output_path, index=False)
    logger.info(f"Saved mapping CSV to '{output_path}'")

def _save_mapping_to_json(mapping_json: dict, output_path: str = "outputs/mappings.json"):
    """
    Save the raw mapping JSON to a file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping_json, f, indent=2)
    logger.info(f"Saved raw mapping JSON to '{output_path}'")

def map_schema_with_gpt(source_columns: list, dest_schema: dict, source_table_name: str, calc_method:str, activity_cat , activity_sub_cat) -> dict:
    """
    Call GPT once to map the entire FACT schema against the provided source_columns list.
    """
    logger.info("Building prompt for mapping FACT schema against source columns.")
    prompt = build_prompt(source_columns, dest_schema, source_table_name,calc_method,activity_cat , activity_sub_cat)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # replace with your actual deployment name
            messages=[
                {"role": "system", "content": "You are a helpful data engineer assistant."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0
        )

        raw_text = response.choices[0].message.content.strip()
        logger.info("Received raw GPT response. Extracting JSON...")

        json_text = _extract_json_from_markdown(raw_text)
        mapping_json = json.loads(json_text)
        logger.info(f"Parsed mapping JSON: {mapping_json}")

        # Save mapping to CSV
        _save_mapping_to_csv(mapping_json, output_path="outputs/mappings.csv")
        # Save raw JSON
        _save_mapping_to_json(mapping_json, output_path="outputs/mappings.json")

        return mapping_json

    except json.JSONDecodeError:
        logger.error(f"Failed to parse GPT response as JSON. Raw response:\n{raw_text}")
        raise
    except Exception as e:
        logger.error(f"Error during GPT call: {e}")
        raise
