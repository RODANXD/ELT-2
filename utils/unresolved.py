import pandas as pd
from pathlib import Path
import langdetect

def is_non_english(text):
    """Detect if the text is not English using langdetect."""
    try:
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            return False
        lang = langdetect.detect(text)
        return lang != 'en'
    except Exception:
        return False

def flag_unresolved(mapping, source_df, output_path="outputs"):
    """Creates UnresolvedData_Report.xlsx with detailed reasons."""
    unresolved = []

    for fact_col, meta in mapping.items():
        src = meta.get('source_column')
        # 1. No matching column in source
        if src is None or src == "null" or src not in source_df.columns:
            unresolved.append({
                "FactColumn": fact_col,
                "Reason": f"No matching column '{src}' in source"
            })
            continue

        # 2. Ambiguous mapping: if source_column is a list or contains multiple candidates
        if isinstance(src, list) and len(src) > 1:
            unresolved.append({
                "FactColumn": fact_col,
                "Reason": f"Ambiguous mapping: multiple possible source columns {src}"
            })

        # 3. Untranslatable non-English entries (scan a sample of the column)
        if source_df[src].dtype == object:
            sample = source_df[src].dropna().astype(str).sample(min(5, len(source_df[src].dropna())), random_state=1)
            for val in sample:
                if is_non_english(val):
                    unresolved.append({
                        "FactColumn": fact_col,
                        "Reason": f"Untranslatable non-English text detected in '{src}': '{val[:30]}...'"
                    })
                    break  # Only flag once per column

    df = pd.DataFrame(unresolved)
    if not df.empty:
        Path(output_path).mkdir(exist_ok=True)
        df.to_excel(f"{output_path}/UnresolvedData_Report.xlsx", index=False)