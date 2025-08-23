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

        # Normalize src: allow list of candidates or single value
        candidate_cols = []
        if src is None or src == "null":
            candidate_cols = []
        elif isinstance(src, list):
            # flatten nested lists if any
            for item in src:
                if isinstance(item, list):
                    candidate_cols.extend(item)
                else:
                    candidate_cols.append(item)
        else:
            candidate_cols = [src]

        # Find which candidates actually exist in source_df
        matched = [c for c in candidate_cols if isinstance(c, str) and c in source_df.columns]

        if not matched:
            unresolved.append({
                "FactColumn": fact_col,
                "Reason": f"No matching column(s) {candidate_cols} in source"
            })
            continue

        # If multiple candidates matched, flag ambiguous mapping but still proceed using the first match
        if len(matched) > 1:
            unresolved.append({
                "FactColumn": fact_col,
                "Reason": f"Ambiguous mapping: multiple possible source columns {matched} - using '{matched[0]}'"
            })

        chosen_col = matched[0]

        # 3. Untranslatable non-English entries (scan a sample of the column)
        if source_df[chosen_col].dtype == object:
            non_null_count = len(source_df[chosen_col].dropna())
            if non_null_count > 0:
                sample = source_df[chosen_col].dropna().astype(str).sample(min(5, non_null_count), random_state=1)
                for val in sample:
                    if is_non_english(val):
                        unresolved.append({
                            "FactColumn": fact_col,
                            "Reason": f"Untranslatable non-English text detected in '{chosen_col}': '{val[:30]}...'"
                        })
                        break  # Only flag once per column

    df = pd.DataFrame(unresolved)
    if not df.empty:
        Path(output_path).mkdir(exist_ok=True)
        df.to_excel(f"{output_path}/UnresolvedData_Report.xlsx", index=False)