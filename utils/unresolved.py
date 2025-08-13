import pandas as pd
from pathlib import Path

def flag_unresolved(mapping, source_df, output_path="outputs"):
    """Creates UnresolvedData_Report.xlsx"""
    unresolved = []

    for fact_col, meta in mapping.items():
        src = meta.get('source_column')
        if src is None or src == "null" or src not in source_df.columns:
            unresolved.append({
                "FactColumn": fact_col,
                "Reason": f"No matching column '{src}' in source"
            })

    df = pd.DataFrame(unresolved)
    if not df.empty:
        Path(output_path).mkdir(exist_ok=True)
        df.to_excel(f"{output_path}/UnresolvedData_Report.xlsx", index=False)