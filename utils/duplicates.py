import pandas as pd
import hashlib

def drop_and_log_duplicates(df: pd.DataFrame, subset_cols=None) -> pd.DataFrame:
    """Return de-duplicated df and write duplicates to CSV."""
    if subset_cols is None:
        subset_cols = df.columns.tolist()

    dup_mask = df.duplicated(subset=subset_cols, keep='first')
    dups = df[dup_mask]

    if not dups.empty:
        dups.to_csv("outputs/duplicates_removed.csv", index=False)

    return df[~dup_mask].reset_index(drop=True)