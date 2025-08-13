import pandas as pd
import os
from logger import setup_logger
logger = setup_logger("file_loader")


# Get the base directory path (parent of utils folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_excel(uploaded_file):
    """
    Enhanced loader that analyzes source data structure dynamically
    """
    logger.info("Loading and analyzing uploaded Excel file.")
    try:
        # Load all sheets
        all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
        
        # Analyze each sheet's structure
        sheet_analysis = {}
        for sheet_name, df in all_sheets.items():
            sheet_analysis[sheet_name] = {
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': list(df.columns)
            }
        
        # Select the most likely data sheet (typically the one with most rows)
        main_sheet = max(sheet_analysis.items(), key=lambda x: x[1]['row_count'])[0]
        df = all_sheets[main_sheet]
        
        logger.info(f"Loaded Excel file. Selected sheet: {main_sheet}")
        return main_sheet, df
        
    except Exception as e:
        logger.error(f"Failed to load uploaded Excel file: {e}")
        raise


def load_destination_schema(path=None):
    if path is None:
        path = os.path.join(BASE_DIR, "data", "DestinationSchema.xlsx")
    logger.info(f"Loading destination schema from: {path}")
    try:
        df = pd.read_excel(path)

        required_cols = {"TableName", "ColumnName", "DataType","IsPrimaryKey"}
        if not required_cols.issubset(df.columns):
            raise ValueError("Excel file must include 'TableName', 'ColumnName', and 'IsPrimaryKey' columns.")

        schema = {}
        for table, group in df.groupby("TableName"):
            columns = group["ColumnName"].tolist()
            datatypes= group["DataType"].tolist()
            pk_rows = group[group["IsPrimaryKey"].astype(str).str.upper() == "YES"]
            primary_key = pk_rows["ColumnName"].iloc[0] if not pk_rows.empty else None

            schema[table] = {
                "columns": columns,
                "datatypes":datatypes,
                "primary_key": primary_key
                
            }

        logger.info(f"Loaded destination schema: {len(schema)} tables, {len(df)} total columns.")
        return schema

    except Exception as e:
        logger.error(f"Failed to load destination schema: {e}")
        raise


def load_destination_tables(path=None, sheet_name=None, columns=None):
    if path is None:
        path = os.path.join(BASE_DIR, "data", "DestinationTables.xlsx")
    logger.info(f"Loading destination tables from: {path}")
    try:
        if sheet_name:
            # Read specific sheet and columns if provided
            df = pd.read_excel(path, sheet_name=sheet_name, usecols=columns if columns else None)
            logger.debug(f"Loaded sheet: {sheet_name} with {len(df)} rows")
            return df
        else:
            # Read all sheets as before
            xls = pd.ExcelFile(path)
            tables = {}
            for sheet in xls.sheet_names:
                tables[sheet] = pd.read_excel(xls, sheet_name=sheet)
                logger.debug(f"Loaded fixed table: {sheet} with {len(tables[sheet])} rows")
            logger.info("Destination tables loaded successfully.")
            return tables
    except Exception as e:
        logger.error(f"Failed to load destination tables: {e}")
        raise



