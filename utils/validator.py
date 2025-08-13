import pandas as pd
import numpy as np
from pathlib import Path
import datetime

def validate_transformed_data(transformed_data, schema, output_folder):
    """Validate transformed data against destination schema and generate validation report"""
    validation_results = []
    
    for table_name, df in transformed_data.items():
        print(f"Validating table: {table_name}")
        if table_name not in schema:
            validation_results.append(('Schema', table_name, f"Table {table_name} not found in schema"))
            continue
            
        # Schema validation
        schema_issues = validate_schema(df, schema[table_name])
        
        # Null check
        null_issues = check_null_values(df)
        
        # Duplicate check
        duplicate_issues = check_duplicates(df, schema[table_name])
        
        validation_results.extend([
            *[('Schema', table_name, issue) for issue in schema_issues],
            *[('Null Values', table_name, issue) for issue in null_issues],
            *[('Duplicates', table_name, issue) for issue in duplicate_issues]
        ])
    
    # Generate report
    if validation_results:
        write_validation_report(validation_results, output_folder)
        return False
    return True

def validate_schema(df, table_schema):
    issues = []
    
    # Check column names
    expected_cols = set(table_schema['columns'])
    actual_cols = set(df.columns)
    
    missing_cols = expected_cols - actual_cols
    extra_cols = actual_cols - expected_cols
    
    if missing_cols:
        issues.append(f"Missing columns: {', '.join(missing_cols)}")
    if extra_cols:
        issues.append(f"Extra columns: {', '.join(extra_cols)}")
    
    # Check data types
    for col in df.columns:
        if col in table_schema['columns']:
            col_idx = table_schema['columns'].index(col)
            expected_type = table_schema['datatypes'][col_idx]
            actual_type = str(df[col].dtype)
            if not are_compatible_types(actual_type, expected_type):
                issues.append(f"Column '{col}' has type {actual_type}, expected {expected_type}")
    
    return issues

def check_null_values(df):
    issues = []
    null_counts = df.isnull().sum()
    
    for col, count in null_counts.items():
        if count > 0:
            issues.append(f"Column '{col}' has {count} null values")
    
    return issues

def check_duplicates(df, table_schema):
    issues = []
    # Get first column as primary key
    pk_column = table_schema['columns'][0]
    
    if pk_column in df.columns:
        duplicates = df[df.duplicated(subset=[pk_column], keep=False)]
        if not duplicates.empty:
            issues.append(f"Found {len(duplicates)} duplicate records for primary key column: {pk_column}")
            # Optionally add duplicate values to the report
            duplicate_values = duplicates[pk_column].unique().tolist()
            issues.append(f"Duplicate values: {duplicate_values}")
    else:
        issues.append(f"Primary key column {pk_column} is missing from the data")
    
    return issues

def are_compatible_types(actual_type, expected_type):
    type_mappings = {
        'int64': ['int', 'integer', 'bigint', 'smallint'],
        'float64': ['float', 'decimal', 'numeric', 'double'],
        'object': ['string', 'varchar', 'text', 'char'],
        'datetime64[ns]': ['datetime', 'timestamp', 'date'],
        'bool': ['boolean', 'bit'],
    }
    
    actual_base = actual_type.lower()
    expected_base = str(expected_type).lower()
    
    # Direct match
    if actual_base == expected_base:
        return True
        
    # Check compatible types
    for pandas_type, compatible_types in type_mappings.items():
        if actual_base == pandas_type and any(exp_type in expected_base for exp_type in compatible_types):
            return True
        
    return False

def write_validation_report(validation_results, output_folder):
    report_df = pd.DataFrame(validation_results, columns=['Check Type', 'Table', 'Issue'])
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = Path(output_folder) / f'validation_report_{timestamp}.xlsx'
    
    # Create output folder if it doesn't exist
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        report_df.to_excel(writer, index=False, sheet_name='Validation Results')
        
    print(f"Validation report saved to: {report_path}")


def validate_business_rules(df: pd.DataFrame) -> list:
    issues = []
    if (df['ConsumptionAmount'] < 0).any():
        issues.append("Negative consumption amounts detected.")
    if (df['PaidAmount'] < 0).any():
        issues.append("Negative paid amounts detected.")
    today = pd.Timestamp.today()
    if (pd.to_datetime(df['DateKey'].astype(str), errors='coerce') > today).any():
        issues.append("Future dates detected.")
    return issues
