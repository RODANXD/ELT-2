import pandas as pd
from utils.transformer import transform_data
from utils.file_loader import load_destination_schema, load_destination_tables

# Load destination schema and tables
dest_schema = load_destination_schema('data/DestinationSchema.xlsx')
dest_tables = load_destination_tables('data/DestinationTables.xlsx')

# Print the organizational units to verify
print("\nOrganizational Units in destination tables:")
print(dest_tables['D_OrganizationalUnit'][['OrganizationalUnitID', 'OrganizationalUnitName', 'CompanyID']])

# Print the companies to verify
print("\nCompanies in destination tables:")
print(dest_tables['D_Company'][['CompanyID', 'CompanyName']])

# Create a simple test source dataframe
source_df = pd.DataFrame({
    'invoice_number': ['INV001'],
    'invoice_date': ['2025-01-01'],
    'invoice_amount': [100],
    'supplier_name': ['Test Supplier'],
    'energy_type': ['green_electricity']
})

# Create a simple mapping
mapping = {
    'ActivityEmissionSourceProviderID': {'source_column': 'supplier_name'}
}

# Company name and organizational unit name
company_name = 'BASF SE'
org_unit_name = 'BASF Agricultural Division'

print(f"\nLooking up company: {company_name}")
print(f"Looking up organizational unit: {org_unit_name}")

# Transform the data
result = transform_data(
    source_df,
    mapping,
    company_name,  # Company name
    'Germany',  # Country
    'Consumption-based',  # Calculation method
    'Energy',  # Activity category
    'Electricity',  # Activity subcategory
    dest_schema,
    dest_tables,
    2025,  # Reporting year
    org_unit_name  # Organizational unit name
)

# Print the results
print('\nTransformed data:\n')
for table_name, df in result.items():
    if table_name == 'FE1_EmissionActivityData':
        print(f'\n{table_name}:\n')
        if 'OrganizationalUnitID ' in df.columns:
            print(df[['OrganizationalUnitID ', 'ActivityEmissionSourceProviderID']].head())
        elif 'OrganizationalUnitID' in df.columns:
            print(df[['OrganizationalUnitID', 'ActivityEmissionSourceProviderID']].head())
        else:
            print('OrganizationalUnitID column not found')