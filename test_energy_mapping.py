import pandas as pd
import logging
from utils.fact import find_emission_ids
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_energy_mapping():
    """Test the energy type mapping functionality"""
    # Load the destination tables
    dest_tables_path = os.path.join('data', 'DestinationTables.xlsx')
    activity_emission_source_df = pd.read_excel(dest_tables_path, sheet_name='DE1_ActivityEmissionSource')
    country_df = pd.read_excel(dest_tables_path, sheet_name='D_Country')
    
    # Create a test DataFrame with different energy types
    test_data = {
        'EnergyOrigin': ['green', 'solar PPA', 'conventional', 'biomass', 'unknown']
    }
    test_df = pd.DataFrame(test_data)
    
    # Print the test data
    print("\nTest Data:")
    print(test_df)
    
    # Print the destination mapping table (first few rows)
    print("\nDestination Mapping Table (first few rows):")
    print(activity_emission_source_df[['ActivityEmissionSourceID', 'ActivityEmissionSourceName']].head())
    
    # Test each energy type
    print("\nTesting Energy Type Mapping:")
    for energy_type in test_data['EnergyOrigin']:
        print(f"\nTesting energy type: '{energy_type}'")
        
        # Create a mock source row with the energy type
        source_row = {'EnergyOrigin': energy_type}
        source_df = pd.DataFrame([source_row])
        
        # Call the function that would normally handle this mapping
        # This is a simplified version just for testing
        # In the real code, this would be part of the generate_fact function
        
        # Normalize the energy type
        energy_type_normalized = energy_type.strip().lower()
        
        # Define the energy type mappings (same as in fact.py)
        energy_type_mappings = {
            # Green electricity mappings
            'green': 'Green Electricity',
            'renewable': 'Green Electricity',
            'solar': 'Green Electricity',  # Keep solar as Green Electricity
            'wind': 'Green Electricity',
            'hydro': 'Green Electricity',
            'clean': 'Green Electricity',
            
            # Conventional electricity mappings
            'conventional': 'Conventional Electricity',
            'fossil': 'Conventional Electricity',
            'coal': 'Conventional Electricity',
            'gas fired': 'Conventional Electricity',
            'non-renewable': 'Conventional Electricity',
            'grid': 'Conventional Electricity',
            
            # Biomass electricity mappings
            # 'biomass': 'Biomass Electricity',
            'organic': 'Biomass Electricity',
            'biofuel': 'Biomass Electricity',
            'waste': 'Biomass Electricity',
            'solar PPA': 'Biomass Electricity'  # Map solar PPA to Biomass Electricity as requested
        }
        
        # Check if we have a direct mapping for this energy type
        mapped_energy_type = None
        # First try exact match
        if energy_type_normalized in energy_type_mappings:
            mapped_energy_type = energy_type_mappings[energy_type_normalized]
            print(f"  ✓ Mapped '{energy_type}' to '{mapped_energy_type}' using exact match")
        # Then try substring match
        else:
            for source_type, dest_type in energy_type_mappings.items():
                if source_type in energy_type_normalized:
                    mapped_energy_type = dest_type
                    print(f"  ✓ Mapped '{energy_type}' to '{mapped_energy_type}' using substring match")
                    break
        
        # If we found a mapping, look up the corresponding ID
        if mapped_energy_type:
            row = activity_emission_source_df[activity_emission_source_df['ActivityEmissionSourceName'] == mapped_energy_type]
            if not row.empty:
                resolved_emission_source_id = int(row['ActivityEmissionSourceID'].iloc[0])
                print(f"  ✓ Found ID {resolved_emission_source_id} for '{mapped_energy_type}'")
                
                # Test EmissionFactorID generation
                country_iso2 = 'IND'  # Example ISO2 code for India
                emission_factor_id = f"{country_iso2}_{str(mapped_energy_type).strip().replace(' ', '_')}"
                print(f"  ✓ Generated EmissionFactorID: {emission_factor_id}")
            else:
                print(f"  ✗ Could not find ID for '{mapped_energy_type}' in the destination table")
        else:
            print(f"  ✗ No mapping found for '{energy_type}'")

def test_emission_factor_id_generation():
    """Test the EmissionFactorID generation functionality"""
    # Load the destination tables
    dest_tables_path = os.path.join('data', 'DestinationTables.xlsx')
    activity_emission_source_df = pd.read_excel(dest_tables_path, sheet_name='DE1_ActivityEmissionSource')
    country_df = pd.read_excel(dest_tables_path, sheet_name='D_Country')
    
    print("\nTesting EmissionFactorID Generation:")
    
    # Test with different countries and energy types
    test_cases = [
        {'country': 'India', 'energy_type': 'Green Electricity'},
        {'country': 'Germany', 'energy_type': 'Conventional Electricity'},
        {'country': 'United States', 'energy_type': 'Biomass Electricity'}
    ]
    
    for case in test_cases:
        country = case['country']
        energy_type = case['energy_type']
        
        # Find the ISO2 code for the country
        country_row = country_df[country_df['CountryName'] == country]
        if not country_row.empty:
            country_iso2 = country_row['ISO2Code'].iloc[0]
            
            # Generate the EmissionFactorID
            emission_factor_id = f"{country_iso2}_{energy_type.replace(' ', '_')}"
            print(f"  ✓ For {country} and {energy_type}: EmissionFactorID = {emission_factor_id}")
        else:
            print(f"  ✗ Could not find ISO2 code for {country}")

if __name__ == "__main__":
    test_energy_mapping()
    test_emission_factor_id_generation()