import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
import re
from fuzzywuzzy import fuzz
from .schema_analyzer import _infer_currency_hints_from_headers,_infer_unit_hints_from_headers

logger = logging.getLogger(__name__)

class DynamicFactMapper:
    """
    Enhanced dynamic mapper for fact table generation that can work with any source schema
    """
    
    def __init__(self):
        self.column_patterns = {
            'financial': ['amount', 'cost', 'price', 'paid', 'expense', 'spend', 'total', 'value', 'fee', 'charge'],
            'temporal': ['date', 'time', 'year', 'month', 'day', 'period', 'timestamp', 'created', 'updated'],
            'location': ['country', 'city', 'location', 'place', 'origin', 'destination', 'from', 'to', 'departure', 'arrival'],
            'organization': ['company', 'organization', 'supplier', 'provider', 'vendor', 'unit', 'department', 'org'],
            'consumption': ['energy', 'fuel', 'electricity', 'gas', 'consumption', 'usage', 'quantity', 'volume', 'kwh', 'mwh'],
            'currency': ['currency', 'curr', 'ccy'],
            'unit': ['unit', 'measure', 'uom', 'measurement'],
            'travel': ['travel', 'trip', 'journey', 'flight', 'hotel', 'distance', 'miles', 'km', 'mileage'],
            'airport': ['airport', 'iata', 'origin', 'destination', 'departure', 'arrival', 'from', 'to'],
            'duration': ['nights', 'days', 'duration', 'stay', 'period', 'length'],
            'provider': ['supplier', 'provider', 'vendor', 'source', 'partner', 'company']
        }
    
    def find_best_column_match(self, source_columns: List[str], target_patterns: List[str], 
                              source_df: pd.DataFrame = None, field_name: str = "") -> Optional[str]:
        """
        Find the best matching source column for a target field using multiple strategies
        """
        if not source_columns:
            return None
        
        # Strategy 1: Exact pattern matching
        for pattern in target_patterns:
            for col in source_columns:
                if pattern.lower() in col.lower():
                    logger.info(f"Exact pattern match for {field_name}: {col} matches pattern '{pattern}'")
                    return col
        
        # Strategy 2: Fuzzy semantic matching
        best_score = 0
        best_match = None
        
        for col in source_columns:
            col_lower = col.lower().strip()
            for pattern in target_patterns:
                score = fuzz.ratio(col_lower, pattern.lower()) / 100.0
                if score > best_score and score > 0.6:  # 60% similarity threshold
                    best_score = score
                    best_match = col
        
        if best_match:
            logger.info(f"Fuzzy match for {field_name}: {best_match} (score: {best_score:.2f})")
            return best_match
        
        # Strategy 3: Data content analysis (if dataframe provided)
        if source_df is not None:
            content_match = self._analyze_column_content(source_df, target_patterns, field_name)
            if content_match:
                logger.info(f"Content-based match for {field_name}: {content_match}")
                return content_match
        
        logger.warning(f"No match found for {field_name} with patterns {target_patterns}")
        return None
    
    def _analyze_column_content(self, source_df: pd.DataFrame, patterns: List[str], field_name: str) -> Optional[str]:
        """
        Analyze column content to find matches based on data patterns
        """
        for col in source_df.columns:
            col_data = source_df[col].dropna()
            if len(col_data) == 0:
                continue
            
            # For financial fields, look for numeric data with currency-like values
            if 'financial' in str(patterns).lower() or 'amount' in field_name.lower():
                if col_data.dtype in ['int64', 'float64']:
                    # Check if values look like monetary amounts
                    if (col_data > 0).any() and col_data.max() > 10:  # Basic heuristic
                        return col
            
            # For currency fields, look for 3-letter codes
            elif 'currency' in str(patterns).lower():
                if col_data.dtype == 'object':
                    sample_values = col_data.astype(str).str.upper().head(10)
                    common_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'CNY', 'INR']
                    if any(curr in sample_values.values for curr in common_currencies):
                        return col
            
            # For airport codes, look for 3-letter alphabetic codes
            elif 'airport' in str(patterns).lower() or field_name in ['origin', 'destination']:
                if col_data.dtype == 'object':
                    sample_values = col_data.astype(str).str.upper().head(5)
                    if all(len(str(v).strip()) == 3 and str(v).isalpha() for v in sample_values if str(v) != 'nan'):
                        return col
        
        return None
    
    def create_dynamic_mapping(self, source_df: pd.DataFrame, mapping: Dict, 
                             calc_method: str, activity_cat: str, activity_subcat: str) -> Dict:
        """
        Create enhanced dynamic mapping based on source data analysis
        """
        source_columns = list(source_df.columns)
        enhanced_mapping = mapping.copy()
        
        # Dynamic field mappings with multiple fallback strategies
        field_mappings = {
            'PaidAmount': {
                'patterns': self.column_patterns['financial'],
                'fallback_patterns': ['total', 'sum', 'value', 'charge', 'fee']
            },
            'ConsumptionAmount': {
                'patterns': self._get_consumption_patterns(calc_method, activity_cat, activity_subcat),
                'fallback_patterns': ['quantity', 'volume', 'count', 'number']
            },
            'DateKey': {
                'patterns': self.column_patterns['temporal'],
                'fallback_patterns': ['created', 'updated', 'transaction']
            },
            'CurrencyID': {
                'patterns': self.column_patterns['currency'],
                'fallback_patterns': ['curr', 'money', 'denomination']
            },
            'UnitID': {
                'patterns': self.column_patterns['unit'],
                'fallback_patterns': ['measure', 'type', 'kind']
            },
            'OrganizationalUnitID': {
                'patterns': self.column_patterns['organization'],
                'fallback_patterns': ['dept', 'division', 'team', 'group']
            },
            'ActivityEmissionSourceProviderID': {
                'patterns': self.column_patterns['provider'],
                'fallback_patterns': ['name', 'source', 'entity']
            }
        }
        
        # Apply dynamic mapping for each field
        for field_name, config in field_mappings.items():
            if field_name not in enhanced_mapping or not enhanced_mapping[field_name].get('source_column'):
                # Try to find the best match
                patterns = config['patterns'] + config.get('fallback_patterns', [])
                best_match = self.find_best_column_match(
                    source_columns, patterns, source_df, field_name
                )
                
                if best_match:
                    if field_name not in enhanced_mapping:
                        enhanced_mapping[field_name] = {}
                    enhanced_mapping[field_name]['source_column'] = best_match
                    enhanced_mapping[field_name]['confidence'] = 'dynamic_match'
                    
                    # Add specific logic for consumption amount
                    if field_name == 'ConsumptionAmount':
                        consumption_type = self._determine_consumption_type(
                            best_match, source_df[best_match], calc_method, activity_cat, activity_subcat
                        )
                        enhanced_mapping[field_name]['consumption_type'] = consumption_type
        
        # Handle special cases for air travel
        if (calc_method == 'Consumption-based' and 
            activity_cat.lower() == 'business travel' and 
            'air' in activity_subcat.lower()):
            
            origin_col = self.find_best_column_match(
                source_columns, 
                ['origin', 'departure', 'from', 'start', 'source', 'dep'],
                source_df, 'origin'
            )
            
            dest_col = self.find_best_column_match(
                source_columns,
                ['destination', 'arrival', 'to', 'end', 'dest', 'arr'],
                source_df, 'destination'
            )
            
            if origin_col and dest_col:
                enhanced_mapping['ConsumptionAmount'] = {
                    'source_column': None,
                    'consumption_type': 'Distance',
                    'transformation': f'calculate_airport_distance_from_{origin_col}_to_{dest_col}',
                    'origin_column': origin_col,
                    'destination_column': dest_col,
                    'relation': None
                }
        
        return enhanced_mapping
    
    def _get_consumption_patterns(self, calc_method: str, activity_cat: str, activity_subcat: str) -> List[str]:
        """
        Get consumption patterns based on activity context
        """
        base_patterns = self.column_patterns['consumption'].copy()
        
        if calc_method == 'Consumption-based':
            if activity_cat.lower() == 'business travel':
                if 'air' in activity_subcat.lower():
                    base_patterns.extend(['distance', 'miles', 'km', 'mileage'])
                elif 'hotel' in activity_subcat.lower():
                    base_patterns.extend(['nights', 'days', 'duration', 'stay'])
                elif 'car' in activity_subcat.lower():
                    base_patterns.extend(['distance', 'miles', 'km', 'fuel', 'gasoline', 'diesel'])
            elif 'energy' in activity_cat.lower():
                base_patterns.extend(['kwh', 'mwh', 'electricity', 'power', 'energy'])
        else:
            # For expense-based, focus on financial amounts
            base_patterns = self.column_patterns['financial']
        
        return base_patterns
    
    def _determine_consumption_type(self, column_name: str, column_data: pd.Series, 
                                  calc_method: str, activity_cat: str, activity_subcat: str) -> str:
        """
        Determine the appropriate consumption type based on column name and data
        """
        col_lower = column_name.lower()
        
        # Distance patterns
        if any(term in col_lower for term in ['distance', 'miles', 'km', 'mileage', 'travel']):
            return 'Distance'
        
        # Energy patterns
        if any(term in col_lower for term in ['kwh', 'mwh', 'energy', 'power', 'electricity']):
            return 'Energy'
        
        # Fuel patterns
        if any(term in col_lower for term in ['fuel', 'gas', 'gasoline', 'diesel', 'petrol']):
            return 'Fuel'
        
        # Time/Duration patterns
        if any(term in col_lower for term in ['days', 'nights', 'duration', 'stay', 'period']):
            return 'Days'
        
        # Heating patterns
        if any(term in col_lower for term in ['heating', 'heat', 'thermal']):
            return 'Heating'
        
        # Electricity patterns (specific)
        if any(term in col_lower for term in ['electricity', 'electric', 'grid']):
            return 'Electricity'
        
        # Default based on activity context
        if activity_cat.lower() == 'business travel':
            if 'air' in activity_subcat.lower():
                return 'Distance'
            elif 'hotel' in activity_subcat.lower():
                return 'Days'
            elif 'car' in activity_subcat.lower():
                return 'Distance'
        elif 'energy' in activity_cat.lower():
            return 'Energy'
        
        # Fallback to Currency for expense-based
        return 'Currency' if calc_method == 'Expense-based' else 'Distance'
    
    def get_dynamic_field_value(self, source_row: pd.Series, mapping: Dict, field_name: str, 
                               lookup_dfs: Dict, default_values: Dict = None) -> Any:
        """
        Get field value using dynamic mapping with intelligent fallbacks
        """
        field_mapping = mapping.get(field_name, {})
        source_column = field_mapping.get('source_column')
        
        # If we have a direct source column mapping
        if source_column and source_column in source_row.index and pd.notna(source_row[source_column]):
            value = source_row[source_column]
            
            # Apply transformations based on field type
            if field_name == 'PaidAmount':
                return float(value) if pd.notna(value) else 0.0
            elif field_name == 'ConsumptionAmount':
                return float(value) if pd.notna(value) else 1.0
            elif field_name in ['ActivityEmissionSourceProviderID']:
                # Handle lookup transformations
                return self._perform_lookup(value, field_name, lookup_dfs)
            else:
                return value
            
        if field_name == 'CurrencyID':
            print(f"✅ Inferring currency for {field_name} from headers ✅ ")
            hint = _infer_currency_hints_from_headers(list(source_row.index))
            if hint:
                looked = self._perform_lookup(hint, field_name, lookup_dfs)
                if looked is not None:
                    return looked
                

        if field_name == 'UnitID':

            print(f" ✅ Inferring unit for {field_name} from headers ✅ ")
            hint = _infer_unit_hints_from_headers(list(source_row.index))
            if hint:
                unit_df = lookup_dfs.get('unit_df')
                if unit_df is not None and not unit_df.empty:
                    match = unit_df[unit_df['UnitName'].astype(str).str.lower() == str(hint).lower()]

                    if not match.empty:
                        return int(match['UnitID'].iloc[0])
        
        # Apply intelligent defaults
        if default_values and field_name in default_values:
            return default_values[field_name]
        
        # Field-specific defaults
        if field_name == 'PaidAmount':
            return 0.0
        elif field_name == 'ConsumptionAmount':
            return 1.0
        elif field_name in ['CurrencyID', 'UnitID', 'ActivityEmissionSourceProviderID']:
            # Return first available ID from lookup tables
            return self._get_default_id(field_name, lookup_dfs)
        

        
        return None
    
    def _perform_lookup(self, value: Any, field_name: str, lookup_dfs: Dict) -> Optional[int]:
        """
        Perform lookup operations for ID fields
        """
        if field_name == 'CurrencyID':
            currency_df = lookup_dfs.get('currency_df')
            if currency_df is not None and not currency_df.empty:
                match = currency_df[currency_df['CurrencyCode'].astype(str).str.upper() == str(value).upper()]
                if not match.empty:
                    return int(match['CurrencyID'].iloc[0])
        
        elif field_name == 'UnitID':
            unit_df = lookup_dfs.get('unit_df')
            if unit_df is not None and not unit_df.empty:
                match = unit_df[unit_df['UnitName'].astype(str).str.lower() == str(value).lower()]
                if not match.empty:
                    return int(match['UnitID'].iloc[0])
        
        elif field_name == 'ActivityEmissionSourceProviderID':
            provider_df = lookup_dfs.get('provider_df')
            if provider_df is not None and not provider_df.empty:
                match = provider_df[provider_df['ProviderName'].astype(str).str.lower() == str(value).lower()]
                if not match.empty:
                    return int(match['ActivityEmissionSourceProviderID'].iloc[0])
        
        return None
    
    def _get_default_id(self, field_name: str, lookup_dfs: Dict) -> Optional[int]:
        """
        Get default ID from lookup tables when no mapping exists
        """
        if field_name == 'CurrencyID':
            currency_df = lookup_dfs.get('currency_df')
            if currency_df is not None and not currency_df.empty:
                return int(currency_df['CurrencyID'].iloc[0])
        
        elif field_name == 'UnitID':
            unit_df = lookup_dfs.get('unit_df')
            if unit_df is not None and not unit_df.empty:
                return int(unit_df['UnitID'].iloc[0])
        
        elif field_name == 'ActivityEmissionSourceProviderID':
            provider_df = lookup_dfs.get('provider_df')
            if provider_df is not None and not provider_df.empty:
                return int(provider_df['ActivityEmissionSourceProviderID'].iloc[0])
        
        return None
    
    def validate_mapping_completeness(self, mapping: Dict, required_fields: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate if mapping covers all required fields
        """
        missing_fields = []
        
        for field in required_fields:
            if field not in mapping:
                missing_fields.append(field)
            elif not mapping[field].get('source_column') and field not in ['EmissionActivityID']:
                # Field exists but has no source mapping (except for auto-increment fields)
                missing_fields.append(f"{field} (no source mapping)")
        
        is_complete = len(missing_fields) == 0
        return is_complete, missing_fields
    
    def enhance_mapping_with_defaults(self, mapping: Dict, required_fields: List[str]) -> Dict:
        """
        Enhance mapping with intelligent defaults for missing fields
        """
        enhanced = mapping.copy()
        
        default_mappings = {
            'EmissionActivityID': {
                'source_column': None,
                'transformation': 'auto_increment',
                'relation': 'primary_key'
            },
            'DateKey': {
                'source_column': None,
                'transformation': 'use_reporting_year',
                'relation': 'D_Date.DateKey->FE1_EmissionActivityData.DateKey'
            },
            'ScopeID': {
                'source_column': 'derived_from_activity_category',
                'transformation': 'lookup_from_activity_category',
                'relation': 'DE1_Scopes.ScopeID->FE1_EmissionActivityData.ScopeID'
            },
            'EmissionFactorID': {
                'source_column': None,
                'transformation': 'generate_from_country_and_source',
                'relation': None
            }
        }
        
        for field in required_fields:
            if field not in enhanced:
                if field in default_mappings:
                    enhanced[field] = default_mappings[field]
                else:
                    enhanced[field] = {
                        'source_column': None,
                        'transformation': f'default_for_{field}',
                        'relation': None
                    }
        
        return enhanced