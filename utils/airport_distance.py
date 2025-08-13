import pandas as pd
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import logging

# Set up logger
logger = logging.getLogger(__name__)

# Major airport coordinates database (IATA code -> (latitude, longitude))
AIRPORT_COORDINATES = {
    'CDG': (49.0097, 2.5479),   # Paris Charles de Gaulle
    'HND': (35.5494, 139.7798), # Tokyo Haneda
    'LHR': (51.4700, -0.4543),  # London Heathrow
    'JFK': (40.6413, -73.7781), # New York JFK
    'LAX': (33.9425, -118.4081), # Los Angeles
    'FRA': (50.0379, 8.5622),   # Frankfurt
    'AMS': (52.3105, 4.7683),   # Amsterdam Schiphol
    'DXB': (25.2532, 55.3657),  # Dubai International
    'SIN': (1.3644, 103.9915),  # Singapore Changi
    'NRT': (35.7653, 140.3864), # Tokyo Narita
    'ICN': (37.4602, 126.4407), # Seoul Incheon
    'SYD': (-33.9399, 151.1753), # Sydney
    'MEL': (-37.6690, 144.8410), # Melbourne
    'BKK': (13.6900, 100.7501), # Bangkok Suvarnabhumi
    'HKG': (22.3080, 113.9185), # Hong Kong
    'TPE': (25.0777, 121.2328), # Taipei Taoyuan
    'KUL': (2.7456, 101.7072),  # Kuala Lumpur
    'BOM': (19.0896, 72.8656),  # Mumbai
    'DEL': (28.5665, 77.1031),  # Delhi
    'PEK': (40.0799, 116.6031), # Beijing Capital
    'SHA': (31.1979, 121.3364), # Shanghai Hongqiao
    'PVG': (31.1443, 121.8083), # Shanghai Pudong
    'CAN': (23.3924, 113.2988), # Guangzhou
    'SZX': (22.6393, 113.8107), # Shenzhen
    'MAD': (40.4839, -3.5680),  # Madrid
    'BCN': (41.2974, 2.0833),   # Barcelona
    'FCO': (41.7943, 12.2531),  # Rome Fiumicino
    'MXP': (45.6306, 8.7231),   # Milan Malpensa
    'VIE': (48.1103, 16.5697),  # Vienna
    'ZUR': (47.4647, 8.5492),   # Zurich
    'MUC': (48.3537, 11.7750),  # Munich
    'DUS': (51.2895, 6.7668),   # Düsseldorf
    'BER': (52.3667, 13.5033),  # Berlin Brandenburg
    'CPH': (55.6181, 12.6561),  # Copenhagen
    'ARN': (59.6519, 17.9186),  # Stockholm Arlanda
    'OSL': (60.1939, 11.1004),  # Oslo
    'HEL': (60.3172, 24.9633),  # Helsinki
    'WAW': (52.1657, 20.9671),  # Warsaw
    'PRG': (50.1008, 14.2632),  # Prague
    'BUD': (47.4269, 19.2618),  # Budapest
    'ATH': (37.9364, 23.9445),  # Athens
    'IST': (41.2753, 28.7519),  # Istanbul
    'CAI': (30.1127, 31.4000),  # Cairo
    'JNB': (-26.1367, 28.2411), # Johannesburg
    'CPT': (-33.9716, 18.6021), # Cape Town
    'YYZ': (43.6777, -79.6248), # Toronto Pearson
    'YVR': (49.1967, -123.1815), # Vancouver
    'MEX': (19.4363, -99.0721), # Mexico City
    'GRU': (-23.4356, -46.4731), # São Paulo Guarulhos
    'GIG': (-22.8099, -43.2505), # Rio de Janeiro Galeão
    'EZE': (-34.8222, -58.5358), # Buenos Aires Ezeiza
    'BOG': (4.7016, -74.1469),  # Bogotá
    'LIM': (-12.0219, -77.1143), # Lima
    'SCL': (-33.3928, -70.7854), # Santiago
    'DFW': (32.8998, -97.0403), # Dallas/Fort Worth
    'ORD': (41.9742, -87.9073), # Chicago O'Hare
    'ATL': (33.6407, -84.4277), # Atlanta
    'MIA': (25.7959, -80.2870), # Miami
    'SFO': (37.6213, -122.3790), # San Francisco
    'SEA': (47.4502, -122.3088), # Seattle
    'LAS': (36.0840, -115.1537), # Las Vegas
    'PHX': (33.4346, -112.0120), # Phoenix
    'DEN': (39.8561, -104.6737), # Denver
    'IAH': (29.9902, -95.3368), # Houston Intercontinental
    'MSP': (44.8848, -93.2223), # Minneapolis-St. Paul
    'DTW': (42.2162, -83.3554), # Detroit
    'BOS': (42.3656, -71.0096), # Boston Logan
    'PHL': (39.8744, -75.2424), # Philadelphia
    'CLT': (35.2144, -80.9473), # Charlotte
    'LGA': (40.7769, -73.8740), # New York LaGuardia
    'BWI': (39.1774, -76.6684), # Baltimore-Washington
    'DCA': (38.8512, -77.0402), # Washington Reagan
    'IAD': (38.9531, -77.4565), # Washington Dulles
}

def get_airport_coordinates(iata_code):
    """
    Get coordinates for an airport by IATA code.
    Returns (latitude, longitude) tuple or None if not found.
    """
    if not iata_code:
        return None
    
    iata_code = iata_code.strip().upper()
    return AIRPORT_COORDINATES.get(iata_code)

def calculate_airport_distance(origin_code, destination_code):
    """
    Calculate the distance between two airports in kilometers.
    
    Args:
        origin_code (str): IATA code of origin airport
        destination_code (str): IATA code of destination airport
    
    Returns:
        float: Distance in kilometers, or None if calculation fails
    """
    try:
        if not origin_code or not destination_code:
            logger.warning(f"Invalid airport codes: origin='{origin_code}', destination='{destination_code}'")
            return None
        
        origin_coords = get_airport_coordinates(origin_code)
        dest_coords = get_airport_coordinates(destination_code)
        
        if not origin_coords or not dest_coords:
            logger.warning(f"Coordinates not found for airports: {origin_code}={origin_coords}, {destination_code}={dest_coords}")
            return None
        
        # Calculate distance using geodesic (great circle) distance
        distance = geodesic(origin_coords, dest_coords).kilometers
        
        logger.info(f"Calculated distance {origin_code} -> {destination_code}: {distance:.2f} km")
        return round(distance, 2)
        
    except Exception as e:
        logger.error(f"Error calculating distance between {origin_code} and {destination_code}: {e}")
        return None

def calculate_consumption_amount_for_air_travel(source_df, origin_column, destination_column):
    """
    Calculate consumption amounts (distances) for air travel data.
    
    Args:
        source_df (pd.DataFrame): Source data containing flight information
        origin_column (str): Column name containing origin airport codes
        destination_column (str): Column name containing destination airport codes
    
    Returns:
        pd.Series: Series of distances in kilometers
    """
    if origin_column not in source_df.columns or destination_column not in source_df.columns:
        logger.error(f"Required columns not found: {origin_column}, {destination_column}")
        return pd.Series([None] * len(source_df))
    
    distances = []
    for idx, row in source_df.iterrows():
        origin = row.get(origin_column)
        destination = row.get(destination_column)
        distance = calculate_airport_distance(origin, destination)
        distances.append(distance)
    
    return pd.Series(distances)

def add_missing_airport(iata_code, latitude, longitude):
    """
    Add a new airport to the coordinates database.
    This function can be used to extend the airport database if needed.
    """
    if iata_code and isinstance(latitude, (int, float)) and isinstance(longitude, (int, float)):
        AIRPORT_COORDINATES[iata_code.upper()] = (latitude, longitude)
        logger.info(f"Added airport {iata_code}: ({latitude}, {longitude})")
    else:
        logger.error(f"Invalid airport data: {iata_code}, {latitude}, {longitude}") 