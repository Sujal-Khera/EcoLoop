import requests
import os
import json
import logging
from geopy.distance import geodesic
from math import radians, cos, sin, asin, sqrt

logger = logging.getLogger(__name__)

# These would normally come from environment variables or a config file
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', 'your_api_key_here')

def find_recycling_centers(latitude, longitude, waste_type, radius=5000, max_results=5):
    """
    Find recycling centers near the specified location using Google Places API
    
    Parameters:
    latitude: User's latitude
    longitude: User's longitude
    waste_type: Type of waste (recyclable, compostable, general_waste)
    radius: Search radius in meters (default 5km)
    max_results: Maximum number of results to return
    
    Returns:
    List of recycling centers with details
    """
    if not GOOGLE_MAPS_API_KEY or GOOGLE_MAPS_API_KEY == 'your_api_key_here':
        logger.error("Google Maps API key not configured")
        return []

    try:
        # Define search parameters based on waste type
        search_types = {
            'recyclable': ['recycling_center', 'recycling_drop_off'],
            'compostable': ['compost_facility', 'garden_center'],
            'general_waste': ['waste_facility', 'landfill']
        }
        
        # Validate input parameters
        if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
            raise ValueError("Invalid coordinates provided")
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError("Invalid radius provided")
        if waste_type not in search_types:
            raise ValueError(f"Invalid waste type: {waste_type}")

        # Get appropriate place types to search for
        place_type = search_types[waste_type]
        
        # Use Google Places API to find nearby recycling facilities
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{latitude},{longitude}",
            "radius": radius,
            "type": "establishment",
            "keyword": " OR ".join(place_type),
            "key": GOOGLE_MAPS_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        results = response.json().get("results", [])
        if not results:
            logger.info(f"No recycling centers found for {waste_type} within {radius}m")
            return []

        # Process and format results
        centers = []
        for place in results[:max_results]:
            location = place.get("geometry", {}).get("location", {})
            if not location:
                continue

            center = {
                "name": place.get("name", "Unknown"),
                "address": place.get("vicinity", "Address not available"),
                "latitude": location.get("lat"),
                "longitude": location.get("lng"),
                "rating": place.get("rating", 0.0),
                "place_id": place.get("place_id"),
                "accepts": place_type
            }
            
            # Add directions URL
            center['directions_url'] = (
                f"https://www.google.com/maps/dir/?api=1&"
                f"origin={latitude},{longitude}&"
                f"destination={center['latitude']},{center['longitude']}&"
                f"destination_place_id={center['place_id']}"
            )
            
            centers.append(center)
        
        return centers
        
    except requests.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return []
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error finding recycling centers: {str(e)}")
        return []

def _mock_recycling_centers(latitude, longitude, place_types, radius, max_results):
    """
    Mock function to generate sample recycling centers
    This would be replaced with actual API calls in production
    """
    # In a real app, you would use the Google Places API like this:
    # url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    # params = {
    #     "location": f"{latitude},{longitude}",
    #     "radius": radius,
    #     "type": "recycling_center",  # Note: Google doesn't have all the specific types we want
    #     "keyword": place_types[0],   # Use as keyword instead
    #     "key": GOOGLE_MAPS_API_KEY
    # }
    # response = requests.get(url, params=params)
    # if response.status_code == 200:
    #     results = response.json().get("results", [])
    
    # For this example, generate mock data
    mock_centers = [
        {
            "name": "City Recycling Center",
            "address": "123 Green St, City Center",
            "latitude": latitude + 0.01,
            "longitude": longitude + 0.01,
            "accepts": ["paper", "plastic", "glass", "metal"],
            "rating": 4.5
        },
        {
            "name": "EcoWaste Solutions",
            "address": "456 Sustainable Ave, Green District",
            "latitude": latitude - 0.015,
            "longitude": longitude + 0.008,
            "accepts": ["electronics", "batteries", "plastics"],
            "rating": 4.2
        },
        {
            "name": "Community Compost Center",
            "address": "789 Garden Rd, East Side",
            "latitude": latitude + 0.02,
            "longitude": longitude - 0.01,
            "accepts": ["food waste", "yard waste", "compostable packaging"],
            "rating": 4.7
        },
        {
            "name": "Green Earth Recycling",
            "address": "101 Environmental Blvd, West End",
            "latitude": latitude - 0.008,
            "longitude": longitude - 0.022,
            "accepts": ["cardboard", "paper", "cans", "bottles"],
            "rating": 4.0
        },
        {
            "name": "Total Waste Management",
            "address": "202 Disposal Lane, Industrial Area",
            "latitude": latitude + 0.025,
            "longitude": longitude + 0.018,
            "accepts": ["general waste", "construction debris", "hazardous materials"],
            "rating": 3.8
        }
    ]
    
    # Calculate distances
    for center in mock_centers:
        center_coords = (center["latitude"], center["longitude"])
        user_coords = (latitude, longitude)
        # Calculate distance in kilometers
        center["distance"] = geodesic(user_coords, center_coords).kilometers
    
    # Sort by distance and limit results
    mock_centers.sort(key=lambda x: x["distance"])
    return mock_centers[:max_results]

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r