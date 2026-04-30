"""
Weather Service Module
Integrates with OpenWeather API to fetch weather data for major ports
"""
import requests
from typing import Dict, List, Optional
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import OPENWEATHER_API_KEY, MAJOR_PORTS, SEVERE_WEATHER_CONDITIONS


class WeatherService:
    """Service to fetch and analyze weather data for shipping ports"""
    
    def __init__(self, api_key: str = OPENWEATHER_API_KEY):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        
    def get_weather_data(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Fetch weather data for given coordinates
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Weather data dictionary or None if request fails
        """
        try:
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric"
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def analyze_weather_risk(self, weather_data: Dict) -> Dict:
        """
        Analyze weather data to determine risk level
        
        Args:
            weather_data: Weather data from API
            
        Returns:
            Dictionary with risk score and analysis
        """
        if not weather_data:
            return {
                "risk_score": 0,
                "risk_level": "unknown",
                "factors": ["No weather data available"]
            }
        
        risk_score = 0
        risk_factors = []
        
        # Check weather conditions
        weather_main = weather_data.get("weather", [{}])[0].get("main", "").lower()
        weather_desc = weather_data.get("weather", [{}])[0].get("description", "").lower()
        
        # Check for severe weather
        for severe_condition in SEVERE_WEATHER_CONDITIONS:
            if severe_condition in weather_desc or severe_condition in weather_main:
                risk_score += 30
                risk_factors.append(f"Severe weather: {weather_desc}")
                break
        
        # Check wind speed (m/s)
        wind_speed = weather_data.get("wind", {}).get("speed", 0)
        if wind_speed > 15:  # Strong winds
            risk_score += 25
            risk_factors.append(f"High wind speed: {wind_speed} m/s")
        elif wind_speed > 10:
            risk_score += 15
            risk_factors.append(f"Moderate wind speed: {wind_speed} m/s")
        
        # Check visibility (meters)
        visibility = weather_data.get("visibility", 10000)
        if visibility < 1000:  # Poor visibility
            risk_score += 20
            risk_factors.append(f"Poor visibility: {visibility}m")
        elif visibility < 5000:
            risk_score += 10
            risk_factors.append(f"Reduced visibility: {visibility}m")
        
        # Check temperature extremes
        temp = weather_data.get("main", {}).get("temp", 20)
        if temp < -10 or temp > 45:
            risk_score += 15
            risk_factors.append(f"Extreme temperature: {temp}°C")
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = "critical"
        elif risk_score >= 50:
            risk_level = "high"
        elif risk_score >= 30:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "risk_score": min(risk_score, 100),
            "risk_level": risk_level,
            "factors": risk_factors if risk_factors else ["Normal weather conditions"],
            "weather_description": weather_desc,
            "temperature": temp,
            "wind_speed": wind_speed,
            "visibility": visibility
        }
    
    def get_all_ports_weather(self) -> List[Dict]:
        """
        Fetch weather data for all major ports
        
        Returns:
            List of dictionaries with port and weather information
        """
        ports_data = []
        
        for port_name, port_info in MAJOR_PORTS.items():
            weather_data = self.get_weather_data(port_info["lat"], port_info["lon"])
            
            # Handle None case - analyze_weather_risk already handles None internally
            if weather_data is None:
                risk_analysis = self.analyze_weather_risk({})
            else:
                risk_analysis = self.analyze_weather_risk(weather_data)
            
            ports_data.append({
                "port_name": port_name,
                "country": port_info["country"],
                "lat": port_info["lat"],
                "lon": port_info["lon"],
                "weather_data": weather_data,
                "risk_analysis": risk_analysis,
                "timestamp": datetime.now().isoformat()
            })
        
        return ports_data
    
    def get_weather_for_event(self, event: Dict) -> Dict:
        """
        Fetch weather data for a specific event using its coordinates
        
        Args:
            event: Event dictionary containing action_geo_lat and action_geo_long
            
        Returns:
            Dictionary with weather data and risk analysis for the event location
        """
        lat = event.get('action_geo_lat')
        lon = event.get('action_geo_long')
        
        if lat is None or lon is None:
            return {
                "weather_data": None,
                "risk_analysis": {
                    "risk_score": 0,
                    "risk_level": "unknown",
                    "factors": ["No coordinates available"]
                }
            }
        
        weather_data = self.get_weather_data(lat, lon)
        
        # Handle None case - analyze_weather_risk already handles None internally
        if weather_data is None:
            risk_analysis = self.analyze_weather_risk({})
        else:
            risk_analysis = self.analyze_weather_risk(weather_data)
        
        return {
            "event_location": event.get('event_location', 'Unknown'),
            "coordinates": {"lat": lat, "lon": lon},
            "weather_data": weather_data,
            "risk_analysis": risk_analysis,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_weather_for_events(self, events: List[Dict]) -> List[Dict]:
        """
        Fetch weather data for multiple events using their coordinates
        
        Args:
            events: List of event dictionaries with action_geo_lat and action_geo_long
            
        Returns:
            List of dictionaries with weather data for each event
        """
        weather_results = []
        
        for event in events:
            weather_info = self.get_weather_for_event(event)
            weather_info['event_description'] = event.get('event_description', '')
            weather_info['event_type'] = event.get('event_type', '')
            weather_info['goldstein_scale'] = event.get('goldstein_scale', 0)
            weather_results.append(weather_info)
        
        return weather_results


if __name__ == "__main__":
    # Test the weather service
    service = WeatherService()
    print("Testing Weather Service...")
    print("\n1. Fetching weather data for all major ports...\n")
    
    ports_data = service.get_all_ports_weather()
    
    for port in ports_data[:3]:  # Show first 3 ports
        print(f"\n{port['port_name']}, {port['country']}")
        print(f"Risk Level: {port['risk_analysis']['risk_level'].upper()}")
        print(f"Risk Score: {port['risk_analysis']['risk_score']}")
        print(f"Weather: {port['risk_analysis']['weather_description']}")
        print(f"Factors: {', '.join(port['risk_analysis']['factors'])}")
    
    # Test with sample event coordinates
    print("\n" + "="*80)
    print("\n2. Testing weather fetch for event coordinates...\n")
    
    sample_event = {
        "event_location": "Texas, United States",
        "action_geo_lat": 31.106,
        "action_geo_long": -97.6475,
        "event_description": "Sample event in Texas",
        "event_type": "CONCERN"
    }
    
    event_weather = service.get_weather_for_event(sample_event)
    print(f"Location: {event_weather['event_location']}")
    print(f"Coordinates: {event_weather['coordinates']}")
    print(f"Weather Risk: {event_weather['risk_analysis']['risk_level'].upper()}")
    print(f"Weather: {event_weather['risk_analysis'].get('weather_description', 'N/A')}")

# Made with Bob
