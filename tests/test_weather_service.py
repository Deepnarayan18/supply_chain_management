"""
Unit tests for Weather Service
"""
import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.weather_service import WeatherService
from config.config import MAJOR_PORTS


class TestWeatherService(unittest.TestCase):
    """Test cases for WeatherService class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.service = WeatherService()
    
    def test_service_initialization(self):
        """Test that service initializes correctly"""
        self.assertIsNotNone(self.service)
        self.assertIsNotNone(self.service.api_key)
    
    def test_get_weather_data(self):
        """Test fetching weather data for a location"""
        # Test with Rotterdam coordinates
        lat, lon = 51.9225, 4.47917
        weather_data = self.service.get_weather_data(lat, lon)
        
        if weather_data:  # Only test if API call succeeds
            self.assertIsInstance(weather_data, dict)
            self.assertIn('weather', weather_data)
            self.assertIn('main', weather_data)
    
    def test_analyze_weather_risk(self):
        """Test weather risk analysis"""
        # Test with sample weather data
        sample_data = {
            'weather': [{'main': 'Clear', 'description': 'clear sky'}],
            'main': {'temp': 20},
            'wind': {'speed': 5},
            'visibility': 10000
        }
        
        risk = self.service.analyze_weather_risk(sample_data)
        
        self.assertIsInstance(risk, dict)
        self.assertIn('risk_score', risk)
        self.assertIn('risk_level', risk)
        self.assertIn('factors', risk)
        self.assertGreaterEqual(risk['risk_score'], 0)
        self.assertLessEqual(risk['risk_score'], 100)
    
    def test_analyze_weather_risk_none(self):
        """Test weather risk analysis with None data"""
        risk = self.service.analyze_weather_risk(None)
        
        self.assertEqual(risk['risk_score'], 0)
        self.assertEqual(risk['risk_level'], 'unknown')
    
    def test_get_all_ports_weather(self):
        """Test fetching weather for all ports"""
        ports_data = self.service.get_all_ports_weather()
        
        self.assertIsInstance(ports_data, list)
        self.assertEqual(len(ports_data), len(MAJOR_PORTS))
        
        for port in ports_data:
            self.assertIn('port_name', port)
            self.assertIn('risk_analysis', port)
            self.assertIn('lat', port)
            self.assertIn('lon', port)


if __name__ == '__main__':
    unittest.main()

# Made with Bob
