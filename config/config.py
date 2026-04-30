"""
Configuration file for Supply Chain Risk Monitor
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENWEATHER_API_KEY = "789f1ee0f9eec1e1b1dfbb9ab1076273"

# Gemini API Key (optional - for LLM summarization)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDzBhaSVXnqbF6YKVL3OGN9LM6QeM3JXXg")
GEMINI_MODEL = "gemini-1.5-flash"

# Application Settings
APP_TITLE = "Global Supply Chain & Brand Risk Monitor"
APP_DESCRIPTION = "AI-Driven Macro-Event & Sentiment Intelligence for Supply Chain Resilience"

# Major shipping hubs and ports coordinates
MAJOR_PORTS = {
    "Rotterdam": {"lat": 51.9225, "lon": 4.47917, "country": "Netherlands"},
    "Singapore": {"lat": 1.3521, "lon": 103.8198, "country": "Singapore"},
    "Shanghai": {"lat": 31.2304, "lon": 121.4737, "country": "China"},
    "Los Angeles": {"lat": 33.7405, "lon": -118.2713, "country": "USA"},
    "Hamburg": {"lat": 53.5511, "lon": 9.9937, "country": "Germany"},
    "Antwerp": {"lat": 51.2194, "lon": 4.4025, "country": "Belgium"},
    "Hong Kong": {"lat": 22.3193, "lon": 114.1694, "country": "Hong Kong"},
    "Dubai": {"lat": 25.2048, "lon": 55.2708, "country": "UAE"},
    "New York": {"lat": 40.7128, "lon": -74.0060, "country": "USA"},
    "Busan": {"lat": 35.1796, "lon": 129.0756, "country": "South Korea"},
    "Mumbai": {"lat": 18.9388, "lon": 72.8354, "country": "India"},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "country": "India"},
}

# Risk thresholds
RISK_THRESHOLDS = {
    "low": 30,
    "medium": 60,
    "high": 85,
    "critical": 100
}

# Weather conditions that impact shipping
SEVERE_WEATHER_CONDITIONS = [
    "thunderstorm", "heavy rain", "snow", "fog", "extreme",
    "storm", "hurricane", "typhoon", "cyclone"
]

# Made with Bob
