"""
Gemini AI Service Module
Uses Google's Gemini API to analyze supply chain risks
"""
from typing import Dict, Optional, Any
import json
import sys
import os
import time
import random

# NEW SDK IMPORT 
from google import genai
from google.genai import errors

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import GEMINI_API_KEY


class GeminiService:
    """Service to use Gemini AI for risk analysis"""
    
    def __init__(self, api_key: str = GEMINI_API_KEY):
        """
        Initialize Gemini service
        
        Args:
            api_key: Gemini API key
        """
        self.client: Any = None
        self.model_name = "gemini-2.5-flash-lite"
        self.available = False
        
        try:
            if api_key:
                self.client = genai.Client(api_key=api_key)
                self.available = True
        except Exception as e:
            print(f"Warning: Gemini API client init failed: {e}")

    def analyze_supply_chain_risk(
        self, 
        location: str,
        gdelt_data: Dict,
        weather_data: Dict
    ) -> Dict:
        """
        Analyze supply chain risk using Gemini AI
        """
        prompt = f"""You will receive input data from two distinct sources. You must evaluate both to determine the overall risk to shipping routes and ports:

1. GDELT (Global Database of Events, Language, and Tone) Data:
   - WHAT IT IS: Real-time global news, sentiment, and geopolitical event data.
   - YOUR TASK: Look for human-driven disruptions. Identify keywords and tones related to worker strikes, political protests, port closures, riots, trade embargoes, or local economic collapses.

2. OpenWeather API Data:
   - WHAT IT IS: Real-time meteorological and environmental forecasts.
   - YOUR TASK: Look for nature-driven disruptions. Identify severe weather conditions such as hurricanes, cyclones, heavy blizzards, floods, or extreme storms that would make logistics operations impossible or unsafe.

INSTRUCTIONS:
- Analyze the provided GDELT and OpenWeather data for the given location.
- Determine if there is a threat to supply chain operations.
- Assign a 'Risk Level' (Low, Medium, High, Critical).
- Generate a concise, 1-2 sentence 'Executive Brief' summarizing exactly WHY the region is flagged (e.g., "Protest at Port of Rotterdam likely to delay shipments due to worker strikes").

LOCATION: {location}

GDELT DATA:
{json.dumps(gdelt_data, indent=2, default=str)}

WEATHER DATA:
{json.dumps(weather_data, indent=2, default=str)}

OUTPUT FORMAT:
Return ONLY a valid JSON object in the following structure, with no extra markdown text:
{{
  "location": "{location}",
  "risk_level": "[Low/Medium/High/Critical]",
  "primary_driver": "[GDELT / OpenWeather / Both]",
  "executive_brief": "[Your 1-2 sentence summary of the disruption]",
  "recommended_action": "[e.g., Reroute shipments to nearby port]"
}}"""

        if not self.available or self.client is None:
            return self._fallback_analysis(location, gdelt_data, weather_data)

        # IMPLEMENTATION OF EXPONENTIAL BACKOFF FOR 429 AND 503 ERRORS
        max_retries = 3
        for attempt in range(max_retries):
            try:
                assert self.client is not None
                
                # Add a baseline pacing delay to avoid hitting the 5/min limit too fast
                if attempt == 0:
                    time.sleep(2) 
                    
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                
                if not response.text:
                    raise ValueError("Empty response from model")
                
                response_text = response.text.strip()
                
                # Cleanup Markdown blocks
                if response_text.startswith("```json"):
                    response_text = response_text.replace("```json", "", 1)
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]
                elif response_text.startswith("```"):
                    response_text = response_text.replace("```", "", 1)
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]
                
                response_text = response_text.strip()
                risk_analysis = json.loads(response_text)
                
                required_fields = ["location", "risk_level", "primary_driver", "executive_brief", "recommended_action"]
                for field in required_fields:
                    if field not in risk_analysis:
                        raise ValueError(f"Missing required field: {field}")
                
                return risk_analysis
                
            except errors.APIError as e:
                error_msg = str(e)
                # Check if it's a rate limit or overloaded server error
                if "429" in error_msg or "503" in error_msg:
                    if attempt < max_retries - 1:
                        # Exponential backoff formula: (2^attempt * 5 seconds) + random jitter
                        wait_time = (2 ** attempt * 5) + random.uniform(1, 3)
                        error_type = "Rate Limit (429)" if "429" in error_msg else "High Demand (503)"
                        print(f"Gemini API {error_type}. Retrying in {wait_time:.1f}s (Attempt {attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Gemini API Error (Exhausted retries): {e}")
                        return self._fallback_analysis(location, gdelt_data, weather_data)
                else:
                    # If it's a different API error (like 400 Bad Request), don't retry, just fallback
                    print(f"Gemini API Error: {e}")
                    return self._fallback_analysis(location, gdelt_data, weather_data)
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing Gemini response as JSON: {e}")
                return self._fallback_analysis(location, gdelt_data, weather_data)
            except Exception as e:
                print(f"Error calling Gemini API: {e}")
                return self._fallback_analysis(location, gdelt_data, weather_data)
                
        # Failsafe return
        return self._fallback_analysis(location, gdelt_data, weather_data)
    
    def _fallback_analysis(self, location: str, gdelt_data: Dict, weather_data: Dict) -> Dict:
        """Fallback risk analysis when Gemini API fails"""
        gdelt_risk = gdelt_data.get('risk_score', 0)
        weather_risk = weather_data.get('risk_score', 0)
        
        total_risk = max(gdelt_risk, weather_risk)
        
        if total_risk >= 70:
            risk_level = "Critical"
        elif total_risk >= 50:
            risk_level = "High"
        elif total_risk >= 30:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        if gdelt_risk > weather_risk:
            primary_driver = "GDELT"
            brief = f"Human-driven disruptions detected at {location} with risk score {gdelt_risk}"
        elif weather_risk > gdelt_risk:
            primary_driver = "OpenWeather"
            brief = f"Weather-related risks detected at {location} with risk score {weather_risk}"
        else:
            primary_driver = "Both"
            brief = f"Multiple risk factors detected at {location}"
        
        return {
            "location": location,
            "risk_level": risk_level,
            "primary_driver": primary_driver,
            "executive_brief": brief,
            "recommended_action": "Monitor situation closely and prepare contingency plans"
        }


if __name__ == "__main__":
    service = GeminiService()
    
    sample_gdelt = {
        "risk_score": 65.5,
        "event_count": 2,
        "avg_tone": -4.8,
        "severity": "high",
        "primary_concerns": ["Port workers union announces 48-hour strike over wage disputes"]
    }
    
    sample_weather = {
        "risk_score": 35,
        "risk_level": "medium",
        "factors": ["Moderate wind speed: 12 m/s", "Reduced visibility: 3000m"],
        "weather_description": "light rain",
        "temperature": 15,
        "wind_speed": 12,
        "visibility": 3000
    }
    
    print("Testing Gemini Service...")
    result = service.analyze_supply_chain_risk("Rotterdam", sample_gdelt, sample_weather)
    print("\nRisk Analysis Result:")
    print(json.dumps(result, indent=2))