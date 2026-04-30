"""
GDELT Service Module
Reads GDELT events data from gdlets.csv file for supply chain risk analysis
"""
import pandas as pd
from typing import Dict, List, Optional, Any, cast
import os
import sys
import requests
from bs4 import BeautifulSoup

# NEW SDK IMPORT
from google import genai
from google.genai import errors

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import GEMINI_API_KEY


class GDELTService:
    """Service to provide GDELT event data with geographic coordinates"""
    
    def __init__(self, csv_path: str = ""):
        """Initialize GDELT service"""
        if not csv_path:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            csv_path = os.path.join(base_dir, 'data', 'gdlets.csv')
            
        self.csv_path = csv_path
        self.events_df: pd.DataFrame = self._load_events_data()
        
        try:
            self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            self.gemini_model = "gemini-2.5-flash-lite"
            self.gemini_available = True
        except Exception as e:
            print(f"Warning: Gemini API not available: {e}")
            self.gemini_available = False
            
        self.url_cache: Dict[str, str] = {}

    def _load_events_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.csv_path)
            
            required_cols = ['SQLDATE', 'ActionGeo_FullName', 'ActionGeo_Lat', 
                           'ActionGeo_Long', 'EventCode', 'GoldsteinScale', 'SOURCEURL']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return pd.DataFrame()
            
            df['SQLDATE'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d', errors='coerce')
            df['ActionGeo_FullName'] = df['ActionGeo_FullName'].fillna('')
            df['SOURCEURL'] = df['SOURCEURL'].fillna('')
            
            df['ActionGeo_Lat'] = pd.to_numeric(df['ActionGeo_Lat'], errors='coerce')
            df['ActionGeo_Long'] = pd.to_numeric(df['ActionGeo_Long'], errors='coerce')
            df['GoldsteinScale'] = pd.to_numeric(df['GoldsteinScale'], errors='coerce')
            df['EventCode'] = pd.to_numeric(df['EventCode'], errors='coerce')
            
            df = df.fillna({
                'ActionGeo_Lat': 0.0, 
                'ActionGeo_Long': 0.0, 
                'GoldsteinScale': 0.0, 
                'EventCode': 0
            })
            
            df = df.dropna(subset=['ActionGeo_Lat', 'ActionGeo_Long'])
            df = df.reset_index(drop=True)
            
            return df
        except Exception:
            return pd.DataFrame()

    def fetch_url_content(self, url: str, timeout: int = 10) -> Optional[str]:
        """Fetch content from a URL with robust anti-bot bypass headers"""
        if url in self.url_cache:
            return self.url_cache[url]
            
        if not url or not str(url).startswith('http'):
            return None
            
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }
            
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
                
            text = soup.get_text(separator=' ', strip=True)
            text = ' '.join(text.split())
            text = text[:3000] 
            
            self.url_cache[url] = text
            return text
        except Exception:
            return None

    def summarize_with_gemini(self, text: str, url: str) -> str:
        """Summarize text content using the new Gemini API (On-Demand Use Only)"""
        if not self.gemini_available or not text:
            return "Summary unavailable - content could not be fetched or processed."
            
        try:
            prompt = (
                "Summarize the following news article in exactly 60 words or less.\n"
                f"Article URL: {url}\n\n"
                f"Article Content:\n{text}\n\n"
                "Provide ONLY the summary, no additional text or formatting:"
            )
            
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt
            )
            
            if not response.text:
                return "Summary unavailable."
                
            return response.text.strip().replace("```json", "").replace("```", "")
        except Exception as e:
            return "Summary generation failed."

    def get_events_for_location(self, location: str) -> List[Dict[str, Any]]:
        """Get GDELT events for a specific location WITHOUT auto-summarizing"""
        if self.events_df.empty:
            return []
            
        mask = self.events_df['ActionGeo_FullName'].str.contains(location, case=False, na=False)
        location_events = self.events_df[mask]
        
        if location_events.empty:
            return []
            
        events: List[Dict[str, Any]] = []
        df_any = cast(Any, location_events)
        records: List[Dict[str, Any]] = df_any.to_dict('records')
        
        for row in records:
            try:
                goldstein_val = float(row.get('GoldsteinScale', 0.0))
                event_location = str(row.get('ActionGeo_FullName', ""))
                
                if goldstein_val < -7:
                    event_type = "CRITICAL_INCIDENT"
                elif goldstein_val < -5:
                    event_type = "NEGATIVE_EVENT"
                elif goldstein_val < -3:
                    event_type = "CONCERN"
                else:
                    event_type = "NEUTRAL_MENTION"
                    
                events.append({
                    "location": location,
                    "event_location": event_location,
                    "event_type": event_type,
                    "event_description": f"Event in {event_location}",
                    "goldstein_scale": goldstein_val,
                    "event_code": int(row.get('EventCode', 0)),
                    "avg_tone": goldstein_val,
                    "tone": goldstein_val,
                    "num_mentions": 1,
                    "confidence": 80.0,
                    "timestamp": row.get('SQLDATE'),
                    "source_url": str(row.get('SOURCEURL', "")).strip(),
                    # DONT SUMMARIZE HERE TO SAVE API LIMITS!
                    "url_summary": "Click 'Generate AI Summary' to view.",
                    "action_geo_lat": float(row.get('ActionGeo_Lat', 0.0)),
                    "action_geo_long": float(row.get('ActionGeo_Long', 0.0))
                })
            except Exception:
                continue
                
        return events

    def get_all_events(self) -> List[Dict[str, Any]]:
        """Get all GDELT events from the CSV WITHOUT auto-summarizing"""
        if self.events_df.empty:
            return []
            
        events: List[Dict[str, Any]] = []
        df_any = cast(Any, self.events_df)
        records: List[Dict[str, Any]] = df_any.to_dict('records')
        
        for row in records:
            try:
                goldstein_val = float(row.get('GoldsteinScale', 0.0))
                event_location = str(row.get('ActionGeo_FullName', ""))
                
                if goldstein_val < -7:
                    event_type = "CRITICAL_INCIDENT"
                elif goldstein_val < -5:
                    event_type = "NEGATIVE_EVENT"
                elif goldstein_val < -3:
                    event_type = "CONCERN"
                else:
                    event_type = "NEUTRAL_MENTION"
                        
                events.append({
                    "location": event_location,
                    "event_location": event_location,
                    "event_type": event_type,
                    "event_description": f"Event in {event_location}",
                    "goldstein_scale": goldstein_val,
                    "event_code": int(row.get('EventCode', 0)),
                    "avg_tone": goldstein_val,
                    "tone": goldstein_val,
                    "num_mentions": 1,
                    "confidence": 80.0,
                    "timestamp": row.get('SQLDATE'),
                    "source_url": str(row.get('SOURCEURL', "")).strip(),
                    # DONT SUMMARIZE HERE TO SAVE API LIMITS!
                    "url_summary": "Click 'Generate AI Summary' to view.",
                    "action_geo_lat": float(row.get('ActionGeo_Lat', 0.0)),
                    "action_geo_long": float(row.get('ActionGeo_Long', 0.0))
                })
            except Exception:
                continue
                
        return events

    def calculate_gdelt_risk_score(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate risk score based on GDELT events"""
        if not events:
            return {
                "risk_score": 0, "event_count": 0, "avg_tone": 0,
                "avg_goldstein": 0, "total_mentions": 0, "avg_confidence": 0,
                "severity": "low", "primary_concerns": []
            }
            
        avg_tone = sum(e.get('avg_tone', 0.0) for e in events) / len(events)
        avg_goldstein = sum(e.get('goldstein_scale', 0.0) for e in events) / len(events)
        total_mentions = sum(e.get('num_mentions', 0) for e in events)
        avg_confidence = sum(e.get('confidence', 0.0) for e in events) / len(events)
        
        tone_risk = min(abs(avg_tone) * 8, 40)
        goldstein_risk = min(abs(avg_goldstein) * 6, 35)
        mention_risk = min(total_mentions / 10, 20)
        confidence_factor = avg_confidence / 100
        
        risk_score = (tone_risk + goldstein_risk + mention_risk) * (0.5 + confidence_factor * 0.5)
        
        if risk_score >= 70:
            severity = "critical"
        elif risk_score >= 50:
            severity = "high"
        elif risk_score >= 30:
            severity = "medium"
        else:
            severity = "low"
            
        primary_concerns = [str(e.get('event_description', '')) for e in events[:3]]
        
        return {
            "risk_score": round(risk_score, 2),
            "event_count": len(events),
            "avg_tone": round(avg_tone, 2),
            "avg_goldstein": round(avg_goldstein, 2),
            "total_mentions": total_mentions,
            "avg_confidence": round(avg_confidence, 2),
            "severity": severity,
            "primary_concerns": primary_concerns
        }