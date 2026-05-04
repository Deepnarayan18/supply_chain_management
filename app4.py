import os
import streamlit as st
import asyncio
import aiohttp
import pycountry
import pandas as pd
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
import searoute as sr
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
from streamlit_folium import st_folium
from dotenv import load_dotenv, find_dotenv

from groq import Groq
from geopy.adapters import AioHTTPAdapter
from geopy.geocoders import ArcGIS

# --- CONFIGURATION & API KEYS ---
st.set_page_config(page_title="Supply Chain Intelligence Nexus", layout="wide", initial_sidebar_state="expanded")

# Aggressively locate and load the .env file
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path, override=True)
else:
    st.warning("⚠️ .env file not found automatically. Make sure it is named exactly '.env' and is in the same folder as app4.py")

TIMEOUT = 15
MAX_RETRIES = 3
OSRM_BASE_URL = "https://router.project-osrm.org"
USER_AGENT = "supply_chain_monitor_v46/deepghosh@youremail.com"

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY or not OPENWEATHER_API_KEY:
    st.warning("⚠️ API keys are missing! Please ensure OPENWEATHER_API_KEY and GROQ_API_KEY are set in your .env file.")

# Only initialize the client if the key exists to prevent crashing
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None


# --- HELPER FUNCTIONS ---
def convert_to_iso2(country_string: str) -> str:
    if not country_string: return ""
    country_string = country_string.strip().upper()
    if len(country_string) == 2: return country_string
    try:
        if len(country_string) == 3:
            country_obj = pycountry.countries.get(alpha_3=country_string)
            if country_obj: return country_obj.alpha_2
        country_obj = pycountry.countries.search_fuzzy(country_string)
        if country_obj and len(country_obj) > 0: return country_obj[0].alpha_2
    except Exception: pass
    return country_string[:2]


# --- 1. WEATHER, NEWS & MARINE API CALLS ---
async def get_openweather(lat: float, lon: float) -> Dict[str, Any]:
    if lat is None or lon is None: return {"temp": "N/A", "condition": "N/A"}
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and isinstance(data, dict):
                        main_data = data.get("main", {})
                        temp = main_data.get("temp", "N/A") if isinstance(main_data, dict) else "N/A"
                        weather_array = data.get("weather", [])
                        condition = "N/A"
                        if isinstance(weather_array, list) and len(weather_array) > 0 and isinstance(weather_array[0], dict):
                            condition = weather_array[0].get("description", "N/A").title()
                        return {"temp": temp, "condition": condition}
    except Exception: pass
    return {"temp": "N/A", "condition": "N/A"}

async def get_marine_weather(lat: float, lon: float) -> str:
    if lat is None or lon is None: return "N/A"
    url = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&current=wave_height"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and isinstance(data, dict):
                        current_data = data.get("current", {})
                        if isinstance(current_data, dict):
                            val = current_data.get("wave_height")
                            return f"{val} m" if val is not None else "N/A"
    except Exception: pass
    return "N/A"

async def fetch_real_news_for_location(full_address: str) -> str:
    if not full_address: return "✅ No disruptive news detected recently."
    parts = [p.strip() for p in full_address.split(',')]
    if len(parts) >= 3:
        city_name = parts[-3].split(' ')[0] 
    elif len(parts) == 2:
        city_name = parts[0]
    else:
        city_name = full_address.replace("Coast of ", "").replace("Port of ", "").replace("Open Ocean Waters", "Maritime")

    query = f"{city_name} logistics OR traffic OR transport OR delay OR strike OR port"
    encoded_query = urllib.parse.quote(query)
    
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    timeout_obj = aiohttp.ClientTimeout(total=8)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout_obj) as response:
                if response.status == 200:
                    xml_data = await response.text()
                    if xml_data:
                        root = ET.fromstring(xml_data)
                        item = root.find('.//item')
                        
                        if item is not None:
                            title_elem = item.find('title')
                            if title_elem is not None and title_elem.text:
                                return f"📰 {title_elem.text}"
    except Exception:
        pass
        
    return f"✅ No disruptive news detected for {city_name} recently."

async def get_marine_region_name(lat: float, lon: float) -> str:
    if lat is None or lon is None: return "Open Ocean Waters"
    url = f"https://marineregions.org/rest/getGazetteerRecordsByLatLong.json/{lat}/{lon}/"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and isinstance(data, list) and len(data) > 0:
                        return data[0].get("preferredGazetteerName", "Open Ocean Waters")
    except Exception: pass
    return "Open Ocean Waters"

# --- 2. AI INTELLIGENCE INTEGRATION ---
def get_llm_risk_assessment(location: str, lat: float, lon: float, mode: str, temp: Any, condition: str, waves: str, news: str) -> Dict[str, str]:
    if not groq_client: return {"Risk": "Medium", "Details": "API Key missing. Cannot generate AI risk assessment."}
    
    loc_str = location if location else "Unknown Location"
    news_str = news if news else "No news available."
    
    prompt = f"""
    You are an expert global supply chain intelligence agent executing a clinical node-level risk assessment.
    Evaluate the real-time logistics risk for a transport vehicle travelling through this exact location right now.
    
    Transport Mode: {mode}
    Location: {loc_str} (Lat: {lat}, Lon: {lon})
    Live Weather: {temp}°C, {condition}
    Wave Height (if applicable): {waves}
    Live Regional News Headline: {news_str}

    Consider known geopolitical chokepoints, the live weather, and heavily weigh the live news headline provided.

    Respond STRICTLY in this format (no other text):
    RISK_LEVEL: [Choose exactly one: Low, Medium, High, Critical]
    DETAILS: [1-2 sentences summarizing the specific geographic, weather, and news-related risks for this location]
    """
    
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_completion_tokens=256,
            top_p=1,
            stream=False,
            stop=None
        )
        
        if not completion or not completion.choices or not completion.choices[0].message:
            return {"Risk": "Medium", "Details": "AI assessment failed to return data."}
            
        content = completion.choices[0].message.content
        if not content:
            return {"Risk": "Medium", "Details": "AI content generation resulted in an empty string."}
            
        text = content.strip().split('\n')
        
        risk = "Medium"
        details = "Risk parameters evaluated without explicit regional warnings."
        
        for line in text:
            if line.startswith("RISK_LEVEL:"):
                risk = line.split("RISK_LEVEL:")[1].strip()
            elif line.startswith("DETAILS:"):
                details = line.split("DETAILS:")[1].strip()
                
        return {"Risk": risk, "Details": details}
    except Exception:
        return {"Risk": "Medium", "Details": f"Regional risk assessment unavailable (API Issue)."}

def generate_rerouting_suggestion(milestones: List[Dict[str, Any]], mode: str, origin: str, dest: str, primary_distance: float, alt_info: str, alt_milestones: List[Dict[str, Any]]) -> str:
    if not groq_client: return "API Key missing. Cannot generate intelligence briefing."
    
    if not milestones:
        return "Intelligence Briefing Error: No routing milestones were generated to evaluate."
        
    context = ""
    primary_total_nodes = len(milestones)
    primary_severe_nodes = sum(1 for m in milestones if m.get('Risk Level') in ['High', 'Critical'])
    primary_risk_pct = (primary_severe_nodes / primary_total_nodes) * 100 if primary_total_nodes > 0 else 0

    for m in milestones:
        if m.get('Risk Level') in ['High', 'Critical']:
            context += f"- High Risk at {m.get('Location', 'Unknown')}: {m.get('AI Intelligence', '')} (Weather: {m.get('Temp (°C)', '')}°C, {m.get('Weather', '')})\n"
            
    if not context:
        context = "No severe high-risk bottlenecks identified along the primary route."
        
    alt_context = "No real-time alternative data available."
    alt_risk_pct = 0.0
    
    if alt_milestones:
        alt_context = ""
        alt_total_nodes = len(alt_milestones)
        alt_severe_nodes = sum(1 for am in alt_milestones if am.get('Risk Level') in ['High', 'Critical'])
        alt_risk_pct = (alt_severe_nodes / alt_total_nodes) * 100 if alt_total_nodes > 0 else 0

        for am in alt_milestones:
            if am.get('Risk Level') in ['High', 'Critical', 'Medium']:
                alt_context += f"- Alt Region: {am.get('Location', 'Unknown')} | Risk: {am.get('Risk Level', 'Low')} | Weather: {am.get('Weather', '')} | Temp: {am.get('Temp (°C)', '')}°C | Intel: {am.get('Local News', '')}\n"

    if mode == "Seaways":
        prompt = f"""
        You are a Chief Maritime Data-Geography Analyst.
        
        Route: {origin} to {dest} ({mode})
        Primary Route Real-Time Bottlenecks & Intel:
        {context}
        Primary Route Severe Risk Percentage: {primary_risk_pct:.1f}% of nodes are High/Critical Risk.
        
        Write a highly detailed, professional Executive Briefing. Evaluate the safety, weather, wave heights, and news risks of the primary maritime path. State exactly whether the vessel is cleared to sail or if it must hold at port. Explain how the calculated risk percentage impacts this.

        CRITICAL RULES:
        - NEVER use words like "API", "Algorithm", "Calculated", "Groq", or "Llama".
        - DO NOT mention alternative routes. Evaluate strictly as a Go/No-Go decision based on the primary path.
        """
    else:
        prompt = f"""
        **System Persona:** You are a Chief Logistics Data-Geography Analyst. You operate on cold mathematical logic. You do not recommend a detour unless it mathematically guarantees a safer outcome based on the calculated risk percentage.

        **Task:** Execute a comprehensive risk audit and deliver a definitive routing mandate.

        **Data Feeds:**
        Route: {origin} to {dest} ({mode})
        Primary Route Severe Risk Percentage: {primary_risk_pct:.1f}%
        Primary Route Severe Bottlenecks:
        {context}
        
        Alternative Route Severe Risk Percentage: {alt_risk_pct:.1f}%
        Alternative Route Severe Bottlenecks:
        {alt_context}
        
        **Final Output Format (You must use exactly these headers):**

        ### Primary Route Assessment
        Deliver a clinical analysis of the primary path based on the telemetry provided. Note the exact risk percentage calculated ({primary_risk_pct:.1f}%).

        ### Alternative Route Assessment
        Deliver a clinical analysis of the alternative path. Note the exact risk percentage calculated ({alt_risk_pct:.1f}%).

        ### Definitive Mandate
        State exactly which route to take in one clear, authoritative sentence based strictly on the lower risk percentage.

        ### Mathematical Justification
        Explain your mandate. Explicitly compare the two risk percentages (e.g., "The primary route holds a {primary_risk_pct:.1f}% risk factor versus the alternative's {alt_risk_pct:.1f}%").
        """
        
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.05, 
            max_completion_tokens=700,
            top_p=0.9,
            stream=False,
            stop=None
        )
        
        if not completion or not completion.choices or not completion.choices[0].message:
            return "Strategic intelligence briefing unavailable (API returned empty response)."
            
        content = completion.choices[0].message.content
        if content:
            return content.strip()
        else:
            return "Strategic intelligence briefing unavailable."
    except Exception:
        return f"Strategic intelligence briefing unavailable at this time. (API Error)"

def generate_chart_insight(milestones: List[Dict[str, Any]], alt_milestones: List[Dict[str, Any]], primary_dist: float, alt_dist: float) -> str:
    if not groq_client: return "Insight unavailable due to missing API Key."
    if not milestones: return "Telemetry data unavailable for insight generation."
    
    primary_risks = [m for m in milestones if m.get('Risk Level') in ['High', 'Critical']]
    alt_risks = [m for m in alt_milestones if m.get('Risk Level', 'Low') in ['High', 'Critical']] if alt_milestones else []
    
    if len(alt_risks) >= len(primary_risks) and len(alt_risks) > 0:
        return f"Telemetry indicates the alternative deviation presents equal or higher severe risk exposure ({len(alt_risks)} critical zones) than the primary path ({len(primary_risks)} critical zones). Maintaining the primary route is operationally superior to optimize fuel and distance."
    
    if not primary_risks:
        return "Logistics telemetry indicates stable conditions across the primary corridor. The proposed deviation yields a negative ROI, as the increased fuel expenditure and transit delay provide no proportional reduction in operational risk."
        
    primary_context = f"Critical bottleneck at {primary_risks[0].get('Location', 'Unknown')}. Weather: {primary_risks[0].get('Weather', '')}, {primary_risks[0].get('Temp (°C)', '')}°C. Intel: {primary_risks[0].get('Local News', '')}."
    
    alt_context = "No alternative data mapped."
    if alt_milestones:
        worst_alt = max(alt_milestones, key=lambda x: ['Low', 'Medium', 'High', 'Critical'].index(x.get('Risk Level', 'Low')))
        alt_context = f"Alt path shows max Risk: {worst_alt.get('Risk Level', 'Low')}. Weather: {worst_alt.get('Weather', '')}, {worst_alt.get('Temp (°C)', '')}°C. Intel: {worst_alt.get('Local News', '')}."

    dist_diff = alt_dist - primary_dist
    
    prompt = f"""
    You are a Senior Logistics Operations Analyst. 
    
    Data Feeds:
    - Primary Route Real-Time Intel: {primary_context}
    - Alternative Route Delta: {dist_diff:.1f} km difference
    - Alternative Route Real-Time Intel: {alt_context}
    
    Task: Write a definitive, 3-4 sentence technical analysis concluding which route is operationally superior. 
    If the alternative route is safer, justify the detour. 
    If the alternative route is worse or identical in risk, explicitly recommend maintaining the primary trajectory to optimize fuel/ROI. 
    Do not use introductory filler.
    """
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, 
            max_completion_tokens=200,
            stream=False
        )
        
        if completion and completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            return completion.choices[0].message.content.strip()
        else:
            return f"Telemetry evaluated. Primary route contains {len(primary_risks)} critical zones versus Alternative {len(alt_risks)} critical zones."
    except Exception:
        return f"Telemetry evaluated. Primary route contains {len(primary_risks)} critical zones versus Alternative {len(alt_risks)} critical zones."


# --- 3. GEOSPATIAL LOGIC ROUTERS ---
async def geocode_place_arcgis(place_name: str, geolocator: ArcGIS) -> Optional[Dict[str, Any]]:
    if not place_name: return None
    for attempt in range(MAX_RETRIES):
        try:
            location = await geolocator.geocode(place_name, exactly_one=True, timeout=TIMEOUT) # type: ignore
            if location and hasattr(location, 'latitude') and hasattr(location, 'longitude'): 
                return {"lat": float(location.latitude), "lon": float(location.longitude)}
        except Exception: await asyncio.sleep(1)
    return None

async def calculate_road_route(src: str, dest: str) -> Dict[str, Any]:
    async with ArcGIS(adapter_factory=AioHTTPAdapter) as geolocator:
        origin = await geocode_place_arcgis(src, geolocator)
        dest_loc = await geocode_place_arcgis(dest, geolocator)
        
        if not origin or not dest_loc: 
            return {"error": "Geospatial coordinate mapping failed. Please verify the locations."}

        url = f"{OSRM_BASE_URL}/route/v1/driving/{origin['lon']},{origin['lat']};{dest_loc['lon']},{dest_loc['lat']}"
        params = {"overview": "full", "geometries": "geojson", "steps": "true", "alternatives": "true"}
        timeout_config = aiohttp.ClientTimeout(total=45)
        
        data = None
        try:
            async with aiohttp.ClientSession(timeout=timeout_config) as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                    else:
                        return {"error": f"Routing server returned status {resp.status}."}
        except asyncio.TimeoutError:
            params['alternatives'] = 'false'
            try:
                async with aiohttp.ClientSession(timeout=timeout_config) as session:
                    async with session.get(url, params=params) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                        else:
                            return {"error": "Routing server timeout."}
            except Exception:
                return {"error": "The public routing server timed out."}
        except Exception as e:
            return {"error": f"Failed to connect to the routing server: {str(e)}"}

        if not data or not isinstance(data, dict) or data.get("code") != "Ok" or not data.get("routes"): 
            return {"error": "Primary route generation failed. The routing server did not return a valid path."}
        
        routes = data.get("routes", [])
        if not routes or not isinstance(routes, list):
            return {"error": "Primary route data missing from routing server."}
            
        primary_route = routes[0]
        coords = primary_route.get("geometry", {}).get("coordinates", [])
        
        if not coords:
            return {"error": "Coordinate data missing from primary route."}
            
        primary_distance = primary_route.get("distance", 0) / 1000
        primary_duration = primary_route.get("duration", 0) / 3600
        
        alt_coords = []
        alt_distance = 0
        alt_duration = 0
        alt_info = "No geographical alternative route is currently mapped for this corridor."
        alt_milestones = []
        
        if len(routes) > 1:
            alt_route = routes[1]
            alt_coords = alt_route.get("geometry", {}).get("coordinates", [])
            
            if alt_coords:
                alt_distance = alt_route.get("distance", 0) / 1000
                alt_duration = alt_route.get("duration", 0) / 3600 
                
                alt_waypoints_text = ""
                if len(alt_coords) > 10:
                    alt_interval = max(1, len(alt_coords) // 12)
                    sample_pts = [alt_coords[i] for i in range(alt_interval, len(alt_coords) - alt_interval, alt_interval)]
                    
                    alt_places = []
                    
                    for pt in sample_pts:
                        if not pt or len(pt) < 2: continue
                        lon, lat = pt[0], pt[1]
                        place = f"Alternative Highway near Lat: {round(lat,2)}"
                        try:
                            rev = await geolocator.reverse(f"{lat}, {lon}", timeout=TIMEOUT) # type: ignore
                            if rev and getattr(rev, "address", None):
                                parts = rev.address.split(',')
                                if len(parts) >= 3:
                                    place = f"{parts[-3].strip()}, {parts[-2].strip()}"
                                else:
                                    place = parts[0].strip()
                        except Exception:
                            pass
                            
                        if place not in alt_places:
                            alt_places.append(place)
                        
                        weather = await get_openweather(lat, lon)
                        local_news = await fetch_real_news_for_location(place)
                        llm_eval = get_llm_risk_assessment(place, lat, lon, "Roadways", weather.get("temp", "N/A"), weather.get("condition", "N/A"), "N/A", local_news)
                        
                        alt_milestones.append({
                            "Step": len(alt_milestones) + 1,
                            "Location": place, "Lat": lat, "Lon": lon, 
                            "Temp (°C)": weather.get("temp", "N/A"), "Weather": weather.get("condition", "N/A"), 
                            "Local News": local_news, "Risk Level": llm_eval.get("Risk", "Medium"), "AI Intelligence": llm_eval.get("Details", "")
                        })
                    
                    if alt_places:
                        alt_waypoints_text = f" This strategic deviation physically routes through {len(alt_places)} distinct regional monitoring zones."
                
                alt_info = f"An alternative geographical deviation is mapped. It measures {alt_distance:.1f} km with an estimated transit time of {alt_duration:.1f} hours.{alt_waypoints_text}"
        
        interval = max(1, len(coords) // 15) 
        
        final_waypoints = []
        final_waypoints.append({"name": f"Origin: {src}", "lon": coords[0][0], "lat": coords[0][1]})
        
        for i in range(interval, len(coords) - interval, interval):
            if coords[i] and len(coords[i]) >= 2:
                lon, lat = coords[i][0], coords[i][1]
                final_waypoints.append({"name": "Route Waypoint", "lon": lon, "lat": lat})
            
        final_waypoints.append({"name": f"Destination: {dest}", "lon": coords[-1][0], "lat": coords[-1][1]})

        milestones = []
        for i, pt in enumerate(final_waypoints, 1):
            lon, lat = pt.get("lon"), pt.get("lat")
            if lon is None or lat is None: continue
                
            place = pt.get("name", "Unknown")
            
            if place == "Route Waypoint":
                try:
                    rev = await geolocator.reverse(f"{lat}, {lon}", timeout=TIMEOUT) # type: ignore
                    if rev and getattr(rev, "address", None):
                        parts = rev.address.split(',')
                        if len(parts) >= 3:
                            place = f"{parts[-3].strip()}, {parts[-2].strip()}"
                        else:
                            place = rev.address
                    else:
                        place = f"Highway near Lat: {round(lat,2)}"
                except Exception:
                    place = f"Highway near Lat: {round(lat,2)}"
            
            weather = await get_openweather(lat, lon)
            local_news = await fetch_real_news_for_location(place)
            llm_eval = get_llm_risk_assessment(place, lat, lon, "Roadways", weather.get("temp", "N/A"), weather.get("condition", "N/A"), "N/A", local_news)
            
            milestones.append({
                "Step": i, "Location": place, "Lat": lat, "Lon": lon, 
                "Temp (°C)": weather.get("temp", "N/A"), "Weather": weather.get("condition", "N/A"), 
                "Wave Height": "N/A", "Local News": local_news, "Risk Level": llm_eval.get("Risk", "Medium"), "AI Intelligence": llm_eval.get("Details", "")
            })
            
        return {
            "coords": coords, 
            "alt_coords": alt_coords,
            "distance": primary_distance,
            "primary_duration": primary_duration,
            "alt_distance": alt_distance,
            "alt_duration": alt_duration,
            "milestones": milestones,
            "alt_milestones": alt_milestones,
            "alt_info": alt_info
        }

async def calculate_sea_route(src: str, dest: str) -> Dict[str, Any]:
    async with ArcGIS(adapter_factory=AioHTTPAdapter) as geolocator:
        origin_data = await geocode_place_arcgis(src, geolocator)
        dest_data = await geocode_place_arcgis(dest, geolocator)
    
    if not origin_data or not dest_data: 
        return {"error": "Geospatial coordinate mapping failed. The geocoder could not locate the port cities."}

    # Searoute expects [longitude, latitude] arrays
    origin = [origin_data['lon'], origin_data['lat']]
    dest_loc = [dest_data['lon'], dest_data['lat']]

    try:
        route = sr.searoute(origin, dest_loc) # type: ignore
    except Exception:
        return {"error": "Maritime searoute generation encountered an internal exception."}
        
    if not route or not isinstance(route, dict): 
        return {"error": "Maritime route generation failed. No data returned."}
        
    coords = route.get('geometry', {}).get('coordinates', [])
    
    if not coords or not isinstance(coords, list): 
        return {"error": "No viable seaway coordinates returned by the engine."}
    
    dynamic_sea_points = []
    if coords[0] and len(coords[0]) >= 2:
        dynamic_sea_points.append({"name": f"Port of {src}", "lon": coords[0][0], "lat": coords[0][1]})
    
    seen_waters = set()
    interval = max(5, len(coords) // 30)
    
    for i in range(1, len(coords)-1, interval):
        if not coords[i] or len(coords[i]) < 2: continue
        lon, lat = coords[i][0], coords[i][1]
        water_name = await get_marine_region_name(lat, lon)
        if water_name and water_name != "Open Ocean Waters" and water_name not in seen_waters:
            dynamic_sea_points.append({"name": water_name, "lon": lon, "lat": lat})
            seen_waters.add(water_name)
            
    if coords[-1] and len(coords[-1]) >= 2:
        dynamic_sea_points.append({"name": f"Port of {dest}", "lon": coords[-1][0], "lat": coords[-1][1]})
    
    milestones = []
    for i, pt in enumerate(dynamic_sea_points, 1):
        lon, lat = pt.get("lon"), pt.get("lat")
        if lon is None or lat is None: continue
            
        place = pt.get("name", "Unknown Waterway")
        
        weather = await get_openweather(lat, lon)
        waves = await get_marine_weather(lat, lon)
        local_news = await fetch_real_news_for_location(place)
        llm_eval = get_llm_risk_assessment(place, lat, lon, "Seaways", weather.get("temp", "N/A"), weather.get("condition", "N/A"), waves, local_news)
        
        milestones.append({
            "Step": i, "Location": place, "Lat": lat, "Lon": lon,
            "Temp (°C)": weather.get("temp", "N/A"), "Weather": weather.get("condition", "N/A"),
            "Wave Height": waves, "Local News": local_news, "Risk Level": llm_eval.get("Risk", "Medium"), "AI Intelligence": llm_eval.get("Details", "")
        })

    return {
        "coords": coords, 
        "alt_coords": [], 
        "distance": len(coords), 
        "primary_duration": 0,
        "alt_distance": 0,
        "alt_duration": 0,
        "milestones": milestones,
        "alt_milestones": [],
        "alt_info": ""
    }

# --- DATA VISUALIZATIONS ---
def generate_matplotlib_charts(df: pd.DataFrame, result: Dict[str, Any], mode: str):
    coords = result.get("coords", [])
    if not coords: return
    
    sns.set_theme(style="whitegrid")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#d62728" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"></path>
            </svg>
            <h4 style="margin: 0; color: #1E293B;">Temperature Trends</h4>
        </div>
        """, unsafe_allow_html=True)
        df_temp = df[df['Temp (°C)'] != 'N/A'].copy()
        if not df_temp.empty:
            x_vals = [str(x) for x in df_temp['Step']]
            y_vals = [float(x) for x in df_temp['Temp (°C)']]
            
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(x_vals, y_vals, marker='o', color='#d62728', linewidth=2.5, markersize=6)
            
            ax1.set_xlabel('Milestone Step')
            ax1.set_ylabel('Temperature (°C)')
            
            if len(x_vals) > 10: ax1.set_xticks(x_vals[::len(x_vals)//10])
            st.pyplot(fig1)
        else:
            st.info("No temperature data available.")

    with col2:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ff7f0e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                <line x1="12" y1="9" x2="12" y2="13"></line>
                <line x1="12" y1="17" x2="12.01" y2="17"></line>
            </svg>
            <h4 style="margin: 0; color: #1E293B;">Risk Level Distribution</h4>
        </div>
        """, unsafe_allow_html=True)
        
        colors_dict = {'Low': '#2ca02c', 'Medium': '#ff7f0e', 'High': '#d62728', 'Critical': '#8c564b'}
        risk_list = list(df['Risk Level'])
        risk_counts = {}
        for r in risk_list: risk_counts[str(r)] = risk_counts.get(str(r), 0) + 1
            
        bar_labels = list(risk_counts.keys())
        bar_heights = list(risk_counts.values())
        
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        
        bars = ax2.bar(bar_labels, bar_heights, width=0.6)
        
        for i, bar in enumerate(bars):
            color_to_use = colors_dict.get(bar_labels[i], '#64748b')
            bar.set_color(color_to_use)
            
        ax2.set_ylabel('Number of Milestones')
        
        if len(bar_heights) > 0: ax2.set_yticks(range(0, max(bar_heights)+2))
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        if mode == "Seaways":
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1f77b4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M2 12h4l2-9 4 18 2-9h4"></path>
                </svg>
                <h4 style="margin: 0; color: #1E293B;">Marine Wave Heights</h4>
            </div>
            """, unsafe_allow_html=True)
            df_waves = df[df['Wave Height'] != 'N/A'].copy()
            if not df_waves.empty:
                y_waves = [float(str(x).replace(' m', '')) for x in df_waves['Wave Height']]
                x_waves = [str(x) for x in df_waves['Step']]
                
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                
                ax3.bar(x_waves, y_waves, color='#1f77b4', width=0.6)
                ax3.set_xlabel('Milestone Step')
                ax3.set_ylabel('Wave Height (meters)')
                
                if len(x_waves) > 10: ax3.set_xticks(x_waves[::len(x_waves)//10])
                st.pyplot(fig3)
            else:
                st.info("No wave data available.")
        else:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                    <line x1="3" y1="9" x2="21" y2="9"></line>
                    <line x1="9" y1="21" x2="9" y2="9"></line>
                </svg>
                <h4 style="margin: 0; color: #1E293B;">Traffic & Road Conditions</h4>
            </div>
            """, unsafe_allow_html=True)
            st.info("This metric is reserved for Seaway maritime tracking.")

    with col4:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#0066cc" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <circle cx="12" cy="12" r="3"></circle>
            </svg>
            <h4 style="margin: 0; color: #1E293B;">Geospatial Drift Analysis</h4>
        </div>
        """, unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        
        route_lons = [float(c[0]) for c in coords if len(c) >= 2]
        route_lats = [float(c[1]) for c in coords if len(c) >= 2]
        if route_lons and route_lats:
            ax4.plot(route_lons, route_lats, color='#0066cc', alpha=0.6, linewidth=2, label='Trajectory')
        
        records = df.to_dict('records')
        m_lons = [float(r['Lon']) for r in records if 'Lon' in r and r['Lon'] is not None]
        m_lats = [float(r['Lat']) for r in records if 'Lat' in r and r['Lat'] is not None]
        
        if m_lons and m_lats:
            ax4.scatter(m_lons, m_lats, color='#d62728', s=40, zorder=5, label='Nodes')
            
            for r in records:
                if 'Step' not in r or 'Lon' not in r or 'Lat' not in r: continue
                step_str = str(r['Step'])
                lon_float = float(r['Lon'])
                lat_float = float(r['Lat'])
                if int(step_str) % 3 == 0 or int(step_str) == 1 or int(step_str) == len(records):
                    ax4.annotate(text=step_str, xy=(lon_float, lat_float), xytext=(5, 5), textcoords='offset points', fontsize=8, color='#475569')
            
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
        
        ax4.legend()
        
        st.pyplot(fig4)

    alt_dist = result.get('alt_distance', 0)
    if alt_dist > 0 and mode == "Roadways":
        st.markdown("<hr style='border-color: #e2e8f0;'>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#7e22ce" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="18" y1="20" x2="18" y2="10"></line>
                <line x1="12" y1="20" x2="12" y2="4"></line>
                <line x1="6" y1="20" x2="6" y2="14"></line>
            </svg>
            <h3 style="margin: 0; color: #1E293B; font-weight: 600;">Percentage Risk Comparison Ratio</h3>
        </div>
        """, unsafe_allow_html=True)

        alt_milestones = result.get('alt_milestones', [])
        
        def calculate_percentages(milestones_list):
            if not milestones_list: return [0,0,0,0]
            total = len(milestones_list)
            low = sum(1 for m in milestones_list if m.get('Risk Level') == 'Low') / total * 100
            med = sum(1 for m in milestones_list if m.get('Risk Level') == 'Medium') / total * 100
            high = sum(1 for m in milestones_list if m.get('Risk Level') == 'High') / total * 100
            crit = sum(1 for m in milestones_list if m.get('Risk Level') == 'Critical') / total * 100
            return [low, med, high, crit]

        prim_pct = calculate_percentages(result.get("milestones", []))
        alt_pct = calculate_percentages(alt_milestones)

        fig_pct, ax_pct = plt.subplots(figsize=(10, 3))
        
        y_pos = [0, 1]
        bars_low = [prim_pct[0], alt_pct[0]]
        bars_med = [prim_pct[1], alt_pct[1]]
        bars_high = [prim_pct[2], alt_pct[2]]
        bars_crit = [prim_pct[3], alt_pct[3]]

        ax_pct.barh(y_pos, bars_low, color='#2ca02c', label='Low Risk %')
        ax_pct.barh(y_pos, bars_med, left=bars_low, color='#ff7f0e', label='Medium Risk %')
        
        left_high = [i+j for i,j in zip(bars_low, bars_med)]
        ax_pct.barh(y_pos, bars_high, left=left_high, color='#d62728', label='High Risk %')
        
        left_crit = [i+j for i,j in zip(left_high, bars_high)]
        ax_pct.barh(y_pos, bars_crit, left=left_crit, color='#8c564b', label='Critical Risk %')

        ax_pct.set_yticks(y_pos)
        ax_pct.set_yticklabels(['Primary Route', 'Alternative Route'], fontweight='bold')
        ax_pct.set_xlabel('Percentage (%)')
        ax_pct.set_xlim(0, 100)
        
        for p_index, p_val in enumerate([prim_pct, alt_pct]):
            if p_val[2] + p_val[3] > 0:
                ax_pct.text(101, p_index, f"Severe Risk: {(p_val[2] + p_val[3]):.1f}%", va='center', fontweight='bold', color='#d62728')

        ax_pct.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)
        st.pyplot(fig_pct)
        st.markdown("<hr style='border-color: #e2e8f0; margin-bottom: 30px;'>", unsafe_allow_html=True)

        col5, col6 = st.columns(2)
        
        with col5:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#DAA520" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="12" y1="3" x2="12" y2="21"></line>
                    <path d="M3 13.5l4-8 4 8H3z"></path>
                    <path d="M13 13.5l4-8 4 8h-8z"></path>
                    <line x1="3" y1="13.5" x2="21" y2="13.5"></line>
                </svg>
                <h3 style="margin: 0; color: #1E293B; font-weight: 600;">Route Viability Comparison</h3>
            </div>
            """, unsafe_allow_html=True)
            
            chart_insight_text = generate_chart_insight(
                result.get("milestones", []), 
                result.get('alt_milestones', []), 
                result.get('distance', 0), 
                alt_dist
            )
            st.info(f"**Logistics Insight:** {chart_insight_text}")
            
            fig5, ax5 = plt.subplots(figsize=(6, 4))
            
            categories = ['Distance (km)', 'Transit Time (Hrs)']
            primary_vals = [result.get('distance', 0), result.get('primary_duration', 0)]
            alt_vals = [alt_dist, result.get('alt_duration', 0)]
            
            x = [0, 1]
            width_bar = 0.35
            
            ax5.bar([i - width_bar/2 for i in x], primary_vals, width_bar, label='Primary', color='#0066cc')
            ax5.bar([i + width_bar/2 for i in x], alt_vals, width_bar, label='Alternative', color='#DAA520')
            
            ax5.set_ylabel('Metrics')
            ax5.set_xticks(x)
            ax5.set_xticklabels(categories)
            
            ax5.legend()
            
            max_height = max(primary_vals + alt_vals)
            for i, val in enumerate(primary_vals):
                ax5.text(i - width_bar/2, val + (max_height * 0.02), f"{val:.1f}", ha='center', fontweight='bold')
            for i, val in enumerate(alt_vals):
                ax5.text(i + width_bar/2, val + (max_height * 0.02), f"{val:.1f}", ha='center', fontweight='bold')

            st.pyplot(fig5)
            
        with col6:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#8c564b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                    <line x1="12" y1="9" x2="12" y2="13"></line>
                    <line x1="12" y1="17" x2="12.01" y2="17"></line>
                </svg>
                <h3 style="margin: 0; color: #1E293B; font-weight: 600;">Hazard Distribution Matrix</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("**Risk Matrix Insight:** Comparing absolute volume of severe vs moderate bottlenecks detected live on both vectors.")
            
            primary_severe = len([m for m in result.get("milestones", []) if m.get('Risk Level') in ['High', 'Critical']])
            alt_severe = len([m for m in alt_milestones if m.get('Risk Level', 'Low') in ['High', 'Critical']])
            
            primary_med = len([m for m in result.get("milestones", []) if m.get('Risk Level') == 'Medium'])
            alt_med = len([m for m in alt_milestones if m.get('Risk Level', 'Low') == 'Medium'])
            
            fig6, ax6 = plt.subplots(figsize=(6, 4))
            
            risk_labels = ['Severe Hazards', 'Moderate Hazards']
            prim_counts = [primary_severe, primary_med]
            alt_counts = [alt_severe, alt_med]
            
            x_risk = [0, 1]
            
            ax6.bar([i - width_bar/2 for i in x_risk], prim_counts, width_bar, label='Primary', color='#0066cc')
            ax6.bar([i + width_bar/2 for i in x_risk], alt_counts, width_bar, label='Alternative', color='#DAA520')
            
            ax6.set_ylabel('Node Count')
            ax6.set_xticks(x_risk)
            ax6.set_xticklabels(risk_labels)
            
            max_y = max(max(prim_counts), max(alt_counts))
            if max_y < 5: ax6.set_ylim(0, max_y + 1.5)
            
            ax6.legend()
            
            for i, val in enumerate(prim_counts):
                ax6.text(i - width_bar/2, val + 0.1, str(val), ha='center', fontweight='bold')
            for i, val in enumerate(alt_counts):
                ax6.text(i + width_bar/2, val + 0.1, str(val), ha='center', fontweight='bold')

            st.pyplot(fig6)

# --- UI ---
def main():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
            
            .stApp {
                font-family: 'Inter', sans-serif;
                background-color: #f8fafc;
            }
            
            .block-container {
                padding-top: 2rem !important;
                padding-bottom: 2rem !important;
                max-width: 95% !important;
            }
            
            .main-header {
                font-family: 'Inter', sans-serif;
                font-size: 2.2rem;
                font-weight: 700;
                color: #0f172a;
                margin-bottom: 0.2rem;
                letter-spacing: -0.02em;
            }
            .sub-header {
                font-family: 'Inter', sans-serif;
                color: #475569;
                font-size: 1.1rem;
                margin-bottom: 2rem;
            }
            
            .strategy-box {
                background-color: #ffffff;
                border-left: 4px solid #0284c7;
                border-top: 1px solid #e2e8f0;
                border-right: 1px solid #e2e8f0;
                border-bottom: 1px solid #e2e8f0;
                padding: 1.5rem;
                border-radius: 6px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
                font-family: 'Inter', sans-serif;
                font-size: 1rem;
                line-height: 1.6;
                color: #334155;
            }
            
            [data-testid="stSidebar"] {
                background-color: #ffffff;
                border-right: 1px solid #e2e8f0;
            }
            
            .stButton>button {
                background: #0284c7;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 0.5rem 1rem;
                font-weight: 600;
                font-family: 'Inter', sans-serif;
                width: 100%;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                background: #0369a1;
                box-shadow: 0 4px 12px rgba(2, 132, 199, 0.2);
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)
    
    col_logo, col_title = st.columns([1, 4])
    
    with col_logo:
        try:
            st.image("asb_logo_light.png", width="stretch")
        except Exception:
            pass
            
    with col_title:
        st.markdown("""
            <div style="display: flex; flex-direction: column; justify-content: center; height: 100%;">
                <h1 class="main-header">Global Supply Chain Nexus</h1>
                <p class="sub-header">Advanced AI Routing, Real-Time Geo-Risk Analysis, & Maritime Tracking.</p>
            </div>
        """, unsafe_allow_html=True)

    if 'report_generated' not in st.session_state:
        st.session_state.report_generated = False
    if 'mode' not in st.session_state:
        st.session_state.mode = ""
    if 'result' not in st.session_state:
        st.session_state.result = {}
    if 'reroute_strategy' not in st.session_state:
        st.session_state.reroute_strategy = ""

    with st.sidebar:
        try:
            st.image("asb_logo_light.png", width="stretch")
            st.markdown("<hr style='margin-top: 5px; margin-bottom: 15px;'>", unsafe_allow_html=True)
        except Exception:
            pass
            
        st.markdown("<h2 style='color: #0f172a; font-family: Inter; font-size: 1.2rem; margin-bottom: 20px;'>Mission Parameters</h2>", unsafe_allow_html=True)
        mode_input = st.selectbox("Transport Vector", ["Roadways", "Seaways"])
        
        default_origin = "Mumbai Port" if mode_input == "Seaways" else "Mumbai, India"
        default_dest = "Port of London" if mode_input == "Seaways" else "Dhaka, Bangladesh"
        
        origin_input = st.text_input("Origin Node", default_origin)
        destination_input = st.text_input("Destination Node", default_dest)
        
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("Generate Intelligence Report")

    if analyze_btn:
        with st.spinner("Establishing secure uplink. Scanning global risk nodes & meteorological data..."):
            
            if mode_input == "Roadways":
                result = asyncio.run(calculate_road_route(origin_input, destination_input))
            else:
                result = asyncio.run(calculate_sea_route(origin_input, destination_input))

            if "error" in result:
                st.error(result["error"])
                return
            
            if result.get("milestones"):
                strategy = generate_rerouting_suggestion(
                    milestones=result["milestones"], 
                    mode=mode_input, 
                    origin=origin_input, 
                    dest=destination_input,
                    primary_distance=result.get("distance", 0),
                    alt_info=result.get("alt_info", ""),
                    alt_milestones=result.get("alt_milestones", [])
                )
                
                st.session_state.mode = mode_input
                st.session_state.result = result
                st.session_state.reroute_strategy = strategy
                st.session_state.report_generated = True
            else:
                st.error("Routing engine could not map the provided locations.")

    if st.session_state.report_generated:
        mode = st.session_state.mode
        result = st.session_state.result
        milestones = result.get("milestones", [])
        
        if not milestones:
            st.error("No valid milestones mapped. Please verify origin and destination.")
            return
        
        def color_risk(val):
            color = '#10b981' if val == 'Low' else '#f59e0b' if val == 'Medium' else '#ef4444' if val in ['High', 'Critical'] else '#334155'
            return f'color: {color}; font-weight: bold; background-color: transparent;'

        st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-top: 20px; margin-bottom: 15px;">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#0f172a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="16" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12.01" y2="8"></line>
            </svg>
            <h3 style="margin: 0; color: #0f172a; font-family: 'Inter', sans-serif; font-size: 1.5rem; font-weight: 700;">Executive Mandate & Strategy</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"<div class='strategy-box'>{st.session_state.reroute_strategy.replace('###', '<br><b><span style=\"color:#0284c7;\">►</span>').replace('**', '')}</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-top: 40px; margin-bottom: 15px;">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#0f172a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"></polygon>
                <line x1="9" y1="3" x2="9" y2="18"></line>
                <line x1="15" y1="6" x2="15" y2="21"></line>
            </svg>
            <h3 style="margin: 0; color: #0f172a; font-family: 'Inter', sans-serif; font-size: 1.5rem; font-weight: 700;">Tactical Geospatial Map</h3>
        </div>
        """, unsafe_allow_html=True)
        
        color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red', 'Critical': 'darkred'}
        
        m = folium.Map(location=[0, 0], zoom_start=2, tiles="CartoDB positron")
        
        folium_coords = []
        for c in result.get("coords", []):
            if c and len(c) >= 2:
                folium_coords.append([float(c[1]), float(c[0])])
                
        if folium_coords:
            route_line = folium.PolyLine(
                locations=folium_coords,
                color="#0ea5e9",
                weight=4,
                opacity=0.8,
                tooltip=f"Primary {mode} Vector"
            )
            route_line.add_to(m)
            
            plugins.PolyLineTextPath(
                route_line,
                "\u25BA", 
                repeat=True,
                offset=6,
                attributes={'fill': '#0ea5e9', 'font-weight': 'bold', 'font-size': '14'}
            ).add_to(m)

        if result.get("alt_coords"):
            alt_folium_coords = []
            for c in result.get("alt_coords", []):
                if c and len(c) >= 2:
                    alt_folium_coords.append([float(c[1]), float(c[0])])
            if alt_folium_coords:
                folium.PolyLine(
                    locations=alt_folium_coords,
                    color="#f59e0b", 
                    weight=3,
                    opacity=0.7,
                    dash_array='5, 5',
                    tooltip="Alternative Vector"
                ).add_to(m)

        def generate_popup_html(location, risk, temp, weather, news, intel, border_color):
            return f"""
            <div style="width:260px; background-color:#ffffff; color:#334155; padding:10px; border-radius:5px; border-left:4px solid {border_color}; font-family:sans-serif; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <h4 style="color:#0f172a; margin-top:0; border-bottom:1px solid #e2e8f0; padding-bottom:5px;">{location}</h4>
                <span style="font-size:0.9rem;">
                <b>Risk:</b> <span style="color:{border_color}; font-weight:bold;">{risk}</span><br>
                <b>Meteorology:</b> {temp}°C, {weather}<br>
                <div style="margin-top:8px; padding-top:8px; border-top:1px solid #e2e8f0;">
                <b>Intel:</b> {news}<br>
                <b>Audit:</b> {intel}
                </div></span>
            </div>
            """

        for row in result.get("alt_milestones", []):
            risk_level = str(row.get('Risk Level', 'Low'))
            marker_color = color_map.get(risk_level, 'gray')
            hex_color = '#10b981' if risk_level == 'Low' else '#f59e0b' if risk_level == 'Medium' else '#ef4444'
            
            lat = row.get('Lat')
            lon = row.get('Lon')
            if lat is None or lon is None: continue
                
            location_name = str(row.get('Location', 'Unknown'))
            
            popup_html = generate_popup_html(location_name, risk_level, row.get('Temp (°C)', 'N/A'), row.get('Weather', 'N/A'), row.get('Local News', 'N/A'), row.get('AI Intelligence', ''), hex_color)
            
            marker = folium.Marker(
                location=[float(lat), float(lon)],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color='orange', icon='star', icon_color='white')
            )
            marker.add_to(m)

        for row in milestones:
            risk_level = str(row.get('Risk Level', 'Low'))
            marker_color = color_map.get(risk_level, 'gray')
            hex_color = '#10b981' if risk_level == 'Low' else '#f59e0b' if risk_level == 'Medium' else '#ef4444'
            
            lat = row.get('Lat')
            lon = row.get('Lon')
            if lat is None or lon is None: continue
                
            location_name = str(row.get('Location', 'Unknown'))
            
            popup_html = generate_popup_html(location_name, risk_level, row.get('Temp (°C)', 'N/A'), row.get('Weather', 'N/A'), row.get('Local News', 'N/A'), row.get('AI Intelligence', ''), hex_color)
            
            marker = folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=7,
                popup=folium.Popup(popup_html, max_width=300),
                color=hex_color,
                fill=True,
                fill_color=hex_color,
                fill_opacity=0.8
            )
            marker.add_to(m)

        all_lats = []
        all_lons = []
        for c in result.get("coords", []):
            if c and len(c) >= 2:
                all_lats.append(float(c[1]))
                all_lons.append(float(c[0]))
        for c in result.get("alt_coords", []):
            if c and len(c) >= 2:
                all_lats.append(float(c[1]))
                all_lons.append(float(c[0]))
            
        if all_lats and all_lons:
            m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])

        st_folium(m, width=1200, height=500, returned_objects=[]) 

        st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-top: 40px; margin-bottom: 15px;">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#0f172a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
                <polyline points="10 9 9 9 8 9"></polyline>
            </svg>
            <h3 style="margin: 0; color: #0f172a; font-family: 'Inter', sans-serif; font-size: 1.5rem; font-weight: 700;">Node Telemetry Matrix</h3>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Primary Vector", "Alternative Vector"])
        
        with tab1:
            df_primary = pd.DataFrame(milestones)
            if 'Temp (°C)' in df_primary.columns: df_primary['Temp (°C)'] = df_primary['Temp (°C)'].astype(str)
            if 'Wave Height' in df_primary.columns: df_primary['Wave Height'] = df_primary['Wave Height'].astype(str)
            styled_primary = df_primary.style.map(color_risk, subset=['Risk Level'])
            
            st.dataframe(styled_primary, width="stretch")
            
            csv_primary = df_primary.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export Primary Route CSV", data=csv_primary, file_name=f"{mode}_primary_report.csv", mime="text/csv", key="btn_primary")
            
        with tab2:
            alt_milestones = result.get('alt_milestones', [])
            if alt_milestones:
                df_alt = pd.DataFrame(alt_milestones)
                if 'Temp (°C)' in df_alt.columns: df_alt['Temp (°C)'] = df_alt['Temp (°C)'].astype(str)
                if 'Wave Height' not in df_alt.columns:
                    df_alt['Wave Height'] = "N/A"
                df_alt['Wave Height'] = df_alt['Wave Height'].astype(str)
                
                styled_alt = df_alt.style.map(color_risk, subset=['Risk Level'])
                
                st.dataframe(styled_alt, width="stretch")
                
                csv_alt = df_alt.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Export Alternative Route CSV", data=csv_alt, file_name=f"{mode}_alt_report.csv", mime="text/csv", key="btn_alt")
            else:
                st.info("No alternative routing available for this corridor.")

        st.markdown("<hr style='border-color: #e2e8f0; margin-top: 30px;'>", unsafe_allow_html=True)
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-top: 10px; margin-bottom: 20px;">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#0f172a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="18" y1="20" x2="18" y2="10"></line>
                <line x1="12" y1="20" x2="12" y2="4"></line>
                <line x1="6" y1="20" x2="6" y2="14"></line>
            </svg>
            <h3 style="margin: 0; color: #0f172a; font-family: 'Inter', sans-serif; font-size: 1.5rem; font-weight: 700;">Advanced Operational Analytics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        df_for_charts = pd.DataFrame(milestones)
        if 'Temp (°C)' in df_for_charts.columns: df_for_charts['Temp (°C)'] = df_for_charts['Temp (°C)'].astype(str)
        if 'Wave Height' in df_for_charts.columns: df_for_charts['Wave Height'] = df_for_charts['Wave Height'].astype(str)
            
        generate_matplotlib_charts(df_for_charts, result, mode)

if __name__ == "__main__":
    main()