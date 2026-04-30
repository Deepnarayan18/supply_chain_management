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

# Folium Imports for the Interactive Map
import folium
from folium import plugins
from streamlit_folium import st_folium

# Groq Import
from groq import Groq

from geopy.adapters import AioHTTPAdapter
from geopy.geocoders import ArcGIS, Nominatim

# --- CONFIGURATION & API KEYS ---
# Set page layout and ensure the sidebar is always expanded by default
st.set_page_config(
    page_title="Supply Chain Intelligence Nexus", 
    layout="wide",
    initial_sidebar_state="expanded"
)

TIMEOUT = 15
MAX_RETRIES = 3
OSRM_BASE_URL = "https://router.project-osrm.org"
USER_AGENT = "supply_chain_monitor_v46/deepghosh@youremail.com"

# API KEYS (Ensure these are your active keys)
OPENWEATHER_API_KEY = "789f1ee0f9eec1e1b1dfbb9ab1076273"
GROQ_API_KEY = "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"

# Initialize Groq Client
groq_client = Groq(api_key=GROQ_API_KEY)


# --- HELPER FUNCTIONS ---
def convert_to_iso2(country_string: str) -> str:
    country_string = country_string.strip().upper()
    if not country_string: return ""
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
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, dict):
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
    url = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&current=wave_height"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, dict):
                        current_data = data.get("current", {})
                        if isinstance(current_data, dict):
                            val = current_data.get("wave_height")
                            return f"{val} m" if val is not None else "N/A"
    except Exception: pass
    return "N/A"


async def fetch_real_news_for_location(full_address: str) -> str:
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
    url = f"https://marineregions.org/rest/getGazetteerRecordsByLatLong.json/{lat}/{lon}/"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, list) and len(data) > 0:
                        return data[0].get("preferredGazetteerName", "Open Ocean Waters")
    except Exception: pass
    return "Open Ocean Waters"


# --- 2. AI INTELLIGENCE INTEGRATION ---
def get_llm_risk_assessment(location: str, lat: float, lon: float, mode: str, temp: Any, condition: str, waves: str, news: str) -> Dict[str, str]:
    prompt = f"""
    You are an expert global supply chain intelligence agent.
    Evaluate the real-time logistics risk for a transport vehicle travelling through this exact location right now.
    
    Transport Mode: {mode}
    Location: {location} (Lat: {lat}, Lon: {lon})
    Live Weather: {temp}°C, {condition}
    Wave Height (if applicable): {waves}
    Live Regional News Headline: {news}

    Consider known geopolitical chokepoints, the live weather, and heavily weigh the live news headline provided.

    Respond STRICTLY in this format (no other text):
    RISK_LEVEL: [Choose exactly one: Low, Medium, High, Critical]
    DETAILS: [1-2 sentences summarizing the specific geographic, weather, and news-related risks for this location]
    """
    
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=256,
            top_p=1,
            stream=False,
            stop=None
        )
        
        content = completion.choices[0].message.content or ""
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
        return {"Risk": "Medium", "Details": f"Regional risk assessment unavailable for: {location}"}


def generate_rerouting_suggestion(milestones: List[Dict[str, Any]], mode: str, origin: str, dest: str, primary_distance: float, alt_info: str, alt_milestones: List[Dict[str, Any]]) -> str:
    context = ""
    for m in milestones:
        if m['Risk Level'] in ['High', 'Critical']:
            context += f"- High Risk at {m['Location']}: {m['AI Intelligence']}\n"
            
    if not context:
        context = "No severe high-risk bottlenecks identified along the primary route."
        
    alt_context = "No real-time alternative data available."
    if alt_milestones:
        alt_context = ""
        for am in alt_milestones:
            alt_context += f"- Alt Region: {am['Location']} | Risk: {am['Risk Level']} | Weather: {am['Weather']}, {am['Temp (°C)']}°C | Intel: {am['Local News']}\n"

    prompt = f"""
    You are the Lead Supply Chain Intelligence Director providing a final, definitive routing decision to fleet operators.
    
    Route: {origin} to {dest} ({mode})
    Primary Route Length: {primary_distance:.1f} km
    Primary Route Real-Time Bottlenecks & Intel:
    {context}
    
    Alternative Route Logistics Data:
    {alt_info}
    Alternative Route Real-Time Intel:
    {alt_context}
    
    Write a highly detailed, professional Executive Briefing with EXACTLY this structure:
    
    ### 🔴 Primary Route Assessment
    [Evaluate the real-time safety, weather, and news risks of the primary path based strictly on the intel provided.]
    
    ### 🟡 Alternative Route Assessment
    [Evaluate the real-time safety, weather, and news risks of the alternative path based strictly on the intel provided.]
    
    ### ✅ Definitive Recommendation
    [State exactly which route to take in one clear sentence.]
    
    ### 💡 Why This Route is Superior
    [Explicitly explain WHY the recommended route is better. Directly compare the real-time intel, weather, risk levels, and the distance trade-off. Explain how avoiding the specific risks on the bad route makes the extra distance mathematically and operationally worth it for the logistics team.]
    
    CRITICAL RULES:
    - NEVER use words like "API", "OSRM", "Algorithm", "Calculated", "Groq", or "Llama".
    - Make the analysis definitive, highly technical, and end-to-end helpful for a logistics manager deciding whether to accept a detour.
    """
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, 
            max_completion_tokens=600,
            top_p=1,
            stream=False,
            stop=None
        )
        content = completion.choices[0].message.content
        if content is not None:
            return content.strip()
        else:
            return "Strategic intelligence briefing unavailable."
    except Exception:
        return "Strategic intelligence briefing unavailable at this time."


def generate_chart_insight(milestones: List[Dict[str, Any]], alt_milestones: List[Dict[str, Any]], primary_dist: float, alt_dist: float) -> str:
    """Generates a detailed, technical logistics insight based on real-time API data comparing both routes."""
    
    primary_risks = [m for m in milestones if m['Risk Level'] in ['High', 'Critical']]
    primary_context = "Clear conditions."
    if primary_risks:
        worst_primary = primary_risks[0]
        primary_context = f"Critical bottleneck at {worst_primary['Location']}. Weather: {worst_primary['Weather']}, {worst_primary['Temp (°C)']}°C. Intel: {worst_primary['Local News']}. Risk: {worst_primary['Risk Level']}."

    alt_context = "No alternative data mapped."
    if alt_milestones:
        worst_alt = max(alt_milestones, key=lambda x: ['Low', 'Medium', 'High', 'Critical'].index(x.get('Risk Level', 'Low')))
        alt_context = f"Alt path via {worst_alt['Location']} shows Risk: {worst_alt['Risk Level']}. Weather: {worst_alt['Weather']}, {worst_alt['Temp (°C)']}°C. Intel: {worst_alt['Local News']}."

    dist_diff = alt_dist - primary_dist
    
    if not primary_risks:
        return "Logistics telemetry indicates stable conditions across the primary corridor. The proposed deviation yields negative ROI, as the increased fuel expenditure and transit delay provide no proportional reduction in operational risk."
        
    prompt = f"""
    You are a Senior Logistics Operations Analyst. Write a technical, data-driven analysis justifying the route deviation shown in the chart.
    
    Data Feeds:
    - Primary Route Real-Time Intel: {primary_context}
    - Alternative Route Delta: +{dist_diff:.1f} km detour
    - Alternative Route Real-Time Intel: {alt_context}
    
    Task:
    Write a definitive, 3-4 sentence technical analysis explaining exactly why accepting the {dist_diff:.1f} km detour is the mathematically and operationally superior choice. 
    Explicitly compare the specific weather conditions or news events provided in the intel to prove why the alternative route is safer.
    Do not use introductory filler.
    """
    
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, 
            max_completion_tokens=200,
            stream=False
        )
        content = completion.choices[0].message.content
        if content is not None:
            return content.strip()
        else:
            return f"Telemetry indicates severe operational friction on the primary path. The +{dist_diff:.1f} km deviation optimizes risk-adjusted transit times based on current weather and event data."
    except Exception:
        return f"Telemetry indicates severe operational friction on the primary path. The +{dist_diff:.1f} km deviation optimizes risk-adjusted transit times based on current weather and event data."


# --- 3. GEOSPATIAL LOGIC ROUTERS ---
async def geocode_place_arcgis(place_name: str, geolocator: ArcGIS) -> Optional[Dict[str, Any]]:
    for attempt in range(MAX_RETRIES):
        try:
            location = await geolocator.geocode(place_name, exactly_one=True, timeout=TIMEOUT) # type: ignore
            if location: return {"lat": float(location.latitude), "lon": float(location.longitude)}
        except Exception: await asyncio.sleep(1)
    return None


async def calculate_road_route(src: str, dest: str) -> Dict[str, Any]:
    async with ArcGIS(adapter_factory=AioHTTPAdapter) as geolocator:
        origin = await geocode_place_arcgis(src, geolocator)
        dest_loc = await geocode_place_arcgis(dest, geolocator)
        if not origin or not dest_loc: return {"error": "Geospatial coordinate mapping failed. Please verify the locations."}

        url = f"{OSRM_BASE_URL}/route/v1/driving/{origin['lon']},{origin['lat']};{dest_loc['lon']},{dest_loc['lat']}"
        params = {"overview": "full", "geometries": "geojson", "steps": "true", "alternatives": "true"}
        
        # INCREASED TIMEOUT: 45 seconds for long international routes
        timeout_config = aiohttp.ClientTimeout(total=45)
        
        data = None
        try:
            async with aiohttp.ClientSession(timeout=timeout_config) as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                    else:
                        return {"error": f"Routing server returned status {resp.status}. The distance may be too long for the public API."}
        except asyncio.TimeoutError:
            # FALLBACK: If alternative routes take too long, try again asking for ONLY the primary route
            params["alternatives"] = "false"
            try:
                async with aiohttp.ClientSession(timeout=timeout_config) as session:
                    async with session.get(url, params=params) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                        else:
                            return {"error": "Routing server timeout. The route between these two cities is too complex."}
            except Exception:
                return {"error": "The public routing server timed out. This usually happens for routes over 2,000 km. Please try closer cities."}
        except Exception as e:
            return {"error": f"Failed to connect to the routing server: {str(e)}"}

        if not data or not isinstance(data, dict) or data.get("code") != "Ok": 
            return {"error": "Primary route generation failed. The routing engine could not find a valid road path between these locations."}
        
        routes = data.get("routes", [])
        primary_route = routes[0]
        coords = primary_route.get("geometry", {}).get("coordinates", [])
        
        primary_distance = primary_route.get("distance", 0) / 1000
        primary_duration = primary_route.get("duration", 0) / 3600
        
        alt_coords = []
        alt_distance = 0
        alt_duration = 0
        alt_info = "No geographical alternative route is currently mapped for this corridor. Logistics must proceed with caution on the primary path."
        alt_milestones = []
        
        if len(routes) > 1:
            alt_route = routes[1]
            alt_coords = alt_route.get("geometry", {}).get("coordinates", [])
            alt_distance = alt_route.get("distance", 0) / 1000
            alt_duration = alt_route.get("duration", 0) / 3600 
            
            alt_waypoints_text = ""
            if len(alt_coords) > 10:
                # Sample points on the alternative route to check real-time risk
                idx_33 = len(alt_coords) // 3
                idx_66 = len(alt_coords) * 2 // 3
                sample_pts = [alt_coords[idx_33], alt_coords[idx_66]]
                alt_places = []
                
                for pt in sample_pts:
                    lon, lat = pt
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
                    
                    # Fetch real-time data for alternative route
                    weather = await get_openweather(lat, lon)
                    local_news = await fetch_real_news_for_location(place)
                    llm_eval = get_llm_risk_assessment(place, lat, lon, "Roadways", weather["temp"], weather["condition"], "N/A", local_news)
                    
                    alt_milestones.append({
                        "Location": place, "Lat": lat, "Lon": lon,
                        "Temp (°C)": weather["temp"], "Weather": weather["condition"],
                        "Local News": local_news, "Risk Level": llm_eval["Risk"], "AI Intelligence": llm_eval["Details"]
                    })

                if alt_places:
                    alt_waypoints_text = f" This strategic deviation physically routes through the following regions: {', '.join(alt_places)}."

            alt_info = f"An alternative geographical deviation is mapped. It measures {alt_distance:.1f} km with an estimated transit time of {alt_duration:.1f} hours.{alt_waypoints_text}"
        
        interval = max(1, len(coords) // 15) 
        
        final_waypoints = []
        final_waypoints.append({"name": f"Origin: {src}", "lon": coords[0][0], "lat": coords[0][1]})
        
        for i in range(interval, len(coords) - interval, interval):
            lon, lat = coords[i]
            final_waypoints.append({"name": "Route Waypoint", "lon": lon, "lat": lat})
            
        final_waypoints.append({"name": f"Destination: {dest}", "lon": coords[-1][0], "lat": coords[-1][1]})

        milestones = []
        for i, pt in enumerate(final_waypoints, 1):
            lon, lat = pt["lon"], pt["lat"]
            place = pt["name"]
            
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
            llm_eval = get_llm_risk_assessment(place, lat, lon, "Roadways", weather["temp"], weather["condition"], "N/A", local_news)
            
            milestones.append({
                "Step": i, "Location": place, "Lat": lat, "Lon": lon, 
                "Temp (°C)": weather["temp"], "Weather": weather["condition"], 
                "Wave Height": "N/A", "Local News": local_news, "Risk Level": llm_eval["Risk"], "AI Intelligence": llm_eval["Details"]
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


async def get_coordinates_nominatim(place_name: str) -> Optional[List[float]]:
    async with Nominatim(user_agent=USER_AGENT, adapter_factory=AioHTTPAdapter) as geolocator:
        try:
            location = await geolocator.geocode(place_name, timeout=10) # type: ignore
            if location: return [location.longitude, location.latitude]
        except Exception: pass
    return None


async def calculate_sea_route(src: str, dest: str) -> Dict[str, Any]:
    origin = await get_coordinates_nominatim(src)
    await asyncio.sleep(1.1)
    dest_loc = await get_coordinates_nominatim(dest)
    if not origin or not dest_loc: return {"error": "Geospatial coordinate mapping failed."}

    route = sr.searoute(origin, dest_loc) # type: ignore
    if not isinstance(route, dict): return {"error": "Maritime route generation failed."}
    coords = route.get('geometry', {}).get('coordinates', [])
    
    if not coords: return {"error": "No viable seaway coordinates returned."}
    
    dynamic_sea_points = []
    dynamic_sea_points.append({"name": f"Port of {src}", "lon": coords[0][0], "lat": coords[0][1]})
    
    seen_waters = set()
    interval = max(5, len(coords) // 30)
    
    for i in range(1, len(coords)-1, interval):
        lon, lat = coords[i]
        water_name = await get_marine_region_name(lat, lon)
        if water_name and water_name != "Open Ocean Waters" and water_name not in seen_waters:
            dynamic_sea_points.append({"name": water_name, "lon": lon, "lat": lat})
            seen_waters.add(water_name)
            
    dynamic_sea_points.append({"name": f"Port of {dest}", "lon": coords[-1][0], "lat": coords[-1][1]})
    
    milestones = []
    for i, pt in enumerate(dynamic_sea_points, 1):
        lon, lat = pt["lon"], pt["lat"]
        place = pt["name"]
        
        weather = await get_openweather(lat, lon)
        waves = await get_marine_weather(lat, lon)
        local_news = await fetch_real_news_for_location(place)
        llm_eval = get_llm_risk_assessment(place, lat, lon, "Seaways", weather["temp"], weather["condition"], waves, local_news)
        
        milestones.append({
            "Step": i, "Location": place, "Lat": lat, "Lon": lon,
            "Temp (°C)": weather["temp"], "Weather": weather["condition"],
            "Wave Height": waves, "Local News": local_news, "Risk Level": llm_eval["Risk"], "AI Intelligence": llm_eval["Details"]
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
        "alt_info": "Alternative maritime deviations must be planned by the harbormaster depending on specific port congestion."
    }


# --- DATA VISUALIZATIONS ---
def generate_matplotlib_charts(df: pd.DataFrame, result: Dict[str, Any], mode: str):
    coords = result.get("coords", [])
    sns.set_theme(style="whitegrid")
    
    # First Row of Charts
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 10px;'>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#d62728" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"></path>
                </svg>
                <h4 style='margin: 0; color: #1E293B;'>Temperature Trends</h4>
            </div>
        """, unsafe_allow_html=True)
        df_temp = df[df['Temp (°C)'] != 'N/A'].copy()
        if not df_temp.empty:
            x_vals = [str(x) for x in df_temp['Step']]
            y_vals = [float(x) for x in df_temp['Temp (°C)']]
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(x_vals, y_vals, marker='o', color='#d62728', linewidth=2)
            ax1.set_xlabel('Milestone Step')
            ax1.set_ylabel('Temperature (°C)')
            if len(x_vals) > 10: ax1.set_xticks(x_vals[::len(x_vals)//10])
            st.pyplot(fig1)
        else:
            st.info("No temperature data available.")

    with col2:
        st.markdown("""
            <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 10px;'>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ff7f0e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                    <line x1="12" y1="9" x2="12" y2="13"></line>
                    <line x1="12" y1="17" x2="12.01" y2="17"></line>
                </svg>
                <h4 style='margin: 0; color: #1E293B;'>Risk Level Distribution</h4>
            </div>
        """, unsafe_allow_html=True)
        colors_dict = {'Low': '#2ca02c', 'Medium': '#ff7f0e', 'High': '#d62728', 'Critical': '#8c564b'}
        risk_list = list(df['Risk Level'])
        risk_counts = {}
        for r in risk_list: risk_counts[str(r)] = risk_counts.get(str(r), 0) + 1
            
        bar_labels = list(risk_counts.keys())
        bar_heights = list(risk_counts.values())
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        bars = ax2.bar(bar_labels, bar_heights)
        
        for i, bar in enumerate(bars):
            color_to_use = colors_dict.get(bar_labels[i], 'gray')
            bar.set_color(color_to_use)
            
        ax2.set_ylabel('Number of Milestones')
        if len(bar_heights) > 0: ax2.set_yticks(range(0, max(bar_heights)+2))
        st.pyplot(fig2)

    # Second Row of Charts
    col3, col4 = st.columns(2)
    with col3:
        if mode == "Seaways":
            st.markdown("""
                <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 10px;'>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1f77b4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M2 12h4l2-9 4 18 2-9h4"></path>
                    </svg>
                    <h4 style='margin: 0; color: #1E293B;'>Marine Wave Heights</h4>
                </div>
            """, unsafe_allow_html=True)
            df_waves = df[df['Wave Height'] != 'N/A'].copy()
            if not df_waves.empty:
                y_waves = [float(str(x).replace(' m', '')) for x in df_waves['Wave Height']]
                x_waves = [str(x) for x in df_waves['Step']]
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                ax3.bar(x_waves, y_waves, color='#1f77b4')
                ax3.set_xlabel('Milestone Step')
                ax3.set_ylabel('Wave Height (meters)')
                if len(x_waves) > 10: ax3.set_xticks(x_waves[::len(x_waves)//10])
                st.pyplot(fig3)
            else:
                st.info("No wave data available.")
        else:
            st.markdown("""
                <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 10px;'>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                        <line x1="3" y1="9" x2="21" y2="9"></line>
                        <line x1="9" y1="21" x2="9" y2="9"></line>
                    </svg>
                    <h4 style='margin: 0; color: #1E293B;'>Traffic & Road Conditions</h4>
                </div>
            """, unsafe_allow_html=True)
            st.info("This metric is reserved for Seaway maritime tracking.")

    with col4:
        st.markdown("""
            <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 10px;'>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0066cc" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <circle cx="12" cy="12" r="3"></circle>
                </svg>
                <h4 style='margin: 0; color: #1E293B;'>Geographic Scatter Plot</h4>
            </div>
        """, unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        route_lons = [float(c[0]) for c in coords]
        route_lats = [float(c[1]) for c in coords]
        ax4.plot(route_lons, route_lats, color='#0066cc', alpha=0.5, label='Route')
        
        records = df.to_dict('records')
        m_lons = [float(r['Lon']) for r in records]
        m_lats = [float(r['Lat']) for r in records]
        ax4.scatter(m_lons, m_lats, color='red', s=50, zorder=5, label='Milestones')
        
        for r in records:
            step_str = str(r['Step'])
            lon_float = float(r['Lon'])
            lat_float = float(r['Lat'])
            if int(step_str) % 3 == 0 or int(step_str) == 1 or int(step_str) == len(records):
                ax4.annotate(text=step_str, xy=(lon_float, lat_float), xytext=(5, 5), textcoords='offset points', fontsize=8)
            
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
        ax4.legend()
        st.pyplot(fig4)

    # Third Row: Distance Viability & New Risk Comparison
    alt_dist = result.get('alt_distance', 0)
    if alt_dist > 0:
        st.markdown("---")
        
        col5, col6 = st.columns(2)
        
        with col5:
            st.markdown("""
                <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 15px;'>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#DAA520" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="12" y1="3" x2="12" y2="21"></line>
                        <path d="M3 13.5l4-8 4 8H3z"></path>
                        <path d="M13 13.5l4-8 4 8h-8z"></path>
                        <line x1="3" y1="13.5" x2="21" y2="13.5"></line>
                    </svg>
                    <h3 style='margin: 0; color: #1E293B; font-weight: 600;'>Route Viability Comparison</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # DYNAMIC REAL-TIME AI EXPLANATION
            chart_insight_text = generate_chart_insight(
                milestones=result.get("milestones", []), 
                alt_milestones=result.get("alt_milestones", []), 
                primary_dist=result.get('distance', 0), 
                alt_dist=alt_dist
            )
            st.info(f"**Logistics Insight:** {chart_insight_text}")
            
            fig5, ax5 = plt.subplots(figsize=(6, 4))
            
            categories = ['Total Distance (km)', 'Est. Transit Time (Hrs)']
            primary_vals = [result.get('distance', 0), result.get('primary_duration', 0)]
            alt_vals = [alt_dist, result.get('alt_duration', 0)]
            
            x = [0, 1]
            width = 0.35
            
            ax5.bar([i - width/2 for i in x], primary_vals, width, label='Primary Route', color='#d62728')
            ax5.bar([i + width/2 for i in x], alt_vals, width, label='Alternative Route', color='#DAA520')
            
            ax5.set_ylabel('Measurement')
            ax5.set_xticks(x)
            ax5.set_xticklabels(categories)
            ax5.legend()
            
            max_height = max(primary_vals + alt_vals)
            for i, val in enumerate(primary_vals):
                ax5.text(i - width/2, val + (max_height * 0.02), f"{val:.1f}", ha='center', fontweight='bold')
            for i, val in enumerate(alt_vals):
                ax5.text(i + width/2, val + (max_height * 0.02), f"{val:.1f}", ha='center', fontweight='bold')
    
            st.pyplot(fig5)

        with col6:
            # --- NEW 6TH CHART: Risk Profile Comparison ---
            st.markdown("""
                <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 15px;'>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#8c564b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                        <line x1="12" y1="9" x2="12" y2="13"></line>
                        <line x1="12" y1="17" x2="12.01" y2="17"></line>
                    </svg>
                    <h3 style='margin: 0; color: #1E293B; font-weight: 600;'>Real-Time Risk Profile Comparison</h3>
                </div>
            """, unsafe_allow_html=True)
            
            st.info("**Risk Matrix Insight:** This graph explicitly shows the count of severe vs moderate bottlenecks detected live on both routes.")

            alt_milestones = result.get("alt_milestones", [])
            
            primary_severe = len([m for m in result.get("milestones", []) if m['Risk Level'] in ['High', 'Critical']])
            alt_severe = len([m for m in alt_milestones if m.get('Risk Level', 'Low') in ['High', 'Critical']])
            
            primary_med = len([m for m in result.get("milestones", []) if m['Risk Level'] == 'Medium'])
            alt_med = len([m for m in alt_milestones if m.get('Risk Level', 'Low') == 'Medium'])

            fig6, ax6 = plt.subplots(figsize=(6, 4))
            
            risk_labels = ['Severe Risks (High/Critical)', 'Moderate Risks (Medium)']
            prim_counts = [primary_severe, primary_med]
            alt_counts = [alt_severe, alt_med]
            
            x_risk = [0, 1]
            
            ax6.bar([i - width/2 for i in x_risk], prim_counts, width, label='Primary Route', color='#d62728')
            ax6.bar([i + width/2 for i in x_risk], alt_counts, width, label='Alternative Route', color='#DAA520')
            
            ax6.set_ylabel('Number of Occurrences')
            ax6.set_xticks(x_risk)
            ax6.set_xticklabels(risk_labels)
            
            max_y = max(max(prim_counts), max(alt_counts))
            if max_y < 5: ax6.set_ylim(0, max_y + 1.5)
            
            ax6.legend()
            
            for i, val in enumerate(prim_counts):
                ax6.text(i - width/2, val + 0.1, str(val), ha='center', fontweight='bold')
            for i, val in enumerate(alt_counts):
                ax6.text(i + width/2, val + 0.1, str(val), ha='center', fontweight='bold')

            st.pyplot(fig6)


# --- UI ---
def main():
    
    # Custom CSS for the Header Logo integration
    st.markdown("""
        <style>
            .header-text { margin: 0; padding: 0; font-size: 2.2rem; font-weight: 700; color: #1E293B; }
            .sub-text { color: #475569; font-size: 1.1rem; margin-top: 0; margin-bottom: 25px; }
        </style>
    """, unsafe_allow_html=True)
    
    # Header Columns Setup
    col_logo, col_title = st.columns([1, 4])
    
    with col_logo:
        try:
            st.image("asb_logo_light.png", width='stretch')
        except Exception:
            st.warning("Logo file not found. Please ensure 'asb_logo_light.png' is in the directory.")
            
    with col_title:
        st.markdown("""
            <div style='display: flex; flex-direction: column; justify-content: center; height: 100%;'>
                <h1 class='header-text'>Global Supply Chain Nexus</h1>
                <p class='sub-text'>Advanced AI Routing, Real-Time Geo-Risk Analysis, & Maritime Tracking.</p>
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
            st.image("asb_logo_light.png", width='stretch')
            st.markdown("<hr style='margin-top: 5px; margin-bottom: 15px;'>", unsafe_allow_html=True)
        except Exception:
            pass
            
        st.markdown("""
            <h2 style='color: #1E293B; font-size: 1.3rem; margin-bottom: 15px;'>Route Configuration</h2>
        """, unsafe_allow_html=True)
        mode_input = st.selectbox("Select Transport Mode", ["Roadways", "Seaways"])
        
        default_origin = "Mumbai Port" if mode_input == "Seaways" else "Mumbai, India"
        default_dest = "Port of London" if mode_input == "Seaways" else "Dhaka, Bangladesh"
        
        origin_input = st.text_input("Origin (City/Port)", default_origin)
        destination_input = st.text_input("Destination (City/Port)", default_dest)
        
        analyze_btn = st.button("Generate Intelligence Report", type="primary")

    if analyze_btn:
        with st.spinner("Initializing Global Supply Chain Analysis Protocol (This may take a moment)..."):
            
            if mode_input == "Roadways":
                result = asyncio.run(calculate_road_route(origin_input, destination_input))
            else:
                result = asyncio.run(calculate_sea_route(origin_input, destination_input))

            if "error" in result:
                st.error(result["error"])
                return
            
            strategy = generate_rerouting_suggestion(
                milestones=result["milestones"], 
                mode=mode_input, 
                origin=origin_input, 
                dest=destination_input,
                primary_distance=result["distance"],
                alt_info=result["alt_info"],
                alt_milestones=result.get("alt_milestones", [])
            )
            
            st.session_state.mode = mode_input
            st.session_state.result = result
            st.session_state.reroute_strategy = strategy
            st.session_state.report_generated = True

    if st.session_state.report_generated:
        mode = st.session_state.mode
        result = st.session_state.result
        milestones = result["milestones"]
        
        def color_risk(val):
            color = 'green' if val == 'Low' else 'orange' if val == 'Medium' else 'red' if val in ['High', 'Critical'] else 'black'
            return f'color: {color}; font-weight: bold'

        st.markdown("""
            <div style='display: flex; align-items: center; gap: 10px; margin-top: 20px; margin-bottom: 10px;'>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="16" x2="12" y2="12"></line>
                    <line x1="12" y1="8" x2="12.01" y2="8"></line>
                </svg>
                <h3 style='margin: 0; color: #334155; font-size: 1.5rem; font-weight: 600;'>Executive Briefing & Rerouting Strategy</h3>
            </div>
        """, unsafe_allow_html=True)
        st.info(st.session_state.reroute_strategy)

        st.markdown("""
            <div style='display: flex; align-items: center; gap: 10px; margin-top: 30px; margin-bottom: 15px;'>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="16" y1="13" x2="8" y2="13"></line>
                    <line x1="16" y1="17" x2="8" y2="17"></line>
                    <polyline points="10 9 9 9 8 9"></polyline>
                </svg>
                <h3 style='margin: 0; color: #334155; font-size: 1.5rem; font-weight: 600;'>Geographic Intelligence Reports</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # --- TABBED DATAFRAME VIEW FOR SIDE-BY-SIDE COMPARISON ---
        tab1, tab2 = st.tabs(["🔴 Primary Route Data", "🟡 Alternative Route Data"])
        
        with tab1:
            df_primary = pd.DataFrame(milestones)
            df_primary['Temp (°C)'] = df_primary['Temp (°C)'].astype(str)
            df_primary['Wave Height'] = df_primary['Wave Height'].astype(str)
            styled_primary = df_primary.style.map(color_risk, subset=['Risk Level'])
            st.dataframe(styled_primary, use_container_width=True) 
            
            csv_primary = df_primary.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export Primary Route (CSV)", data=csv_primary, file_name=f"{mode}_primary_report.csv", mime="text/csv", key="btn_primary")

        with tab2:
            alt_milestones = result.get("alt_milestones", [])
            if alt_milestones:
                df_alt = pd.DataFrame(alt_milestones)
                df_alt['Temp (°C)'] = df_alt['Temp (°C)'].astype(str)
                if 'Wave Height' not in df_alt.columns:
                    df_alt['Wave Height'] = 'N/A'
                df_alt['Wave Height'] = df_alt['Wave Height'].astype(str)
                
                styled_alt = df_alt.style.map(color_risk, subset=['Risk Level'])
                st.dataframe(styled_alt, use_container_width=True)
                
                csv_alt = df_alt.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Export Alternative Route (CSV)", data=csv_alt, file_name=f"{mode}_alt_report.csv", mime="text/csv", key="btn_alt")
            else:
                st.info("No alternative route data available for this journey.")

        st.markdown("""
            <div style='display: flex; align-items: center; gap: 10px; margin-top: 30px; margin-bottom: 15px;'>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"></polygon>
                    <line x1="9" y1="3" x2="9" y2="18"></line>
                    <line x1="15" y1="6" x2="15" y2="21"></line>
                </svg>
                <h3 style='margin: 0; color: #334155; font-size: 1.5rem; font-weight: 600;'>Dynamic Risk Terrain Map</h3>
            </div>
        """, unsafe_allow_html=True)
        
        color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red', 'Critical': 'darkred'}
        
        m = folium.Map(location=[0, 0], zoom_start=2)
        
        # Plot Primary Route
        folium_coords = [[float(c[1]), float(c[0])] for c in result["coords"]]
        route_line = folium.PolyLine(
            locations=folium_coords,
            color="#0066cc",
            weight=5,
            opacity=0.8,
            tooltip=f"Primary {mode} Route"
        )
        route_line.add_to(m)
        
        plugins.PolyLineTextPath(
            route_line,
            "\u25BA", 
            repeat=True,
            offset=6,
            attributes={'fill': '#0066cc', 'font-weight': 'bold', 'font-size': '16'}
        ).add_to(m)

        # Plot Alternative Route (Now Dark Yellow)
        if result.get("alt_coords"):
            alt_folium_coords = [[float(c[1]), float(c[0])] for c in result["alt_coords"]]
            folium.PolyLine(
                locations=alt_folium_coords,
                color="#DAA520", # Goldenrod / Dark Yellow 
                weight=5,
                opacity=0.9,
                dash_array='8, 8',
                tooltip="Recommended Alternative Route"
            ).add_to(m)

        # Plot Primary Route Markers
        for row in milestones:
            risk_level = str(row['Risk Level'])
            marker_color = color_map.get(risk_level, 'gray')
            
            lat = float(row['Lat'])
            lon = float(row['Lon'])
            location_name = str(row['Location'])
            
            popup_html = f"""
            <div style="width:250px;">
                <h4>{location_name}</h4>
                <b>Risk:</b> <span style="color:{marker_color}; font-weight:bold;">{risk_level}</span><br>
                <b>Weather:</b> {row['Temp (°C)']}°C, {row['Weather']}<br>
                <b>Regional Intel:</b> {row['Local News']}<br>
                <b>Security Note:</b> {row['AI Intelligence']}
            </div>
            """
            
            marker = folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color=marker_color, icon="info-sign")
            )
            
            tooltip_html = f'<span style="font-weight: bold; font-size: 12px; white-space: nowrap; background-color: rgba(255,255,255,0.7);">{location_name}</span>'
            marker.add_child(folium.Tooltip(tooltip_html, permanent=False, direction="top", opacity=0.9))
            marker.add_to(m)

        # Plot Alternative Route Markers
        for row in result.get("alt_milestones", []):
            risk_level = str(row['Risk Level'])
            marker_color = color_map.get(risk_level, 'gray')
            
            lat = float(row['Lat'])
            lon = float(row['Lon'])
            location_name = str(row['Location']) + " (ALT ROUTE)"
            
            popup_html = f"""
            <div style="width:250px;">
                <h4 style="color:#DAA520;">{location_name}</h4>
                <b>Risk:</b> <span style="color:{marker_color}; font-weight:bold;">{risk_level}</span><br>
                <b>Weather:</b> {row['Temp (°C)']}°C, {row['Weather']}<br>
                <b>Regional Intel:</b> {row['Local News']}<br>
                <b>Security Note:</b> {row['AI Intelligence']}
            </div>
            """
            
            marker = folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color='orange', icon='star', icon_color='white')
            )
            
            tooltip_html = f'<span style="font-weight: bold; font-size: 12px; white-space: nowrap; background-color: rgba(255,255,255,0.7);">{location_name}</span>'
            marker.add_child(folium.Tooltip(tooltip_html, permanent=False, direction="top", opacity=0.9))
            marker.add_to(m)

        # Calculate bounds to automatically zoom and center the map
        all_lats = []
        all_lons = []
        
        for c in result.get("coords", []):
            all_lats.append(float(c[1]))
            all_lons.append(float(c[0]))
            
        for c in result.get("alt_coords", []):
            all_lats.append(float(c[1]))
            all_lons.append(float(c[0]))
            
        if all_lats and all_lons:
            m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])

        st_folium(m, width=1200, height=600, use_container_width=True, returned_objects=[])

        st.markdown("---")
        st.markdown("""
            <div style='display: flex; align-items: center; gap: 10px; margin-top: 10px; margin-bottom: 20px;'>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="20" x2="18" y2="10"></line>
                    <line x1="12" y1="20" x2="12" y2="4"></line>
                    <line x1="6" y1="20" x2="6" y2="14"></line>
                </svg>
                <h3 style='margin: 0; color: #334155; font-size: 1.5rem; font-weight: 600;'>Advanced Operational Analytics</h3>
            </div>
        """, unsafe_allow_html=True)
        # Fix for Dataframe passing to charts
        df_for_charts = pd.DataFrame(milestones)
        df_for_charts['Temp (°C)'] = df_for_charts['Temp (°C)'].astype(str)
        df_for_charts['Wave Height'] = df_for_charts['Wave Height'].astype(str)
        generate_matplotlib_charts(df_for_charts, result, mode)

if __name__ == "__main__":
    main()