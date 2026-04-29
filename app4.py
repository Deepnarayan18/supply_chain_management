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
st.set_page_config(page_title="Supply Chain Intelligence Nexus", layout="wide")

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

def generate_rerouting_suggestion(milestones: List[Dict[str, Any]], mode: str, origin: str, dest: str, primary_distance: float, alt_info: str) -> str:
    context = ""
    for m in milestones:
        if m['Risk Level'] in ['High', 'Critical']:
            context += f"- High Risk at {m['Location']}: {m['AI Intelligence']}\n"
            
    if not context:
        context = "No severe high-risk bottlenecks identified along the primary route."
        
    prompt = f"""
    You are the Lead Supply Chain Intelligence Director. Write a professional intelligence briefing for the logistics team.
    
    Route: {origin} to {dest} ({mode})
    Primary Route Length: {primary_distance:.1f} km
    
    Identified Bottlenecks on Primary Route:
    {context}
    
    Alternative Route Logistics Data:
    {alt_info}
    
    Write a 2-paragraph operational briefing:
    1. Assess the viability and safety of the primary route based on the bottlenecks.
    2. Provide a clear, authoritative recommendation on whether logistics managers should authorize the primary route or switch to the alternative route.
    
    CRITICAL RULES:
    - NEVER use words like "API", "OSRM", "Algorithm", "Calculated", "Groq", or "Llama".
    - Frame the alternative route as "Intelligence gathered by our geospatial systems" or "Our recommended deviation."
    - MUST explicitly name the alternative cities/regions provided in the Alternative Route Logistics Data so the driver knows where to go.
    """
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, 
            max_completion_tokens=512,
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

def generate_chart_insight(milestones: List[Dict[str, Any]], primary_dist: float, alt_dist: float) -> str:
    """Generates a dynamic 1-sentence explanation for the chart comparing the two routes."""
    high_risks = [m['Location'] for m in milestones if m['Risk Level'] in ['High', 'Critical']]
    
    if not high_risks:
        return "Since there are no critical bottlenecks on the primary path, deviations are unnecessary and would only result in increased fuel consumption and delay."
        
    places = ", ".join(high_risks[:2])
    dist_diff = alt_dist - primary_dist
    
    prompt = f"""
    Write a ONE SENTENCE professional logistics insight.
    The primary route passes through dangerous high-risk zones ({places}).
    The alternative route avoids these zones but adds {dist_diff:.1f} km to the journey.
    Explain why accepting this extra distance is the safer, strategic choice based on real-time risks.
    Keep it strictly under 30 words.
    """
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4, 
            max_completion_tokens=100,
            stream=False
        )
        content = completion.choices[0].message.content
        if content is not None:
            return content.strip()
        else:
            return f"Although the deviation adds {dist_diff:.1f} km, it strategically bypasses the high-risk disruptions detected in {places}."
    except Exception:
        return f"Although the deviation adds {dist_diff:.1f} km, it strategically bypasses the high-risk disruptions detected in {places}."


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
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                data = await resp.json()

        if not isinstance(data, dict) or data.get("code") != "Ok": return {"error": "Primary route generation failed."}
        
        routes = data.get("routes", [])
        primary_route = routes[0]
        coords = primary_route.get("geometry", {}).get("coordinates", [])
        
        primary_distance = primary_route.get("distance", 0) / 1000
        primary_duration = primary_route.get("duration", 0) / 3600
        
        alt_coords = []
        alt_distance = 0
        alt_duration = 0
        alt_info = "No geographical alternative route is currently mapped for this corridor. Logistics must proceed with caution on the primary path."
        
        if len(routes) > 1:
            alt_route = routes[1]
            alt_coords = alt_route.get("geometry", {}).get("coordinates", [])
            alt_distance = alt_route.get("distance", 0) / 1000
            alt_duration = alt_route.get("duration", 0) / 3600 
            
            alt_waypoints_text = ""
            if len(alt_coords) > 10:
                idx_25 = len(alt_coords) // 4
                idx_50 = len(alt_coords) // 2
                idx_75 = len(alt_coords) * 3 // 4
                sample_pts = [alt_coords[idx_25], alt_coords[idx_50], alt_coords[idx_75]]
                alt_places = []
                
                for pt in sample_pts:
                    lon, lat = pt
                    try:
                        rev = await geolocator.reverse(f"{lat}, {lon}", timeout=TIMEOUT) # type: ignore
                        if rev and getattr(rev, "address", None):
                            parts = rev.address.split(',')
                            if len(parts) >= 3:
                                place = f"{parts[-3].strip()}, {parts[-2].strip()}"
                            else:
                                place = parts[0].strip()
                            if place not in alt_places:
                                alt_places.append(place)
                    except Exception:
                        pass
                
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
        "alt_info": "Alternative maritime deviations must be planned by the harbormaster depending on specific port congestion."
    }


# --- DATA VISUALIZATIONS ---
def generate_matplotlib_charts(df: pd.DataFrame, result: Dict[str, Any], mode: str):
    coords = result.get("coords", [])
    sns.set_theme(style="whitegrid")
    
    # First Row of Charts
    col1, col2 = st.columns(2)
    with col1:
        st.write("### 🌡️ Temperature Trends")
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
        st.write("### ⚠️ Risk Level Distribution")
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
            st.write("### 🌊 Marine Wave Heights")
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
            st.write("### 🛣️ Traffic & Road Conditions")
            st.info("This metric is reserved for Seaway maritime tracking.")

    with col4:
        st.write("### 📍 Route Geographic Scatter Plot")
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

    # Third Row: NEW 5th Chart - Route Deviation Comparison
    alt_dist = result.get('alt_distance', 0)
    if alt_dist > 0:
        st.markdown("---")
        st.write("### ⚖️ Route Viability & Deviation Comparison")
        
        # DYNAMIC REAL-TIME AI EXPLANATION
        chart_insight_text = generate_chart_insight(result.get("milestones", []), result.get('distance', 0), alt_dist)
        st.info(f"**Logistics Insight:** {chart_insight_text}")
        
        fig5, ax5 = plt.subplots(figsize=(10, 4))
        
        categories = ['Total Distance (km)', 'Est. Transit Time (Hours)']
        primary_vals = [result.get('distance', 0), result.get('primary_duration', 0)]
        alt_vals = [alt_dist, result.get('alt_duration', 0)]
        
        x = [0, 1]
        width = 0.35
        
        ax5.bar([i - width/2 for i in x], primary_vals, width, label='Primary Route', color='#d62728')
        ax5.bar([i + width/2 for i in x], alt_vals, width, label='Alternative Route', color='#2ca02c')
        
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


# --- UI ---
def main():
    st.title("🌍 Global Supply Chain Nexus")
    st.markdown("##### Advanced AI Routing, Real-Time Geo-Risk Analysis, & Maritime Tracking.")

    if 'report_generated' not in st.session_state:
        st.session_state.report_generated = False
    if 'mode' not in st.session_state:
        st.session_state.mode = ""
    if 'result' not in st.session_state:
        st.session_state.result = {}
    if 'reroute_strategy' not in st.session_state:
        st.session_state.reroute_strategy = ""

    with st.sidebar:
        st.header("Route Configuration")
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
                alt_info=result["alt_info"]
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

        st.subheader("🤖 Executive Briefing & Rerouting Strategy")
        st.info(st.session_state.reroute_strategy)

        st.subheader(f"📋 Geographic Intelligence Report ({len(milestones)} Milestones Found)")
        df = pd.DataFrame(milestones)
        styled_df = df.style.map(color_risk, subset=['Risk Level'])
        st.dataframe(styled_df, width="stretch")

        st.subheader("🗺️ Dynamic Risk Terrain Map")
        
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

        # Plot Alternative Route
        if result.get("alt_coords"):
            alt_folium_coords = [[float(c[1]), float(c[0])] for c in result["alt_coords"]]
            folium.PolyLine(
                locations=alt_folium_coords,
                color="darkred", 
                weight=5,
                opacity=0.9,
                dash_array='8, 8',
                tooltip="Recommended Alternative Route"
            ).add_to(m)

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
            
            tooltip_html = f'<span style="font-weight: bold; font-size: 11px; white-space: nowrap;">{location_name}</span>'
            marker.add_child(folium.Tooltip(tooltip_html, permanent=True, direction="auto", opacity=0.85))
            marker.add_to(m)

        raw_bounds = route_line.get_bounds()
        clean_bounds = []
        for point in raw_bounds:
            if point and len(point) >= 2 and point[0] is not None and point[1] is not None:
                clean_bounds.append([float(point[0]), float(point[1])])
                
        if clean_bounds:
            m.fit_bounds(clean_bounds)

        st_folium(m, width=1200, height=600, use_container_width=True, returned_objects=[])

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Logistics Data (CSV)", data=csv, file_name=f"{mode}_supply_chain_report.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("📊 Advanced Operational Analytics")
        generate_matplotlib_charts(df, result, mode)

if __name__ == "__main__":
    main()