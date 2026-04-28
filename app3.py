import streamlit as st
import asyncio
import aiohttp
import pycountry
import pandas as pd
from typing import Any, Dict, List, Optional
import searoute as sr
import matplotlib.pyplot as plt
import seaborn as sns

# Plotly Imports for the Interactive Map
import plotly.express as px
import plotly.graph_objects as go

# Groq Import
from groq import Groq

from geopy.adapters import AioHTTPAdapter
from geopy.geocoders import ArcGIS, Nominatim

# --- CONFIGURATION & API KEYS ---
st.set_page_config(page_title="AI Supply Chain Intelligence", layout="wide")

TIMEOUT = 15
MAX_RETRIES = 3
OSRM_BASE_URL = "https://router.project-osrm.org"
USER_AGENT = "supply_chain_monitor_v12/deepghosh@youremail.com"

# API KEYS
OPENWEATHER_API_KEY = "789f1ee0f9eec1e1b1dfbb9ab1076273"
GROQ_API_KEY = "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"

# NEW: Your GNews API Key goes here!
GNEWS_API_KEY = "50806093fa93b13062815b9c7bb5fd47"

# Initialize Groq Client
groq_client = Groq(api_key=GROQ_API_KEY)


# --- HELPER FUNCTIONS ---
def extract_best_place_name(location: Any) -> str:
    if not location: return "Unknown place"
    if getattr(location, "address", None): return str(location.address)
    return "Unknown place"

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

def extract_key_waypoints(steps_or_coords: List[Any], coordinates: List[List[float]], max_points: int = 5, is_road: bool = True) -> List[int]:
    if not coordinates: return []
    if len(coordinates) <= max_points: return list(range(len(coordinates)))
        
    key_indices = [0] 
    
    if is_road:
        steps = steps_or_coords
        step_spacing = max(1, len(steps) // (max_points - 2))
        for i in range(1, max_points - 1):
            target_step_idx = i * step_spacing
            if target_step_idx < len(steps):
                coord_idx = int((target_step_idx / len(steps)) * len(coordinates))
                if coord_idx not in key_indices: key_indices.append(coord_idx)
    else:
        step_spacing = max(1, len(coordinates) // (max_points - 1))
        for i in range(1, max_points - 1):
            target_idx = i * step_spacing
            if target_idx not in key_indices: key_indices.append(target_idx)
                
    if (len(coordinates) - 1) not in key_indices: 
        key_indices.append(len(coordinates) - 1)
        
    return sorted(key_indices)[:max_points]


# --- 1. WEATHER & NEWS API CALLS ---
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
    """
    NEW: Fetches localized news using the GNews API (gnews.io).
    """
    if not GNEWS_API_KEY or GNEWS_API_KEY == "YOUR_GNEWS_API_KEY_HERE":
        return "GNews API Key missing. Cannot fetch news."

    # Extract the core city/region name for a cleaner search
    parts = [p.strip() for p in full_address.split(',')]
    if len(parts) >= 3:
        city_name = parts[-3].split(' ')[0] 
    elif len(parts) == 2:
        city_name = parts[0]
    else:
        city_name = full_address.replace("Coast of ", "").replace("Open Ocean Waters", "Maritime")

    # GNews API search endpoint
    url = "https://gnews.io/api/v4/search"
    params = {
        "q": f'"{city_name}" AND (traffic OR logistics OR transport OR strike OR delay)',
        "lang": "en",
        "sortby": "publishedAt",
        "max": 1,
        "apikey": GNEWS_API_KEY
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, dict) and data.get("totalArticles", 0) > 0:
                        articles = data.get("articles", [])
                        if isinstance(articles, list) and len(articles) > 0:
                            first_article = articles[0]
                            title = first_article.get("title", "No Title")
                            source = first_article.get("source", {}).get("name", "Unknown Source")
                            return f"📰 [{source}] {title}"
    except Exception: pass
    return f"✅ No disruptive news detected for {city_name} recently."


# --- 2. GROQ LLM INTEGRATION ---
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
        details = "AI Analysis generated without specific details."
        
        for line in text:
            if line.startswith("RISK_LEVEL:"):
                risk = line.split("RISK_LEVEL:")[1].strip()
            elif line.startswith("DETAILS:"):
                details = line.split("DETAILS:")[1].strip()
                
        return {"Risk": risk, "Details": details}
    except Exception as e:
        print(f"Groq API Error: {e}")
        return {"Risk": "Medium", "Details": f"AI Analysis failed via Groq. Region: {location}"}

def generate_rerouting_suggestion(milestones: List[Dict[str, Any]], mode: str, origin: str, dest: str) -> str:
    context = ""
    for m in milestones:
        context += f"- {m['Location']}: Risk={m['Risk Level']}, Details={m['AI Intelligence']}\n"
        
    prompt = f"""
    You are an expert global supply chain architect. 
    Analyze this {mode} route from {origin} to {dest}.
    
    Here are the milestones and live risks:
    {context}
    
    1. Summarize the overall safety of this route.
    2. Identify the biggest chokepoint or danger based on the risk levels above.
    3. Suggest a specific, alternative rerouting strategy (e.g., specific alternative ports, different highways, or multi-modal options) to lower the risk. 
    Keep it highly professional, geographical, and under 4 sentences. Do not use bullet points.
    """
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_completion_tokens=512,
            top_p=1,
            stream=False,
            stop=None
        )
        content = completion.choices[0].message.content or "Could not generate rerouting suggestion."
        return content.strip()
    except Exception:
        return "Could not generate rerouting suggestion at this time."


# --- 3. LOGIC ROUTERS ---
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
        if not origin or not dest_loc: return {"error": "Geocoding failed."}

        url = f"{OSRM_BASE_URL}/route/v1/driving/{origin['lon']},{origin['lat']};{dest_loc['lon']},{dest_loc['lat']}"
        params = {"overview": "full", "geometries": "geojson", "steps": "true"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                data = await resp.json()

        if not isinstance(data, dict) or data.get("code") != "Ok": return {"error": "OSRM routing failed."}
        route = data.get("routes", [])[0]
        coords = route.get("geometry", {}).get("coordinates", [])
        
        steps = []
        for leg in route.get("legs", []):
            if isinstance(leg, dict):
                for s in leg.get("steps", []):
                    if isinstance(s, dict): steps.append(s)

        idxs = extract_key_waypoints(steps, coords, max_points=5, is_road=True)
        milestones = []
        
        for i, idx in enumerate(idxs, 1):
            lon, lat = coords[idx]
            try:
                rev = await geolocator.reverse(f"{lat}, {lon}", timeout=TIMEOUT) # type: ignore
                place = extract_best_place_name(rev)
            except Exception: place = "Unknown Road"
            
            weather = await get_openweather(lat, lon)
            local_news = await fetch_real_news_for_location(place)
            
            llm_eval = get_llm_risk_assessment(place, lat, lon, "Roadways", weather["temp"], weather["condition"], "N/A", local_news)
            
            milestones.append({
                "Step": i, "Location": place, "Lat": lat, "Lon": lon, 
                "Temp (°C)": weather["temp"], "Weather": weather["condition"], 
                "Wave Height": "N/A", "Local News": local_news, "Risk Level": llm_eval["Risk"], "AI Intelligence": llm_eval["Details"]
            })
            
        return {"coords": coords, "distance": route.get("distance",0)/1000, "milestones": milestones}

async def get_coordinates_nominatim(place_name: str) -> Optional[List[float]]:
    async with Nominatim(user_agent=USER_AGENT, adapter_factory=AioHTTPAdapter) as geolocator:
        try:
            location = await geolocator.geocode(place_name, timeout=10) # type: ignore
            if location: return [location.longitude, location.latitude]
        except Exception: pass
    return None

async def get_dynamic_sea_name(lat: float, lon: float, geolocator: Any) -> str:
    try:
        await asyncio.sleep(1.1)
        location = await geolocator.reverse(f"{lat}, {lon}", timeout=10, language='en') # type: ignore
        if location and location.raw.get('address'):
            addr = location.raw['address']
            water = addr.get('ocean') or addr.get('sea') or addr.get('strait') or addr.get('bay')
            if water: return water
            land = addr.get('country') or addr.get('state') or addr.get('city')
            if land: return f"Coast of {land}"
            return location.address.split(',')[0]
        return "Open Ocean Waters"
    except Exception: return "Unmapped Marine Area"

async def calculate_sea_route(src: str, dest: str) -> Dict[str, Any]:
    origin = await get_coordinates_nominatim(src)
    await asyncio.sleep(1.1)
    dest_loc = await get_coordinates_nominatim(dest)
    if not origin or not dest_loc: return {"error": "Geocoding failed."}

    route = sr.searoute(origin, dest_loc) # type: ignore
    if not isinstance(route, dict): return {"error": "Searoute failed to calculate path."}
    coords = route.get('geometry', {}).get('coordinates', [])
    if not coords: return {"error": "No sea coordinates returned."}
    
    idxs = extract_key_waypoints([], coords, max_points=5, is_road=False)
    milestones = []
    
    async with Nominatim(user_agent=USER_AGENT, adapter_factory=AioHTTPAdapter) as geolocator:
        for i, idx in enumerate(idxs, 1):
            lon, lat = coords[idx]
            place = await get_dynamic_sea_name(lat, lon, geolocator)
            weather = await get_openweather(lat, lon)
            waves = await get_marine_weather(lat, lon)
            local_news = await fetch_real_news_for_location(place)
            
            llm_eval = get_llm_risk_assessment(place, lat, lon, "Seaways", weather["temp"], weather["condition"], waves, local_news)
            
            milestones.append({
                "Step": i, "Location": place, "Lat": lat, "Lon": lon,
                "Temp (°C)": weather["temp"], "Weather": weather["condition"],
                "Wave Height": waves, "Local News": local_news, "Risk Level": llm_eval["Risk"], "AI Intelligence": llm_eval["Details"]
            })

    return {"coords": coords, "distance": route.get('properties', {}).get('length', 0), "milestones": milestones}


# --- MATPLOTLIB VISUALIZATIONS ---
def generate_matplotlib_charts(df: pd.DataFrame, coords: List[List[float]], mode: str):
    """Generates 4 beautiful Matplotlib graphs safely typed to satisfy VS Code (Pylance)."""
    
    sns.set_theme(style="whitegrid")
    
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    # 1. Temperature Line Chart
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
            st.pyplot(fig1)
        else:
            st.info("No temperature data available.")

    # 2. Risk Distribution Bar Chart
    with col2:
        st.write("### ⚠️ Risk Level Distribution")
        colors_dict = {'Low': '#2ca02c', 'Medium': '#ff7f0e', 'High': '#d62728', 'Critical': '#8c564b'}
        
        risk_list = list(df['Risk Level'])
        risk_counts = {}
        for r in risk_list:
            risk_counts[str(r)] = risk_counts.get(str(r), 0) + 1
            
        bar_labels = list(risk_counts.keys())
        bar_heights = list(risk_counts.values())
        
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        bars = ax2.bar(bar_labels, bar_heights)
        
        for i, bar in enumerate(bars):
            color_to_use = colors_dict.get(bar_labels[i], 'gray')
            bar.set_color(color_to_use)
            
        ax2.set_ylabel('Number of Milestones')
        if len(bar_heights) > 0:
            ax2.set_yticks(range(0, max(bar_heights)+2))
        st.pyplot(fig2)

    # 3. Wave Height Chart
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
                st.pyplot(fig3)
            else:
                st.info("No wave data available.")
        else:
            st.write("### 🛣️ Traffic / Road Conditions")
            st.info("This metric is reserved for Seaway tracking.")

    # 4. Route Scatter Map
    with col4:
        st.write("### 📍 Matplotlib Coordinate View")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        
        route_lons = [float(c[0]) for c in coords]
        route_lats = [float(c[1]) for c in coords]
        ax4.plot(route_lons, route_lats, color='#1f77b4', alpha=0.5, label='Route')
        
        records = df.to_dict('records')
        m_lons = [float(r['Lon']) for r in records]
        m_lats = [float(r['Lat']) for r in records]
        
        ax4.scatter(m_lons, m_lats, color='red', s=100, zorder=5, label='Milestones')
        
        for r in records:
            step_str = str(r['Step'])
            lon_float = float(r['Lon'])
            lat_float = float(r['Lat'])
            
            ax4.annotate(
                text=step_str, 
                xy=(lon_float, lat_float), 
                xytext=(5, 5), 
                textcoords='offset points'
            )
            
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
        ax4.legend()
        st.pyplot(fig4)


# --- UI ---
def main():
    st.title("🌍 AI-Powered Global Supply Chain Monitor")
    st.markdown("Track **Roadways** or **Seaways**. Powered by Groq (Llama-3) & GNews API for Risk Intelligence.")

    with st.sidebar:
        st.header("Route Configuration")
        mode = st.selectbox("Select Transport Mode", ["Roadways", "Seaways"])
        
        default_origin = "Mumbai Port" if mode == "Seaways" else "Mumbai, India"
        default_dest = "Port of London" if mode == "Seaways" else "Pune, India"
        
        origin = st.text_input("Origin (City/Port)", default_origin)
        destination = st.text_input("Destination (City/Port)", default_dest)
        analyze_btn = st.button("Generate Supply Chain Report", type="primary")

    if analyze_btn:
        with st.spinner(f"Analyzing {mode} via Groq Llama-3 AI..."):
            if mode == "Roadways":
                result = asyncio.run(calculate_road_route(origin, destination))
            else:
                result = asyncio.run(calculate_sea_route(origin, destination))

            if "error" in result:
                st.error(result["error"])
                return
                
            milestones = result["milestones"]
            
            def color_risk(val):
                color = 'green' if val == 'Low' else 'orange' if val == 'Medium' else 'red' if val in ['High', 'Critical'] else 'black'
                return f'color: {color}; font-weight: bold'

            # --- REROUTING SUGGESTION ---
            st.subheader("🤖 AI Rerouting & Risk Strategy")
            with st.spinner("Llama-3 is formulating a rerouting strategy..."):
                reroute_strategy = generate_rerouting_suggestion(milestones, mode, origin, destination)
                st.info(reroute_strategy)

            # --- DATAFRAME REPORT ---
            st.subheader("📋 Point-by-Point Intelligence Report")
            df = pd.DataFrame(milestones)
            
            styled_df = df.style.map(color_risk, subset=['Risk Level'])
            st.dataframe(styled_df, width="stretch")

            # --- PLOTLY INTERACTIVE MAP ---
            st.subheader("🗺️ Geographic Risk Map")
            route_df = pd.DataFrame(result["coords"], columns=["Lon", "Lat"])
            
            fig_map = px.line_map(route_df, lat="Lat", lon="Lon", zoom=2, height=550)
            
            color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red', 'Critical': 'darkred', 'Unknown': 'gray'}
            df['Marker Color'] = df['Risk Level'].apply(lambda x: color_map.get(str(x), 'gray'))
            
            hover_text = df["Location"] + "<br><b>News:</b> " + df["Local News"] + "<br><b>AI Summary:</b> " + df["AI Intelligence"]
            
            fig_map.add_trace(go.Scattermap(
                lat=df["Lat"], lon=df["Lon"], mode='markers+text',
                marker=go.scattermap.Marker(size=14, color=df['Marker Color']),
                text=df["Step"], textposition="bottom right",
                hovertext=hover_text
            ))
            
            fig_map.update_layout(map_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_map, width="stretch")

            # --- DOWNLOAD BUTTON ---
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Report (CSV)", data=csv, file_name=f"{mode}_supply_chain_report.csv", mime="text/csv")

            # --- 4 MATPLOTLIB CHARTS ---
            st.markdown("---")
            st.subheader("📊 Detailed Graphical Analysis")
            generate_matplotlib_charts(df, result["coords"], mode)

if __name__ == "__main__":
    main()
