"""
Supply Chain Risk Monitor - Main Streamlit Application
AI-Driven Macro-Event & Sentiment Intelligence for Supply Chain Resilience
"""
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import APP_TITLE, APP_DESCRIPTION, MAJOR_PORTS
from weather_service import WeatherService
from risk_analyzer import RiskAnalyzer

# --- ROUTING MATH HELPER FUNCTIONS ---
def generate_waypoints(origin, dest, mode):
    points = [{"lat": origin["lat"], "lon": origin["lon"], "name": origin["name"]}]
    o_lon, o_lat = origin["lon"], origin["lat"]
    d_lon, d_lat = dest["lon"], dest["lat"]
    
    if mode == "Waterways":
        if (o_lon > 60 and d_lon < 20) or (o_lon < 20 and d_lon > 60): 
            points.append({"lat": 1.29, "lon": 103.85, "name": "Strait of Malacca"})
            points.append({"lat": 12.50, "lon": 43.30, "name": "Gulf of Aden"})
            points.append({"lat": 29.97, "lon": 32.52, "name": "Suez Canal"})
            points.append({"lat": 36.14, "lon": -5.35, "name": "Strait of Gibraltar"})
        elif (o_lon < -40 and d_lon > 100) or (o_lon > 100 and d_lon < -40):
            points.append({"lat": 9.14, "lon": -79.73, "name": "Panama Canal"})
            
    elif mode == "Roadways":
        points.append({"lat": (o_lat + d_lat)/2, "lon": (o_lon + d_lon)/2, "name": "Overland Transit Node"})
        
    elif mode == "Airways":
        points.append({"lat": (o_lat + d_lat)/2, "lon": (o_lon + d_lon)/2, "name": "Airspace Transit Node"})

    points.append({"lat": dest["lat"], "lon": dest["lon"], "name": dest["name"]})
    return points

def is_in_segment(lat, lon, lat1, lon1, lat2, lon2, padding):
    min_lat, max_lat = min(lat1, lat2) - padding, max(lat1, lat2) + padding
    min_lon, max_lon = min(lon1, lon2) - padding, max(lon1, lon2) + padding
    return (min_lat <= lat <= max_lat) and (min_lon <= lon <= max_lon)

def is_event_on_route(event_lat, event_lon, waypoints, mode):
    if mode == "Airways": padding = 5.0       
    elif mode == "Waterways": padding = 3.0   
    else: padding = 1.5                       
    
    for i in range(len(waypoints) - 1):
        if is_in_segment(event_lat, event_lon, waypoints[i]["lat"], waypoints[i]["lon"], waypoints[i+1]["lat"], waypoints[i+1]["lon"], padding):
            return True
    return False

# Page configuration
st.set_page_config(page_title=APP_TITLE, page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_base_events():
    return RiskAnalyzer().gdelt_service.get_all_events()

@st.cache_data(ttl=900)
def fetch_specific_ports_data(port_configs):
    weather_service = WeatherService()
    risk_analyzer = RiskAnalyzer()
    events = load_base_events()
    
    analyzed_ports = []
    for port in port_configs:
        weather_data = weather_service.get_weather_data(port["lat"], port["lon"])
        
        if weather_data is None:
            weather_data = {}
            
        risk_analysis = weather_service.analyze_weather_risk(weather_data)
        
        comp_risk = risk_analyzer.calculate_comprehensive_risk(port["name"], risk_analysis, events)
        comp_risk["lat"], comp_risk["lon"], comp_risk["country"] = port["lat"], port["lon"], port["country"]
        analyzed_ports.append(comp_risk)
        
    return analyzed_ports

def create_risk_map(risks_data, waypoints=None):
    m = folium.Map(location=[30, 20], zoom_start=2, tiles="CartoDB positron")
    
    if waypoints:
        route_coords = [[p["lat"], p["lon"]] for p in waypoints]
        folium.PolyLine(locations=route_coords, color='#00E5FF', weight=4, opacity=0.8, dash_array='10').add_to(m)
        for wp in waypoints[1:-1]:
            folium.CircleMarker(
                location=[wp["lat"], wp["lon"]], radius=4, color='white', fill=True, 
                fill_color='blue', tooltip=f"Transit Node: {wp['name']}"
            ).add_to(m)

    for risk in risks_data:
        color = "red" if risk["risk_level"] == "critical" else "orange" if risk["risk_level"] == "high" else "yellow" if risk["risk_level"] == "medium" else "green"
        icon = "exclamation-triangle" if color == "red" else "exclamation-circle" if color == "orange" else "info-circle" if color == "yellow" else "check-circle"
        
        popup_html = f"<h4>{risk['port_name']}</h4><p>Risk: {risk['risk_level'].upper()}</p><p>Score: {risk['total_risk_score']}</p>"
        folium.Marker(
            location=[risk["lat"], risk["lon"]], popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{risk['port_name']}: {risk['risk_level'].upper()}", icon=folium.Icon(color=color, icon=icon, prefix='fa')
        ).add_to(m)
    return m


def main():
    st.markdown(f'<div class="main-header">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{APP_DESCRIPTION}</div>', unsafe_allow_html=True)
    
    # --- Initialize Session State for the "Click to Fetch" logic ---
    if "data_fetched" not in st.session_state:
        st.session_state.data_fetched = False
        
    with st.sidebar:
        st.header("⚙️ Controls")
        
        if st.button("🔄 Reset / Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.session_state.data_fetched = False
            st.rerun()
            
        st.markdown("---")
        st.subheader("🗺️ Supply Chain Routing")
        
        port_names = list(MAJOR_PORTS.keys())
        origin_name = st.selectbox("Origin Port", port_names, index=0)
        dest_name = st.selectbox("Destination Port", port_names, index=1 if len(port_names) > 1 else 0)
        transport_mode = st.radio("Transport Mode", ["Waterways", "Airways", "Roadways"])
        
        st.markdown("---")
        st.subheader("📊 Risk Filter")
        risk_filter = st.multiselect("Risk Levels", ["low", "medium", "high", "critical"], default=["medium", "high", "critical"])
        
        st.markdown("---")
        # THE CRITICAL BUTTON - Nothing happens until this is clicked
        if st.button("🚀 Analyze Route Risk", type="primary", use_container_width=True):
            st.session_state.data_fetched = True

    # 🛑 STOP HERE if the user hasn't clicked the button yet.
    if not st.session_state.data_fetched:
        st.info("👋 Welcome! Please configure your supply chain route in the sidebar and click **'Analyze Route Risk'** to begin.")
        
        # Draw a placeholder blank map to keep the UI looking nice
        m = folium.Map(location=[30, 20], zoom_start=2, tiles="CartoDB positron")
        st_folium(m, width=None, height=600)
        return

    # =========================================================
    # IF WE REACH HERE, THE USER CLICKED THE BUTTON!
    # =========================================================
    
    origin_config = MAJOR_PORTS[origin_name]
    origin_config["name"] = origin_name
    
    dest_config = MAJOR_PORTS[dest_name]
    dest_config["name"] = dest_name

    waypoints = generate_waypoints(origin_config, dest_config, transport_mode)
    
    with st.spinner(f"Fetching data for route: {origin_name} ➡️ {dest_name}..."):
        active_ports_data = fetch_specific_ports_data([origin_config, dest_config])
        raw_events = load_base_events()

    filtered_events = []
    for event in raw_events:
        goldstein_raw = event.get('goldstein_scale')
        if goldstein_raw is None: goldstein_raw = 0.0
        severity_score = abs(float(goldstein_raw)) * 10
        event_severity = "critical" if severity_score > 60 else "high" if severity_score > 40 else "medium" if severity_score > 20 else "low"
        
        if event_severity in risk_filter:
            e_lat, e_lon = event.get('action_geo_lat'), event.get('action_geo_long')
            if e_lat is not None and e_lon is not None:
                if is_event_on_route(float(e_lat), float(e_lon), waypoints, transport_mode):
                    filtered_events.append(event)
                
    col1, col2, col3 = st.columns(3)
    col1.metric("⚓ Ports Monitored", len(active_ports_data))
    col2.metric("📰 Route Events Found", len(filtered_events))
    col3.metric("🛣️ Active Transport Mode", transport_mode)
    st.markdown("---")
    
    tab1, tab4 = st.tabs(["🗺️ Route & Risk Map", "📰 Events on Route"])
    
    with tab1:
        st.subheader("Active Supply Chain Tracker")
        st.success(f"Tracking {transport_mode} route from **{origin_name}** to **{dest_name}**.")
        risk_map = create_risk_map(active_ports_data, waypoints)
        st_folium(risk_map, width=None, height=600)
    
    with tab4:
        st.subheader("Disruptions Along Selected Path")
        if not filtered_events:
            st.info("✅ Route is clear! No active events match the current filter criteria for this corridor.")
        else:
            events_df = pd.DataFrame(filtered_events).sort_values('goldstein_scale', ascending=True)
            
            for idx, event in events_df.head(10).iterrows(): 
                goldstein_raw = event.get('goldstein_scale', 0.0)
                if goldstein_raw is None: goldstein_raw = 0.0
                severity_score = abs(float(goldstein_raw)) * 10
                severity_color = "🔴" if severity_score > 60 else "🟠" if severity_score > 40 else "🟡"
                
                with st.container():
                    st.markdown(f"### {severity_color} {event.get('event_location', 'Unknown Location')}")
                    colA, colB, colC = st.columns([2, 1, 1])
                    
                    with colA:
                        st.write(f"**{event.get('event_description', 'N/A')}**")
                        lat, lon = event.get('action_geo_lat'), event.get('action_geo_long')
                        if lat is None: lat = 0.0
                        if lon is None: lon = 0.0
                        st.write(f"📍 Coordinates: ({float(lat):.4f}, {float(lon):.4f})")
                        
                    with colB:
                        st.write(f"Goldstein: **{float(goldstein_raw):.1f}**")
                    with colC:
                        st.write(f"Type: {event.get('event_type', 'Unknown')}")
                    
                    source_url = str(event.get('source_url', ''))
                    st.caption(f"Source: {source_url[:80]}...")
                    
                    if source_url.startswith('http'):
                        if st.button("🤖 Summarize Event (Gemini 2.5 Flash-Lite)", key=f"btn_sum_{idx}"):
                            with st.spinner("Analyzing event context..."):
                                try:
                                    gdelt_srv = RiskAnalyzer().gdelt_service
                                    content = gdelt_srv.fetch_url_content(source_url)
                                    if content:
                                        summary = gdelt_srv.summarize_with_gemini(content, source_url)
                                        st.success(f"**AI Summary:** {summary}")
                                    else:
                                        st.error("Website firewall blocked content extraction.")
                                except Exception:
                                    st.error("Error summarizing content.")
                    st.markdown("---")

if __name__ == "__main__":
    main()