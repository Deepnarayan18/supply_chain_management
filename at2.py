import asyncio
import pycountry
from typing import Any, Dict, List, Optional

import aiohttp
from geopy.adapters import AioHTTPAdapter
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from geopy.geocoders import ArcGIS

# --- CONFIGURATION VARIABLES ---
TIMEOUT = 15
MAX_RETRIES = 3
OSRM_BASE_URL = "https://router.project-osrm.org"

# Your Real Currents API Key
CURRENTS_API_KEY = "knKi5jXU7QuT_fI9K2gdXwKs6hCZCQLv5YUbtkHVZp67leYx"
# -------------------------------


def format_duration(seconds: float) -> str:
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def extract_key_waypoints(steps: List[Dict[str, Any]], coordinates: List[List[float]], max_points: int = 5) -> List[int]:
    """
    Mathematical Heuristic Algorithm for Route Parsing:
    Selects the most 'newsworthy' points along a route by looking for long 
    stretches of continuous driving (highways) and evenly spacing them.
    """
    if not coordinates:
        return []
    
    if len(coordinates) <= max_points:
        return list(range(len(coordinates)))
        
    key_indices = []
    
    # 1. Anchor Heuristic: Always include the origin (Start point)
    key_indices.append(0)
    
    # 2. Long Stretch Heuristic: Find major highway segments
    long_steps = []
    for step in steps:
        if step.get("distance_km", 0) > 5.0:
            long_steps.append(step)
            
    # 3. Even Distribution Heuristic: Spread points across the route
    step_spacing = max(1, len(steps) // (max_points - 2))
    
    for i in range(1, max_points - 1):
        target_step_idx = i * step_spacing
        if target_step_idx < len(steps):
            # Roughly map the OSRM step back to a GPS coordinate index
            coord_idx = int((target_step_idx / len(steps)) * len(coordinates))
            if coord_idx not in key_indices:
                key_indices.append(coord_idx)
                
    # 4. Anchor Heuristic: Always include the destination (End point)
    if (len(coordinates) - 1) not in key_indices:
        key_indices.append(len(coordinates) - 1)
        
    return sorted(key_indices)[:max_points]


def extract_best_place_name(location: Any) -> str:
    if not location:
        return "Unknown place"
    if getattr(location, "address", None):
        return str(location.address)
    return "Unknown place"


def convert_to_iso2(country_string: str) -> str:
    """
    Dynamically converts any country string (like a 3-letter ISO code or full name)
    into the strict 2-letter ISO code required by the Currents API.
    """
    country_string = country_string.strip().upper()
    
    if not country_string:
        return ""
        
    if len(country_string) == 2:
        return country_string
        
    try:
        if len(country_string) == 3:
            country_obj = pycountry.countries.get(alpha_3=country_string)
            if country_obj:
                return country_obj.alpha_2
                
        country_obj = pycountry.countries.search_fuzzy(country_string)
        if country_obj and len(country_obj) > 0:
            return country_obj[0].alpha_2
            
    except Exception:
        pass
        
    return country_string[:2]


async def geocode_place(place_name: str, geolocator: ArcGIS) -> Optional[Dict[str, Any]]:
    _geolocator: Any = geolocator 
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            location = await _geolocator.geocode(place_name, exactly_one=True, timeout=TIMEOUT)
            if not location: return None
            return {
                "query": place_name,
                "full_address": location.address,
                "lat": float(location.latitude),
                "lon": float(location.longitude),
                "lonlat": [float(location.longitude), float(location.latitude)],
            }
        except (GeocoderTimedOut, GeocoderUnavailable):
            if attempt == MAX_RETRIES: return None
            await asyncio.sleep(2)
        except Exception as e:
            print(f"Exception during geocode: {e}")
            return None
    return None


async def reverse_place_name(lat: float, lon: float, geolocator: ArcGIS) -> str:
    _geolocator: Any = geolocator 
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            location = await _geolocator.reverse(f"{lat}, {lon}", exactly_one=True, timeout=TIMEOUT)
            return extract_best_place_name(location)
        except (GeocoderTimedOut, GeocoderUnavailable):
            if attempt == MAX_RETRIES: return "Unknown place"
            await asyncio.sleep(2)
        except Exception as e:
            print(f"Exception during reverse geocode: {e}")
            return "Unknown place"
    return "Unknown place"


async def get_road_route(origin: Dict[str, Any], destination: Dict[str, Any]) -> Dict[str, Any]:
    lon1, lat1 = origin["lon"], origin["lat"]
    lon2, lat2 = destination["lon"], destination["lat"]
    url = f"{OSRM_BASE_URL}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    params = {"overview": "full", "geometries": "geojson", "steps": "true", "alternatives": "false"}

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        async with session.get(url, params=params) as response:
            if response.status != 200:
                return {"status": "error", "error": f"OSRM API failed with status {response.status}"}
            data = await response.json()

    if data.get("code") != "Ok":
        return {"status": "error", "error": data.get("message", "OSRM could not build a route")}

    routes = data.get("routes", [])
    if not routes:
        return {"status": "error", "error": "No road route found"}

    route = routes[0]
    geometry = route.get("geometry", {})
    coordinates = geometry.get("coordinates", [])

    steps_output: List[Dict[str, Any]] = []
    step_no = 1
    for leg in route.get("legs", []):
        for step in leg.get("steps", []):
            steps_output.append({
                "step_no": step_no,
                "instruction_type": step.get("maneuver", {}).get("type", "continue"),
                "road_name": step.get("name") or "Unnamed road",
                "distance_km": round(step.get("distance", 0.0) / 1000, 2),
                "duration_min": round(step.get("duration", 0.0) / 60, 1),
            })
            step_no += 1

    return {
        "status": "success",
        "distance_km": round(route.get("distance", 0.0) / 1000, 2),
        "duration_seconds": round(route.get("duration", 0.0), 2),
        "total_waypoints_count": len(coordinates),
        "geometry": geometry,
        "coordinates": coordinates,
        "steps": steps_output,
    }


async def calculate_road_route_with_places(source_place: str, destination_place: str) -> Dict[str, Any]:
    async with ArcGIS(adapter_factory=AioHTTPAdapter) as geolocator:
        print(f"Looking up coordinates for {source_place} and {destination_place} using ArcGIS...")
        origin = await geocode_place(source_place, geolocator)
        await asyncio.sleep(0.5)
        destination = await geocode_place(destination_place, geolocator)

        if not origin or not destination:
            return {"status": "error", "error": "Could not geocode locations"}

        route_result = await get_road_route(origin, destination)
        if route_result.get("status") != "success": return route_result

        coordinates = route_result["coordinates"]
        steps = route_result["steps"]
        
        # Using the mathematical heuristic algorithm to find the 5 best milestones
        idxs = extract_key_waypoints(steps, coordinates, max_points=5) 

        milestones: List[Dict[str, Any]] = []
        for display_no, idx in enumerate(idxs, start=1):
            lon, lat = coordinates[idx]
            place_name = await reverse_place_name(lat, lon, geolocator)
            milestones.append({
                "milestone_no": display_no,
                "lon": lon,
                "lat": lat,
                "place_name": place_name,
            })
            await asyncio.sleep(0.5)

        return {
            "status": "success",
            "mode": "roadways",
            "from_query": source_place,
            "to_query": destination_place,
            "distance_km": route_result["distance_km"],
            "duration_text": format_duration(route_result["duration_seconds"]),
            "milestones": milestones,
        }


# --- REAL NEWS INTEGRATION (CURRENTS API) ---
async def fetch_real_news_for_location(city_name: str, country_code: str = "") -> Optional[Dict[str, str]]:
    """Fetches real news for a specific geographic location using Currents API."""
    url = "https://api.currentsapi.services/v1/search"
    headers = {"Authorization": CURRENTS_API_KEY}
    
    # We dynamically pass the city name and a transport keyword to filter the news
    params: Dict[str, Any] = {
        "keywords": f"{city_name} traffic",
        "language": "en",
        "limit": 1 
    }
    
    # Dynamically convert the country string to a 2-letter ISO code using pycountry
    iso2_country = convert_to_iso2(country_code)
    if iso2_country:
        params["country"] = iso2_country
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get("news", [])
                    
                    if articles and len(articles) > 0:
                        top_article = articles[0]
                        return {
                            "title": top_article.get("title", "No Title"),
                            "source": top_article.get("author", "Unknown Source")
                        }
                return None
    except Exception as e:
        print(f"Currents API Error: {e}")
        return None


async def check_real_news_for_route(milestones: List[Dict[str, Any]]):
    print(f"\n{'=' * 70}")
    print("REAL-TIME GLOBAL NEWS SCANNER (CURRENTS API)")
    print(f"{'=' * 70}")
    
    async def process_milestone(milestone):
        full_address = milestone['place_name']
        parts = [p.strip() for p in full_address.split(',')]
        
        if len(parts) >= 3:
            city_name = parts[-3].split(' ')[0] # Attempt to get clean city name
            country = parts[-1]                 # Attempt to get country code (e.g., PRT, ITA)
        elif len(parts) == 2:
            city_name = parts[0]
            country = parts[1]
        else:
            city_name = full_address
            country = ""
            
        news = await fetch_real_news_for_location(city_name, country)
        return city_name, country, news

    tasks = [process_milestone(m) for m in milestones]
    results = await asyncio.gather(*tasks)

    for city_name, country, news in results:
        display_name = f"{city_name}, {country}".strip(', ')
        if news:
            source = news.get("source") or "Unknown"
            print(f"📰 {display_name.upper()} | [{source}] {news['title']}")
        else:
            print(f"✅ {display_name.upper()} | No recent disruptive news detected.")

    print(f"{'=' * 70}")


async def main():
    SOURCE_PLACE = "Mumbai, India"
    DESTINATION_PLACE = "Pune, India"

    result = await calculate_road_route_with_places(SOURCE_PLACE, DESTINATION_PLACE)

    if result.get("status") != "success":
        print(f"\nError: {result.get('error')}")
        return

    print(f"\n{'=' * 70}")
    print("FINAL RESULT")
    print(f"{'=' * 70}")
    print(f"From            : {result['from_query']}")
    print(f"To              : {result['to_query']}")
    print(f"Distance        : {result['distance_km']} km")
    print(f"Estimated time  : {result['duration_text']}")

    print(f"\n{'=' * 70}")
    print("AI ROUTE MILESTONES")
    print(f"{'=' * 70}")
    for m in result["milestones"]:
        print(f"Milestone {m['milestone_no']:>2} | [{m['lon']:.4f}, {m['lat']:.4f}] -> {m['place_name']}")

    # Pass the milestones to the Currents News API function
    await check_real_news_for_route(result["milestones"])


if __name__ == "__main__":
    asyncio.run(main())