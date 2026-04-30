import asyncio
from typing import Optional, List, Dict, Any
from geopy.adapters import AioHTTPAdapter
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import searoute as sr

USER_AGENT = "supply_chain_monitor_v1/deepghosh@youremail.com"

async def get_coordinates(place_name: str) -> Optional[List[float]]:
    """
    Forward geocodes a place name into [longitude, latitude] 
    specifically formatted for the searoute library.
    """
    async with Nominatim(user_agent=USER_AGENT, adapter_factory=AioHTTPAdapter) as geolocator:
        try:
            # type: ignore suppresses the false VS Code Pylance warning about await
            location = await geolocator.geocode(place_name, timeout=10) # type: ignore
            if location:
                # searoute expects [longitude, latitude]
                return [location.longitude, location.latitude]
        except Exception as e:
            print(f"Error geocoding {place_name}: {e}")
    return None


# Notice geolocator is typed as Any to stop the strict typing error
async def get_dynamic_place_name(lat: float, lon: float, geolocator: Any) -> str:
    """
    Dynamically asks OpenStreetMap for the name of the ocean, sea, or coast
    at this exact latitude and longitude.
    """
    try:
        # Nominatim strictly limits to 1 request per second
        await asyncio.sleep(1.1) 
        
        # reverse() expects "latitude, longitude" (opposite of searoute!)
        # type: ignore tells the linter that this IS awaitable at runtime
        location = await geolocator.reverse(f"{lat}, {lon}", timeout=10, language='en') # type: ignore
        
        if location and location.raw.get('address'):
            addr = location.raw['address']
            # Try to grab water bodies first
            water_body = addr.get('ocean') or addr.get('sea') or addr.get('strait') or addr.get('bay')
            if water_body:
                return water_body
            
            # If close to land, grab the country or coastal city
            land_body = addr.get('country') or addr.get('state') or addr.get('city')
            if land_body:
                return f"Coast of {land_body}"
                
            # Fallback to the first part of the address string
            return location.address.split(',')[0]
            
        return "Open Ocean Waters"
        
    except Exception as e:
        return "Unmapped Marine Area"


async def calculate_sea_route(source_city: str, destination_city: str) -> Dict[str, Any]:
    # 1. Forward Geocode the Start and End
    origin = await get_coordinates(source_city)
    await asyncio.sleep(1.1)
    destination = await get_coordinates(destination_city)

    if not origin or not destination:
        return {"error": "Could not find coordinates"}

    # 2. Calculate the route using searoute
    route: Dict[str, Any] = sr.searoute(origin, destination)  # type: ignore
    waypoints = route.get('geometry', {}).get('coordinates', [])
    
    # 3. DYNAMICALLY REVERSE GEOCODE MILESTONES FOR AI
    # To save time, we only look up ~10 milestones evenly spaced across the journey
    milestones = []
    
    # Calculate step size to get exactly 10 milestones (or fewer if route is short)
    step_size = max(1, len(waypoints) // 10)
    sampled_waypoints = waypoints[::step_size]
    
    # Always ensure the exact final destination is included
    if waypoints[-1] not in sampled_waypoints:
        sampled_waypoints.append(waypoints[-1])

    print(f"\nDynamically identifying {len(sampled_waypoints)} milestone regions for AI... (This will take ~{len(sampled_waypoints)} seconds)")

    async with Nominatim(user_agent=USER_AGENT, adapter_factory=AioHTTPAdapter) as geolocator:
        for idx, point in enumerate(sampled_waypoints, start=1):
            lon, lat = point[0], point[1]
            
            # Fetch the dynamic place name from the API
            place_name = await get_dynamic_place_name(lat=lat, lon=lon, geolocator=geolocator)
            
            milestone_data = {
                "step": idx,
                "coordinates": [lon, lat],
                "dynamic_location": place_name
            }
            milestones.append(milestone_data)
            print(f"  Milestone {idx:>2}: {place_name}")

    return {
        "status": "success",
        "distance": route.get('properties', {}).get('length', 0),
        "total_waypoints_count": len(waypoints),
        "ai_route_milestones": milestones
    }


async def main():
    result = await calculate_sea_route("Mumbai Port", "Port of London")

    if result.get("status") == "success":
        print(f"\n{'='*60}")
        print("DYNAMIC AI ROUTE CONTEXT:")
        print(f"{'='*60}")
        
        for m in result["ai_route_milestones"]:
            lon, lat = m['coordinates']
            print(f"Step {m['step']:>2} | [ {lon:>9.4f}, {lat:>8.4f} ] -> {m['dynamic_location']}")
            
        print(f"{'='*60}")
    else:
        print("Error calculating route:", result.get("error"))

if __name__ == "__main__":
    asyncio.run(main())