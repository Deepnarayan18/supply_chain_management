import requests

url = "https://marine-api.open-meteo.com/v1/marine"
params = {
    "latitude": 18.96,       # Mumbai coordinates
    "longitude": 72.82,
    "hourly": "wave_height,wave_direction,wave_period",
    "timezone": "Asia/Kolkata"
}

response = requests.get(url, params=params)
data = response.json()
print(data["hourly"])