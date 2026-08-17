"""OpenWeather API client used by the AgriVision workflow.

Fetches current weather for a farmer-provided city/state so the disease
explanation can be grounded in local conditions. All failures are surfaced as
exceptions; callers decide how to degrade gracefully.
"""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

GEOCODING_URL = "https://api.openweathermap.org/geo/1.0/direct"
CURRENT_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
UNITS = "metric"  # Celsius
REQUEST_TIMEOUT = 15


class WeatherService:
    """Resolves a location to coordinates and fetches its current weather."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def geocode(self, location: str) -> tuple[float, float, str | None, str | None]:
        """Return (lat, lon, city, country) for a city/state name."""
        resp = requests.get(
            GEOCODING_URL,
            params={"q": location, "limit": 1, "appid": self.api_key},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json()
        if not results:
            raise ValueError(f"Could not find any location matching '{location}'.")
        first = results[0]
        return first["lat"], first["lon"], first.get("name"), first.get("country")

    def fetch(self, location: str) -> dict:
        """Return a small weather dict for the given location.

        Raises if the API key is missing or the request fails, so the caller
        can degrade gracefully without disrupting the crop pipeline.
        """
        if not self.available:
            raise ValueError(
                "OPENWEATHER_API_KEY is not set. Set it in the .env file to enable weather."
            )

        lat, lon, city, country = self.geocode(location)
        resp = requests.get(
            CURRENT_WEATHER_URL,
            params={
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": UNITS,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        main = data.get("main", {})
        weather = (data.get("weather") or [{}])[0]
        wind = data.get("wind", {})

        return {
            "location": location,
            "city": city or location,
            "country": country,
            "temperature_c": main.get("temp"),
            "feels_like_c": main.get("feels_like"),
            "humidity": main.get("humidity"),
            "description": weather.get("description"),
            "wind_speed_ms": wind.get("speed"),
        }