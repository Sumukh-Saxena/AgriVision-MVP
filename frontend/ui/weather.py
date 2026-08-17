"""Renders weather info fetched from OpenWeather for the crop's location."""

import streamlit as st


def render_weather(weather: dict) -> None:
    """Render a compact weather card for a location.

    Expected keys: city, country, location, temperature_c, feels_like_c,
    humidity, description, wind_speed_ms.
    """
    if not weather:
        return

    city = weather.get("city") or weather.get("location") or "Location"
    country = weather.get("country")
    title = f"{city}{', ' + country if country else ''}"

    with st.container(border=True):
        st.markdown(f"**\u2601\uFE0F Weather \u00b7 {title}**")

        temp = weather.get("temperature_c")
        desc = weather.get("description")
        header = ""
        if temp is not None:
            header += f"**{temp:.0f}\u00b0C**"
        if desc:
            header += f"  \u2014  {desc.capitalize()}"
        st.markdown(header)

        if temp is not None or weather.get("feels_like_c") is not None:
            parts = []
            if weather.get("feels_like_c") is not None:
                parts.append(f"Feels like {weather['feels_like_c']:.0f}\u00b0C")
            if weather.get("humidity") is not None:
                parts.append(f"Humidity {weather['humidity']}%")
            if weather.get("wind_speed_ms") is not None:
                parts.append(f"Wind {weather['wind_speed_ms']} m/s")
            if parts:
                st.caption(" \u00b7 ".join(parts))