"""Renders the regional crop-suitability report (survival/nativity/yield)."""

import streamlit as st


def render_regional(regional: str, location: str | None = None) -> None:
    """Render a compact regional suitability card for a crop + location."""
    if not regional:
        return

    title = "Regional Crop Suitability"
    if location:
        title += f" \u00b7 {location}"

    with st.container(border=True):
        st.markdown(f"**\U0001F33E {title}**")
        st.markdown(regional)