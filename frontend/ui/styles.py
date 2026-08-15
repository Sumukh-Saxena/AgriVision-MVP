"""Calm, minimal styling for the AgriVision assistant.

Colors come from the Streamlit theme in .streamlit/config.toml
(off-white background, dark-green accent, neutral grays). This CSS only
adds light touches that are hard to express with plain Streamlit widgets.
"""

import streamlit as st


def apply() -> None:
    st.markdown(
        """
        <style>
        /* Branding */
        .av-brand {
            font-size: 1.3rem;
            font-weight: 700;
            color: #2F5233;
        }
        .av-brand-sub {
            font-size: 0.82rem;
            color: #6b6b6b;
        }
        .av-section-title {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #6b6b6b;
            margin: 1.1rem 0 0.35rem 0;
        }

        /* Suggested-prompt chips */
        .stButton button[kind="secondary"] {
            border-radius: 999px;
            border-color: #c9c6bd;
            color: #2F5233;
            background: transparent;
        }
        .stButton button[kind="secondary"]:hover {
            border-color: #2F5233;
            background: rgba(47, 82, 51, 0.06);
        }

        /* Sidebar subtle background */
        [data-testid="stSidebar"] {
            background-color: #f2f1ec;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
