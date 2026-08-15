"""Renders the Mistral explanation as structured, farmer-friendly sections.

The raw explanation is a single text block; we split it into the sections the
AgriExpert prompt is asked to produce (disease, cause, symptoms, spread,
management, prevention). If no headings are recognised, the full text is shown
inside one expander instead of being lost.
"""

import re

import streamlit as st

SECTION_RULES = [
    (re.compile(r"what is it", re.IGNORECASE), "What is it?"),
    (re.compile(r"what the disease", re.IGNORECASE), "What is it?"),
    (re.compile(r"\bcause\b", re.IGNORECASE), "Possible Cause"),
    (re.compile(r"symptom", re.IGNORECASE), "Symptoms"),
    (re.compile(r"spread", re.IGNORECASE), "How it spreads"),
    (re.compile(r"treatment|management", re.IGNORECASE), "Management"),
    (re.compile(r"prevent", re.IGNORECASE), "Prevention"),
    (re.compile(r"disease", re.IGNORECASE), "Disease"),
]


def _is_heading(line: str) -> str | None:
    """Return a section title if the line looks like a short heading."""
    stripped = line.strip(" *#")
    if not stripped or len(stripped) > 80:
        return None
    words = stripped.split()
    numbered = bool(re.match(r"^\d+[.)]\s", stripped))
    looks_like_heading = (
        stripped.endswith(":")
        or len(words) <= 8
        or (numbered and len(words) <= 12)
    )
    if not looks_like_heading:
        return None
    for pattern, title in SECTION_RULES:
        if pattern.search(stripped):
            return title
    return None


def _parse_sections(text: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_body: list[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        title = _is_heading(line)
        if title and title != current_title:
            if current_title:
                sections.append((current_title, current_body))
            current_title = title
            current_body = [line]
        else:
            if current_title is None:
                current_title = "AI Explanation"
            current_body.append(line)

    if current_title:
        sections.append((current_title, current_body))

    # Ignore "sections" that are really just one stray heading with no body.
    return [(t, b) for t, b in sections if b and len(" ".join(b)) > 2]


def render_explanation(text: str) -> None:
    """Render an explanation string as expandable sections."""
    if not text:
        return

    sections = _parse_sections(text)
    if not sections:
        with st.expander("AI Explanation", expanded=True):
            st.markdown(text)
        return

    st.markdown("#### AI Explanation")
    for title, body in sections:
        with st.expander(title, expanded=(title in ("What is it?", "Possible Cause"))):
            st.markdown("\n\n".join(body))
