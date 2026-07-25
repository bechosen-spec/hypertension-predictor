from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable

import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def safe_text(value: object) -> str:
    return html.escape(str(value))


def asset_exists(filename: str) -> bool:
    """Return False for missing assets so callers can render a safe fallback."""
    return bool(filename) and (ASSETS_DIR / filename).is_file()


def section_header(kicker: str, title: str, copy: str = "") -> None:
    st.markdown(
        f'<div class="section-kicker">{safe_text(kicker)}</div>'
        f'<div class="section-title">{safe_text(title)}</div>'
        f'<div class="section-copy">{safe_text(copy)}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, detail: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{safe_text(label)}</div>'
        f'<div class="metric-value">{safe_text(value)}</div>'
        f'<div class="small-muted">{safe_text(detail)}</div></div>',
        unsafe_allow_html=True,
    )


def notice(text: str, tone: str = "") -> None:
    tone_class = tone if tone in {"warning", "danger"} else ""
    st.markdown(
        f'<div class="notice {tone_class}">{safe_text(text)}</div>',
        unsafe_allow_html=True,
    )


def image_or_fallback(filename: str, caption: str, alt: str) -> None:
    path = ASSETS_DIR / filename
    if asset_exists(filename):
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.markdown(
            f'<div class="surface" role="img" aria-label="{safe_text(alt)}">'
            "🫀<br><strong>Heart-health illustration</strong></div>",
            unsafe_allow_html=True,
        )


def progress_steps(current: int, total: int = 4) -> None:
    dots = "".join(
        f'<span class="step-dot {"active" if index <= current else ""}"></span>'
        for index in range(1, total + 1)
    )
    st.markdown(
        f'<div class="small-muted">Step {current} of {total}</div><div class="step-line">{dots}</div>',
        unsafe_allow_html=True,
    )


def risk_category(probability: float | None) -> tuple[str, str]:
    value = float(probability or 0)
    if value < 0.35:
        return "Lower estimated risk", "risk-low"
    if value < 0.65:
        return "Moderate estimated risk", "risk-moderate"
    return "Higher estimated risk", "risk-high"


def probability_bar(probability: float | None) -> None:
    pct = max(0.0, min(100.0, float(probability or 0) * 100))
    st.markdown(
        f'<div class="prob-track" aria-label="Estimated hypertension probability {pct:.1f} percent">'
        f'<div class="prob-fill" style="width:{pct:.1f}%"></div></div>',
        unsafe_allow_html=True,
    )


def recommendation_cards(items: Iterable[str]) -> None:
    for index, item in enumerate(items, 1):
        st.markdown(
            f'<div class="recommendation"><strong>{index}</strong>{safe_text(item)}</div>',
            unsafe_allow_html=True,
        )


def footer() -> None:
    st.markdown(
        '<div class="footer">PulseWise Risk • Educational decision-support demonstration • '
        "Not a medical device or diagnostic service</div>",
        unsafe_allow_html=True,
    )
