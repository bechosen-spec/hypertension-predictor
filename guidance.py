"""Gemini-backed guidance generation with a deterministic safe fallback."""
from __future__ import annotations

import json
import logging
from typing import Callable, Optional

from utils.app_logic import parse_recommendations
from vertex_config import (
    VertexSettings,
    get_default_gen_config,
    get_vertex_client,
    load_vertex_settings,
)

LOGGER = logging.getLogger(__name__)

GENERIC_GUIDANCE = [
    "Check your blood pressure with a validated monitor and keep a simple record to discuss with a healthcare professional.",
    "Choose lower-sodium foods when practical, and check packaged-food labels for sodium content.",
    "Aim for regular, comfortable physical activity that suits your health and ability.",
    "Support consistent sleep and use a simple stress-management habit, such as slow breathing or a short walk.",
    "Arrange a professional review if readings are repeatedly high, and never start, stop, or change medication without clinical advice.",
]


def extract_response_text(response) -> str:
    text = getattr(response, "text", None)
    if text:
        return text.strip()
    parts: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        for part in getattr(getattr(candidate, "content", None), "parts", []) or []:
            if getattr(part, "text", None):
                parts.append(part.text)
    return "\n".join(parts).strip()


def fallback_result() -> tuple[str, list[str], bool]:
    text = "\n".join(
        f"{index}. {item}" for index, item in enumerate(GENERIC_GUIDANCE, 1)
    )
    return text, list(GENERIC_GUIDANCE), False


def build_guidance_prompt(
    patient: dict,
    label: str,
    probability: Optional[float],
    factors: dict,
) -> str:
    return f"""
The Random Forest model—not you—produced this educational hypertension-risk estimate.
Patient inputs: {json.dumps(patient)}
Stored model classification: {label}
Estimated probability of hypertension class: {probability}
Relevant active inputs ranked using global importance: {json.dumps(factors)}

Write exactly these sections:
Prediction Summary:
Key Factors Influencing Prediction:
Personalized Recommendations:

Under Personalized Recommendations, give EXACTLY five short numbered recommendations.
Use cautious plain language. Do not diagnose, prescribe, or suggest medication changes.
Encourage appropriate blood-pressure measurement and professional review.
End the fifth recommendation with a reminder that this is not medical advice.
""".strip()


def generate_guidance(
    patient: dict,
    label: str,
    probability: Optional[float],
    factors: dict,
    *,
    settings_loader: Callable[[], VertexSettings] = load_vertex_settings,
    client_factory: Callable[[Optional[VertexSettings]], object] = get_vertex_client,
) -> tuple[str, list[str], bool]:
    """Generate five recommendations, falling back only on a genuine failure."""
    settings: Optional[VertexSettings] = None
    client = None
    try:
        settings = settings_loader()
        client = client_factory(settings)
        response = client.models.generate_content(
            model=settings.model_name,
            contents=build_guidance_prompt(patient, label, probability, factors),
            config=get_default_gen_config(0.3, 0.95, 2048),
        )
        raw = extract_response_text(response)
        items = parse_recommendations(raw)
        if len(items) != 5:
            raise ValueError(
                "Gemini returned text, but it did not contain exactly five "
                "parseable recommendations."
            )
        LOGGER.info(
            "Gemini guidance succeeded: model=%s project=%s location=%s",
            settings.model_name,
            settings.project_id,
            settings.location,
        )
        return raw, items, True
    except Exception as exc:
        context = (
            f"model={settings.model_name} project={settings.project_id} "
            f"location={settings.location} auth={settings.auth_mode}"
            if settings
            else "settings_unavailable"
        )
        LOGGER.exception(
            "Gemini guidance failed (%s): %s: %s",
            context,
            type(exc).__name__,
            exc,
        )
        return fallback_result()
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
