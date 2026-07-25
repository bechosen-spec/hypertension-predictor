"""Validated Google Vertex AI / Gemini configuration.

Configuration priority:
1. Streamlit ``[gcp]`` secrets.
2. Environment variables.
3. Application Default Credentials for authentication only.

Supported Streamlit credential formats:
- ``[gcp.service_account]`` as a TOML table.
- ``gcp.service_account_json`` as a JSON string or mapping.

Supported environment credential formats:
- ``GOOGLE_APPLICATION_CREDENTIALS`` path.
- ``SERVICE_ACCOUNT_JSON`` inline JSON.
- Application Default Credentials (ADC).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import google.auth
from google import genai
from google.genai import types
from google.oauth2 import service_account

try:
    import streamlit as st
except ImportError:  # pragma: no cover - Streamlit is an app dependency
    st = None

LOGGER = logging.getLogger(__name__)
DEFAULT_LOCATION = "global"
DEFAULT_MODEL_NAME = "gemini-2.5-pro"
VERTEX_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)


class VertexConfigurationError(RuntimeError):
    """Raised when local Vertex configuration is absent or malformed."""


@dataclass(frozen=True)
class VertexSettings:
    project_id: str
    location: str
    model_name: str
    credentials: Any
    config_source: str
    auth_mode: str


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping) or hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


def _service_account_info(value: Any, source: str) -> dict[str, Any]:
    value = _plain(value)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise VertexConfigurationError(
                f"{source} is not valid service-account JSON: {exc.msg}"
            ) from exc
    if not isinstance(value, dict):
        raise VertexConfigurationError(f"{source} must be a JSON object or TOML table.")
    required = {"type", "project_id", "private_key", "client_email", "token_uri"}
    missing = sorted(required.difference(value))
    if missing:
        raise VertexConfigurationError(
            f"{source} is missing required fields: {', '.join(missing)}"
        )
    info = dict(value)
    if isinstance(info.get("private_key"), str):
        info["private_key"] = info["private_key"].replace("\\n", "\n")
    return info


def _credentials_from_info(value: Any, source: str):
    info = _service_account_info(value, source)
    try:
        return service_account.Credentials.from_service_account_info(
            info,
            scopes=VERTEX_SCOPES,
        )
    except Exception as exc:
        raise VertexConfigurationError(
            f"Could not construct credentials from {source}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _streamlit_gcp_config() -> Optional[dict[str, Any]]:
    if st is None:
        return None
    try:
        if "gcp" not in st.secrets:
            return None
        return _plain(st.secrets["gcp"])
    except Exception as exc:
        raise VertexConfigurationError(
            f"Streamlit secrets could not be parsed: {type(exc).__name__}: {exc}"
        ) from exc


def _settings_from_streamlit() -> Optional[VertexSettings]:
    conf = _streamlit_gcp_config()
    if not conf:
        return None
    project_id = str(conf.get("project_id", "")).strip()
    if not project_id:
        raise VertexConfigurationError("Streamlit secrets are missing gcp.project_id.")
    location = str(conf.get("location") or DEFAULT_LOCATION).strip()
    model_name = str(conf.get("model_name") or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL_NAME).strip()

    if conf.get("service_account"):
        credentials = _credentials_from_info(
            conf["service_account"], "gcp.service_account"
        )
        auth_mode = "streamlit_service_account"
    elif conf.get("service_account_json"):
        credentials = _credentials_from_info(
            conf["service_account_json"], "gcp.service_account_json"
        )
        auth_mode = "streamlit_service_account_json"
    else:
        credentials, detected_project = google.auth.default(scopes=VERTEX_SCOPES)
        if not project_id and detected_project:
            project_id = detected_project
        auth_mode = "application_default_credentials"

    return VertexSettings(
        project_id=project_id,
        location=location,
        model_name=model_name,
        credentials=credentials,
        config_source="streamlit_secrets",
        auth_mode=auth_mode,
    )


def _settings_from_environment() -> VertexSettings:
    project_id = (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("PROJECT_ID")
        or ""
    ).strip()
    location = (
        os.getenv("GOOGLE_CLOUD_LOCATION")
        or os.getenv("LOCATION")
        or DEFAULT_LOCATION
    ).strip()
    model_name = (os.getenv("GEMINI_MODEL") or DEFAULT_MODEL_NAME).strip()

    inline = os.getenv("SERVICE_ACCOUNT_JSON")
    credential_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if inline:
        credentials = _credentials_from_info(inline, "SERVICE_ACCOUNT_JSON")
        auth_mode = "environment_service_account_json"
    elif credential_path:
        path = Path(credential_path).expanduser()
        if not path.is_file():
            raise VertexConfigurationError(
                f"GOOGLE_APPLICATION_CREDENTIALS does not point to a readable file: {path}"
            )
        try:
            credentials = service_account.Credentials.from_service_account_file(
                str(path), scopes=VERTEX_SCOPES
            )
        except Exception as exc:
            raise VertexConfigurationError(
                "GOOGLE_APPLICATION_CREDENTIALS could not be loaded: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        auth_mode = "google_application_credentials"
    else:
        try:
            credentials, detected_project = google.auth.default(scopes=VERTEX_SCOPES)
        except Exception as exc:
            raise VertexConfigurationError(
                "No Vertex credentials found. Configure Streamlit secrets, "
                "GOOGLE_APPLICATION_CREDENTIALS, SERVICE_ACCOUNT_JSON, or ADC. "
                f"ADC error: {type(exc).__name__}: {exc}"
            ) from exc
        project_id = project_id or (detected_project or "")
        auth_mode = "application_default_credentials"

    if not project_id:
        project_id = str(getattr(credentials, "project_id", "") or "").strip()
    if not project_id:
        raise VertexConfigurationError(
            "No Google Cloud project ID found. Set GOOGLE_CLOUD_PROJECT or PROJECT_ID."
        )

    return VertexSettings(
        project_id=project_id,
        location=location,
        model_name=model_name,
        credentials=credentials,
        config_source="environment",
        auth_mode=auth_mode,
    )


def load_vertex_settings() -> VertexSettings:
    """Load and validate settings without making a network request."""
    streamlit_error: Optional[Exception] = None
    try:
        settings = _settings_from_streamlit()
        if settings:
            return settings
    except VertexConfigurationError as exc:
        streamlit_error = exc
        LOGGER.error("Streamlit Vertex configuration is invalid: %s", exc)

    try:
        return _settings_from_environment()
    except VertexConfigurationError as env_error:
        if streamlit_error:
            raise VertexConfigurationError(
                f"{streamlit_error} Environment fallback also failed: {env_error}"
            ) from env_error
        raise


def get_vertex_client(settings: Optional[VertexSettings] = None) -> genai.Client:
    """Create a stable-v1 Vertex AI Gemini client with explicit credentials."""
    settings = settings or load_vertex_settings()
    LOGGER.info(
        "Initializing Vertex AI: project=%s location=%s model=%s source=%s auth=%s",
        settings.project_id,
        settings.location,
        settings.model_name,
        settings.config_source,
        settings.auth_mode,
    )
    return genai.Client(
        vertexai=True,
        project=settings.project_id,
        location=settings.location,
        credentials=settings.credentials,
        http_options=types.HttpOptions(api_version="v1"),
    )


def get_default_gen_config(
    temperature: float = 0.3,
    top_p: float = 0.95,
    max_output_tokens: int = 1024,
) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        temperature=float(temperature),
        top_p=float(top_p),
        max_output_tokens=int(max_output_tokens),
    )
