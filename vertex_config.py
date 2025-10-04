# vertex_config.py
"""
Vertex AI (Gemini) client helpers.

Reads credentials from Streamlit secrets (recommended) or environment variables.

Streamlit secrets (.streamlit/secrets.toml):
[gcp]
project_id = "your-project-id"
location   = "us-central1"
# Paste full Service Account JSON (multiline string)
service_account_json = """
# { ... full SA JSON ... }
"""

Environment variable fallback (if not using Streamlit secrets):
- PROJECT_ID, LOCATION
- EITHER:
    * GOOGLE_APPLICATION_CREDENTIALS (path to SA JSON file), OR
    * SERVICE_ACCOUNT_JSON (the full JSON string)
"""

from __future__ import annotations

import os
import json
import tempfile
from typing import Optional

from google import genai

# Attempt to import Streamlit; allow running outside Streamlit too.
try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None  # type: ignore


def _write_sa_to_tempfile(sa_json_str: str) -> str:
    """Write a service account JSON string to a temp file and return its path."""
    fd, path = tempfile.mkstemp(prefix="sa_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        f.write(sa_json_str)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
    return path


def _load_conf_from_streamlit() -> Optional[dict]:
    """Return GCP conf from Streamlit secrets if available, else None."""
    if st is None:
        return None
    try:
        gcp = st.secrets.get("gcp", None)
        if not gcp:
            return None
        # Validate minimal fields
        if "project_id" not in gcp:
            raise KeyError("gcp.project_id missing in secrets.")
        # location is optional; default to us-central1
        gcp.setdefault("location", "us-central1")
        return dict(gcp)
    except Exception:
        return None


def _load_conf_from_env() -> dict:
    """Load GCP conf from environment variables."""
    project_id = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION", "us-central1")

    if not project_id:
        raise RuntimeError(
            "PROJECT_ID is not set. Provide via Streamlit secrets [gcp.project_id] "
            "or env var PROJECT_ID."
        )

    conf = {"project_id": project_id, "location": location}

    # Prefer GOOGLE_APPLICATION_CREDENTIALS if already set
    gac = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if gac and os.path.isfile(gac):
        conf["auth_mode"] = "gac_path"
        conf["gac_path"] = gac
        return conf

    # Else accept full JSON via env
    sa_json_str = os.getenv("SERVICE_ACCOUNT_JSON")
    if sa_json_str:
        conf["auth_mode"] = "inline_json"
        conf["service_account_json"] = sa_json_str
        return conf

    # If neither provided, client may still work on GCE/Cloud Run via default ADC
    conf["auth_mode"] = "adc"  # Application Default Credentials
    return conf


def get_vertex_client() -> genai.Client:
    """
    Build a google-genai client for Vertex AI.

    Priority:
      1) Streamlit secrets [gcp]: project_id, location, service_account_json (optional)
      2) Environment variables: PROJECT_ID, LOCATION, and either
         GOOGLE_APPLICATION_CREDENTIALS or SERVICE_ACCOUNT_JSON (or ADC on GCP)
    """
    conf = _load_conf_from_streamlit() or _load_conf_from_env()

    project_id = conf["project_id"]
    location = conf.get("location", "us-central1")

    # If Streamlit secrets provided inline JSON, write to temp and set GAC
    if "service_account_json" in conf and conf["service_account_json"]:
        sa_str = conf["service_account_json"]
        # If Streamlit secret is a structured object, convert to string
        if isinstance(sa_str, (dict, list)):
            sa_str = json.dumps(sa_str)
        _write_sa_to_tempfile(sa_str)

    # If env provided inline JSON, write it too
    elif conf.get("auth_mode") == "inline_json":
        _write_sa_to_tempfile(conf["service_account_json"])

    # Otherwise:
    # - GOOGLE_APPLICATION_CREDENTIALS may already be set (good)
    # - Or ADC will be used in GCP environments

    return genai.Client(vertexai=True, project=project_id, location=location)


def get_default_gen_config(temperature: float = 0.3, top_p: float = 0.95, max_output_tokens: int = 1024) -> dict:
    """Default generation config used by the app."""
    return {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_output_tokens": int(max_output_tokens),
    }
