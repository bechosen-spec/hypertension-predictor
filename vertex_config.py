from __future__ import annotations

# vertex_config.py
# Vertex AI (Gemini) client helpers.
# Reads credentials from Streamlit secrets or environment variables.

import os
import json
import tempfile
from typing import Optional

from google import genai

try:
    import streamlit as st  # optional; file also works outside Streamlit
except Exception:
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
        if "project_id" not in gcp:
            raise KeyError("gcp.project_id missing in secrets.")
        gcp = dict(gcp)
        gcp.setdefault("location", "us-central1")
        return gcp
    except Exception:
        return None


def _load_conf_from_env() -> dict:
    """Load GCP conf from environment variables."""
    project_id = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION", "us-central1")
    if not project_id:
        raise RuntimeError(
            "PROJECT_ID is not set. Provide via Streamlit secrets [gcp.project_id] or env var PROJECT_ID."
        )

    conf = {"project_id": project_id, "location": location}

    gac = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if gac and os.path.isfile(gac):
        conf["auth_mode"] = "gac_path"
        conf["gac_path"] = gac
        return conf

    sa_json_str = os.getenv("SERVICE_ACCOUNT_JSON")
    if sa_json_str:
        conf["auth_mode"] = "inline_json"
        conf["service_account_json"] = sa_json_str
        return conf

    conf["auth_mode"] = "adc"  # GCP default credentials
    return conf


def get_vertex_client() -> genai.Client:
    """
    Build a google-genai client for Vertex AI.

    Priority:
      1) Streamlit secrets [gcp]: project_id, location, service_account_json (optional)
      2) Environment: PROJECT_ID, LOCATION, and either GOOGLE_APPLICATION_CREDENTIALS
         or SERVICE_ACCOUNT_JSON (or ADC on GCP)
    """
    conf = _load_conf_from_streamlit() or _load_conf_from_env()

    project_id = conf["project_id"]
    location = conf.get("location", "us-central1")

    if "service_account_json" in conf and conf["service_account_json"]:
        sa_str = conf["service_account_json"]
        if isinstance(sa_str, (dict, list)):
            sa_str = json.dumps(sa_str)
        _write_sa_to_tempfile(sa_str)
    elif conf.get("auth_mode") == "inline_json":
        _write_sa_to_tempfile(conf["service_account_json"])

    return genai.Client(vertexai=True, project=project_id, location=location)


def get_default_gen_config(temperature: float = 0.3, top_p: float = 0.95, max_output_tokens: int = 1024) -> dict:
    return {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_output_tokens": int(max_output_tokens),
    }
