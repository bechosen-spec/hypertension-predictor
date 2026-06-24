# from __future__ import annotations

# # vertex_config.py
# # Vertex AI (Gemini) client helpers.
# # Reads credentials from Streamlit secrets or environment variables.

# import os
# import json
# import tempfile
# from typing import Optional

# from google import genai

# try:
#     import streamlit as st  # optional; file also works outside Streamlit
# except Exception:
#     st = None  # type: ignore


# def _write_sa_to_tempfile(sa_json_str: str) -> str:
#     """Write a service account JSON string to a temp file and return its path."""
#     fd, path = tempfile.mkstemp(prefix="sa_", suffix=".json")
#     with os.fdopen(fd, "w") as f:
#         f.write(sa_json_str)
#     os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
#     return path


# def _load_conf_from_streamlit() -> Optional[dict]:
#     """Return GCP conf from Streamlit secrets if available, else None."""
#     if st is None:
#         return None
#     try:
#         gcp = st.secrets.get("gcp", None)
#         if not gcp:
#             return None
#         if "project_id" not in gcp:
#             raise KeyError("gcp.project_id missing in secrets.")
#         gcp = dict(gcp)
#         gcp.setdefault("location", "us-central1")
#         return gcp
#     except Exception:
#         return None


# def _load_conf_from_env() -> dict:
#     """Load GCP conf from environment variables."""
#     project_id = os.getenv("PROJECT_ID")
#     location = os.getenv("LOCATION", "us-central1")
#     if not project_id:
#         raise RuntimeError(
#             "PROJECT_ID is not set. Provide via Streamlit secrets [gcp.project_id] or env var PROJECT_ID."
#         )

#     conf = {"project_id": project_id, "location": location}

#     gac = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
#     if gac and os.path.isfile(gac):
#         conf["auth_mode"] = "gac_path"
#         conf["gac_path"] = gac
#         return conf

#     sa_json_str = os.getenv("SERVICE_ACCOUNT_JSON")
#     if sa_json_str:
#         conf["auth_mode"] = "inline_json"
#         conf["service_account_json"] = sa_json_str
#         return conf

#     conf["auth_mode"] = "adc"  # GCP default credentials
#     return conf


# def get_vertex_client() -> genai.Client:
#     """
#     Build a google-genai client for Vertex AI.

#     Priority:
#       1) Streamlit secrets [gcp]: project_id, location, service_account_json (optional)
#       2) Environment: PROJECT_ID, LOCATION, and either GOOGLE_APPLICATION_CREDENTIALS
#          or SERVICE_ACCOUNT_JSON (or ADC on GCP)
#     """
#     conf = _load_conf_from_streamlit() or _load_conf_from_env()

#     project_id = conf["project_id"]
#     location = conf.get("location", "us-central1")

#     if "service_account_json" in conf and conf["service_account_json"]:
#         sa_str = conf["service_account_json"]
#         if isinstance(sa_str, (dict, list)):
#             sa_str = json.dumps(sa_str)
#         _write_sa_to_tempfile(sa_str)
#     elif conf.get("auth_mode") == "inline_json":
#         _write_sa_to_tempfile(conf["service_account_json"])

#     return genai.Client(vertexai=True, project=project_id, location=location)


# def get_default_gen_config(temperature: float = 0.3, top_p: float = 0.95, max_output_tokens: int = 1024) -> dict:
#     return {
#         "temperature": float(temperature),
#         "top_p": float(top_p),
#         "max_output_tokens": int(max_output_tokens),
#     }


from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

from google import genai

try:
    import streamlit as st
except ImportError:
    st = None


def _write_sa_to_tempfile(service_account_data) -> str:
    """
    Write service account credentials to a temporary file and return the path.
    Supports both dict and JSON string inputs.
    """

    try:
        if isinstance(service_account_data, str):
            service_account_data = json.loads(service_account_data)

        if not isinstance(service_account_data, dict):
            raise ValueError("Service account credentials must be a JSON object.")

    except Exception as e:
        raise RuntimeError(
            f"Invalid service account JSON: {e}"
        ) from e

    fd, path = tempfile.mkstemp(prefix="sa_", suffix=".json")

    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(service_account_data, f)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path

    return path


def _load_conf_from_streamlit() -> Optional[dict]:
    """Load GCP config from Streamlit secrets."""

    if st is None:
        return None

    try:
        if "gcp" not in st.secrets:
            return None

        gcp = dict(st.secrets["gcp"])

        if "project_id" not in gcp:
            raise RuntimeError("Missing gcp.project_id in Streamlit secrets")

        gcp.setdefault("location", "us-central1")

        return gcp

    except Exception as e:
        raise RuntimeError(f"Failed to load Streamlit secrets: {e}")


def _load_conf_from_env() -> dict:
    """Load GCP config from environment variables."""

    project_id = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION", "us-central1")

    if not project_id:
        raise RuntimeError(
            "PROJECT_ID is missing. Set PROJECT_ID or configure Streamlit secrets."
        )

    conf = {
        "project_id": project_id,
        "location": location,
    }

    gac_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if gac_path and os.path.isfile(gac_path):
        conf["gac_path"] = gac_path
        return conf

    service_account_json = os.getenv("SERVICE_ACCOUNT_JSON")

    if service_account_json:
        conf["service_account_json"] = service_account_json

    return conf


def get_vertex_client() -> genai.Client:
    """
    Create and return a Vertex AI Gemini client.

    Supported auth methods:
    1. Streamlit secrets
    2. GOOGLE_APPLICATION_CREDENTIALS
    3. SERVICE_ACCOUNT_JSON environment variable
    """

    conf = _load_conf_from_streamlit() or _load_conf_from_env()

    project_id = conf["project_id"]
    location = conf.get("location", "us-central1")

    if "service_account_json" in conf:
        _write_sa_to_tempfile(conf["service_account_json"])

    return genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )


def get_default_gen_config(
    temperature: float = 0.3,
    top_p: float = 0.95,
    max_output_tokens: int = 1024,
) -> dict:
    return {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_output_tokens": int(max_output_tokens),
    }
