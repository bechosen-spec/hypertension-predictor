#!/usr/bin/env python3
"""Test the exact Vertex AI / Gemini path used by the Streamlit app."""
from __future__ import annotations

import argparse
import logging
import socket
import sys
import traceback
from importlib.metadata import version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vertex_config import get_vertex_client, load_vertex_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate PulseWise Vertex credentials and make one Gemini request."
    )
    parser.add_argument("--model", help="Override the configured Gemini model.")
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: GEMINI_DIAGNOSTIC_OK",
        help="Small diagnostic prompt.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    print("PulseWise Gemini diagnostic")
    print(f"python={sys.version.split()[0]}")
    for package in ("google-genai", "google-auth", "streamlit"):
        try:
            print(f"{package}={version(package)}")
        except Exception:
            print(f"{package}=not-installed")

    try:
        socket.getaddrinfo("aiplatform.googleapis.com", 443)
        print("dns_aiplatform=ok")
        socket.getaddrinfo("oauth2.googleapis.com", 443)
        print("dns_oauth2=ok")
    except OSError as exc:
        print(f"dns=failed ({type(exc).__name__}: {exc})")

    client = None
    try:
        settings = load_vertex_settings()
        model = args.model or settings.model_name
        print(f"config_source={settings.config_source}")
        print(f"auth_mode={settings.auth_mode}")
        print(f"project_id={settings.project_id}")
        print(f"location={settings.location}")
        print(f"model={model}")
        print(f"credentials_type={type(settings.credentials).__name__}")

        client = get_vertex_client(settings)
        response = client.models.generate_content(
            model=model,
            contents=args.prompt,
        )
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise RuntimeError("Gemini request succeeded but returned no text.")
        print("request=success")
        print(f"response={text}")
        return 0
    except Exception as exc:
        print("request=failed")
        print(f"exception_type={type(exc).__name__}")
        print(f"exception={exc}")
        if args.verbose:
            traceback.print_exc()
        print("\nConfiguration checklist:")
        print("1. Make .streamlit/secrets.toml valid TOML with exactly one [gcp] table.")
        print("2. Enable aiplatform.googleapis.com in the configured project.")
        print("3. Grant the service account roles/aiplatform.user.")
        print("4. Ensure billing is enabled and the model is available in the region.")
        print("5. Allow outbound HTTPS/DNS to oauth2.googleapis.com and aiplatform.googleapis.com.")
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
