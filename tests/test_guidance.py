from __future__ import annotations

from types import SimpleNamespace

import guidance
import vertex_config


def settings():
    return vertex_config.VertexSettings(
        project_id="test-project",
        location="global",
        model_name="gemini-2.5-pro",
        credentials=object(),
        config_source="test",
        auth_mode="test_credentials",
    )


class FakeModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, models):
        self.models = models
        self.closed = False

    def close(self):
        self.closed = True


def test_successful_gemini_guidance_uses_ai_response():
    text = """Prediction Summary:
The model produced an educational estimate.
Key Factors Influencing Prediction:
- Age was considered.
Personalized Recommendations:
1. Check blood pressure using a validated home monitor.
2. Compare food labels and choose lower-sodium options.
3. Add comfortable physical activity to the weekly routine.
4. Keep a consistent sleep schedule and manage daily stress.
5. Discuss repeated high readings with a professional; this is not medical advice.
"""
    models = FakeModels(SimpleNamespace(text=text))
    client = FakeClient(models)
    raw, items, available = guidance.generate_guidance(
        {"Age": 55},
        "No Hypertension",
        0.2,
        {},
        settings_loader=settings,
        client_factory=lambda _: client,
    )
    assert available is True
    assert raw == text.strip()
    assert len(items) == 5
    assert models.calls[0]["model"] == "gemini-2.5-pro"
    assert client.closed is True


def test_gemini_exception_uses_fallback(caplog):
    client = FakeClient(FakeModels(error=PermissionError("IAM denied")))
    _, items, available = guidance.generate_guidance(
        {},
        "No Hypertension",
        0.1,
        {},
        settings_loader=settings,
        client_factory=lambda _: client,
    )
    assert available is False
    assert items == guidance.GENERIC_GUIDANCE
    assert "PermissionError: IAM denied" in caplog.text
    assert client.closed is True


def test_partial_gemini_output_uses_fallback():
    client = FakeClient(
        FakeModels(SimpleNamespace(text="Personalized Recommendations:\n1. One recommendation only."))
    )
    _, items, available = guidance.generate_guidance(
        {},
        "No Hypertension",
        0.1,
        {},
        settings_loader=settings,
        client_factory=lambda _: client,
    )
    assert available is False
    assert len(items) == 5


def test_streamlit_service_account_json_is_supported(monkeypatch):
    sentinel_credentials = object()
    monkeypatch.setattr(
        vertex_config,
        "_streamlit_gcp_config",
        lambda: {
            "project_id": "test-project",
            "location": "global",
            "service_account_json": '{"placeholder": true}',
        },
    )
    monkeypatch.setattr(
        vertex_config,
        "_credentials_from_info",
        lambda value, source: sentinel_credentials,
    )
    loaded = vertex_config._settings_from_streamlit()
    assert loaded.credentials is sentinel_credentials
    assert loaded.auth_mode == "streamlit_service_account_json"


def test_client_uses_explicit_vertex_v1_configuration(monkeypatch):
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(vertex_config.genai, "Client", fake_client)
    vertex_config.get_vertex_client(settings())
    assert captured["vertexai"] is True
    assert captured["project"] == "test-project"
    assert captured["location"] == "global"
    assert captured["credentials"] is not None
    assert captured["http_options"].api_version == "v1"
