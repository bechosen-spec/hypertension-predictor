from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import db
import inference
from ui_components import asset_exists, risk_category
from utils.app_logic import (
    build_html_report,
    history_csv,
    history_json,
    parse_recommendations,
    validate_patient,
)
from utils.auth import AUTH_STATUS_KEY


def test_auth_key_is_consistent_across_entry_points():
    root = Path(__file__).resolve().parents[1]
    app_source = (root / "app.py").read_text()
    assert "authentication_status" not in app_source
    assert AUTH_STATUS_KEY == "auth_status"


def test_no_streamlit_multipage_directory_is_exposed():
    root = Path(__file__).resolve().parents[1]
    pages = root / "pages"
    assert not pages.exists() or not list(pages.glob("*.py"))


def test_fetch_user_logs_enforces_ownership(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    testing_session = sessionmaker(bind=engine)
    db.Base.metadata.create_all(engine)
    monkeypatch.setattr(db, "SessionLocal", testing_session)
    with testing_session() as session:
        session.add_all(
            [
                db.PredictionLog(
                    username="alice", created_at=dt.datetime.utcnow(),
                    prediction="No Hypertension", probability=0.2,
                    patient_json="{}", llm_feedback="",
                ),
                db.PredictionLog(
                    username="bob", created_at=dt.datetime.utcnow(),
                    prediction="Has Hypertension", probability=0.8,
                    patient_json="{}", llm_feedback="",
                ),
            ]
        )
        session.commit()
    rows = db.fetch_user_logs("alice")
    assert len(rows) == 1
    assert rows[0].username == "alice"
    assert db.fetch_user_logs("") == []


def test_missing_image_has_safe_detection():
    assert asset_exists("heart_health_hero.png")
    assert not asset_exists("does-not-exist.webp")


def test_risk_categories_cover_probability_range():
    assert risk_category(0.1)[0] == "Lower estimated risk"
    assert risk_category(0.5)[0] == "Moderate estimated risk"
    assert risk_category(0.9)[0] == "Higher estimated risk"


def sample_patient():
    return {
        "Age": 55,
        "BMI": 28.0,
        "Salt_Intake": 8.0,
        "Stress_Score": 5,
        "Sleep_Duration": 7.0,
        "BP_History": "Normal",
        "Medication": "None",
        "Family_History": "No",
        "Exercise_Level": "Moderate",
        "Smoking_Status": "Non-Smoker",
    }


def test_patient_validation_boundaries_and_categories():
    patient = sample_patient()
    patient.update({"Age": 1, "BMI": 10, "Salt_Intake": 0, "Stress_Score": 10, "Sleep_Duration": 0})
    assert validate_patient(patient) == []
    patient.update({"Age": 120, "BMI": 60, "Salt_Intake": 50, "Sleep_Duration": 16})
    assert validate_patient(patient) == []
    patient["Smoking_Status"] = "unexpected"
    assert any("smoking status" in error for error in validate_patient(patient))
    assert validate_patient({})  # missing fields fail safely rather than raising


def test_gemini_recommendation_normalization():
    six = "Personalized Recommendations:\n" + "\n".join(
        f"{number}. Recommendation number {number} is long enough."
        for number in range(1, 7)
    )
    assert len(parse_recommendations(six)) == 5
    partial = "Personalized Recommendations:\n1. Only one sufficiently long recommendation."
    assert parse_recommendations(partial) == []
    assert parse_recommendations("") == []


def test_download_payloads_are_valid_and_escape_html():
    records = [{"prediction": "No Hypertension", "notes": "café, follow-up"}]
    csv_payload = history_csv(records).decode("utf-8")
    assert "prediction,notes" in csv_payload
    assert "café, follow-up" in csv_payload
    assert json.loads(history_json(records)) == records
    result = {
        "prediction": "<unsafe>",
        "probability": 0.25,
        "assessed_at": "2026-07-25",
        "patient": {"Age": "<script>"},
        "recommendations": ["Use <care> safely."],
    }
    report = build_html_report(result, {"Age": "Age"})
    assert "<script>" not in report
    assert "&lt;script&gt;" in report
    assert "&lt;unsafe&gt;" in report


def test_two_user_history_and_duplicate_account(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    testing_session = sessionmaker(bind=engine)
    db.Base.metadata.create_all(engine)
    monkeypatch.setattr(db, "SessionLocal", testing_session)
    assert db.create_user("alice", "Alice", "password123")
    assert not db.create_user("alice", "Another Alice", "password123")
    assert db.create_user("bob", "Bob", "password123")
    db.log_prediction("alice", "No Hypertension", 0.2, {}, {}, "")
    db.log_prediction("bob", "Has Hypertension", 0.8, {}, {}, "")
    assert {row.username for row in db.fetch_user_logs("alice")} == {"alice"}
    assert {row.username for row in db.fetch_user_logs("bob")} == {"bob"}
    assert db.fetch_user_logs("not-a-user") == []


def test_prediction_submission_has_duplicate_guard():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    assert "completed_assessment_id" in source
    assert "assessment_id" in source
    assert "pending_navigation" in source


def test_missing_artifact_has_clear_error(monkeypatch, tmp_path):
    missing = tmp_path / "missing-model.joblib"
    monkeypatch.setattr(inference, "MODEL_PATH", str(missing))
    try:
        inference.load_artifacts()
    except FileNotFoundError as exc:
        assert "Model file not found" in str(exc)
    else:
        raise AssertionError("Missing model should raise FileNotFoundError")


def test_database_read_failure_is_not_misreported_as_empty(monkeypatch):
    def broken_session():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(db, "SessionLocal", broken_session)
    try:
        db.fetch_user_logs("alice")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Database failures must reach the UI error handler")


def test_secure_streamlit_configuration():
    config = (Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml").read_text()
    assert "enableCORS = true" in config
    assert "enableXsrfProtection = true" in config
    assert "showErrorDetails = false" in config
