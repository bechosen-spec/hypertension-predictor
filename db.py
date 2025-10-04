# db.py
"""
SQLite logging for predictions and LLM feedback.

Creates a lightweight local DB (default: ./app.db) with a single table:
- prediction_logs: stores user, timestamp, model output, raw patient input,
  top local features, and generated guidance.

Usage:
    from db import init_db, log_prediction, fetch_logs
    init_db()
    log_prediction(username, prediction, probability, patient, top_features, llm_feedback)
    rows = fetch_logs(limit=200, username="admin")
"""

from __future__ import annotations

import json
import os
import datetime as dt
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

# --------------------------------------------------------------------
# Engine / Session
# --------------------------------------------------------------------
DB_URL = os.getenv("DB_URL", "sqlite:///app.db")  # file in repo root by default
engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


# --------------------------------------------------------------------
# ORM Model
# --------------------------------------------------------------------
class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(120), nullable=True, index=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True)

    prediction = Column(String(64), nullable=False)  # "Has Hypertension" / "No Hypertension"
    probability = Column(Float, nullable=True)       # P(class=1) if available

    # JSON payloads stored as text
    patient_json = Column(Text, nullable=False)          # original input dict
    top_features_json = Column(Text, nullable=True)      # {feature: value} for this case
    llm_feedback = Column(Text, nullable=True)           # generated guidance text

    # Optional fields you may want later:
    model_version = Column(String(64), nullable=True)    # e.g., "rf_v1"
    notes = Column(Text, nullable=True)


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------
def init_db() -> None:
    """Create tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def log_prediction(
    username: Optional[str],
    prediction: str,
    probability: Optional[float],
    patient: Dict[str, Any],
    top_features: Dict[str, Any],
    llm_feedback: str,
    model_version: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    """Insert a new prediction log row."""
    with SessionLocal() as s:
        row = PredictionLog(
            username=username or "anonymous",
            prediction=prediction,
            probability=probability,
            patient_json=json.dumps(patient),
            top_features_json=json.dumps(top_features) if top_features else None,
            llm_feedback=llm_feedback or "",
            model_version=model_version,
            notes=notes,
        )
        s.add(row)
        s.commit()


def fetch_logs(limit: int = 200, username: Optional[str] = None) -> List[PredictionLog]:
    """
    Retrieve latest logs (optionally filtered by username).
    Returns a list of ORM objects (PredictionLog).
    """
    with SessionLocal() as s:
        q = s.query(PredictionLog).order_by(PredictionLog.created_at.desc())
        if username:
            q = q.filter(PredictionLog.username == username)
        return q.limit(limit).all()


# --------------------------------------------------------------------
# Optional helpers
# --------------------------------------------------------------------
def to_dict(row: PredictionLog) -> Dict[str, Any]:
    """Convert an ORM row to a plain dict (with parsed JSON fields)."""
    return {
        "id": row.id,
        "username": row.username,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "prediction": row.prediction,
        "probability": row.probability,
        "patient": _safe_json_loads(row.patient_json),
        "top_features": _safe_json_loads(row.top_features_json),
        "llm_feedback": row.llm_feedback,
        "model_version": row.model_version,
        "notes": row.notes,
    }


def _safe_json_loads(s: Optional[str]) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return s  # fallback to raw text if not valid JSON
