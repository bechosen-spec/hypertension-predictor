# scripts/verify_artifacts.py
"""
Quick sanity checks for model artifacts and preprocessing pipeline.

What it does:
1) Verifies that required files exist in ./models/
2) Loads model, scaler, feature_names
3) Prints summary: n_features, model type, has predict_proba, etc.
4) Runs a tiny round-trip using a dummy patient dict to ensure
   prepare_single_row() returns the correct columns and the model predicts.

Usage:
  python scripts/verify_artifacts.py
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Dict

import numpy as np

# Allow running from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from inference import (
    MODELS_DIR,
    MODEL_PATH,
    SCALER_PATH,
    FEATURES_PATH,
    load_artifacts,
    prepare_single_row,
)

REQUIRED_FILES = [MODEL_PATH, SCALER_PATH, FEATURES_PATH]


def _ok(msg: str) -> None:
    print(f"✅ {msg}")


def _warn(msg: str) -> None:
    print(f"⚠️  {msg}")


def _fail(msg: str) -> None:
    print(f"❌ {msg}")


def check_files() -> bool:
    print(f"Looking for artifacts in: {os.path.abspath(MODELS_DIR)}")
    ok = True
    for path in REQUIRED_FILES:
        if os.path.exists(path):
            _ok(f"Found: {path}")
        else:
            _fail(f"Missing: {path}")
            ok = False
    return ok


def dummy_patient() -> Dict:
    """A minimal, schema-consistent dummy input."""
    return {
        "Age": 55,
        "Salt_Intake": 10.0,
        "Stress_Score": 7,
        "BP_History": "Normal",
        "Sleep_Duration": 7.5,
        "BMI": 28.0,
        "Medication": "None",
        "Family_History": "No",
        "Exercise_Level": "Moderate",
        "Smoking_Status": "Non-Smoker",
    }


def main() -> int:
    print("\n=== Artifact Verification ===")
    if not check_files():
        print("\nFix the missing files above and rerun.")
        return 1

    try:
        model, scaler, feature_names = load_artifacts()
        _ok(f"Loaded model: {type(model).__name__}")
        _ok(f"Loaded scaler: {type(scaler).__name__}")
        _ok(f"Feature names: {len(feature_names)} columns")
        if len(set(feature_names)) != len(feature_names):
            _warn("feature_names contains duplicates.")
    except Exception as e:
        _fail(f"Failed to load artifacts: {e}")
        traceback.print_exc(limit=1)
        return 1

    # Check model API
    has_proba = hasattr(model, "predict_proba")
    _ok(f"Model supports predict_proba: {has_proba}")

    # Build a single row
    try:
        patient = dummy_patient()
        X = prepare_single_row(patient, feature_names, scaler)
        if X.shape[1] != len(feature_names):
            _fail(
                f"Prepared row has {X.shape[1]} columns but feature_names has {len(feature_names)}."
            )
            return 1
        _ok(f"Prepared single row shape: {X.shape}")
        # Basic sanity on dtypes (floats/ints)
        if not np.issubdtype(X.dtypes[0], np.number):
            _warn("First column is not numeric; check preprocessing.")
    except Exception as e:
        _fail(f"Failed to prepare single row: {e}")
        traceback.print_exc(limit=1)
        return 1

    # Try predicting
    try:
        y = model.predict(X)
        _ok(f"Model.predict() returned: {y} (dtype={type(y).__name__})")
        if has_proba:
            p = model.predict_proba(X)[0][1]
            _ok(f"Model.predict_proba() P(class=1): {float(p):.4f}")
    except Exception as e:
        _fail(f"Prediction failed: {e}")
        traceback.print_exc(limit=1)
        return 1

    print("\nAll checks passed ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
