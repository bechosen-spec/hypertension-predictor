# tests/test_inference.py
"""
Lightweight tests for the inference pipeline.

These tests are robust to missing local artifacts:
- If required files aren't present in ./models, model-dependent tests are skipped.
- Schema-level tests still run.

Run:
    pytest -q
"""

from __future__ import annotations

import os
import sys
import types
import importlib
import pytest

# Allow running tests from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import module under test
inference = importlib.import_module("inference")


def artifacts_exist() -> bool:
    """Return True if all expected artifact files exist."""
    return all(
        os.path.exists(p)
        for p in (inference.MODEL_PATH, inference.SCALER_PATH, inference.FEATURES_PATH)
    )


def test_module_exports():
    """Basic sanity: module and key attributes/functions exist."""
    assert isinstance(inference, types.ModuleType)
    for name in [
        "CATEGORICAL_COLS",
        "NUMERICAL_COLS",
        "load_artifacts",
        "prepare_single_row",
        "predict_hypertension",
        "get_feature_importance",
    ]:
        assert hasattr(inference, name), f"missing {name} in inference.py"


def test_schema_lists_nonempty():
    """Ensure schema lists are defined and non-empty."""
    assert isinstance(inference.CATEGORICAL_COLS, list)
    assert isinstance(inference.NUMERICAL_COLS, list)
    assert len(inference.CATEGORICAL_COLS) > 0
    assert len(inference.NUMERICAL_COLS) > 0


@pytest.mark.skipif(not artifacts_exist(), reason="Model artifacts not found in ./models")
def test_load_artifacts_ok():
    """Artifacts load successfully when present."""
    model, scaler, feature_names = inference.load_artifacts()
    assert model is not None
    assert scaler is not None
    assert isinstance(feature_names, list)
    assert len(feature_names) > 0
    # No duplicates in feature names
    assert len(set(feature_names)) == len(feature_names)


@pytest.mark.skipif(not artifacts_exist(), reason="Model artifacts not found in ./models")
def test_prepare_single_row_alignment():
    """prepare_single_row returns a 1 x n_features frame aligned to training columns."""
    _, scaler, feature_names = inference.load_artifacts()
    sample = {
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
    X = inference.prepare_single_row(sample, feature_names, scaler)
    assert X.shape[0] == 1
    assert X.shape[1] == len(feature_names)
    # Columns exactly match the saved order
    assert list(X.columns) == feature_names
    # All values numeric
    assert all(str(dtype).startswith(("float", "int")) for dtype in X.dtypes)


@pytest.mark.skipif(not artifacts_exist(), reason="Model artifacts not found in ./models")
def test_predict_hypertension_returns_label_and_proba():
    """predict_hypertension returns a valid label and an optional probability."""
    label, proba = inference.predict_hypertension(
        {
            "Age": 60,
            "Salt_Intake": 9.5,
            "Stress_Score": 6,
            "BP_History": "Prehypertension",
            "Sleep_Duration": 6.5,
            "BMI": 30.0,
            "Medication": "None",
            "Family_History": "Yes",
            "Exercise_Level": "Light",
            "Smoking_Status": "Non-Smoker",
        }
    )
    assert label in {"Has Hypertension", "No Hypertension"}
    if proba is not None:
        assert 0.0 <= float(proba) <= 1.0


@pytest.mark.skipif(not artifacts_exist(), reason="Model artifacts not found in ./models")
def test_get_feature_importance_shape_and_order():
    """get_feature_importance returns a dataframe with (feature, importance) columns."""
    model, _, feature_names = inference.load_artifacts()
    fi = inference.get_feature_importance(model, feature_names)
    # Some models (e.g., without .coef_/.feature_importances_) may return None
    if fi is None:
        pytest.skip("Model does not expose feature importances/coefficients.")
    assert list(fi.columns) == ["feature", "importance"]
    assert len(fi) == len(feature_names)
    # Sorted descending
    importances = fi["importance"].to_list()
    assert importances == sorted(importances, reverse=True)
