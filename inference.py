# inference.py
"""
Model artifacts loading, preprocessing for a single patient row,
prediction, and global feature importance utilities.

Expected artifacts (place them in ./models/):
- best_rf_model.joblib        : trained classifier (e.g., RandomForest)
- scaler.joblib               : fitted StandardScaler for numeric columns
- feature_names.joblib        : list of X_train_final.columns after all preprocessing (post-dummies)

NOTE: Keep CATEGORICAL_COLS and NUMERICAL_COLS consistent with training.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd


# ---------- Paths (override via env vars if needed) ----------
MODELS_DIR = os.getenv("MODELS_DIR", "models")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(MODELS_DIR, "best_rf_model.joblib"))
SCALER_PATH = os.getenv("SCALER_PATH", os.path.join(MODELS_DIR, "scaler.joblib"))
FEATURES_PATH = os.getenv(
    "FEATURES_PATH", os.path.join(MODELS_DIR, "feature_names.joblib")
)

# ---------- Columns used during training ----------
CATEGORICAL_COLS: List[str] = [
    "BP_History",
    "Medication",
    "Family_History",
    "Exercise_Level",
    "Smoking_Status",
]
NUMERICAL_COLS: List[str] = ["Age", "Salt_Intake", "Stress_Score", "Sleep_Duration", "BMI"]


# ============================================================
# Artifacts
# ============================================================

def load_artifacts():
    """
    Load model, scaler, and feature names from disk.
    Returns:
        model: trained estimator
        scaler: fitted StandardScaler (or compatible transformer for NUMERICAL_COLS)
        feature_names: list[str] exact columns used to fit the model
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Feature names file not found: {FEATURES_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    if not isinstance(feature_names, (list, tuple)):
        raise ValueError("feature_names.joblib must contain a list/tuple of column names.")

    return model, scaler, list(feature_names)


# ============================================================
# Preprocessing for a single row
# ============================================================

def prepare_single_row(new_data: Dict, feature_names: List[str], scaler) -> pd.DataFrame:
    """
    Build a single-row DataFrame in the EXACT format the model expects:
    - ensure base columns exist
    - one-hot encode categorical columns (drop_first=False)
    - reindex to training feature set (missing -> 0)
    - scale numeric columns (same transformer as training)
    """
    df = pd.DataFrame([new_data])

    # Ensure all base columns are present
    for col in set(CATEGORICAL_COLS + NUMERICAL_COLS):
        if col not in df.columns:
            df[col] = np.nan

    # One-hot encode categoricals WITHOUT drop_first
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False, dtype=np.int64)

    # Align to training feature columns (create any missing dummies with 0)
    df = df.reindex(columns=feature_names, fill_value=0)

    # Scale only numeric columns that exist in the aligned frame
    numeric_to_scale = [c for c in NUMERICAL_COLS if c in df.columns]
    if numeric_to_scale:
        df[numeric_to_scale] = scaler.transform(df[numeric_to_scale])

    return df


# ============================================================
# Prediction
# ============================================================

def predict_hypertension(new_data: Dict) -> Tuple[str, Optional[float]]:
    """
    Predicts hypertension status for a new patient.

    Returns:
        label: "Has Hypertension" or "No Hypertension"
        proba: probability of 'Has Hypertension' (class 1) if available, else None
    """
    model, scaler, feature_names = load_artifacts()
    X = prepare_single_row(new_data, feature_names, scaler)

    # Predict class
    pred = model.predict(X)[0]
    label = "Has Hypertension" if int(pred) == 1 else "No Hypertension"

    # Probability if supported
    proba: Optional[float] = None
    if hasattr(model, "predict_proba"):
        try:
            proba = float(model.predict_proba(X)[0][1])
        except Exception:
            proba = None

    return label, proba


# ============================================================
# Importances (global) and helpers
# ============================================================

def get_feature_importance(model, feature_names: List[str]) -> Optional[pd.DataFrame]:
    """
    Returns a DataFrame with global feature importances.
    Supports:
      - Tree models with .feature_importances_
      - Linear models with .coef_ (abs value as proxy)
    """
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = getattr(model, "feature_importances_", None)

    elif hasattr(model, "coef_"):
        coef = getattr(model, "coef_", None)
        if coef is not None:
            # Binary linear classifier/regressor
            if np.ndim(coef) == 1:
                importances = np.abs(coef)
            else:
                # e.g., shape (1, n_features)
                importances = np.abs(coef[0])

    if importances is None:
        return None

    fi_df = pd.DataFrame(
        {"feature": feature_names, "importance": np.asarray(importances).ravel()}
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    return fi_df


def top_k_features_from_global_importance(
    model, feature_names: List[str], k: int = 10
) -> List[str]:
    """
    Convenience: names of the top-k features by global importance.
    """
    ranks = get_feature_importance(model, feature_names)
    if ranks is None or ranks.empty:
        return []
    return ranks.head(k)["feature"].tolist()


def extract_patient_feature_values(X_row: pd.DataFrame, features: List[str]) -> Dict[str, float]:
    """
    For a single aligned row (1 x n_features), return the values of the selected features.
    Works for both scaled numeric and one-hot columns.
    """
    out: Dict[str, float] = {}
    for f in features:
        if f in X_row.columns:
            out[f] = float(X_row.iloc[0][f])
    return out
