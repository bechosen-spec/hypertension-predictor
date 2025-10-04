# scripts/export_feature_names.py
"""
Export the EXACT feature column order used for inference to: ./models/feature_names.joblib

You can run this in TWO ways:

A) From a DESIGN MATRIX (already one-hot encoded; i.e., the exact frame you fed to model.fit)
   -------------------------------------------------------------------------
   python scripts/export_feature_names.py \
       --design-csv path/to/X_train_final.csv

   → Reads the CSV, takes its columns in order, and saves feature_names.joblib.

B) From a RAW TRAINING CSV (pre-dummies; script will one-hot encode like inference)
   -------------------------------------------------------------------------
   python scripts/export_feature_names.py \
       --raw-csv path/to/train.csv \
       --target-column target

   → Applies get_dummies(drop_first=False) on the categorical columns defined below,
     keeps numeric columns as-is, then writes the resulting column order.

Notes:
- Keep CATEGORICAL_COLS and NUMERICAL_COLS consistent with training.
- Scaling is NOT needed to export names; we only need the column order.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import joblib
import pandas as pd

# Allow running from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.joblib")

# ─────────────────────────────────────────────────────────────
# Define the training-time schema (edit if your training schema changed)
# ─────────────────────────────────────────────────────────────
CATEGORICAL_COLS: List[str] = [
    "BP_History",
    "Medication",
    "Family_History",
    "Exercise_Level",
    "Smoking_Status",
]
NUMERICAL_COLS: List[str] = ["Age", "Salt_Intake", "Stress_Score", "Sleep_Duration", "BMI"]


def _ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


def export_from_design_csv(design_csv: str) -> List[str]:
    """Read already-encoded design matrix and export its columns as-is."""
    if not os.path.exists(design_csv):
        raise FileNotFoundError(f"Design CSV not found: {design_csv}")
    df = pd.read_csv(design_csv)
    feature_names = list(df.columns)
    return feature_names


def export_from_raw_csv(raw_csv: str, target_col: str | None) -> List[str]:
    """Read raw training CSV, one-hot encode categorical cols (drop_first=False), return column order."""
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Raw CSV not found: {raw_csv}")

    df = pd.read_csv(raw_csv)

    # Drop target if provided
    if target_col and target_col in df.columns:
        df = df.drop(columns=[target_col])

    # Sanity: ensure required columns exist (create if missing—will be dropped later if empty)
    for col in set(CATEGORICAL_COLS + NUMERICAL_COLS):
        if col not in df.columns:
            # If you truly trained without this column, that's fine; it just won't appear.
            # We add it empty so get_dummies can still create a stable set if needed.
            df[col] = pd.Series(dtype="object" if col in CATEGORICAL_COLS else "float")

    # Keep only the declared columns to avoid accidental extras
    df = df[CATEGORICAL_COLS + NUMERICAL_COLS]

    # One-hot categoricals (important: drop_first=False to match inference.py)
    df_enc = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False, dtype="int64")

    feature_names = list(df_enc.columns)
    return feature_names


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export feature_names.joblib for inference.")
    m = parser.add_mutually_exclusive_group(required=True)
    m.add_argument("--design-csv", help="Path to X_train_final.csv (already one-hot encoded).")
    m.add_argument("--raw-csv", help="Path to raw training CSV (pre-dummies).")
    parser.add_argument("--target-column", default=None, help="Target column name in raw CSV (if any).")

    args = parser.parse_args(argv)

    _ensure_models_dir()

    if args.design_csv:
        feature_names = export_from_design_csv(args.design_csv)
        source = args.design_csv
    else:
        feature_names = export_from_raw_csv(args.raw_csv, args.target_column)
        source = args.raw_csv

    # Save
    joblib.dump(feature_names, FEATURES_PATH)

    # Report
    print(f"✅ Exported {len(feature_names)} feature names → {FEATURES_PATH}")
    print("First 20 columns:", feature_names[:20])
    print(f"Source file: {source}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
