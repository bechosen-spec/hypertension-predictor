# tests/conftest.py
"""
Shared pytest fixtures and test configuration.
Placed alongside tests so pytest auto-loads it.

- Adds the repo root to sys.path so tests can import local modules.
- Provides a reusable `sample_patient` fixture matching the expected schema.
"""

from __future__ import annotations

import os
import sys
import pytest


# Ensure repo root is importable (so `import inference` works when running `pytest` from anywhere)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@pytest.fixture(scope="session")
def sample_patient() -> dict:
    """A schema-consistent sample input for inference tests."""
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
