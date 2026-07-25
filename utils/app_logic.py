from __future__ import annotations

import csv
import html
import io
import json
import re
from typing import Any, Iterable

EXPECTED_CATEGORIES = {
    "BP_History": {"Normal", "Prehypertension", "Hypertension"},
    "Medication": {
        "None",
        "ACE Inhibitor",
        "Beta Blocker",
        "Diuretic",
        "Calcium Channel Blocker",
    },
    "Family_History": {"No", "Yes"},
    "Exercise_Level": {"Sedentary", "Light", "Moderate", "Vigorous"},
    "Smoking_Status": {"Non-Smoker", "Former Smoker", "Current Smoker"},
}


def validate_patient(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    numeric_rules = [
        ("Age", 1, 120, "Age must be between 1 and 120."),
        ("BMI", 10, 60, "BMI must be between 10 and 60."),
        ("Salt_Intake", 0, 50, "Daily salt intake must be between 0 and 50 grams."),
        ("Stress_Score", 0, 10, "Stress score must be between 0 and 10."),
        ("Sleep_Duration", 0, 16, "Sleep duration must be between 0 and 16 hours."),
    ]
    for field, minimum, maximum, message in numeric_rules:
        try:
            value = float(data[field])
            if not minimum <= value <= maximum:
                errors.append(message)
        except (KeyError, TypeError, ValueError):
            errors.append(message)
    for field, allowed in EXPECTED_CATEGORIES.items():
        if data.get(field) not in allowed:
            errors.append(f"Please choose a valid {field.replace('_', ' ').lower()} option.")
    return errors


def parse_recommendations(text: str) -> list[str]:
    """Extract exactly five numbered/bulleted recommendations or return none."""
    if not text:
        return []
    section = text.split("Personalized Recommendations:", 1)[-1]
    lines: list[str] = []
    for raw in section.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw).strip()
        if cleaned and not cleaned.endswith(":") and len(cleaned) > 12:
            lines.append(cleaned)
    return lines[:5] if len(lines) >= 5 else []


def history_csv(records: Iterable[dict[str, Any]]) -> bytes:
    rows = list(records)
    if not rows:
        return b""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def history_json(records: Iterable[dict[str, Any]]) -> str:
    return json.dumps(list(records), indent=2, default=str, ensure_ascii=False)


def build_html_report(
    result: dict[str, Any],
    display_features: dict[str, str],
) -> str:
    patient_rows = "".join(
        f"<tr><th>{html.escape(display_features.get(key, key))}</th>"
        f"<td>{html.escape(str(value))}</td></tr>"
        for key, value in result["patient"].items()
    )
    guidance = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in result["recommendations"]
    )
    probability = (
        f'{float(result["probability"]):.1%}'
        if result.get("probability") is not None
        else "Unavailable"
    )
    assessed_at = html.escape(str(result.get("assessed_at", "")))
    prediction = html.escape(str(result.get("prediction", "Unavailable")))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PulseWise assessment</title>
<style>body{{font-family:Arial,sans-serif;max-width:760px;margin:40px auto;padding:0 18px;color:#102a43}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #dce8ef;text-align:left}}
.note{{background:#edf7fc;padding:14px;border-radius:8px}}</style></head>
<body><h1>PulseWise educational assessment</h1><p>{assessed_at}</p>
<h2>{prediction}</h2><p>Estimated hypertension-class probability: <strong>{probability}</strong></p>
<h3>Submitted information</h3><table>{patient_rows}</table><h3>Guidance</h3><ol>{guidance}</ol>
<p class="note">This model output is educational and is not a diagnosis or a substitute for professional medical care.</p>
</body></html>"""
