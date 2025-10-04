---
name: "🐛 Bug report"
about: "Report a problem to help us improve the Hypertension Predictor app"
title: "[Bug] <short summary>"
labels: bug, needs-triage
assignees: ""
---

## 🧾 Summary
_A clear, concise description of the problem._

## 🔁 Steps to Reproduce
1. Go to `<page or URL>`
2. Click `<button/link>`
3. Enter inputs (example):
   - Age: `55`
   - Salt_Intake: `10.0`
   - Stress_Score: `7`
   - BP_History: `Normal`
   - Sleep_Duration: `7.5`
   - BMI: `28.0`
   - Medication: `None`
   - Family_History: `No`
   - Exercise_Level: `Moderate`
   - Smoking_Status: `Non-Smoker`
4. Press **Predict**
5. See error `<message or behavior>`

> If LLM-related, include the prompt (remove private data).

## ✅ Expected Behavior
_What you expected to happen._

## ❌ Actual Behavior
_What actually happened. Include full error text if available._

<details>
<summary>📜 Logs / Traceback</summary>

</details>

## 🖼️ Screenshots / Recording
_If applicable, attach screenshots or a short screen recording._

## 🧩 Environment
- Git commit / App version: `<SHA or vX.Y.Z>`
- Deployment: ☐ Local ☐ Streamlit Cloud ☐ Docker ☐ Other: `<specify>`
- OS: ☐ macOS ☐ Windows ☐ Linux
- Browser: ☐ Chrome ☐ Firefox ☐ Safari ☐ Edge
- Python: `3.11.x`

### Key Packages
- streamlit: `<version>`
- scikit-learn: `<version>`
- google-genai: `<version>`
- pandas: `<version>`
- numpy: `<version>`

## 📦 Model Artifacts Present?
- `models/best_rf_model.joblib`: ☐ Yes ☐ No
- `models/scaler.joblib`: ☐ Yes ☐ No
- `models/feature_names.joblib`: ☐ Yes ☐ No

## 🔐 Configuration
- `.streamlit/secrets.toml` configured? ☐ Yes ☐ No
  - `[gcp] project_id` / `location`: `<values>`
  - `service_account_json` present? ☐ Yes ☐ No
- Auth enabled? ☐ Yes ☐ No
  - Users configured in `[auth]`? ☐ Yes ☐ No

## 🚨 Severity & Impact
- Severity: ☐ Critical ☐ High ☐ Medium ☐ Low  
- Regression: ☐ Yes (worked before on `v...`) ☐ No  
- Affected pages: ☐ `app.py` ☐ `pages/1_🗂️_Dashboard.py` ☐ Both

## 🧠 Possible Cause / Notes
_Any clues, recent changes, or related issues. If reproduction requires special data, attach a minimal sample._
