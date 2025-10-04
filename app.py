# app.py — Sign Up / Sign In in main area, fixed LLM settings, gated Dashboard
from __future__ import annotations

import json
import streamlit as st
import pandas as pd

from inference import (
    load_artifacts,
    prepare_single_row,
    predict_hypertension,
    get_feature_importance,
)
from vertex_config import get_vertex_client, get_default_gen_config
from db import (
    init_db, log_prediction,
    create_user, get_user, verify_password,
)

st.set_page_config(page_title="Hypertension Predictor", page_icon="🩺", layout="centered")

# ─────────────────────────────
# Fixed LLM settings (no UI controls)
# ─────────────────────────────
LLM_MODEL_NAME = "gemini-2.5-pro"
LLM_TEMPERATURE = 0.3
LLM_TOP_P = 0.95
LLM_MAX_TOKENS = 2048  # bumped to avoid truncation

def _extract_text(resp) -> str:
    """Robustly stitch Gemini response text."""
    txt = getattr(resp, "text", None)
    if txt:
        return txt.strip()
    parts = []
    for cand in getattr(resp, "candidates", []) or []:
        content = getattr(cand, "content", None)
        if content and getattr(content, "parts", None):
            for part in content.parts:
                t = getattr(part, "text", None)
                if t:
                    parts.append(t)
        if parts:
            break
    return "\n".join(parts).strip()

# ─────────────────────────────
# Init DB & session auth state
# ─────────────────────────────
init_db()

if "auth_status" not in st.session_state:
    st.session_state.auth_status = None
if "username" not in st.session_state:
    st.session_state.username = None
if "name" not in st.session_state:
    st.session_state.name = None

def do_login(username: str, password: str) -> bool:
    user = get_user((username or "").strip())
    if not user:
        return False
    if not verify_password(password or "", user.password_hash):
        return False
    st.session_state.auth_status = True
    st.session_state.username = user.username
    st.session_state.name = user.name
    return True

def do_logout():
    st.session_state.auth_status = None
    st.session_state.username = None
    st.session_state.name = None

def do_signup(name: str, username: str, password: str):
    name = (name or "").strip()
    username = (username or "").strip()
    if len(name) < 2:
        return False, "Please enter your full name."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password or "") < 6:
        return False, "Password must be at least 6 characters."
    ok = create_user(username=username, name=name, password=password)
    if not ok:
        return False, "Username already exists."
    return True, "Account created. Please sign in."

# ─────────────────────────────
# AUTH VIEW (main area)
# ─────────────────────────────
if st.session_state.auth_status is not True:
    st.title("🩺 Hypertension Risk Predictor")
    st.caption("Sign in or create an account to continue.")

    mode = st.radio("Authentication", options=["Sign In", "Sign Up"], horizontal=True, index=0, label_visibility="visible")

    if mode == "Sign In":
        with st.form("signin_form", clear_on_submit=False):
            u = st.text_input("Username", autocomplete="username")
            p = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Sign In")
        if submitted:
            if do_login(u, p):
                st.success(f"Welcome back, {st.session_state.name}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
    else:
        with st.form("signup_form", clear_on_submit=False):
            name = st.text_input("Full name")
            u = st.text_input("Choose a username", autocomplete="username")
            p1 = st.text_input("Create password", type="password", autocomplete="new-password")
            p2 = st.text_input("Confirm password", type="password", autocomplete="new-password")
            submitted = st.form_submit_button("Create Account")
        if submitted:
            if p1 != p2:
                st.error("Passwords do not match.")
            else:
                ok, msg = do_signup(name, u, p1)
                if ok:
                    st.success(msg)
                    st.info("Now sign in with your new account.")
                else:
                    st.error(msg)

    st.stop()

# ─────────────────────────────
# AUTHENTICATED APP (Predict + Dashboard)
# ─────────────────────────────
with st.sidebar:
    st.success(f"Signed in as {st.session_state.username}")
    if st.button("Logout"):
        do_logout()
        st.rerun()

st.title("🩺 Hypertension Risk Predictor")
st.caption("Provide patient info → Predict risk → See key factors → Get personalized guidance.")

tabs = st.tabs(["Predict", "Dashboard"])

# Load artifacts (cached)
@st.cache_resource(show_spinner=True)
def _load_artifacts_cached():
    return load_artifacts()

try:
    model, scaler, feature_names = _load_artifacts_cached()
except Exception as e:
    st.error(f"Failed to load model artifacts: {e}")
    st.stop()

# ─────────────────────────────
# TAB 1: Predict
# ─────────────────────────────
with tabs[0]:
    st.subheader("Enter Patient Data")
    with st.form("patient_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=1, max_value=120, value=55)
            salt_intake = st.number_input("Salt Intake (g/day)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
            stress_score = st.number_input("Stress Score (0-10)", min_value=0, max_value=10, value=7)
            sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=16.0, value=7.5, step=0.5)
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.0, step=0.1)
        with c2:
            bp_history = st.selectbox("BP History", ["Normal", "Prehypertension", "Hypertension"], index=0)
            medication = st.selectbox(
                "Medication",
                ["None", "ACE Inhibitor", "Beta Blocker", "Diuretic", "Calcium Channel Blocker"],
                index=0,
            )
            family_history = st.selectbox("Family History", ["No", "Yes"], index=0)
            exercise_level = st.selectbox("Exercise Level", ["Sedentary", "Light", "Moderate", "Vigorous"], index=2)
            smoking_status = st.selectbox("Smoking Status", ["Non-Smoker", "Former Smoker", "Current Smoker"], index=0)
        submitted = st.form_submit_button("Predict 🧠")

    new_patient_data = {
        "Age": age,
        "Salt_Intake": float(salt_intake),
        "Stress_Score": int(stress_score),
        "BP_History": bp_history,
        "Sleep_Duration": float(sleep_duration),
        "BMI": float(bmi),
        "Medication": medication,
        "Family_History": family_history,
        "Exercise_Level": exercise_level,
        "Smoking_Status": smoking_status,
    }

    if submitted:
        try:
            # Prepare aligned, scaled row
            X_row = prepare_single_row(new_patient_data, feature_names, scaler)

            # Predict
            pred_label, pred_proba = predict_hypertension(new_patient_data)

            # Display result
            st.markdown("### Results")
            badge_color = "red" if pred_label == "Has Hypertension" else "green"
            st.markdown(
                f"**Prediction:** <span style='color:{badge_color}; font-weight:700'>{pred_label}</span>",
                unsafe_allow_html=True,
            )
            if pred_proba is not None:
                st.write(f"Probability of Hypertension: **{pred_proba:.2%}**")

            # Global feature importance
            fi_df = get_feature_importance(model, feature_names)
            if fi_df is not None and not fi_df.empty:
                st.write("#### Top Global Feature Importances")
                st.dataframe(fi_df.head(15), use_container_width=True)

                # Simple local view: top active columns (by global importance) for this row
                active_cols = [c for c in X_row.columns if abs(float(X_row.iloc[0][c])) > 1e-9]
                local_fi = fi_df[fi_df["feature"].isin(active_cols)].copy()
                local_fi = local_fi.sort_values("importance", ascending=False).head(5)
            else:
                local_fi = pd.DataFrame(columns=["feature", "importance"])

            if not local_fi.empty:
                st.write("#### Likely Key Factors for this Prediction")
                st.table(local_fi.rename(columns={"feature": "Feature", "importance": "Importance"}))

            # Build compact dict for logging/LLM
            new_patient_top_features_values = {
                f: float(X_row.iloc[0][f]) for f in local_fi["feature"].tolist() if f in X_row.columns
            }

            # Personalized guidance via Gemini (fixed config) — using your structured prompt
            llm_text = ""
            try:
                client = get_vertex_client()
                gen_config = get_default_gen_config(LLM_TEMPERATURE, LLM_TOP_P, LLM_MAX_TOKENS)

                # Human-readable sentence for the prediction
                if pred_label == "Has Hypertension":
                    prediction_result = (
                        f"has hypertension risk (estimated probability {pred_proba:.1%})"
                        if pred_proba is not None else
                        "has hypertension risk"
                    )
                else:
                    prediction_result = (
                        f"is unlikely to have hypertension (estimated probability {(1 - pred_proba):.1%})"
                        if pred_proba is not None else
                        "is unlikely to have hypertension"
                    )

                # Your structured prompt with sections
                prompt = f"""
Analyze the following patient data and the model's prediction regarding hypertension. Provide personalized feedback based on the prediction and the key factors that most influenced this prediction. Structure your response with clear sections.

Patient Data:
{json.dumps(new_patient_data, indent=2)}

Model Prediction:
The model predicts that this patient {prediction_result}.

Key Factors Influencing Prediction (Top Features and their values for this patient):
These are the features that the model identified as most important in making this prediction for this specific patient:
{json.dumps(new_patient_top_features_values, indent=2)}

Please provide feedback structured as follows:

Prediction Summary: Briefly state the prediction in plain language, including probability if provided.
Key Factors Influencing Prediction: Explain which features were most important for this prediction, referencing their values for this patient. Keep this to 3–6 short bullet points.
Personalized Recommendations: Based on the prediction and influential factors, provide EXACTLY 5 concise, actionable, non-judgmental bullet points a layperson can follow. Favor guidance on salt/sodium, activity, stress, sleep, weight/BMI, smoking, alcohol, and when to seek medical advice. Do NOT diagnose or provide medication changes. End with a gentle note to consult a healthcare professional for personalized advice.

Constraints:
- Use the three section headings exactly as specified.
- Keep sentences short and avoid medical jargon.
- Do not include preambles or trailing ellipses.
"""

                resp = client.models.generate_content(
                    model=LLM_MODEL_NAME,
                    contents=prompt,
                    config=gen_config,
                )
                llm_text = _extract_text(resp)

                if llm_text:
                    st.write("#### Personalized Guidance")
                    st.write(llm_text)
                else:
                    st.info("LLM returned no text.")
            except Exception as e:
                llm_text = ""
                st.warning(f"LLM feedback unavailable: {e}")

            # Log the run
            try:
                log_prediction(
                    username=st.session_state.username,
                    prediction=pred_label,
                    probability=pred_proba,
                    patient=new_patient_data,
                    top_features=new_patient_top_features_values,
                    llm_feedback=llm_text or "",
                )
            except Exception as e:
                st.warning(f"Could not save log: {e}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

# ─────────────────────────────
# TAB 2: Dashboard (only after auth)
# ─────────────────────────────
with tabs[1]:
    st.subheader("📊 Prediction & Feedback Dashboard")
    from db import fetch_logs, to_dict

    only_me = st.checkbox("Show only my logs", value=True)
    limit = st.slider("Max rows", 50, 1000, 300, step=50)

    rows = fetch_logs(limit=limit, username=st.session_state.username if only_me else None)
    if not rows:
        st.info("No logs yet. Run a prediction first.")
    else:
        records = [to_dict(r) for r in rows]
        df = pd.DataFrame(records)
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"])

        st.dataframe(
            df[[c for c in ["id", "created_at", "username", "prediction", "probability"] if c in df.columns]]
            .sort_values("created_at", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        st.write("---")
        idx = st.number_input("Inspect row #", 1, max(1, len(df)), 1)
        if len(df) >= idx:
            row = df.iloc[idx - 1]
            cA, cB = st.columns([1, 1])
            with cA:
                st.markdown("**Patient input**")
                st.json(row.get("patient", {}))
            with cB:
                st.markdown("**Top features (this case)**")
                st.json(row.get("top_features", {}) or {})
            st.markdown("**Personalized feedback**")
            st.write(row.get("llm_feedback", "") or "_(no feedback)_")

        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_logs.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download JSON",
            json.dumps(records, indent=2, default=str).encode("utf-8"),
            file_name="prediction_logs.json",
            mime="application/json",
        )

st.markdown("---")
st.caption("This tool provides educational information and does not substitute professional medical advice.")
