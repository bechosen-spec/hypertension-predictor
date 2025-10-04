# app.py (LLM settings fixed — no user controls)
import json
import streamlit as st
import pandas as pd
import streamlit_authenticator as stauth

from inference import (
    load_artifacts,
    prepare_single_row,
    predict_hypertension,
    get_feature_importance,
)
from vertex_config import get_vertex_client, get_default_gen_config
from db import init_db, log_prediction

st.set_page_config(page_title="Hypertension Predictor", page_icon="🩺", layout="centered")

# ─────────────────────────────
# Fixed LLM settings (no UI controls)
# ─────────────────────────────
LLM_MODEL_NAME = "gemini-2.5-pro"
LLM_TEMPERATURE = 0.3
LLM_TOP_P = 0.95
LLM_MAX_TOKENS = 1024

# ─────────────────────────────
# Auth & DB
# ─────────────────────────────
init_db()

auth_conf = st.secrets.get("auth", {})
credentials = {
    "usernames": {
        u: {"name": n, "password": p}
        for u, n, p in zip(
            auth_conf.get("usernames", []),
            auth_conf.get("names", []),
            auth_conf.get("passwords_hashed", []),
        )
    }
}
authenticator = stauth.Authenticate(
    credentials=credentials,
    cookie_name=auth_conf.get("cookie_name", "hpredictor_auth"),
    key=auth_conf.get("cookie_key", "change_me"),
    cookie_expiry_days=int(auth_conf.get("cookie_expiry_days", 7)),
)

with st.sidebar:
    st.header("Login")
    name, auth_status, username = authenticator.login("Login", "sidebar")

if auth_status is False:
    st.error("Invalid username/password.")
    st.stop()
elif auth_status is None:
    st.info("Please log in to continue.")
    st.stop()
else:
    with st.sidebar:
        authenticator.logout("Logout", "sidebar")

# ─────────────────────────────
# Title & description
# ─────────────────────────────
st.title("🩺 Hypertension Risk Predictor")
st.caption("Provide patient info → Predict risk → See key factors → Get personalized guidance.")

# ─────────────────────────────
# Load artifacts (cached)
# ─────────────────────────────
@st.cache_resource(show_spinner=True)
def _load_artifacts_cached():
    return load_artifacts()

try:
    model, scaler, feature_names = _load_artifacts_cached()
except Exception as e:
    st.error(f"Failed to load model artifacts: {e}")
    st.stop()

# ─────────────────────────────
# Form inputs
# ─────────────────────────────
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

# ─────────────────────────────
# Predict + Importances + LLM (fixed settings)
# ─────────────────────────────
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
            active_cols = []
            for col in X_row.columns:
                val = float(X_row.iloc[0][col])
                if abs(val) > 1e-9:
                    active_cols.append(col)
            local_fi = fi_df[fi_df["feature"].isin(active_cols)].copy()
            local_fi = local_fi.sort_values("importance", ascending=False).head(5)
        else:
            local_fi = pd.DataFrame(columns=["feature", "importance"])

        if not local_fi.empty:
            st.write("#### Likely Key Factors for this Prediction")
            st.table(local_fi.rename(columns={"feature": "Feature", "importance": "Importance"}))

        # Build compact dict for logging/LLM
        new_patient_top_features_values = {}
        for f in local_fi["feature"].tolist():
            if f in X_row.columns:
                new_patient_top_features_values[f] = float(X_row.iloc[0][f])

        # Personalized guidance via Gemini (fixed config)
        try:
            client = get_vertex_client()
            gen_config = get_default_gen_config(LLM_TEMPERATURE, LLM_TOP_P, LLM_MAX_TOKENS)

            if pred_label == "Has Hypertension":
                pred_sentence = (
                    f"is at risk of hypertension (probability {pred_proba:.1%})"
                    if pred_proba is not None
                    else "is at risk of hypertension"
                )
            else:
                pred_sentence = (
                    f"is unlikely to have hypertension (probability {(1-pred_proba):.1%})"
                    if pred_proba is not None
                    else "is unlikely to have hypertension"
                )

            context = {
                "prediction": pred_label,
                "probability": None if pred_proba is None else round(float(pred_proba), 4),
                "patient": new_patient_data,
                "top_local_factors": list(new_patient_top_features_values.keys()),
            }

            prompt = (
                "You are a health coach assistant. Based on the JSON below, provide 5 concise, actionable, "
                "non-judgmental recommendations tailored to the user. If risk is present, include practical steps on salt, "
                "activity, stress, sleep, and follow-up with healthcare. Avoid diagnoses.\n\n"
                f"Model summary: The model predicts the patient {pred_sentence}.\n\n"
                f"JSON:\n{json.dumps(context, indent=2)}\n\n"
                "Output format:\n"
                "- 5 bullet points, one sentence each\n"
                "- Simple language, avoid medical jargon\n"
                "- Include a gentle note to consult a healthcare professional for personalized advice\n"
            )

            resp = client.models.generate_content(
                model=LLM_MODEL_NAME,
                contents=prompt,
                config=gen_config,
            )
            llm_text = getattr(resp, "text", None)
            if not llm_text and getattr(resp, "candidates", None):
                llm_text = resp.candidates[0].content.parts[0].text

            if llm_text:
                st.write("#### Personalized Guidance")
                st.write(llm_text)
            else:
                llm_text = ""
                st.info("LLM returned no text.")

        except Exception as e:
            llm_text = ""
            st.warning(f"LLM feedback unavailable: {e}")

        # Log the run
        try:
            log_prediction(
                username=username,
                prediction=pred_label,
                probability=pred_proba,
                patient=new_patient_data,
                top_features=new_patient_top_features_values,
                llm_feedback=llm_text,
            )
        except Exception as e:
            st.warning(f"Could not save log: {e}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown("---")
st.caption(
    "This tool provides educational information and does not substitute professional medical advice."
)
