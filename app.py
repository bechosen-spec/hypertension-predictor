from __future__ import annotations

import logging
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

from db import (
    create_user,
    fetch_user_logs,
    get_user,
    init_db,
    log_prediction,
    to_dict,
    verify_password,
)
from inference import (
    get_feature_importance,
    load_artifacts,
    predict_hypertension,
    prepare_single_row,
)
from styles import apply_styles
from ui_components import (
    footer,
    image_or_fallback,
    metric_card,
    notice,
    probability_bar,
    progress_steps,
    recommendation_cards,
    risk_category,
    section_header,
)
from utils.auth import (
    clear_auth,
    init_auth_state,
    is_authenticated,
    set_authenticated,
)
from utils.app_logic import (
    build_html_report,
    history_json,
    validate_patient,
)
from guidance import generate_guidance

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

APP_NAME = "PulseWise Risk"
APP_VERSION = "2.0"
DISPLAY_FEATURES = {
    "Age": "Age",
    "BMI": "Body mass index",
    "Salt_Intake": "Daily salt intake",
    "Stress_Score": "Stress score",
    "Sleep_Duration": "Sleep duration",
    "BP_History": "Blood-pressure history",
    "Medication": "Medication category",
    "Family_History": "Family history",
    "Exercise_Level": "Exercise level",
    "Smoking_Status": "Smoking status",
}
ASSESSMENT_WIDGET_KEYS = (
    "assessment_age",
    "assessment_bmi",
    "assessment_salt",
    "assessment_stress",
    "assessment_sleep",
    "assessment_exercise",
    "assessment_smoking",
    "assessment_bp_history",
    "assessment_medication",
    "assessment_family_history",
)

st.set_page_config(page_title=APP_NAME, page_icon="🫀", layout="wide")
apply_styles()
init_auth_state()

try:
    init_db()
except Exception:
    LOGGER.exception("Database initialization failed")
    st.error("The secure data store is temporarily unavailable. Please try again later.")
    st.stop()


@st.cache_resource(show_spinner=False)
def cached_artifacts():
    return load_artifacts()


def human_feature(name: str) -> str:
    base = name.replace("_", " ")
    for raw, display in DISPLAY_FEATURES.items():
        if name == raw:
            return display
        if name.startswith(f"{raw}_"):
            return f"{display}: {name[len(raw) + 1:].replace('_', ' ')}"
    return base.title()


def queue_navigation(destination: str) -> None:
    """Schedule navigation for the next rerun without mutating a live widget."""
    st.session_state.pending_navigation = destination


def start_new_assessment() -> None:
    for key in ASSESSMENT_WIDGET_KEYS:
        st.session_state.pop(key, None)
    st.session_state.assessment_data = {}
    st.session_state.assessment_step = 1
    st.session_state.assessment_id = uuid.uuid4().hex
    st.session_state.pop("completed_assessment_id", None)
    queue_navigation("New Assessment")


def landing() -> None:
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown(
            '<div class="hero"><div class="eyebrow">Educational heart-health assessment</div>'
            "<h1>Understand your estimated hypertension risk.</h1>"
            "<p>PulseWise uses a trained Random Forest model to estimate risk from ten health "
            "and lifestyle inputs, then asks Gemini to explain the result in practical language.</p>"
            '<div class="notice">Your result is an educational estimate—not a diagnosis. '
            "Clinical blood-pressure measurements and professional review remain essential.</div></div>",
            unsafe_allow_html=True,
        )
    with right:
        image_or_fallback(
            "heart_health_hero.png",
            "A healthcare professional reviewing heart-health information with a patient.",
            "Healthcare professional checking a patient's blood pressure",
        )

    st.write("")
    benefit_cols = st.columns(4)
    benefits = [
        ("⚡ Fast assessment", "Complete a guided, four-step check."),
        ("✨ Clear guidance", "Five cautious, practical suggestions."),
        ("🔒 Private history", "Only your account can query your records."),
        ("⬇️ Portable exports", "Download your history as CSV or JSON."),
    ]
    for col, (title, copy) in zip(benefit_cols, benefits):
        with col:
            metric_card(title, copy)

    st.write("")
    auth_col, info_col = st.columns([1, 1], gap="large")
    with auth_col:
        section_header("Welcome", "Sign in or create an account", "Your assessment history is linked to your private account.")
        sign_in, sign_up = st.tabs(["Sign in", "Create account"])
        with sign_in:
            with st.form("signin_form"):
                username = st.text_input("Username", autocomplete="username", key="signin_username")
                password = st.text_input("Password", type="password", autocomplete="current-password", key="signin_password")
                submitted = st.form_submit_button("Sign in", width="stretch")
            if submitted:
                with st.spinner("Signing you in securely…"):
                    user = get_user(username.strip())
                    if user and verify_password(password, user.password_hash):
                        set_authenticated(user.username, user.name)
                        st.success("Welcome back.")
                        st.rerun()
                    else:
                        st.error("We could not sign you in. Check your username and password.")
        with sign_up:
            with st.form("signup_form"):
                name = st.text_input("Full name", key="signup_name")
                new_username = st.text_input("Choose a username", autocomplete="username", key="signup_username")
                password1 = st.text_input("Create password", type="password", autocomplete="new-password", key="signup_password")
                password2 = st.text_input("Confirm password", type="password", autocomplete="new-password", key="signup_password_confirm")
                created = st.form_submit_button("Create account", width="stretch")
            if created:
                if len(name.strip()) < 2:
                    st.error("Please enter your name.")
                elif len(new_username.strip()) < 3:
                    st.error("Choose a username with at least three characters.")
                elif len(password1) < 8:
                    st.error("Use at least eight characters for your password.")
                elif password1 != password2:
                    st.error("The passwords do not match.")
                elif create_user(new_username.strip(), name.strip(), password1):
                    st.success("Your account is ready. Use the Sign in tab to continue.")
                else:
                    st.error("That username is unavailable. Please choose another.")
    with info_col:
        section_header("How it works", "A simple, transparent flow")
        st.markdown(
            '<div class="surface"><strong>1 · Share</strong><p>Enter ten health and lifestyle details.</p>'
            "<strong>2 · Estimate</strong><p>The saved Random Forest model calculates an estimated probability.</p>"
            "<strong>3 · Understand</strong><p>Review the result, model-level importance, and relevant submitted factors.</p>"
            "<strong>4 · Act thoughtfully</strong><p>Use the guidance to support a conversation with a professional.</p></div>",
            unsafe_allow_html=True,
        )
        st.write("")
        notice("Health information is sensitive. Use a unique password and avoid entering identifying information beyond what the form requests.", "warning")
    footer()


def sidebar() -> str:
    pending = st.session_state.pop("pending_navigation", None)
    if pending:
        st.session_state.nav_destination = pending
    with st.sidebar:
        st.markdown("## 🫀 PulseWise")
        st.caption(f"Risk assessment · v{APP_VERSION}")
        st.markdown(f"**👤 {st.session_state.name}**")
        st.caption(st.session_state.username)
        destination = st.radio(
            "Navigation",
            ["Overview", "New Assessment", "Results", "History", "Health Education", "Account & Privacy"],
            label_visibility="collapsed",
            key="nav_destination",
        )
        st.write("")
        notice("Your history view is always restricted to your account.")
        st.write("")
        if st.button("Log out", width="stretch"):
            clear_auth()
            st.rerun()
        st.caption(f"{APP_NAME} · {APP_VERSION}")
    return destination


def overview(logs: list) -> None:
    name = (st.session_state.name or st.session_state.username or "there").split()[0]
    section_header("Overview", f"Welcome, {name}", "A small step toward understanding your heart-health risk.")
    latest = to_dict(logs[0]) if logs else None
    cols = st.columns(4)
    values = [
        ("Total assessments", str(len(logs)), "Saved to your account"),
        ("Most recent result", latest["prediction"] if latest else "No assessment", "Model classification"),
        ("Latest probability", f'{latest["probability"]:.1%}' if latest and latest["probability"] is not None else "—", "Estimated class probability"),
        ("Latest date", pd.to_datetime(latest["created_at"]).strftime("%d %b %Y") if latest else "—", "Assessment date"),
    ]
    for col, valueset in zip(cols, values):
        with col:
            metric_card(*valueset)
    st.write("")
    st.button(
        "Start New Assessment →",
        type="primary",
        width="stretch",
        on_click=start_new_assessment,
    )
    st.write("")
    section_header("Recent activity", "Your latest assessments", "Only records owned by your account appear here.")
    if not logs:
        notice("You have no saved assessments yet. Start a new assessment when you are ready.")
        return
    df = pd.DataFrame([to_dict(row) for row in logs[:5]])
    df["Date"] = pd.to_datetime(df["created_at"]).dt.strftime("%d %b %Y, %H:%M")
    df["Estimated probability"] = df["probability"].map(lambda value: f"{value:.1%}" if pd.notna(value) else "—")
    st.dataframe(
        df[["Date", "prediction", "Estimated probability"]].rename(columns={"prediction": "Model classification"}),
        width="stretch",
        hide_index=True,
    )


def assessment() -> None:
    st.session_state.setdefault("assessment_step", 1)
    st.session_state.setdefault("assessment_data", {})
    st.session_state.setdefault("assessment_id", uuid.uuid4().hex)
    step = st.session_state.assessment_step
    section_header("New assessment", "A guided four-step check", "Your entries are used by the existing prediction pipeline.")
    progress_steps(step)
    data = st.session_state.assessment_data

    if step == 1:
        with st.form("assessment_basic"):
            st.subheader("Basic information")
            age = st.number_input("Age (years)", 1, 120, int(data.get("Age", 55)), help="Enter your age in completed years.", key="assessment_age")
            bmi = st.number_input("BMI (kg/m²)", 10.0, 60.0, float(data.get("BMI", 28.0)), 0.1, help="BMI relates weight to height. It is a screening measure, not a diagnosis.", key="assessment_bmi")
            with st.expander("How should I interpret BMI?"):
                st.write("BMI is one broad indicator considered by the model. It does not describe overall health on its own.")
            if st.form_submit_button("Continue →", type="primary", width="stretch"):
                data.update({"Age": age, "BMI": float(bmi)})
                st.session_state.assessment_step = 2
                st.rerun()
    elif step == 2:
        with st.form("assessment_lifestyle"):
            st.subheader("Lifestyle")
            salt = st.number_input("Estimated salt intake (g/day)", 0.0, 50.0, float(data.get("Salt_Intake", 8.0)), 0.1, help="Estimate added salt plus salt in packaged and prepared food.", key="assessment_salt")
            stress = st.slider("Typical stress score", 0, 10, int(data.get("Stress_Score", 5)), help="0 means very little stress; 10 means extremely high stress.", key="assessment_stress")
            sleep = st.number_input("Typical sleep duration (hours/night)", 0.0, 16.0, float(data.get("Sleep_Duration", 7.0)), 0.5, help="Use your typical nightly sleep, not time spent in bed.", key="assessment_sleep")
            exercise = st.selectbox("Exercise level", ["Sedentary", "Light", "Moderate", "Vigorous"], index=["Sedentary", "Light", "Moderate", "Vigorous"].index(data.get("Exercise_Level", "Moderate")), key="assessment_exercise")
            smoking = st.selectbox("Smoking status", ["Non-Smoker", "Former Smoker", "Current Smoker"], index=["Non-Smoker", "Former Smoker", "Current Smoker"].index(data.get("Smoking_Status", "Non-Smoker")), key="assessment_smoking")
            previous, onward = st.columns(2)
            if previous.form_submit_button("← Previous", width="stretch"):
                data.update({"Salt_Intake": float(salt), "Stress_Score": stress, "Sleep_Duration": float(sleep), "Exercise_Level": exercise, "Smoking_Status": smoking})
                st.session_state.assessment_step = 1
                st.rerun()
            if onward.form_submit_button("Continue →", type="primary", width="stretch"):
                data.update({"Salt_Intake": float(salt), "Stress_Score": stress, "Sleep_Duration": float(sleep), "Exercise_Level": exercise, "Smoking_Status": smoking})
                st.session_state.assessment_step = 3
                st.rerun()
    elif step == 3:
        with st.form("assessment_medical"):
            st.subheader("Medical background")
            bp = st.selectbox("Blood-pressure history", ["Normal", "Prehypertension", "Hypertension"], index=["Normal", "Prehypertension", "Hypertension"].index(data.get("BP_History", "Normal")), help="Choose the category previously communicated by a healthcare professional, if known.", key="assessment_bp_history")
            medication = st.selectbox("Current medication category", ["None", "ACE Inhibitor", "Beta Blocker", "Diuretic", "Calcium Channel Blocker"], index=["None", "ACE Inhibitor", "Beta Blocker", "Diuretic", "Calcium Channel Blocker"].index(data.get("Medication", "None")), help="Reflect current use only. This app never recommends medication changes.", key="assessment_medication")
            family = st.selectbox("Family history of hypertension", ["No", "Yes"], index=["No", "Yes"].index(data.get("Family_History", "No")), key="assessment_family_history")
            previous, onward = st.columns(2)
            if previous.form_submit_button("← Previous", width="stretch"):
                data.update({"BP_History": bp, "Medication": medication, "Family_History": family})
                st.session_state.assessment_step = 2
                st.rerun()
            if onward.form_submit_button("Review answers →", type="primary", width="stretch"):
                data.update({"BP_History": bp, "Medication": medication, "Family_History": family})
                st.session_state.assessment_step = 4
                st.rerun()
    else:
        st.subheader("Review your information")
        review = pd.DataFrame(
            [
                {
                    "Information": DISPLAY_FEATURES.get(key, key),
                    "Your answer": str(value),
                }
                for key, value in data.items()
            ]
        )
        st.dataframe(review, width="stretch", hide_index=True)
        notice("Submitting runs an educational model estimate. It does not perform a medical diagnosis.", "warning")
        back, submit = st.columns(2)
        if back.button("← Edit answers", width="stretch"):
            st.session_state.assessment_step = 3
            st.rerun()
        if submit.button("Analyze information", type="primary", width="stretch"):
            errors = validate_patient(data)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                run_prediction(dict(data))


def run_prediction(patient: dict) -> None:
    assessment_id = st.session_state.get("assessment_id")
    if assessment_id and st.session_state.get("completed_assessment_id") == assessment_id:
        queue_navigation("Results")
        st.rerun()
    try:
        with st.status("Analyzing your health information…", expanded=True) as status:
            st.write("Aligning your information with the trained model")
            model, scaler, feature_names = cached_artifacts()
            row = prepare_single_row(patient, feature_names, scaler)
            st.write("Estimating risk probability")
            label, probability = predict_hypertension(patient)
            global_importance = get_feature_importance(model, feature_names)
            active = [name for name in row.columns if abs(float(row.iloc[0][name])) > 1e-9]
            relevant_df = (
                global_importance[global_importance["feature"].isin(active)].head(5).copy()
                if global_importance is not None else pd.DataFrame(columns=["feature", "importance"])
            )
            factors = {name: float(row.iloc[0][name]) for name in relevant_df["feature"] if name in row.columns}
            st.write("Preparing personalized guidance")
            raw_guidance, recommendations, ai_available = generate_guidance(patient, label, probability, factors)
            assessed_at = datetime.now().isoformat(timespec="seconds")
            result = {
                "prediction": label,
                "probability": probability,
                "patient": patient,
                "factors": factors,
                "recommendations": recommendations,
                "guidance": raw_guidance,
                "ai_available": ai_available,
                "assessed_at": assessed_at,
                "global_importance": global_importance.head(10).to_dict("records") if global_importance is not None else [],
            }
            log_prediction(
                username=st.session_state.username,
                prediction=label,
                probability=probability,
                patient=patient,
                top_features=factors,
                llm_feedback=raw_guidance,
                model_version=f"rf / app-{APP_VERSION}",
            )
            st.session_state.latest_result = result
            st.session_state.completed_assessment_id = assessment_id
            status.update(label="Assessment complete", state="complete", expanded=False)
        st.success("Your result was saved to your private history.")
        queue_navigation("Results")
        st.rerun()
    except Exception:
        LOGGER.exception("Prediction failed")
        st.error("We could not complete this assessment. Your answers were not changed; please try again.")


def results() -> None:
    result = st.session_state.get("latest_result")
    if not result:
        section_header("Results", "No new result to display", "Complete an assessment to see a detailed result here.")
        notice("Your previous assessments remain available under History.")
        return
    probability = result["probability"]
    category, css_class = risk_category(probability)
    section_header("Assessment result", "Your educational risk estimate", pd.to_datetime(result["assessed_at"]).strftime("%d %B %Y at %H:%M"))
    st.markdown(f'<span class="risk-pill {css_class}">{category}</span>', unsafe_allow_html=True)
    st.write("")
    cols = st.columns(4)
    cards = [
        ("Model classification", result["prediction"], "Stored model output"),
        ("Estimated probability", f"{probability:.1%}" if probability is not None else "—", "Hypertension class"),
        ("Relevant factors", str(len(result["factors"])), "Active inputs shown below"),
        ("Guidance status", "Gemini generated" if result["ai_available"] else "Generic fallback", f"App v{APP_VERSION}"),
    ]
    for col, values in zip(cols, cards):
        with col:
            metric_card(*values)
    st.write("")
    st.subheader("Estimated probability")
    probability_bar(probability)
    if category == "Higher estimated risk":
        interpretation = "The model estimated a higher likelihood of hypertension-related risk from the information provided."
    elif category == "Moderate estimated risk":
        interpretation = "The model estimate falls in a middle range and should be interpreted cautiously."
    else:
        interpretation = "The model estimated a lower likelihood, but a lower estimate cannot rule out hypertension."
    notice(f"{interpretation} This is not a diagnosis. A qualified professional can assess blood pressure using appropriate clinical measurements.", "warning")

    st.write("")
    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        section_header("This assessment", "Relevant submitted factors")
        st.caption("These are active inputs ranked using global model importance. They do not prove how strongly a factor changed this individual prediction.")
        if result["factors"]:
            for feature, value in result["factors"].items():
                metric_card(human_feature(feature), f"{value:.2f}", "Model-preprocessed value")
                st.write("")
        else:
            notice("No relevant-factor summary is available for this result.")
    with right:
        section_header("Across the training data", "Global model importance")
        st.caption("This chart shows which variables generally influence the model across the dataset. It is not a patient-level explanation.")
        fi = pd.DataFrame(result["global_importance"])
        if not fi.empty:
            fi["feature"] = fi["feature"].map(human_feature)
            st.bar_chart(fi.set_index("feature")["importance"].sort_values(), horizontal=True, color="#168c83")
        else:
            notice("The model does not expose a global feature-importance summary.")

    st.write("")
    section_header("AI-supported explanation", "Personalized health guidance", "Five practical ideas based on the submitted information.")
    if not result["ai_available"]:
        notice("Gemini was unavailable, so these are generic wellness suggestions rather than personalized AI guidance.", "warning")
    recommendation_cards(result["recommendations"])
    notice("AI-generated guidance may contain errors and should not replace advice from a qualified healthcare professional.")

    report = build_html_report(result, DISPLAY_FEATURES)
    download, restart = st.columns(2)
    download.download_button("Download assessment report (HTML)", report, "pulsewise_assessment.html", "text/html", width="stretch")
    restart.button(
        "Start another assessment",
        width="stretch",
        on_click=start_new_assessment,
    )


def history(logs: list) -> None:
    section_header("Private history", "Your saved assessments", "Search, filter, inspect, and export records owned by your account.")
    if not logs:
        notice("No history yet. Complete an assessment to create your first record.")
        return
    records = [to_dict(row) for row in logs]
    df = pd.DataFrame(records)
    df["created_at"] = pd.to_datetime(df["created_at"])
    search_col, risk_col, sort_col = st.columns([1.3, 1, 1])
    query = search_col.text_input("Search", placeholder="Classification or notes", key="history_search")
    risk = risk_col.selectbox("Risk filter", ["All", "Has Hypertension", "No Hypertension"], key="history_risk")
    sort = sort_col.selectbox("Sort", ["Newest first", "Oldest first", "Highest probability"], key="history_sort")
    date_min = df["created_at"].min().date()
    date_max = df["created_at"].max().date()
    date_col1, date_col2 = st.columns(2)
    start_date = date_col1.date_input(
        "From date",
        value=date_min,
        min_value=date_min,
        max_value=date_max,
        key="history_start_date",
    )
    end_date = date_col2.date_input(
        "To date",
        value=date_max,
        min_value=date_min,
        max_value=date_max,
        key="history_end_date",
    )
    filtered = df.copy()
    if start_date > end_date:
        st.warning("The start date must be on or before the end date.")
        filtered = filtered.iloc[0:0]
    else:
        record_dates = filtered["created_at"].dt.date
        filtered = filtered[(record_dates >= start_date) & (record_dates <= end_date)]
    if query:
        mask = filtered.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
        filtered = filtered[mask]
    if risk != "All":
        filtered = filtered[filtered["prediction"] == risk]
    if sort == "Newest first":
        filtered = filtered.sort_values("created_at", ascending=False)
    elif sort == "Oldest first":
        filtered = filtered.sort_values("created_at")
    else:
        filtered = filtered.sort_values("probability", ascending=False)
    display = filtered.copy()
    display["Date"] = display["created_at"].dt.strftime("%d %b %Y, %H:%M")
    display["Probability"] = display["probability"].map(lambda value: f"{value:.1%}" if pd.notna(value) else "—")
    display["Guidance"] = display["llm_feedback"].map(lambda value: "Available" if value else "Unavailable")
    st.dataframe(
        display[["Date", "prediction", "Probability", "Guidance", "model_version"]].rename(columns={"prediction": "Classification", "model_version": "Version"}),
        width="stretch",
        hide_index=True,
    )
    if not filtered.empty:
        choices = filtered.index.tolist()
        selected = st.selectbox(
            "View details",
            choices,
            format_func=lambda idx: f'{filtered.loc[idx, "created_at"]:%d %b %Y, %H:%M} · {filtered.loc[idx, "prediction"]}',
        )
        record = filtered.loc[selected]
        with st.expander("Assessment details", expanded=True):
            st.json(record["patient"], expanded=False)
            st.markdown("**Guidance**")
            st.write(record["llm_feedback"] or "Guidance was unavailable.")
            st.caption(f'Notes: {record["notes"] or "None"} · Version: {record["model_version"] or "Not recorded"}')
    trend = filtered.dropna(subset=["probability"]).sort_values("created_at")
    if not trend.empty:
        st.subheader("Your saved probability estimates")
        st.line_chart(trend.set_index("created_at")["probability"], color="#247ba0")
        st.caption("This displays saved model estimates over time; it is not a clinical blood-pressure trend.")
    export_df = filtered.copy()
    csv_col, json_col = st.columns(2)
    csv_col.download_button("Download filtered CSV", export_df.to_csv(index=False).encode(), "pulsewise_history.csv", "text/csv", width="stretch")
    export_records = [records[df.index.get_loc(index)] for index in filtered.index]
    json_col.download_button("Download filtered JSON", history_json(export_records), "pulsewise_history.json", "application/json", width="stretch")


def education() -> None:
    section_header("Health education", "Hypertension, in plain language", "General information to support—not replace—professional advice.")
    topics = {
        "What is hypertension?": "Hypertension means blood pressure remains higher than recommended over time. It is confirmed through appropriate measurements, not through this risk estimate.",
        "Common risk factors": "Age, family history, tobacco use, lower activity, high sodium intake, excess weight, stress, sleep, and some health conditions may be associated with risk.",
        "Why checks matter": "Hypertension may not cause obvious symptoms. Regular checks can help a qualified professional identify patterns and decide whether follow-up is needed.",
        "Everyday heart-health habits": "Balanced meals, lower-sodium choices, regular activity, adequate sleep, avoiding tobacco, and following professional advice can support heart health.",
        "Risk estimate versus diagnosis": "A model estimate uses patterns in data. A diagnosis requires clinical assessment, valid measurements, and professional judgment.",
    }
    for title, copy in topics.items():
        with st.expander(title):
            st.write(copy)
    notice("Seek urgent medical care if you experience severe chest pain, difficulty breathing, fainting, sudden weakness, confusion, or other serious symptoms. This app cannot assess emergencies.", "danger")


def privacy() -> None:
    section_header("Account & privacy", "Your information deserves care", "Plain-language limitations for this demonstration.")
    st.markdown(
        """
- Your history is queried using your authenticated username; there is no control for viewing another user's records.
- Passwords are hashed with bcrypt. Never share your password.
- SQLite supports this local demonstration. A production deployment needs managed storage, access controls, backups, encryption, monitoring, and a formal privacy review.
- Gemini receives the assessment context needed to produce guidance when configured.
- This project does not claim HIPAA, GDPR, NDPR, or other regulatory compliance.
- Avoid using the application for identifiable or emergency medical information.
"""
    )
    notice("This application is educational and does not provide diagnosis, emergency screening, or medication advice.", "warning")


if not is_authenticated():
    landing()
    st.stop()

try:
    user_logs = fetch_user_logs(st.session_state.username, limit=1000)
except Exception:
    LOGGER.exception("History loading failed")
    user_logs = []
    st.warning("Your history could not be loaded right now.")

page = sidebar()
if page == "Overview":
    overview(user_logs)
elif page == "New Assessment":
    assessment()
elif page == "Results":
    results()
elif page == "History":
    history(user_logs)
elif page == "Health Education":
    education()
else:
    privacy()
footer()
