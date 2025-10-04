# pages/1_🗂️_Dashboard.py
import json
import pandas as pd
import streamlit as st

from db import fetch_logs, to_dict

st.set_page_config(page_title="Prediction Dashboard", page_icon="📊", layout="wide")
st.title("📊 Prediction & Feedback Dashboard")

# ───────────────────────────────────────────────────────────
# Auth check (uses session set by app.py / streamlit-authenticator)
# ───────────────────────────────────────────────────────────
auth_ok = st.session_state.get("authentication_status", None)
username = st.session_state.get("username", None)

if not auth_ok:
    st.info("Please log in from the main app to view the dashboard.")
    st.stop()

# ───────────────────────────────────────────────────────────
# Sidebar filters
# ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    only_me = st.checkbox("Show only my logs", value=False)
    limit = st.slider("Max rows", min_value=50, max_value=2000, value=300, step=50)
    pred_filter = st.selectbox(
        "Prediction",
        ["All", "Has Hypertension", "No Hypertension"],
        index=0,
    )
    st.caption("Tip: Use the table's built-in column filters for interactive search.")

# ───────────────────────────────────────────────────────────
# Load & filter data
# ───────────────────────────────────────────────────────────
rows = fetch_logs(limit=limit, username=username if only_me else None)

if not rows:
    st.info("No logs yet. Run a prediction in the main app first.")
    st.stop()

records = [to_dict(r) for r in rows]
df = pd.DataFrame.from_records(records)

# Convert timestamp for nicer display
if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"])

# Apply simple prediction filter
if pred_filter != "All":
    df = df[df["prediction"] == pred_filter]

if df.empty:
    st.info("No rows match your current filters.")
    st.stop()

# ───────────────────────────────────────────────────────────
# KPIs
# ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
total_runs = len(df)
pos_rate = (
    (df["prediction"] == "Has Hypertension").mean() * 100
    if "prediction" in df.columns and total_runs > 0
    else 0.0
)
avg_proba = (
    df["probability"].dropna().mean() if "probability" in df.columns else None
)

with c1:
    st.metric("Total Logs", f"{total_runs:,}")
with c2:
    st.metric("Hypertension (Has) %", f"{pos_rate:.1f}%")
with c3:
    st.metric("Avg. P(Has)", f"{avg_proba:.2f}" if avg_proba is not None else "—")
with c4:
    st.metric("Users", f"{df['username'].nunique()}")

st.write("---")

# ───────────────────────────────────────────────────────────
# Summary table
# ───────────────────────────────────────────────────────────
summary_cols = [c for c in ["id", "created_at", "username", "prediction", "probability"] if c in df.columns]
st.subheader("Recent Predictions")
st.dataframe(
    df[summary_cols].sort_values("created_at", ascending=False),
    use_container_width=True,
    hide_index=True,
)

# Quick distribution chart
if "prediction" in df.columns:
    st.write("")
    st.bar_chart(df["prediction"].value_counts())

st.write("---")

# ───────────────────────────────────────────────────────────
# Detailed inspector
# ───────────────────────────────────────────────────────────
st.subheader("Inspect a Log")
# Build a label like "#123 • 2025-10-04 11:30 • user • Has"
def _label_row(r):
    ts = pd.to_datetime(r["created_at"]).strftime("%Y-%m-%d %H:%M") if r.get("created_at") else "—"
    return f"#{r.get('id','?')} • {ts} • {r.get('username','—')} • {r.get('prediction','—')}"

options = [{k: v for k, v in rec.items()} for rec in records]  # keep raw dicts
labels = [_label_row(rec) for rec in options]

sel = st.selectbox("Pick a log entry", options=range(len(options)), format_func=lambda i: labels[i])
row = options[sel]

cA, cB = st.columns([1, 1])
with cA:
    st.markdown("**Patient Input**")
    st.json(row.get("patient", {}), expanded=False)

with cB:
    st.markdown("**Top Features (this case)**")
    st.json(row.get("top_features", {}) or {}, expanded=False)

st.markdown("**Personalized Feedback**")
st.write(row.get("llm_feedback", "") or "_(no feedback logged)_")


# ───────────────────────────────────────────────────────────
# Export
# ───────────────────────────────────────────────────────────
st.write("---")
exp_c1, exp_c2 = st.columns([1, 1])

csv_bytes = df.to_csv(index=False).encode("utf-8")
with exp_c1:
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="prediction_logs.csv",
        mime="text/csv",
        use_container_width=True,
    )

# JSON export (indent for readability)
json_str = json.dumps(records, indent=2, default=str)
with exp_c2:
    st.download_button(
        "Download JSON",
        data=json_str.encode("utf-8"),
        file_name="prediction_logs.json",
        mime="application/json",
        use_container_width=True,
    )

st.caption("Note: This dashboard shows stored predictions for analysis and auditing. Handle data responsibly.")
