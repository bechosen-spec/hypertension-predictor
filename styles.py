from __future__ import annotations

import streamlit as st


CSS = """
<style>
:root {
  --navy:#102a43; --blue:#247ba0; --teal:#168c83; --green:#2f855a;
  --soft:#f4f8fb; --line:#dce8ef; --muted:#52677a; --amber:#b7791f; --red:#b42318;
}
.stApp { background:linear-gradient(180deg,#fbfdff 0%,#f7fafc 100%); color:var(--navy); }
.block-container { max-width:1180px; padding-top:1.4rem; padding-bottom:3rem; }
h1,h2,h3 { color:var(--navy); letter-spacing:-.02em; }
[data-testid="stSidebar"] { background:#f2f7fa; border-right:1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top:1.25rem; }
.hero { padding:2.2rem; border:1px solid var(--line); border-radius:24px;
  background:linear-gradient(135deg,#fff 5%,#edf8fb 100%); box-shadow:0 16px 40px rgba(16,42,67,.08); }
.eyebrow { color:var(--teal); font-size:.76rem; font-weight:800; letter-spacing:.11em; text-transform:uppercase; }
.hero h1 { font-size:clamp(2.1rem,5vw,3.8rem); line-height:1.04; margin:.45rem 0 1rem; }
.hero p { color:var(--muted); font-size:1.08rem; max-width:42rem; line-height:1.7; }
.surface,.metric-card,.factor-card,.recommendation { background:#fff; border:1px solid var(--line);
  border-radius:18px; padding:1.25rem; box-shadow:0 8px 24px rgba(16,42,67,.055); height:100%; }
.metric-label { color:var(--muted); font-size:.82rem; font-weight:700; }
.metric-value { color:var(--navy); font-size:1.55rem; font-weight:800; margin-top:.25rem; }
.section-kicker { color:var(--teal); font-size:.75rem; font-weight:800; text-transform:uppercase; letter-spacing:.1em; }
.section-title { font-size:1.65rem; font-weight:800; margin:.2rem 0 .4rem; }
.section-copy { color:var(--muted); margin-bottom:1.1rem; }
.notice { border-left:4px solid var(--blue); background:#edf7fc; padding:1rem 1.1rem; border-radius:10px; color:#274c66; }
.notice.warning { border-left-color:var(--amber); background:#fff8e8; color:#6b4c13; }
.notice.danger { border-left-color:var(--red); background:#fff3f2; color:#78261f; }
.risk-pill { display:inline-block; padding:.38rem .7rem; border-radius:999px; font-weight:800; font-size:.82rem; }
.risk-low { color:#246b49; background:#e8f7ef; }.risk-moderate { color:#805b10; background:#fff4d6; }
.risk-high { color:#922b21; background:#fdeceb; }
.prob-track { width:100%; height:14px; border-radius:999px; background:#e6eef3; overflow:hidden; margin:.7rem 0; }
.prob-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,#1d9b88,#efb34c,#d35d52); }
.step-line { display:flex; gap:.45rem; align-items:center; margin:.2rem 0 1.35rem; }
.step-dot { flex:1; height:6px; border-radius:99px; background:#dce8ef; }
.step-dot.active { background:var(--teal); }
.recommendation { border-left:4px solid var(--teal); margin-bottom:.65rem; padding:1rem 1.1rem; }
.recommendation strong { color:var(--teal); margin-right:.45rem; }
.small-muted { color:var(--muted); font-size:.85rem; }
.footer { margin-top:3rem; padding-top:1.25rem; border-top:1px solid var(--line); color:var(--muted); font-size:.82rem; }
#MainMenu, footer, [data-testid="stAppDeployButton"] { visibility:hidden; }
div.stButton > button, div.stDownloadButton > button { border-radius:10px; min-height:2.75rem; font-weight:750; }
div[data-testid="stForm"] { background:#fff; border:1px solid var(--line); border-radius:18px; padding:1.2rem; }
div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); padding:1rem; border-radius:15px; }
@media (max-width:700px) {
  .block-container { padding:1rem .85rem 2rem; }
  .hero { padding:1.35rem; border-radius:18px; }
  .hero h1 { font-size:2.15rem; }
}
</style>
"""


def apply_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
