# app.py - Streamlit dashboard for the healthcare assistant
# Pages: chat, patients, doctors, appointments, evaluation, memory/logs

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from src.agent import HealthcareAgent
from src.database import (
    get_all_patients, get_all_doctors, get_appointments,
    get_available_slots, find_patient,
)
from src.evaluation import AgentEvaluator
from src.rag_pipeline import build_vector_store


# Lucide SVG icons - compact helper

_ICONS = {
    "message-circle": '<path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "stethoscope": '<path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/><path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"/><circle cx="20" cy="10" r="2"/>',
    "calendar": '<path d="M8 2v4"/><path d="M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/>',
    "bar-chart": '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "brain": '<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/>',
    "heart-pulse": '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/><path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "user": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "file-text": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>',
    "sparkles": '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/><path d="M20 3v4"/><path d="M22 5h-4"/><path d="M4 17v2"/><path d="M5 18H3"/>',
    "wrench": '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>',
    "zap": '<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>',
    "check-circle": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/>',
    "calendar-check": '<path d="M8 2v4"/><path d="M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/><path d="m9 16 2 2 4-4"/>',
    "x-circle": '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    "rotate-ccw": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "trending-up": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "book-open": '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
}


def lucide(name, size=18, color="currentColor"):
    path = _ICONS.get(name, "")
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{path}</svg>'


def icon_text(name, text, size=18, color="#2563EB"):
    return f'<span style="display:inline-flex;align-items:center;gap:6px;">{lucide(name, size, color)}<span>{text}</span></span>'


# page config
st.set_page_config(
    page_title="MedAssist AI",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary: #2563EB;
    --primary-light: #EFF6FF;
    --text-primary: #111827;
    --text-secondary: #6B7280;
    --text-muted: #9CA3AF;
    --border: #E5E7EB;
    --border-light: #F3F4F6;
    --bg: #FFFFFF;
    --bg-muted: #F9FAFB;
    --success: #059669;
    --success-bg: #ECFDF5;
    --danger: #DC2626;
    --danger-bg: #FEF2F2;
    --warning: #D97706;
    --radius: 10px;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 2rem;
    max-width: 900px;
}

/* sidebar */
section[data-testid="stSidebar"] {
    background: #FAFBFC;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}
section[data-testid="stSidebar"] .stRadio > div { gap: 0px; }
section[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent;
    border-radius: 8px;
    padding: 9px 12px;
    font-weight: 500;
    font-size: 14px;
    color: var(--text-secondary);
    border: none;
    transition: all 0.12s ease;
    margin: 1px 0;
}
section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: var(--primary-light);
    color: var(--primary);
}
section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio > div [data-baseweb="radio"] input:checked ~ div {
    background: var(--primary-light);
    color: var(--primary);
    font-weight: 600;
}
section[data-testid="stSidebar"] hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 14px 0;
}

/* metric cards */
div[data-testid="stMetric"] {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
}
div[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    font-size: 24px !important;
}

/* buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    font-size: 13px;
    padding: 7px 16px;
    border: 1px solid var(--border);
    transition: all 0.12s ease;
}
.stButton > button:hover {
    border-color: var(--primary);
    color: var(--primary);
}

.stDataFrame {
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--border);
}

.stChatMessage {
    border-radius: var(--radius);
    border: 1px solid var(--border-light);
    margin-bottom: 4px;
}

hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 20px 0;
}

.streamlit-expanderHeader {
    font-weight: 600;
    font-size: 14px;
    color: var(--text-primary);
    border-radius: 8px;
}

/* page header */
.page-hdr {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 2px;
}
.page-hdr h1 {
    margin: 0;
    font-size: 22px;
    font-weight: 700;
    color: var(--text-primary);
}
.page-sub {
    color: var(--text-secondary);
    font-size: 14px;
    margin-bottom: 24px;
    line-height: 1.5;
}

/* sidebar brand */
.sb-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
}
.sb-brand-name {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
}
.sb-brand-sub {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 500;
}

/* quick-start suggestions (chat page) */
.qs-label {
    font-size: 13px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 8px;
}
.qs-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 16px;
}

/* empty state */
.empty-card {
    text-align: center;
    padding: 40px 24px;
    background: var(--bg-muted);
    border-radius: var(--radius);
    border: 1px solid var(--border);
}
.empty-card svg { margin-bottom: 10px; }
.empty-title { font-size: 14px; font-weight: 600; color: var(--text-secondary); margin-top: 4px; }
.empty-sub { font-size: 13px; color: var(--text-muted); margin-top: 4px; }

/* stat row */
.srow {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 7px 0;
    border-bottom: 1px solid var(--border-light);
    font-size: 14px;
}
.srow:last-child { border-bottom: none; }
.srow-label { color: var(--text-muted); font-weight: 500; min-width: 90px; font-size: 13px; }
.srow-value { color: var(--text-primary); font-weight: 500; }

/* slot pill */
.slot-pill {
    display: inline-block;
    background: var(--success-bg);
    border: 1px solid rgba(5,150,105,.15);
    border-radius: 6px;
    padding: 3px 9px;
    font-size: 12px;
    font-weight: 500;
    color: var(--success);
    margin: 2px;
}

/* badge */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}
.badge-ok { background: var(--success-bg); color: var(--success); }

/* section label */
.sec-label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 16px 0 10px 0;
}

/* summary box */
.summary-box {
    margin-top: 8px;
    padding: 10px 14px;
    background: var(--bg-muted);
    border-radius: 8px;
    border: 1px solid var(--border);
}
.summary-box-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: .5px;
    margin-bottom: 4px;
}
.summary-box-text {
    font-size: 13px;
    color: var(--text-primary);
    line-height: 1.5;
}

/* metric card html */
.mcard {
    text-align: center;
    padding: 18px 14px;
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
}
.mcard-val {
    font-size: 24px;
    font-weight: 700;
    color: #111827;
    margin-top: 6px;
}
.mcard-label {
    font-size: 11px;
    font-weight: 500;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: .5px;
    margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)


# session state

def init_session():
    if "agent" not in st.session_state:
        st.session_state.agent = HealthcareAgent()
    if "evaluator" not in st.session_state:
        st.session_state.evaluator = AgentEvaluator()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store_built" not in st.session_state:
        st.session_state.vector_store_built = False
    if "auto_evaluate" not in st.session_state:
        st.session_state.auto_evaluate = False

init_session()


# helpers

def page_header(icon_name, title, subtitle=""):
    st.markdown(f"""
    <div class="page-hdr">
        {lucide(icon_name, 24, "#2563EB")}
        <h1>{title}</h1>
    </div>
    {"<p class='page-sub'>" + subtitle + "</p>" if subtitle else "<div style='height:16px'></div>"}
    """, unsafe_allow_html=True)


def metric_card(icon_name, label, value, color="#2563EB"):
    return f"""
    <div class="mcard">
        {lucide(icon_name, 22, color)}
        <div class="mcard-val">{value}</div>
        <div class="mcard-label">{label}</div>
    </div>
    """


def empty_state(icon_name, title, sub=""):
    st.markdown(f"""
    <div class="empty-card">
        {lucide(icon_name, 32, "#D1D5DB")}
        <div class="empty-title">{title}</div>
        {"<div class='empty-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """, unsafe_allow_html=True)


def section_label(icon_name, text):
    st.markdown(
        f'<div class="sec-label">{lucide(icon_name, 17, "#6B7280")} {text}</div>',
        unsafe_allow_html=True,
    )


def clean_layout(fig, title=""):
    fig.update_layout(
        title=title,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter", size=12),
        title_font=dict(size=15, color="#374151"),
        xaxis=dict(gridcolor="#F3F4F6"),
        yaxis=dict(gridcolor="#F3F4F6"),
        showlegend=False,
    )
    return fig


# sidebar

with st.sidebar:
    st.markdown(f"""
    <div class="sb-brand">
        {lucide("heart-pulse", 22, "#2563EB")}
        <div>
            <div class="sb-brand-name">MedAssist AI</div>
            <div class="sb-brand-sub">Healthcare Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Chat",
            "Patients",
            "Doctors",
            "Appointments",
            "Evaluation",
            "Memory & Logs",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # knowledge base
    if not st.session_state.vector_store_built:
        if st.button("Build Knowledge Base", use_container_width=True):
            with st.spinner("Indexing patient PDFs..."):
                try:
                    build_vector_store(force_rebuild=True)
                    st.session_state.vector_store_built = True
                    st.success("Knowledge base ready")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.markdown(f"""
        <span class="badge badge-ok">{lucide("check-circle", 13, "#059669")} Knowledge Base Ready</span>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.session_state.auto_evaluate = st.checkbox(
        "Auto-evaluate responses",
        value=st.session_state.auto_evaluate,
    )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    def _reset():
        st.session_state.agent.reset()
        st.session_state.chat_history = []
        st.session_state.vector_store_built = False

    st.button("Reset Agent Session", use_container_width=True, on_click=_reset)


# page: chat

EXAMPLE_QUERIES = [
    "Search for patient Ramesh Kulkarni",
    "Find a nephrologist and show available slots",
    "Book an appointment with Dr. Priya Sharma for Ramesh Kulkarni on 2026-03-16 at 09:00",
    "Retrieve medical history for Anjali Mehra",
    "What are the latest treatments for chronic kidney disease?",
    "My 70-year-old father has chronic kidney disease. Book a nephrologist and summarize treatments.",
    "List all patients in the system",
    "Show all available doctors",
]


def render_chat():
    page_header("message-circle", "Customer Support Agent",
                "Send a message and watch the multi-agent system classify and respond.")

    st.markdown("---")

    # quick-start suggestions (compact, inside chat area)
    if not st.session_state.chat_history:
        st.markdown('<div class="qs-label">Quick-start example queries</div>',
                    unsafe_allow_html=True)
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLE_QUERIES):
            with cols[i % 2]:
                if st.button(ex, key=f"qs_{i}", use_container_width=True):
                    st.session_state.pending_query = ex
                    st.rerun()
        st.markdown("---")

    # chat history
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if "eval_scores" in entry:
                scores = entry["eval_scores"]
                cols = st.columns(5)
                for j, key in enumerate(
                    ["relevance", "accuracy", "helpfulness", "completeness", "overall"]
                ):
                    cols[j].metric(key.capitalize(), f"{scores.get(key, 0)}/5")

    pending = st.session_state.pop("pending_query", None)
    user_input = st.chat_input("Type your healthcare query here...")
    query = pending or user_input

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    response = st.session_state.agent.chat(query)
                    st.markdown(response)
                    entry = {"role": "assistant", "content": response}

                    if st.session_state.auto_evaluate:
                        eval_result = st.session_state.evaluator.evaluate_response(query, response)
                        scores = eval_result.get("scores", {})
                        entry["eval_scores"] = scores
                        for tl in st.session_state.agent.get_tool_log()[-5:]:
                            st.session_state.evaluator.log_tool_usage(tl["tool"], True)
                        cols = st.columns(5)
                        for j, key in enumerate(
                            ["relevance", "accuracy", "helpfulness", "completeness", "overall"]
                        ):
                            cols[j].metric(key.capitalize(), f"{scores.get(key, 0)}/5")

                    st.session_state.chat_history.append(entry)
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_msg}
                    )


# page: patients

def render_patients():
    page_header("users", "Patient Directory",
                "View and manage all registered patient records.")

    patients = get_all_patients()
    if not patients:
        empty_state("users", "No patients in the system yet")
        return

    male = sum(1 for p in patients if str(p.get("gender", "")).lower() == "male")
    female = sum(1 for p in patients if str(p.get("gender", "")).lower() == "female")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_card("users", "Total Patients", len(patients)), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("user", "Male", male, "#3B82F6"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("user", "Female", female, "#EC4899"), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    section_label("file-text", "Patient Records")

    df = pd.DataFrame(patients)
    display_cols = ["name", "age", "gender", "phone", "address"]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available], use_container_width=True, hide_index=True)

    if "age" in df.columns:
        ages = df["age"].dropna().astype(int)
        if not ages.empty:
            fig = px.histogram(ages, nbins=10,
                               labels={"value": "Age", "count": "Count"},
                               color_discrete_sequence=["#2563EB"])
            clean_layout(fig, "Patient Age Distribution")
            st.plotly_chart(fig, use_container_width=True)

    section_label("user", "Patient Details")

    for pat in patients:
        with st.expander(f"{pat['name']}  —  {pat.get('phone', '')}"):
            c1, c2 = st.columns(2)
            c1.markdown(f"""
            <div class="srow"><span class="srow-label">Age</span><span class="srow-value">{pat.get('age', 'N/A')}</span></div>
            <div class="srow"><span class="srow-label">Gender</span><span class="srow-value">{pat.get('gender', 'N/A')}</span></div>
            """, unsafe_allow_html=True)
            c2.markdown(f"""
            <div class="srow"><span class="srow-label">Email</span><span class="srow-value">{pat.get('email') or 'N/A'}</span></div>
            <div class="srow"><span class="srow-label">Address</span><span class="srow-value">{pat.get('address', 'N/A')}</span></div>
            """, unsafe_allow_html=True)

            summary = pat.get("summary") or "No summary available."
            st.markdown(f"""
            <div class="summary-box">
                <div class="summary-box-label">Medical Summary</div>
                <div class="summary-box-text">{summary}</div>
            </div>
            """, unsafe_allow_html=True)

            if pat.get("history"):
                for h in pat["history"]:
                    st.markdown(f"""
                    <div style="display:flex;gap:8px;align-items:baseline;margin-top:5px;padding-left:6px;">
                        {lucide("clock", 12, "#9CA3AF")}
                        <span style="font-size:11px;color:#9CA3AF;">{h['date'][:19]}</span>
                        <span style="font-size:13px;color:#4B5563;">{h['note']}</span>
                    </div>
                    """, unsafe_allow_html=True)


# page: doctors

def render_doctors():
    page_header("stethoscope", "Doctor Directory",
                "Browse specialists and their appointment availability.")

    doctors = get_all_doctors()
    total_slots = sum(len(get_available_slots(d["doctor_id"])) for d in doctors)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(metric_card("stethoscope", "Doctors", len(doctors)), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("calendar", "Open Slots", total_slots, "#059669"), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    for doc in doctors:
        slots = get_available_slots(doc["doctor_id"])
        with st.expander(f"{doc['name']}  ·  {doc['specialty']}  ·  {len(slots)} slots"):
            st.markdown(f"""
            <div class="srow"><span class="srow-label">Doctor ID</span><span class="srow-value">{doc['doctor_id']}</span></div>
            <div class="srow"><span class="srow-label">Specialty</span><span class="srow-value">{doc['specialty']}</span></div>
            """, unsafe_allow_html=True)
            if slots:
                pills = "".join(
                    f'<span class="slot-pill">{lucide("clock", 11, "#059669")} {s["date"]} at {s["time"]}</span>'
                    for s in slots
                )
                st.markdown(f"""
                <div style="margin-top:8px;">
                    <div style="font-size:11px;font-weight:600;color:#9CA3AF;text-transform:uppercase;
                                letter-spacing:.5px;margin-bottom:4px;">Available Slots</div>
                    <div style="display:flex;flex-wrap:wrap;gap:3px;">{pills}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No available slots.")

    specialties = [d["specialty"] for d in doctors]
    spec_counts = pd.Series(specialties).value_counts()
    fig = px.pie(values=spec_counts.values, names=spec_counts.index,
                 color_discrete_sequence=["#2563EB", "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE"])
    clean_layout(fig, "Doctors by Specialty")
    fig.update_traces(textposition="inside", textinfo="label+percent")
    st.plotly_chart(fig, use_container_width=True)


# page: appointments

def render_appointments():
    page_header("calendar", "Appointments",
                "Real-time appointment tracking and management.")

    appointments = get_appointments()
    if not appointments:
        empty_state("calendar", "No appointments booked yet",
                    "Use the Chat Assistant to book one.")
        return

    confirmed = sum(1 for a in appointments if a["status"] == "confirmed")
    cancelled = sum(1 for a in appointments if a["status"] == "cancelled")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_card("calendar", "Total", len(appointments)), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("calendar-check", "Confirmed", confirmed, "#059669"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("x-circle", "Cancelled", cancelled, "#DC2626"), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    df = pd.DataFrame(appointments)
    display_cols = ["appointment_id", "doctor_name", "specialty", "date", "time", "status", "patient_id"]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available], use_container_width=True, hide_index=True)

    if len(appointments) > 0:
        status_counts = df["status"].value_counts()
        fig = px.bar(x=status_counts.index, y=status_counts.values,
                     labels={"x": "Status", "y": "Count"},
                     color=status_counts.index,
                     color_discrete_map={"confirmed": "#059669", "cancelled": "#DC2626"})
        clean_layout(fig, "Appointment Status Breakdown")
        st.plotly_chart(fig, use_container_width=True)


# page: evaluation

def render_evaluation():
    page_header("bar-chart", "Evaluation Metrics",
                "Assess response quality and track tool performance.")

    evaluator = st.session_state.evaluator

    section_label("sparkles", "Manual Evaluation")

    with st.form("eval_form"):
        eval_query = st.text_input("Query to evaluate")
        eval_response = st.text_area("Response to evaluate")
        submitted = st.form_submit_button("Run Evaluation")

        if submitted and eval_query and eval_response:
            with st.spinner("Evaluating..."):
                result = evaluator.evaluate_response(eval_query, eval_response)
                scores = result.get("scores", {})
                cols = st.columns(5)
                for i, key in enumerate(
                    ["relevance", "accuracy", "helpfulness", "completeness", "overall"]
                ):
                    cols[i].metric(key.capitalize(), f"{scores.get(key, 0)}/5")
                if scores.get("feedback"):
                    st.info(f"**Feedback:** {scores['feedback']}")

    st.markdown("---")

    section_label("trending-up", "Aggregate Scores")

    summary = evaluator.get_evaluation_summary()

    if summary["total_evaluations"] == 0:
        empty_state("bar-chart", "No evaluations yet",
                    "Enable auto-evaluate or use manual evaluation above.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(metric_card("zap", "Total Evaluations",
                                    summary["total_evaluations"]), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("check-circle", "Valid Evaluations",
                                    summary["valid_evaluations"], "#059669"), unsafe_allow_html=True)

        avg = summary.get("avg_scores", {})
        if avg:
            fig = go.Figure(data=[go.Bar(
                x=list(avg.keys()), y=list(avg.values()),
                marker_color=["#2563EB", "#059669", "#D97706", "#7C3AED", "#DC2626"],
                marker_line_width=0,
            )])
            clean_layout(fig, "Average Evaluation Scores")
            fig.update_layout(yaxis=dict(range=[0, 5], gridcolor="#F3F4F6"),
                              xaxis_title="Criteria", yaxis_title="Score (1-5)")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    section_label("wrench", "Tool Performance")

    tool_metrics = evaluator.get_tool_metrics()
    if not tool_metrics:
        empty_state("wrench", "No tool usage tracked yet")
    else:
        tool_df = pd.DataFrame([
            {"Tool": name, **data} for name, data in tool_metrics.items()
        ])
        st.dataframe(tool_df, use_container_width=True, hide_index=True)

        fig = px.bar(tool_df, x="Tool", y="success_rate",
                     color="success_rate",
                     color_continuous_scale=["#DC2626", "#D97706", "#059669"],
                     range_color=[0, 100])
        clean_layout(fig, "Tool Success Rate (%)")
        fig.update_layout(yaxis=dict(gridcolor="#F3F4F6"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    section_label("clock", "Recent Evaluations")

    recent = evaluator.get_recent_evaluations()
    if recent:
        for entry in reversed(recent):
            overall = entry["scores"].get("overall", 0)
            with st.expander(f"{entry['timestamp'][:19]}  ·  Overall: {overall}/5"):
                st.write(f"**Query:** {entry['query']}")
                st.write(f"**Response:** {entry['response_preview']}...")
                scores = entry["scores"]
                cols = st.columns(5)
                for i, key in enumerate(
                    ["relevance", "accuracy", "helpfulness", "completeness", "overall"]
                ):
                    cols[i].metric(key.capitalize(), f"{scores.get(key, 0)}/5")
                if scores.get("feedback"):
                    st.info(scores["feedback"])
    else:
        st.caption("No evaluations recorded yet.")


# page: memory & logs

def render_memory():
    page_header("brain", "Memory & Logs",
                "Inspect agent memory, planning breakdowns, and tool invocation logs.")

    agent = st.session_state.agent
    trace = agent.get_memory_trace()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(metric_card("message-circle", "Conversations",
                                trace["conversation_count"]), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("brain", "Context Keys",
                                len(trace["patient_context"]), "#7C3AED"), unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    if trace["patient_context"]:
        section_label("shield", "Patient Context")
        for key, val in trace["patient_context"].items():
            st.code(f"{key}: {val}", language="text")

    st.markdown("---")

    section_label("message-circle", "Recent Conversations")

    recent = trace.get("recent_history", [])
    if not recent:
        empty_state("message-circle", "No conversation history yet")
    else:
        for entry in recent:
            role_label = "User" if entry["role"] == "user" else "Assistant"
            with st.expander(f"{role_label}  ·  {entry['timestamp'][:19]}"):
                st.write(entry["content"])

    st.markdown("---")

    section_label("wrench", "Tool Invocation Log")

    tool_log = agent.get_tool_log()
    if not tool_log:
        empty_state("wrench", "No tools have been invoked yet")
    else:
        for entry in reversed(tool_log[-20:]):
            with st.expander(f"{entry['tool']}  ·  {entry['timestamp'][:19]}"):
                st.json(entry["args"])

        tool_names = [t["tool"] for t in tool_log]
        tool_counts = pd.Series(tool_names).value_counts()
        fig = px.bar(x=tool_counts.index, y=tool_counts.values,
                     labels={"x": "Tool", "y": "Invocations"},
                     color_discrete_sequence=["#2563EB"])
        clean_layout(fig, "Tool Usage Frequency")
        st.plotly_chart(fig, use_container_width=True)


# router

PAGE_MAP = {
    "Chat": render_chat,
    "Patients": render_patients,
    "Doctors": render_doctors,
    "Appointments": render_appointments,
    "Evaluation": render_evaluation,
    "Memory & Logs": render_memory,
}

PAGE_MAP[page]()
