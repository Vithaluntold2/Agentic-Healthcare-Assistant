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


# page config
st.set_page_config(
    page_title="MedAssist AI",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# clean, sharp CSS
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
    --warning-bg: #FFFBEB;
    --radius: 10px;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* main container spacing */
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 2rem;
    max-width: 900px;
}

/* sidebar */
section[data-testid="stSidebar"] {
    background: #FAFBFC;
    border-right: 1px solid var(--border);
    padding-top: 0;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* sidebar nav radio */
section[data-testid="stSidebar"] .stRadio > div {
    gap: 0px;
}
section[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent;
    border-radius: 8px;
    padding: 10px 14px;
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

/* sidebar dividers */
section[data-testid="stSidebar"] hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 16px 0;
}

/* metric cards */
div[data-testid="stMetric"] {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
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
    font-size: 26px !important;
}

/* buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    font-size: 14px;
    padding: 8px 20px;
    border: 1px solid var(--border);
    transition: all 0.12s ease;
}
.stButton > button:hover {
    border-color: var(--primary);
    color: var(--primary);
}

/* dataframes */
.stDataFrame {
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--border);
}

/* chat messages */
.stChatMessage {
    border-radius: var(--radius);
    border: 1px solid var(--border-light);
    margin-bottom: 6px;
}

/* dividers */
hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 24px 0;
}

/* expanders */
.streamlit-expanderHeader {
    font-weight: 600;
    font-size: 14px;
    color: var(--text-primary);
    border-radius: 8px;
}

/* page header styling */
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
    margin-bottom: 28px;
    line-height: 1.5;
}

/* sidebar try-these section */
.try-section-label {
    font-size: 13px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 10px;
}
.try-msg {
    font-size: 13px;
    color: var(--text-secondary);
    padding: 6px 0;
    cursor: pointer;
    line-height: 1.5;
}
.try-msg:hover {
    color: var(--primary);
}

/* sidebar brand */
.sb-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0 4px 0;
}
.sb-brand-icon {
    font-size: 22px;
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

/* info card (empty states) */
.empty-card {
    text-align: center;
    padding: 48px 24px;
    color: var(--text-secondary);
    background: var(--bg-muted);
    border-radius: var(--radius);
    border: 1px solid var(--border);
}
.empty-card .empty-icon { font-size: 32px; margin-bottom: 12px; }
.empty-card .empty-title { font-size: 15px; font-weight: 600; color: var(--text-secondary); }
.empty-card .empty-sub { font-size: 13px; color: var(--text-muted); margin-top: 4px; }

/* stat row */
.srow {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
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
    padding: 4px 10px;
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
.badge-err { background: var(--danger-bg); color: var(--danger); }
.badge-info { background: var(--primary-light); color: var(--primary); }

/* section label */
.sec-label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 20px 0 12px 0;
}

/* patient summary box */
.summary-box {
    margin-top: 10px;
    padding: 12px 16px;
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
    margin-bottom: 6px;
}
.summary-box-text {
    font-size: 13px;
    color: var(--text-primary);
    line-height: 1.6;
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

def page_header(emoji, title, subtitle=""):
    st.markdown(f"""
    <div class="page-hdr">
        <span style="font-size:24px;">{emoji}</span>
        <h1>{title}</h1>
    </div>
    {"<p class='page-sub'>" + subtitle + "</p>" if subtitle else "<div style='height:20px'></div>"}
    """, unsafe_allow_html=True)


def metric_card(emoji, label, value, color="#2563EB"):
    return f"""
    <div style="text-align:center;padding:20px 16px;background:white;
                border:1px solid #E5E7EB;border-radius:10px;">
        <div style="font-size:20px;margin-bottom:6px;">{emoji}</div>
        <div style="font-size:26px;font-weight:700;color:#111827;">{value}</div>
        <div style="font-size:11px;font-weight:500;color:#9CA3AF;text-transform:uppercase;
                    letter-spacing:.5px;margin-top:4px;">{label}</div>
    </div>
    """


def empty_state(emoji, title, sub=""):
    st.markdown(f"""
    <div class="empty-card">
        <div class="empty-icon">{emoji}</div>
        <div class="empty-title">{title}</div>
        {"<div class='empty-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """, unsafe_allow_html=True)


def section_label(emoji, text):
    st.markdown(f'<div class="sec-label"><span>{emoji}</span> {text}</div>',
                unsafe_allow_html=True)


# plotly defaults

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
    # brand
    st.markdown("""
    <div class="sb-brand">
        <span class="sb-brand-icon">⚕️</span>
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
            "💬  Chat",
            "👥  Patients",
            "🩺  Doctors",
            "📅  Appointments",
            "📊  Evaluation",
            "🧠  Memory & Logs",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # try these messages (only on chat page)
    if "Chat" in page:
        st.markdown('<div class="try-section-label">Try these messages:</div>',
                    unsafe_allow_html=True)

        examples = [
            "Search for patient Ramesh Kulkarni",
            "Find a nephrologist and show available slots",
            "Book an appointment with Dr. Priya Sharma for Ramesh Kulkarni on 2026-03-16 at 09:00",
            "Retrieve medical history for Anjali Mehra",
            "What are the latest treatments for chronic kidney disease?",
        ]
        for ex in examples:
            if st.button(f'"{ex}"', key=f"ex_{hash(ex)}", use_container_width=True):
                st.session_state.pending_query = ex

        st.markdown("---")

    # knowledge base
    if not st.session_state.vector_store_built:
        if st.button("📚  Build Knowledge Base", use_container_width=True):
            with st.spinner("Indexing patient PDFs..."):
                try:
                    build_vector_store(force_rebuild=True)
                    st.session_state.vector_store_built = True
                    st.success("Knowledge base ready")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.markdown('<span class="badge badge-ok">✓ Knowledge Base Ready</span>',
                    unsafe_allow_html=True)

    st.markdown("---")

    st.session_state.auto_evaluate = st.checkbox(
        "Auto-evaluate responses",
        value=st.session_state.auto_evaluate,
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    def _reset():
        st.session_state.agent.reset()
        st.session_state.chat_history = []
        st.session_state.vector_store_built = False

    st.button("Reset Agent Session", use_container_width=True, on_click=_reset)


# page: chat

def render_chat():
    page_header("💬", "Chat Assistant",
                "Ask about patients, book appointments, retrieve medical history, or search for disease information.")

    st.markdown("---")

    # chat history
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if "eval_scores" in entry:
                scores = entry["eval_scores"]
                cols = st.columns(5)
                for i, key in enumerate(
                    ["relevance", "accuracy", "helpfulness", "completeness", "overall"]
                ):
                    cols[i].metric(key.capitalize(), f"{scores.get(key, 0)}/5")

    pending = st.session_state.pop("pending_query", None)
    user_input = st.chat_input("Type your message here...")
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
                        for i, key in enumerate(
                            ["relevance", "accuracy", "helpfulness", "completeness", "overall"]
                        ):
                            cols[i].metric(key.capitalize(), f"{scores.get(key, 0)}/5")

                    st.session_state.chat_history.append(entry)
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_msg}
                    )


# page: patients

def render_patients():
    page_header("👥", "Patient Directory",
                "View and manage all registered patient records.")

    patients = get_all_patients()
    if not patients:
        empty_state("👥", "No patients in the system yet")
        return

    male = sum(1 for p in patients if str(p.get("gender", "")).lower() == "male")
    female = sum(1 for p in patients if str(p.get("gender", "")).lower() == "female")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_card("👥", "Total Patients", len(patients)), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("♂", "Male", male, "#3B82F6"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("♀", "Female", female, "#EC4899"), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    section_label("📋", "Patient Records")

    df = pd.DataFrame(patients)
    display_cols = ["name", "age", "gender", "phone", "address"]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available], use_container_width=True, hide_index=True)

    # age distribution
    if "age" in df.columns:
        ages = df["age"].dropna().astype(int)
        if not ages.empty:
            fig = px.histogram(ages, nbins=10,
                               labels={"value": "Age", "count": "Count"},
                               color_discrete_sequence=["#2563EB"])
            clean_layout(fig, "Patient Age Distribution")
            st.plotly_chart(fig, use_container_width=True)

    section_label("👤", "Patient Details")

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
                    <div style="display:flex;gap:8px;align-items:baseline;margin-top:6px;padding-left:8px;">
                        <span style="font-size:11px;color:#9CA3AF;">🕐 {h['date'][:19]}</span>
                        <span style="font-size:13px;color:#4B5563;">{h['note']}</span>
                    </div>
                    """, unsafe_allow_html=True)


# page: doctors

def render_doctors():
    page_header("🩺", "Doctor Directory",
                "Browse specialists and their appointment availability.")

    doctors = get_all_doctors()
    total_slots = sum(len(get_available_slots(d["doctor_id"])) for d in doctors)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(metric_card("🩺", "Doctors", len(doctors)), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("📅", "Open Slots", total_slots, "#059669"), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    for doc in doctors:
        slots = get_available_slots(doc["doctor_id"])
        with st.expander(f"{doc['name']}  ·  {doc['specialty']}  ·  {len(slots)} slots"):
            st.markdown(f"""
            <div class="srow"><span class="srow-label">Doctor ID</span><span class="srow-value">{doc['doctor_id']}</span></div>
            <div class="srow"><span class="srow-label">Specialty</span><span class="srow-value">{doc['specialty']}</span></div>
            """, unsafe_allow_html=True)
            if slots:
                pills = "".join(
                    f'<span class="slot-pill">{s["date"]} at {s["time"]}</span>'
                    for s in slots
                )
                st.markdown(f"""
                <div style="margin-top:10px;">
                    <div style="font-size:11px;font-weight:600;color:#9CA3AF;text-transform:uppercase;
                                letter-spacing:.5px;margin-bottom:6px;">Available Slots</div>
                    <div style="display:flex;flex-wrap:wrap;gap:4px;">{pills}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No available slots.")

    # specialty chart
    specialties = [d["specialty"] for d in doctors]
    spec_counts = pd.Series(specialties).value_counts()
    fig = px.pie(values=spec_counts.values, names=spec_counts.index,
                 color_discrete_sequence=["#2563EB", "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE"])
    clean_layout(fig, "Doctors by Specialty")
    fig.update_traces(textposition="inside", textinfo="label+percent")
    st.plotly_chart(fig, use_container_width=True)


# page: appointments

def render_appointments():
    page_header("📅", "Appointments",
                "Real-time appointment tracking and management.")

    appointments = get_appointments()
    if not appointments:
        empty_state("📅", "No appointments booked yet",
                    "Use the Chat Assistant to book one.")
        return

    confirmed = sum(1 for a in appointments if a["status"] == "confirmed")
    cancelled = sum(1 for a in appointments if a["status"] == "cancelled")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_card("📅", "Total", len(appointments)), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("✅", "Confirmed", confirmed, "#059669"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("❌", "Cancelled", cancelled, "#DC2626"), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

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
    page_header("📊", "Evaluation Metrics",
                "Assess response quality and track tool performance.")

    evaluator = st.session_state.evaluator

    section_label("✨", "Manual Evaluation")

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

    section_label("📈", "Aggregate Scores")

    summary = evaluator.get_evaluation_summary()

    if summary["total_evaluations"] == 0:
        empty_state("📊", "No evaluations yet",
                    "Enable auto-evaluate or use manual evaluation above.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(metric_card("⚡", "Total Evaluations",
                                    summary["total_evaluations"]), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("✅", "Valid Evaluations",
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

    section_label("🔧", "Tool Performance")

    tool_metrics = evaluator.get_tool_metrics()
    if not tool_metrics:
        empty_state("🔧", "No tool usage tracked yet")
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

    section_label("🕐", "Recent Evaluations")

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
    page_header("🧠", "Memory & Logs",
                "Inspect agent memory, planning breakdowns, and tool invocation logs.")

    agent = st.session_state.agent
    trace = agent.get_memory_trace()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(metric_card("💬", "Conversations",
                                trace["conversation_count"]), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("🧠", "Context Keys",
                                len(trace["patient_context"]), "#7C3AED"), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    if trace["patient_context"]:
        section_label("🛡️", "Patient Context")
        for key, val in trace["patient_context"].items():
            st.code(f"{key}: {val}", language="text")

    st.markdown("---")

    section_label("💬", "Recent Conversations")

    recent = trace.get("recent_history", [])
    if not recent:
        empty_state("💬", "No conversation history yet")
    else:
        for entry in recent:
            role_label = "User" if entry["role"] == "user" else "Assistant"
            with st.expander(f"{role_label}  ·  {entry['timestamp'][:19]}"):
                st.write(entry["content"])

    st.markdown("---")

    section_label("🔧", "Tool Invocation Log")

    tool_log = agent.get_tool_log()
    if not tool_log:
        empty_state("🔧", "No tools have been invoked yet")
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
    "💬  Chat": render_chat,
    "👥  Patients": render_patients,
    "🩺  Doctors": render_doctors,
    "📅  Appointments": render_appointments,
    "📊  Evaluation": render_evaluation,
    "🧠  Memory & Logs": render_memory,
}

PAGE_MAP[page]()
