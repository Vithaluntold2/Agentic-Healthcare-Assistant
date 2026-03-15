# app.py - Streamlit dashboard for the healthcare assistant
# Has 6 pages: chat, patients, doctors, appointments, evaluation, memory/logs

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


# inline SVG icons (Lucide) so we don't need any external icon library

def icon(name: str, size: int = 20, color: str = "currentColor") -> str:
    """Returns inline SVG markup for the given icon name."""
    icons = {
        "message-circle": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z"/></svg>',
        "users": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
        "stethoscope": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/><path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"/><circle cx="20" cy="10" r="2"/></svg>',
        "calendar": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 2v4"/><path d="M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/></svg>',
        "bar-chart-3": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>',
        "brain": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/><path d="M17.599 6.5a3 3 0 0 0 .399-1.375"/><path d="M6.003 5.125A3 3 0 0 0 6.401 6.5"/><path d="M3.477 10.896a4 4 0 0 1 .585-.396"/><path d="M19.938 10.5a4 4 0 0 1 .585.396"/><path d="M6 18a4 4 0 0 1-1.967-.516"/><path d="M19.967 17.484A4 4 0 0 1 18 18"/></svg>',
        "activity": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
        "heart-pulse": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/><path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27"/></svg>',
        "database": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/></svg>',
        "rotate-ccw": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>',
        "check-circle": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/></svg>',
        "search": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>',
        "clock": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
        "user": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
        "file-text": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/></svg>',
        "shield": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/></svg>',
        "sparkles": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/><path d="M20 3v4"/><path d="M22 5h-4"/><path d="M4 17v2"/><path d="M5 18H3"/></svg>',
        "wrench": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>',
        "zap": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/></svg>',
        "calendar-check": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 2v4"/><path d="M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/><path d="m9 16 2 2 4-4"/></svg>',
        "x-circle": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/></svg>',
        "info": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>',
    }
    return icons.get(name, "")


def icon_label(icon_name: str, text: str, size: int = 18,
               color: str = "#0F6FFF") -> str:
    """Icon + text label as an inline-flex span."""
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;">'
        f'{icon(icon_name, size, color)}'
        f'<span>{text}</span></span>'
    )


# page config

st.set_page_config(
    page_title="MedAssist AI – Healthcare Assistant",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Global CSS – Professional Light Theme

st.markdown("""
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Root Variables */
:root {
    --primary: #0F6FFF;
    --primary-light: #EBF2FF;
    --primary-dark: #0A4FBF;
    --success: #10B981;
    --success-light: #ECFDF5;
    --warning: #F59E0B;
    --warning-light: #FFFBEB;
    --danger: #EF4444;
    --danger-light: #FEF2F2;
    --gray-50: #F9FAFB;
    --gray-100: #F3F4F6;
    --gray-200: #E5E7EB;
    --gray-300: #D1D5DB;
    --gray-400: #9CA3AF;
    --gray-500: #6B7280;
    --gray-600: #4B5563;
    --gray-700: #374151;
    --gray-800: #1F2937;
    --gray-900: #111827;
    --radius: 12px;
    --radius-sm: 8px;
    --shadow-sm: 0 1px 2px rgba(0,0,0,.04);
    --shadow: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
    --shadow-md: 0 4px 6px -1px rgba(0,0,0,.07), 0 2px 4px -2px rgba(0,0,0,.05);
}

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 Roboto, sans-serif !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FAFBFD 0%, #F0F2F7 100%);
    border-right: 1px solid var(--gray-200);
}
section[data-testid="stSidebar"] .stRadio > div {
    gap: 2px;
}
section[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent;
    border-radius: var(--radius-sm);
    padding: 10px 14px;
    font-weight: 500;
    font-size: 14px;
    transition: all 0.15s ease;
    border: 1px solid transparent;
    color: var(--gray-600);
}
section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: var(--primary-light);
    color: var(--primary);
    border-color: rgba(15, 111, 255, .15);
}
section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio > div [data-baseweb="radio"] input:checked ~ div {
    background: var(--primary-light);
    color: var(--primary);
    border-color: rgba(15, 111, 255, .2);
}

/* Metric Cards */
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--gray-200);
    border-radius: var(--radius);
    padding: 20px 24px;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-md);
}
div[data-testid="stMetric"] label {
    color: var(--gray-500) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--gray-900) !important;
    font-weight: 700 !important;
    font-size: 28px !important;
}

/* Buttons */
.stButton > button {
    border-radius: var(--radius-sm);
    font-weight: 500;
    font-size: 14px;
    padding: 8px 20px;
    border: 1px solid var(--gray-200);
    transition: all 0.15s ease;
}
.stButton > button:hover {
    border-color: var(--primary);
    color: var(--primary);
    box-shadow: var(--shadow);
}

/* Data frames */
.stDataFrame {
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--gray-200);
}

/* Expanders */
.streamlit-expanderHeader {
    font-weight: 600;
    font-size: 14px;
    color: var(--gray-700);
    border-radius: var(--radius-sm);
}

/* Chat Messages */
.stChatMessage {
    border-radius: var(--radius);
    border: 1px solid var(--gray-100);
    box-shadow: var(--shadow-sm);
    margin-bottom: 8px;
}

/* Dividers */
hr {
    border: none;
    border-top: 1px solid var(--gray-200);
    margin: 20px 0;
}

/* Custom card class */
.card {
    background: white;
    border: 1px solid var(--gray-200);
    border-radius: var(--radius);
    padding: 24px;
    box-shadow: var(--shadow-sm);
    margin-bottom: 16px;
}
.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--gray-100);
}
.card-header h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: var(--gray-800);
}

/* Badge */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-success {
    background: var(--success-light);
    color: var(--success);
}
.badge-danger {
    background: var(--danger-light);
    color: var(--danger);
}
.badge-primary {
    background: var(--primary-light);
    color: var(--primary);
}
.badge-warning {
    background: var(--warning-light);
    color: var(--warning);
}

/* Stat Row */
.stat-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid var(--gray-100);
}
.stat-row:last-child { border-bottom: none; }
.stat-label {
    color: var(--gray-500);
    font-size: 13px;
    font-weight: 500;
    min-width: 100px;
}
.stat-value {
    color: var(--gray-800);
    font-size: 14px;
    font-weight: 500;
}

/* Page header */
.page-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 4px;
}
.page-header h1 {
    margin: 0;
    font-size: 24px;
    font-weight: 700;
    color: var(--gray-900);
}
.page-subtitle {
    color: var(--gray-500);
    font-size: 14px;
    margin-bottom: 24px;
}

/* Patient card */
.patient-card {
    background: white;
    border: 1px solid var(--gray-200);
    border-radius: var(--radius);
    padding: 20px;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s ease;
}
.patient-card:hover {
    box-shadow: var(--shadow-md);
}
.patient-name {
    font-size: 16px;
    font-weight: 600;
    color: var(--gray-800);
    margin-bottom: 8px;
}
.patient-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin-bottom: 10px;
}
.patient-meta-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    color: var(--gray-500);
}

/* Doctor slot pill */
.slot-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: var(--success-light);
    border: 1px solid rgba(16,185,129,.2);
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 500;
    color: var(--success);
    margin: 3px;
}

/* Sidebar brand */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0 16px 0;
}
.sidebar-brand-text {
    font-size: 18px;
    font-weight: 700;
    color: var(--gray-900);
}
.sidebar-brand-sub {
    font-size: 11px;
    color: var(--gray-400);
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Scroll & General */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)


# Session State

def init_session_state():
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

init_session_state()


# Sidebar

with st.sidebar:
    # Brand
    st.markdown(f"""
    <div class="sidebar-brand">
        {icon("heart-pulse", 28, "#0F6FFF")}
        <div>
            <div class="sidebar-brand-text">MedAssist AI</div>
            <div class="sidebar-brand-sub">Healthcare Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Chat Assistant",
            "Patient Directory",
            "Doctor Directory",
            "Appointments",
            "Evaluation Metrics",
            "Memory & Logs",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Knowledge base
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
        {icon("database", 16, "#6B7280")}
        <span style="font-size:12px;font-weight:600;color:#6B7280;text-transform:uppercase;letter-spacing:.5px;">Knowledge Base</span>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.vector_store_built:
        if st.button("Build Knowledge Base", use_container_width=True):
            with st.spinner("Indexing patient PDFs into FAISS..."):
                try:
                    build_vector_store(force_rebuild=True)
                    st.session_state.vector_store_built = True
                    st.success("Knowledge base ready")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.markdown(f"""
        <div class="badge badge-success" style="margin-bottom:12px;">
            {icon("check-circle", 14, "#10B981")} Indexed & Ready
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.session_state.auto_evaluate = st.checkbox(
        "Auto-evaluate responses",
        value=st.session_state.auto_evaluate,
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    def _reset_agent():
        st.session_state.agent.reset()
        st.session_state.chat_history = []
        st.session_state.vector_store_built = False

    st.button("Reset Agent Session", use_container_width=True,
              on_click=_reset_agent)


# Helper: Render page header

def page_header(icon_name: str, title: str, subtitle: str = ""):
    st.markdown(f"""
    <div class="page-header">
        {icon(icon_name, 26, "#0F6FFF")}
        <h1>{title}</h1>
    </div>
    {"<p class='page-subtitle'>" + subtitle + "</p>" if subtitle else ""}
    """, unsafe_allow_html=True)


# Helper: Metric card with icon

def metric_card_html(icon_name: str, label: str, value, color: str = "#0F6FFF"):
    return f"""
    <div class="card" style="text-align:center;padding:20px 16px;">
        <div style="margin-bottom:8px;">{icon(icon_name, 24, color)}</div>
        <div style="font-size:28px;font-weight:700;color:#1F2937;">{value}</div>
        <div style="font-size:12px;font-weight:500;color:#9CA3AF;text-transform:uppercase;letter-spacing:.5px;margin-top:4px;">{label}</div>
    </div>
    """


# Page: Chat Assistant

def render_chat_page():
    page_header("message-circle", "Chat Assistant",
                "Ask about patients, book appointments, retrieve medical history, or search for disease information.")

    # Example queries
    with st.expander("Quick-start example queries", expanded=False):
        examples = [
            "Search for patient Ramesh Kulkarni",
            "Find a nephrologist and show available slots",
            "Book an appointment with Dr. Priya Sharma for Ramesh Kulkarni on 2026-03-16 at 09:00",
            "Retrieve medical history for Anjali Mehra",
            "What are the latest treatments for chronic kidney disease?",
            "My 70-year-old father has chronic kidney disease. Book a nephrologist and summarize treatments.",
            "List all patients in the system",
            "Show all available doctors",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{hash(ex)}"):
                st.session_state.pending_query = ex

    # Chat history
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
                        for i, key in enumerate(
                            ["relevance", "accuracy", "helpfulness", "completeness", "overall"]
                        ):
                            cols[i].metric(key.capitalize(), f"{scores.get(key, 0)}/5")

                    st.session_state.chat_history.append(entry)
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})


# Page: Patient Directory

def render_patient_page():
    page_header("users", "Patient Directory", "View and manage all registered patient records.")

    patients = get_all_patients()
    if not patients:
        st.info("No patients in the system yet.")
        return

    male_count = sum(1 for p in patients if str(p.get("gender", "")).lower() == "male")
    female_count = sum(1 for p in patients if str(p.get("gender", "")).lower() == "female")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_card_html("users", "Total Patients", len(patients), "#0F6FFF"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card_html("user", "Male", male_count, "#3B82F6"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card_html("user", "Female", female_count, "#EC4899"), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Table
    st.markdown(f"""
    <div class="card">
        <div class="card-header">
            {icon("file-text", 18, "#6B7280")}
            <h3>Patient Records</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)

    df = pd.DataFrame(patients)
    display_cols = ["name", "age", "gender", "phone", "address"]
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available_cols], use_container_width=True, hide_index=True)

    # Age distribution
    if "age" in df.columns:
        ages = df["age"].dropna().astype(int)
        if not ages.empty:
            fig = px.histogram(
                ages, nbins=10,
                title="Patient Age Distribution",
                labels={"value": "Age", "count": "Count"},
                color_discrete_sequence=["#0F6FFF"],
            )
            fig.update_layout(
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Inter", size=12),
                title_font=dict(size=16, color="#374151"),
                xaxis=dict(gridcolor="#F3F4F6"),
                yaxis=dict(gridcolor="#F3F4F6"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Patient detail cards
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin:20px 0 12px 0;">
        {icon("user", 18, "#6B7280")}
        <span style="font-size:16px;font-weight:600;color:#374151;">Patient Details</span>
    </div>
    """, unsafe_allow_html=True)

    for pat in patients:
        with st.expander(f"{pat['name']}  —  {pat.get('phone', '')}"):
            c1, c2 = st.columns(2)
            c1.markdown(f"""
            <div class="stat-row"><span class="stat-label">Age</span><span class="stat-value">{pat.get('age', 'N/A')}</span></div>
            <div class="stat-row"><span class="stat-label">Gender</span><span class="stat-value">{pat.get('gender', 'N/A')}</span></div>
            """, unsafe_allow_html=True)
            c2.markdown(f"""
            <div class="stat-row"><span class="stat-label">Email</span><span class="stat-value">{pat.get('email') or 'N/A'}</span></div>
            <div class="stat-row"><span class="stat-label">Address</span><span class="stat-value">{pat.get('address', 'N/A')}</span></div>
            """, unsafe_allow_html=True)
            summary = pat.get("summary") or "No summary available."
            st.markdown(f"""
            <div style="margin-top:12px;padding:12px 16px;background:#F9FAFB;border-radius:8px;border:1px solid #E5E7EB;">
                <div style="font-size:12px;font-weight:600;color:#9CA3AF;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;">Medical Summary</div>
                <div style="font-size:13px;color:#374151;line-height:1.6;">{summary}</div>
            </div>
            """, unsafe_allow_html=True)
            if pat.get("history"):
                for h in pat["history"]:
                    st.markdown(f"""
                    <div style="display:flex;gap:8px;align-items:baseline;margin-top:6px;padding-left:8px;">
                        {icon("clock", 12, "#9CA3AF")}
                        <span style="font-size:11px;color:#9CA3AF;">{h['date'][:19]}</span>
                        <span style="font-size:13px;color:#4B5563;">{h['note']}</span>
                    </div>
                    """, unsafe_allow_html=True)


# Page: Doctor Directory

def render_doctor_page():
    page_header("stethoscope", "Doctor Directory", "Browse specialists and their appointment availability.")

    doctors = get_all_doctors()
    total_slots = sum(len(get_available_slots(d["doctor_id"])) for d in doctors)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(metric_card_html("stethoscope", "Doctors", len(doctors), "#0F6FFF"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card_html("calendar", "Open Slots", total_slots, "#10B981"), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    for doc in doctors:
        slots = get_available_slots(doc["doctor_id"])
        with st.expander(f"{doc['name']}  ·  {doc['specialty']}  ·  {len(slots)} slots open"):
            st.markdown(f"""
            <div class="stat-row"><span class="stat-label">Doctor ID</span><span class="stat-value">{doc['doctor_id']}</span></div>
            <div class="stat-row"><span class="stat-label">Specialty</span><span class="stat-value">{doc['specialty']}</span></div>
            """, unsafe_allow_html=True)
            if slots:
                slots_html = "".join(
                    f'<span class="slot-pill">{icon("clock", 12, "#10B981")} {s["date"]} at {s["time"]}</span>'
                    for s in slots
                )
                st.markdown(f"""
                <div style="margin-top:12px;">
                    <div style="font-size:12px;font-weight:600;color:#9CA3AF;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;">Available Slots</div>
                    <div style="display:flex;flex-wrap:wrap;gap:4px;">{slots_html}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No available slots.")

    # Specialty chart
    specialties = [d["specialty"] for d in doctors]
    spec_counts = pd.Series(specialties).value_counts()
    fig = px.pie(
        values=spec_counts.values,
        names=spec_counts.index,
        title="Doctors by Specialty",
        color_discrete_sequence=["#0F6FFF", "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE"],
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter", size=12),
        title_font=dict(size=16, color="#374151"),
    )
    fig.update_traces(textposition="inside", textinfo="label+percent")
    st.plotly_chart(fig, use_container_width=True)


# Page: Appointments

def render_appointments_page():
    page_header("calendar", "Appointments", "Real-time appointment tracking and management.")

    appointments = get_appointments()
    if not appointments:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:48px 24px;">
            <div style="margin-bottom:12px;">{icon("calendar", 36, "#D1D5DB")}</div>
            <div style="font-size:16px;font-weight:600;color:#6B7280;">No appointments booked yet</div>
            <div style="font-size:13px;color:#9CA3AF;margin-top:4px;">Use the Chat Assistant to book one.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    confirmed = sum(1 for a in appointments if a["status"] == "confirmed")
    cancelled = sum(1 for a in appointments if a["status"] == "cancelled")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_card_html("calendar", "Total", len(appointments), "#0F6FFF"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card_html("calendar-check", "Confirmed", confirmed, "#10B981"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card_html("x-circle", "Cancelled", cancelled, "#EF4444"), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    df = pd.DataFrame(appointments)
    display_cols = ["appointment_id", "doctor_name", "specialty", "date", "time", "status", "patient_id"]
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available_cols], use_container_width=True, hide_index=True)

    if len(appointments) > 0:
        status_counts = df["status"].value_counts()
        fig = px.bar(
            x=status_counts.index,
            y=status_counts.values,
            title="Appointment Status Breakdown",
            labels={"x": "Status", "y": "Count"},
            color=status_counts.index,
            color_discrete_map={"confirmed": "#10B981", "cancelled": "#EF4444"},
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter", size=12),
            title_font=dict(size=16, color="#374151"),
            xaxis=dict(gridcolor="#F3F4F6"),
            yaxis=dict(gridcolor="#F3F4F6"),
        )
        st.plotly_chart(fig, use_container_width=True)


# Page: Evaluation Metrics

def render_evaluation_page():
    page_header("bar-chart-3", "Evaluation Metrics",
                "Assess response quality and track tool performance across the system.")

    evaluator = st.session_state.evaluator

    # Manual evaluation
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
        {icon("sparkles", 18, "#F59E0B")}
        <span style="font-size:16px;font-weight:600;color:#374151;">Manual Evaluation</span>
    </div>
    """, unsafe_allow_html=True)

    with st.form("eval_form"):
        eval_query = st.text_input("Query to evaluate")
        eval_response = st.text_area("Response to evaluate")
        submitted = st.form_submit_button("Run Evaluation")

        if submitted and eval_query and eval_response:
            with st.spinner("Evaluating..."):
                result = evaluator.evaluate_response(eval_query, eval_response)
                scores = result.get("scores", {})
                cols = st.columns(5)
                for i, key in enumerate(["relevance", "accuracy", "helpfulness", "completeness", "overall"]):
                    cols[i].metric(key.capitalize(), f"{scores.get(key, 0)}/5")
                if scores.get("feedback"):
                    st.info(f"**Feedback:** {scores['feedback']}")

    st.markdown("---")

    # Aggregate
    summary = evaluator.get_evaluation_summary()

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
        {icon("bar-chart-3", 18, "#6B7280")}
        <span style="font-size:16px;font-weight:600;color:#374151;">Aggregate Scores</span>
    </div>
    """, unsafe_allow_html=True)

    if summary["total_evaluations"] == 0:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:36px 24px;">
            <div style="margin-bottom:10px;">{icon("bar-chart-3", 32, "#D1D5DB")}</div>
            <div style="font-size:14px;color:#6B7280;">No evaluations yet. Enable auto-evaluate or use manual evaluation above.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(metric_card_html("zap", "Total Evaluations", summary["total_evaluations"], "#3B82F6"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card_html("check-circle", "Valid Evaluations", summary["valid_evaluations"], "#10B981"), unsafe_allow_html=True)

        avg = summary.get("avg_scores", {})
        if avg:
            fig = go.Figure(data=[go.Bar(
                x=list(avg.keys()),
                y=list(avg.values()),
                marker_color=["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#EF4444"],
                marker_line_width=0,
            )])
            fig.update_layout(
                title="Average Evaluation Scores",
                yaxis=dict(range=[0, 5], gridcolor="#F3F4F6"),
                xaxis=dict(gridcolor="#F3F4F6"),
                xaxis_title="Criteria",
                yaxis_title="Score (1-5)",
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Inter", size=12),
                title_font=dict(size=16, color="#374151"),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Tool metrics
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
        {icon("wrench", 18, "#6B7280")}
        <span style="font-size:16px;font-weight:600;color:#374151;">Tool Performance</span>
    </div>
    """, unsafe_allow_html=True)

    tool_metrics = evaluator.get_tool_metrics()
    if not tool_metrics:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:36px 24px;">
            <div style="margin-bottom:10px;">{icon("wrench", 32, "#D1D5DB")}</div>
            <div style="font-size:14px;color:#6B7280;">No tool usage tracked yet.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        tool_df = pd.DataFrame([{"Tool": name, **data} for name, data in tool_metrics.items()])
        st.dataframe(tool_df, use_container_width=True, hide_index=True)

        fig = px.bar(
            tool_df, x="Tool", y="success_rate",
            title="Tool Success Rate (%)",
            color="success_rate",
            color_continuous_scale=["#EF4444", "#F59E0B", "#10B981"],
            range_color=[0, 100],
        )
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Inter", size=12),
            title_font=dict(size=16, color="#374151"),
            yaxis=dict(gridcolor="#F3F4F6"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Recent evaluations
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
        {icon("clock", 18, "#6B7280")}
        <span style="font-size:16px;font-weight:600;color:#374151;">Recent Evaluations</span>
    </div>
    """, unsafe_allow_html=True)

    recent = evaluator.get_recent_evaluations()
    if recent:
        for entry in reversed(recent):
            overall = entry['scores'].get('overall', 0)
            with st.expander(f"{entry['timestamp'][:19]}  ·  Overall: {overall}/5"):
                st.write(f"**Query:** {entry['query']}")
                st.write(f"**Response:** {entry['response_preview']}...")
                scores = entry["scores"]
                cols = st.columns(5)
                for i, key in enumerate(["relevance", "accuracy", "helpfulness", "completeness", "overall"]):
                    cols[i].metric(key.capitalize(), f"{scores.get(key, 0)}/5")
                if scores.get("feedback"):
                    st.info(scores["feedback"])
    else:
        st.caption("No evaluations recorded yet.")


# Page: Memory & Logs

def render_memory_page():
    page_header("brain", "Memory & Logs", "Inspect agent memory traces, planning breakdowns, and tool invocation logs.")

    agent = st.session_state.agent
    memory_trace = agent.get_memory_trace()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(metric_card_html("message-circle", "Conversations", memory_trace["conversation_count"], "#3B82F6"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card_html("brain", "Context Keys", len(memory_trace["patient_context"]), "#8B5CF6"), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    if memory_trace["patient_context"]:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
            {icon("shield", 18, "#6B7280")}
            <span style="font-size:16px;font-weight:600;color:#374151;">Patient Context</span>
        </div>
        """, unsafe_allow_html=True)
        for key, val in memory_trace["patient_context"].items():
            st.code(f"{key}: {val}", language="text")

    st.markdown("---")

    # Conversation history
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
        {icon("message-circle", 18, "#6B7280")}
        <span style="font-size:16px;font-weight:600;color:#374151;">Recent Conversations</span>
    </div>
    """, unsafe_allow_html=True)

    recent = memory_trace.get("recent_history", [])
    if not recent:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:36px 24px;">
            <div style="margin-bottom:10px;">{icon("message-circle", 32, "#D1D5DB")}</div>
            <div style="font-size:14px;color:#6B7280;">No conversation history yet.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for entry in recent:
            role_label = "User" if entry["role"] == "user" else "Assistant"
            with st.expander(f"{role_label}  ·  {entry['timestamp'][:19]}"):
                st.write(entry["content"])

    st.markdown("---")

    # Tool log
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
        {icon("wrench", 18, "#6B7280")}
        <span style="font-size:16px;font-weight:600;color:#374151;">Tool Invocation Log</span>
    </div>
    """, unsafe_allow_html=True)

    tool_log = agent.get_tool_log()
    if not tool_log:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:36px 24px;">
            <div style="margin-bottom:10px;">{icon("wrench", 32, "#D1D5DB")}</div>
            <div style="font-size:14px;color:#6B7280;">No tools have been invoked yet.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for entry in reversed(tool_log[-20:]):
            with st.expander(f"{entry['tool']}  ·  {entry['timestamp'][:19]}"):
                st.json(entry["args"])

        tool_names = [t["tool"] for t in tool_log]
        tool_counts = pd.Series(tool_names).value_counts()
        fig = px.bar(
            x=tool_counts.index,
            y=tool_counts.values,
            title="Tool Usage Frequency",
            labels={"x": "Tool", "y": "Invocations"},
            color_discrete_sequence=["#0F6FFF"],
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter", size=12),
            title_font=dict(size=16, color="#374151"),
            xaxis=dict(gridcolor="#F3F4F6"),
            yaxis=dict(gridcolor="#F3F4F6"),
        )
        st.plotly_chart(fig, use_container_width=True)


# Page Router

PAGE_MAP = {
    "Chat Assistant": render_chat_page,
    "Patient Directory": render_patient_page,
    "Doctor Directory": render_doctor_page,
    "Appointments": render_appointments_page,
    "Evaluation Metrics": render_evaluation_page,
    "Memory & Logs": render_memory_page,
}

PAGE_MAP[page]()
