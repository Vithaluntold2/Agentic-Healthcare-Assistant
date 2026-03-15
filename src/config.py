# config.py - loads env vars and defines project paths

import os
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key, default=""):
    """Check os.environ first, fall back to streamlit secrets for cloud deploy."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


# project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RECORDS_PATH = os.path.join(DATA_DIR, "records.xlsx")
PATIENT_REPORTS_DIR = os.path.join(DATA_DIR, "patient_reports")
FAISS_INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")

# azure openai config
AZURE_OPENAI_ENDPOINT = _get_secret("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = _get_secret("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = _get_secret("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2-chat")
AZURE_OPENAI_API_VERSION = _get_secret("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
LLM_TEMPERATURE = float(_get_secret("LLM_TEMPERATURE", "1"))

EMBEDDING_MODEL = _get_secret("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

MEDLINE_BASE_URL = "https://wsearch.nlm.nih.gov/ws/query"
