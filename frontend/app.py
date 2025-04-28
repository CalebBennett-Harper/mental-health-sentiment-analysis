"""
Streamlit frontend for Mental Health Sentiment Analysis System.
Provides UI for text and speech emotion analysis and history tracking.
"""

import streamlit as st
import requests

# --- App Configuration ---
st.set_page_config(
    page_title="Mental Health Sentiment Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API Connection Utility ---
def get_api_url():
    return st.session_state.get("api_url", "http://localhost:8000/api/v1")

def set_api_url():
    st.sidebar.text_input(
        "FastAPI Backend URL",
        value=get_api_url(),
        key="api_url",
        help="Set the URL of your FastAPI backend (e.g., http://localhost:8000/api/v1)"
    )

# --- Sidebar Navigation ---
set_api_url()
page = st.sidebar.radio(
    "Navigation",
    ["Text Analysis", "Speech Analysis", "History"],
    icons=["✍️", "🎤", "📈"]
)

st.title("🧠 Mental Health Sentiment Analysis")
st.write("Analyze your emotions from text or speech in real time.")

# --- Page Routing ---
if page == "Text Analysis":
    st.header("Text Emotion Analysis")
    st.info("Enter your thoughts and get instant emotional feedback.")
    st.write("(Feature coming soon)")
elif page == "Speech Analysis":
    st.header("Speech Emotion Analysis")
    st.info("Record or upload audio to analyze your emotions.")
    st.write("(Feature coming soon)")
elif page == "History":
    st.header("Emotional History")
    st.info("View your emotional timeline and export your data.")
    st.write("(Feature coming soon)")