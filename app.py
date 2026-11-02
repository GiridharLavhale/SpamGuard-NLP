"""
SpamGuard NLP — Intelligent SMS Spam Detection & Analysis System.
Main application entry point and navigation coordinator.
"""

import os
import streamlit as st

# Must be the very first Streamlit call
st.set_page_config(
    page_title="SpamGuard NLP — Intelligent SMS Spam Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.utils import load_custom_css, render_sidebar
from src.database import init_db

# Initialize database schema safely
init_db()

# Inject modern SaaS styling
st.markdown(load_custom_css(), unsafe_allow_html=True)

# Define Navigation Structure
pages = [
    st.Page("pages/1_Home.py", title="Home Dashboard", icon="🏠", default=True),
    st.Page("pages/2_Single_Detection.py", title="Single Message Detection", icon="✉️"),
    st.Page("pages/3_Bulk_Detection.py", title="Bulk Message Detection", icon="📂"),
    st.Page("pages/4_Analytics.py", title="Analytics Dashboard", icon="📊"),
    st.Page("pages/5_Model_Performance.py", title="Model Performance", icon="⚡"),
    st.Page("pages/6_About.py", title="About Project", icon="ℹ️"),
]

# Run navigation
nav = st.navigation(pages)

# Render customized sidebar branding, model indicator, and footer
render_sidebar()

# Run the selected page
nav.run()
