import streamlit as st
import os
import sys
import uuid
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_unique_id() -> str:
    """Generate a unique ID with timestamp and UUID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    uid = str(uuid.uuid4())[:8]
    return f"{timestamp}_{uid}"

def init_session_state():
    """Initialize session state variables if they don't exist"""
    if "resumes" not in st.session_state:
        st.session_state.resumes = []
    
    if "job_descriptions" not in st.session_state:
        st.session_state.job_descriptions = []
    
    if "match_results" not in st.session_state:
        st.session_state.match_results = []
    
    if "similarity_threshold" not in st.session_state:
        st.session_state.similarity_threshold = 0.6
    
    if "hybrid_weight" not in st.session_state:
        st.session_state.hybrid_weight = 0.7
    
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 300
    
    if "start_time" not in st.session_state:
        st.session_state.start_time = datetime.now().strftime("%H:%M:%S")

def render_page_header(title: str, description: str = None):
    """Render a consistent page header with title and optional description"""
    st.title(title)
    if description:
        st.write(description)
    st.divider()