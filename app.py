import streamlit as st
import os
import sys
from streamlit_option_menu import option_menu
from datetime import datetime

# Import pages directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import render functions directly
from app.pages.upload_page import render as render_upload
from app.pages.matching_page import render as render_matching
from app.pages.analysis_page import render as render_analysis
from app.pages.interview_page import render as render_interview
from app.pages.analytics_page import render as render_analytics
from app.components.sidebar import render_sidebar

# Initialize session state
def init_session_state():
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
        
    # Initialize selected page
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Upload"

# Set page configuration
st.set_page_config(
    page_title="AI Resume & Job Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS with gradient title and better menu icon colors
gradient_title_html = """
<style>
/* Import Roboto Serif with all weights */
@import url('https://fonts.googleapis.com/css2?family=Roboto+Serif:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700;900&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');

/* Set Roboto Serif as default font for entire app */
html, body, [class*="css"] {
  font-family: 'Roboto Serif', serif !important;
}

/* Override Streamlit's default fonts */
.stMarkdown, .stText, p, div, h1, h2, h3, h4, h5, h6, li, span, button, label {
  font-family: 'Roboto Serif', serif !important;
}

/* Apply to form elements */
input, textarea, select, .stSelectbox, .stMultiselect {
  font-family: 'Roboto Serif', serif !important;
}

/* Apply to sidebar */
.st-emotion-cache-16idsys p, .st-emotion-cache-j7qwjs p {
  font-family: 'Roboto Serif', serif !important;
}

.gradient-title {
  font-family: 'Poppins', sans-serif !important;
  font-weight: 900;
  font-size: 2.5em;
  background: linear-gradient(90deg, #4287f5, #6c3fe6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-shadow: 2px 2px 5px rgba(0,0,0,0.2);
  margin: 0;
  padding: 20px 0 10px 0;
  text-align: center;
}

.title-caption {
  font-family: 'Roboto Serif', serif;
  text-align: center;
  color: #666;
  margin: 0 auto 25px auto;
  font-size: 1rem;
  font-weight: 400;
  border: 1px solid #ddd;
  border-radius: 30px;
  padding: 8px 25px;
  display: inline-block;
  max-width: 80%;
}

.caption-container {
  text-align: center;
  width: 100%;
}

.footer {
    font-family: 'Roboto Serif', serif;
    text-align: center;
    color: #666;
    padding: 1rem 0;
    font-size: 0.8rem;
    margin-top: 2rem;
    border-top: 1px solid #eee;
}

/* Fix width and spacing of option menu */
nav.st-emotion-cache-1lqmwj5 {
    width: 100% !important;
}

nav.st-emotion-cache-1lqmwj5 ul {
    display: flex !important;
    justify-content: space-between !important;
    width: 100% !important;
}

nav.st-emotion-cache-1lqmwj5 ul li {
    flex: 1 !important;
    margin: 0 8px !important;
}

/* Style for primary (selected) button */
div.stButton > button[kind="primary"] {
    background-color: #4287f5 !important;
    border-color: #4287f5 !important;
    color: #ffffff !important;
}

/* Icon styling */
.icon {
    font-family: 'Material Icons' !important;
    font-size: 20px !important;
    vertical-align: middle !important;
    margin-right: 5px !important;
    display: inline-block !important;
}

/* Orange icon for primary button */
div.stButton > button[kind="primary"] .icon {
    color: #ff6b00 !important;
}

/* Better styling for info cards */
.stAlert {
    border-radius: 8px !important;
    margin-bottom: 1.5rem !important;
    font-family: 'Roboto Serif', serif !important;
}

/* Apply font to tables */
.stDataFrame, .stTable {
    font-family: 'Roboto Serif', serif !important;
}

/* Apply font to code blocks */
.stCodeBlock, code {
    font-family: monospace !important;  /* Keep monospace for code blocks */
}

/* Fix any metrics */
.stMetric label, .stMetric div {
    font-family: 'Roboto Serif', serif !important;
}
</style>
<div class="gradient-title">Automated Screener & AI Job Matching</div>
<div class="caption-container">
  <div class="title-caption">Intelligent resume parsing and job matching powered by NLP</div>
</div>
<div class="caption-container">
<div class="title-caption">BETA VERSION - Still under development, but you can explore some features</div>
</div>
"""

init_session_state()

st.markdown(gradient_title_html, unsafe_allow_html=True)

render_sidebar()

# Use streamlit_option_menu which already supports icons
with st.container():
    # This creates the menu with icons but wraps it in a container to control spacing
    selected = option_menu(
        menu_title=None,
        options=["Upload", "Match", "Analysis", "Interview", "Analytics"],
        icons=["cloud-arrow-up", "search", "bar-chart", "chat-dots", "graph-up"],
        default_index=["Upload", "Match", "Analysis", "Interview", "Analytics"].index(st.session_state.selected_page),
        orientation="horizontal",
        styles={
            "container": {
                "padding": "0.5rem",
                "background-color": "#fafafa",
                "border-radius": "8px",
                "display": "flex",
                "justify-content": "space-between",
                "width": "100%",
            },
            "icon": {
                "color": "#000000",
                "font-size": "18px",
            },
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px 10px",
                "--hover-color": "#eef",
                "font-weight": "500",
                "padding": "0.75rem 1rem",  # Increased padding
                "border": "1px solid #000000",
                "border-radius": "5px",
                "color": "#000000",
                "width": "225px",  # Increased width by 1.5x from 150px
            },
            "nav-link-selected": {
                "background-color": "#4287f5",
                "color": "#000000",
                "border": "1px solid #4287f5",
            },
            "icon-selected": {"color": "#ff6b00"},
        },
    )

# Additional CSS to ensure the menu items are evenly spaced
st.markdown("""
<style>
/* Force menu items to be wider and evenly spaced */
nav ul {
    display: flex !important;
    width: 100% !important;
    justify-content: space-between !important;
}

nav ul li {
    flex: 1 !important;
    max-width: 225px !important;
}

nav ul li a {
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# Update the session state with the selected page
if selected != st.session_state.selected_page:
    st.session_state.selected_page = selected
    st.rerun()

# Get the currently selected page
selected = st.session_state.selected_page

# Display info card for selected page
if selected == "Upload":
    st.info("""
    📄 In this section, you can upload your resumes and job descriptions.
    - Upload multiple resumes at once for batch processing
    - Add job descriptions to match against
    - View and verify extracted information from documents
    """)
    
elif selected == "Match":
    st.info("""
    🔍 In this section, you can match resumes to job descriptions.
    - Compare resumes against job descriptions
    - Adjust matching parameters
    - View detailed matching scores and analysis
    """)
    
elif selected == "Analysis":
    st.info("""
    📊 In this section, you can analyze resumes and job descriptions in detail.
    - View extracted skills and experience
    - Compare candidate qualifications to job requirements
    - Get detailed insights about each document
    """)
    
elif selected == "Interview":
    st.info("""
    💬 In this section, you can generate AI-powered interview questions.
    - Create personalized questions based on resume and job description
    - Customize question types and difficulty
    - Export questions for your interview process
    """)
    
elif selected == "Analytics":
    st.info("""
    📈 In this section, you can view analytics about your documents and matches.
    - See visualizations of matching trends
    - Understand common skills and requirements
    - Get insights about your candidate pool
    """)

# Display selected page content
if selected == "Upload":
    render_upload()
    
elif selected == "Match":
    render_matching()
    
elif selected == "Analysis":
    render_analysis()
    
elif selected == "Interview":
    render_interview()
    
elif selected == "Analytics":
    render_analytics()

# Footer
st.markdown("""
<div class="footer">
    © 2025 Shubham - Automated First Screener and AI Job Matching NLP WebEngine
</div>
""", unsafe_allow_html=True)