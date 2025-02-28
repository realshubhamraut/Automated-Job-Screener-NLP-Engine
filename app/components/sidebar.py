import streamlit as st
import os
import sys
from datetime import datetime
import json

# Add project root to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config import APP_NAME, EMBEDDING_MODEL, VECTOR_DB_TYPE
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Define paths for checking document counts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

def count_documents_on_disk():
    """Count documents directly from disk storage"""
    resume_count = 0
    job_desc_count = 0
    
    # Count resumes
    resumes_dir = os.path.join(PROCESSED_DIR, "resumes")
    if os.path.exists(resumes_dir):
        resume_count = len([f for f in os.listdir(resumes_dir) if f.endswith('.json')])
        
    # Count job descriptions
    jobs_dir = os.path.join(PROCESSED_DIR, "job_descriptions")
    if os.path.exists(jobs_dir):
        job_desc_count = len([f for f in os.listdir(jobs_dir) if f.endswith('.json')])
        
    return resume_count, job_desc_count

def render_sidebar():
    """Render the sidebar components"""
    
    # Apply Roboto Serif font family to sidebar - with enhanced targeting for all content
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Serif:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&family=Roboto+Mono&display=swap');
    
    /* Target the entire sidebar */
    [data-testid="stSidebar"] {
        font-family: 'Roboto Serif', serif !important;
    }
    
    /* Target all elements in the sidebar */
    [data-testid="stSidebar"] * {
        font-family: 'Roboto Serif', serif !important;
    }
    
    /* Target specific markdown elements */
    [data-testid="stSidebar"] .stMarkdown p, 
    [data-testid="stSidebar"] .stMarkdown li, 
    [data-testid="stSidebar"] .stMarkdown ol,
    [data-testid="stSidebar"] .stMarkdown ul,
    [data-testid="stSidebar"] .stMarkdown span,
    [data-testid="stSidebar"] .stMarkdown div {
        font-family: 'Roboto Serif', serif !important;
    }
    
    /* Target heading elements inside markdown */
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown h4 {
        font-family: 'Roboto Serif', serif !important;
        font-weight: 600;
    }
    
    /* Target code blocks */
    [data-testid="stSidebar"] code {
        font-family: 'Roboto Mono', monospace !important;
    }
    
    /* Target titles */
    [data-testid="stSidebar"] .stTitle {
        font-family: 'Roboto Serif', serif !important;
        font-weight: 600;
    }
    
    /* Target instruction cards */
    [data-testid="stSidebar"] .stMarkdown ol li,
    [data-testid="stSidebar"] .stMarkdown ul li {
        font-family: 'Roboto Serif', serif !important;
        margin-bottom: 8px;
    }

    /* Social media buttons styling */
    .social-links {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        margin: 15px 0;
        padding: 10px 0;
        border-bottom: 1px solid #eee;
    }

    .social-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 8px 16px;
        background-color: #f8f9fa;
        border: 1px solid #ddd;
        border-radius: 6px;
        color: #333;
        text-decoration: none !important;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .github-btn {
        background-color: #24292e;
        color: white !important;
    }

    .linkedin-btn {
        background-color: #0077b5;
        color: white !important;
    }

    .social-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-decoration: none !important;
        color: white !important;
    }

    .social-icon {
        margin-right: 6px;
        font-size: 18px;
        fill: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Social Media Links with icons instead of title
    social_links_html = """
    <div class="social-links">
        <a href="https://github.com/realshubhamraut/Automated-First-Screener-and-AI-Job-Matching-NLP-WebEngine" target="_blank" class="social-btn github-btn">
            <svg class="social-icon" xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="white" viewBox="0 0 16 16">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
            </svg>
            <span style="color: white;">GitHub</span>
        </a>
        <a href="https://www.linkedin.com/in/contactshubhamraut/" target="_blank" class="social-btn linkedin-btn">
            <svg class="social-icon" xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="white" viewBox="0 0 16 16">
                <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854V1.146zm4.943 12.248V6.169H2.542v7.225h2.401zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248-.822 0-1.359.54-1.359 1.248 0 .694.521 1.248 1.327 1.248h.016zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016a5.54 5.54 0 0 1 .016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225h2.4z"/>
            </svg>
            <span style="color: white;">LinkedIn</span>
        </a>
    </div>
    """
    
    st.sidebar.markdown(social_links_html, unsafe_allow_html=True)
    
    # System information
    with st.sidebar.expander("Engine Information"):
        st.write(f"**Embedding Model**: {EMBEDDING_MODEL}")
        st.write(f"**Vector Database**: {VECTOR_DB_TYPE}")
        st.write(f"**Session Started**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get document counts directly from disk
        resume_count, job_desc_count = count_documents_on_disk()
        
        # Display document counts
        st.write(f"**Resumes Uploaded**: {resume_count}")
        st.write(f"**Job Descriptions Uploaded**: {job_desc_count}")
    
    with st.sidebar.expander("Instructions"):
        st.markdown("""
        ### Quick Start Guide
        
        1. **Upload**: Upload resumes and job descriptions
        2. **Match**: Find the best matching candidates for each job
        3. **Analysis**: View detailed analysis of candidates
        4. **Interview**: Generate interview questions
        5. **Analytics**: View analytics and insights
        
        ### Tips
        
        - Upload multiple resumes for batch processing
        - Adjust similarity threshold based on your needs
        - Use filters to refine matches
        - Export results for sharing
        """)    

    # Settings
    with st.sidebar.expander("Matching Settings"):
        # Similarity threshold slider
        similarity_threshold = st.slider(
            "Minimum Similarity Score",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Minimum similarity score required for a match"
        )
        
        # Store in session state
        st.session_state.similarity_threshold = similarity_threshold
        
        # Hybrid search weight
        hybrid_weight = st.slider(
            "Semantic vs Keyword Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Weight for semantic search vs keyword matching (1.0 = semantic only)"
        )
        
        # Store in session state
        st.session_state.hybrid_weight = hybrid_weight
        
        # Chunk size for document processing
        chunk_size = st.select_slider(
            "Chunk Size",
            options=[100, 200, 300, 400, 500, 750, 1000],
            value=500,
            help="Size of text chunks for processing (smaller chunks are more precise, larger chunks have more context)"
        )
        
        # Store in session state
        st.session_state.chunk_size = chunk_size
    
    # About section
    with st.sidebar.expander("About"):
        st.markdown("""
        **Automated First Screener and AI Job Matching NLP WebEngine**
        
        This application uses advanced NLP techniques to match job candidates
        with appropriate positions based on resume and job description analysis.
        
        It leverages vector embeddings and semantic search to find matches based
        on skill compatibility, experience relevance, and overall fit.
        """)