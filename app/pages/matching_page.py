import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import os
import importlib
import sys
from typing import Dict, List, Any
from streamlit_pdf_viewer import pdf_viewer

from app.components import charts, document_view
from src.document_processor.embedder import TextEmbedder
from src.utils.logger import get_logger

# Force reload of the hybrid_search module
if "src.matching_engine.hybrid_search" in sys.modules:
    importlib.reload(sys.modules["src.matching_engine.hybrid_search"])

from src.matching_engine.hybrid_search import HybridSearchEngine
from src.config import DEFAULT_SIMILARITY_THRESHOLD, DEFAULT_HYBRID_WEIGHT

logger = get_logger(__name__)

def render():
    # Add CSS for left-aligned buttons
    st.markdown("""
    <style>
    /* Left alignment for buttons */
    .left-align-button {
        display: flex;
        justify-content: flex-start;
    }
    
    /* Make buttons take appropriate width in their container */
    .left-align-button button {
        width: auto;
    }
    
    /* Fix for download button alignment */
    .left-align-button .stDownloadButton {
        width: auto;
        display: flex;
        justify-content: flex-start;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state variables
    if "shortlisted_candidates" not in st.session_state:
        st.session_state.shortlisted_candidates = []
    
    if "pdf_viewer_open" not in st.session_state:
        st.session_state.pdf_viewer_open = None
        
    # Check if we're viewing a PDF in separate page
    if st.session_state.pdf_viewer_open:
        display_pdf_viewer()
        return

    # Main page content
    st.title("Know best fit candidates against job description")
    st.write("Find the best candidates with ATS score.")
    
    # Check if documents are uploaded
    if not st.session_state.resumes:
        st.warning("Please upload at least one resume first.")
        return
        
    if not st.session_state.job_descriptions:
        st.warning("Please upload at least one job description first.")
        return
    
    # Matching Settings in sidebar    
    st.sidebar.subheader("Matching Settings")
    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.similarity_threshold,
        step=0.05,
        help="Minimum similarity score to consider a match"
    )
    
    hybrid_weight = st.sidebar.slider(
        "Semantic vs Keyword Weight", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.hybrid_weight,
        step=0.1,
        help="Weight between semantic (1.0) and keyword (0.0) matching"
    )
    
    # Update session state
    st.session_state.similarity_threshold = similarity_threshold
    st.session_state.hybrid_weight = hybrid_weight
    
    # Initialize NLP models
    try:
        # Initialize embedder and search engine
        embedder = TextEmbedder()
        search_engine = HybridSearchEngine(hybrid_weight=hybrid_weight)
    except Exception as e:
        logger.error(f"Error initializing NLP models: {str(e)}")
        st.error(f"Error initializing models: {str(e)}")
        return
    
    # Form for matching selection
    with st.form("matching_form"):
        # Create columns for side-by-side selection
        left_col, right_col = st.columns(2)
        
        # Left column: Job description selection
        with left_col:
            st.subheader("Select Job Description")
            
            # Create job options with safeguards against empty lists
            if len(st.session_state.job_descriptions) == 0:
                st.error("No job descriptions available.")
                st.form_submit_button("Match Resumes", disabled=True)
                return
                
            job_options = {doc['filename']: i for i, doc in enumerate(st.session_state.job_descriptions)}
            
            # Use direct indices as options
            selected_job_idx = st.selectbox(
                "Choose a job description", 
                options=list(job_options.values()),
                format_func=lambda x: list(job_options.keys())[list(job_options.values()).index(x)],
                key="job_select"
            )
            selected_job = st.session_state.job_descriptions[selected_job_idx]
        
        # Right column: Resume selection
        with right_col:
            st.subheader("Select Resumes to Match")
            resume_options = {doc['filename']: i for i, doc in enumerate(st.session_state.resumes)}
            
            # First show the multiselect for choosing resumes
            selected_resume_indices = st.multiselect(
                "Choose resumes",
                options=list(resume_options.values()),
                format_func=lambda x: list(resume_options.keys())[list(resume_options.values()).index(x)],
                default=[],
                key="resume_multiselect"
            )
            
            # Then show the "Select All" checkbox below
            all_selected = st.checkbox("Select All Resumes", value=False)
            
            # Update selected_resume_indices if "Select All" is checked
            if all_selected:
                selected_resume_indices = list(range(len(st.session_state.resumes)))
        
        # Create a container for the submit button positioned on the far left
        submit_cols = st.columns([1, 4])
        with submit_cols[0]:
            # Using a custom div to ensure the button is at the far left
            st.markdown('<div class="left-align-button">', unsafe_allow_html=True)
            submitted = st.form_submit_button("Match Resumes")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Process matches if form submitted
    if submitted and selected_resume_indices:
        with st.spinner("Matching resumes to job description..."):
            start_time = time.time()
            
            # Get job description text and embed it
            job_text = selected_job['processed']['clean_text']
            job_embedding = embedder.embed_text(job_text)
            
            # Process each resume
            match_results = []
            
            for idx in selected_resume_indices:
                resume = st.session_state.resumes[idx]
                resume_text = resume['processed']['clean_text']
                
                # Generate embedding
                resume_embedding = embedder.embed_text(resume_text)
                
                # Calculate match score
                score, details = search_engine.hybrid_match(
                    resume_embedding, 
                    job_embedding,
                    resume_text,
                    job_text
                )
                
                # Create match result
                match_result = {
                    'resume_id': resume['id'],
                    'resume_filename': resume['filename'],
                    'job_id': selected_job['id'],
                    'job_filename': selected_job['filename'],
                    'score': score,
                    'details': details,
                    'timestamp': time.time()
                }
                
                match_results.append(match_result)
            
            # Sort by score
            match_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Store in session state
            st.session_state.match_results = match_results
            
            # Show execution time
            end_time = time.time()
            st.success(f"Matched {len(match_results)} resumes in {end_time - start_time:.2f} seconds")
    
    # Display match results
    if st.session_state.get('match_results'):
        display_match_results(st.session_state.match_results, similarity_threshold)

def display_pdf_viewer():
    """Display the PDF viewer page"""
    pdf_info = st.session_state.pdf_viewer_open
    
    # Create header with back button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"Resume: {pdf_info['name']}")
    with col2:
        if st.button("← Back to Results", key="back_to_results"):
            st.session_state.pdf_viewer_open = None
            st.rerun()
    
    # Show the PDF with fixed settings
    try:
        # Display PDF with fixed settings (width=1000, height=600)
        pdf_viewer(
            pdf_info['path'], 
            width=1000,
            height=600,
            render_text=True
        )
    except Exception as e:
        st.error(f"Could not load PDF: {str(e)}")
        st.info(f"PDF path: {pdf_info['path']}")

def display_match_results(match_results: List[Dict], threshold: float = 0.6):
    """Display match results in a table and visualizations with embedded PDF viewer"""
    # Create header with shortlist and export buttons aligned left
    st.header("Matching Results")
    
    # Add styles for buttons and tags
    st.markdown(
        """
        <style>
        .tag-pill {
            display: inline-block;
            background-color: #e3f2fd;
            color: #1565c0;
            border: 1px solid #90caf9;
            border-radius: 16px;
            padding: 4px 10px;
            margin: 2px;
            font-size: 13px;
            font-weight: 500;
        }
        .match-score {
            color: #1e88e5;
            font-weight: 500;
        }
        .semantic-score {
            color: #43a047;
            font-weight: 500;
        }
        .keyword-score {
            color: #f57c00;
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Create a row with two columns for the buttons
    button_cols = st.columns([1, 1, 3])
    
    with button_cols[0]:
        # Export button aligned left
        export_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "threshold": threshold,
            "results": [
                {
                    "resume": match["resume_filename"],
                    "job_description": match["job_filename"],
                    "score": match["score"],
                    "semantic_similarity": match["details"]["semantic_similarity"],
                    "keyword_similarity": match["details"]["keyword_similarity"],
                    "common_keywords": match["details"].get("common_keywords", []),
                    "missing_keywords": match["details"].get("missing_keywords", [])
                }
                for match in match_results
            ]
        }
        export_json = json.dumps(export_data, indent=2)
        st.download_button(
            label="Export JSON",
            data=export_json,
            file_name=f"match_results_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="export_json_btn"
        )
        
    with button_cols[1]:
        # Shortlist button aligned left
        shortlist_button = st.button("📋 Shortlist Selected", key="shortlist_btn_header")
    
    # Initialize shortlist selection in session state
    if 'shortlist_selection' not in st.session_state:
        st.session_state.shortlist_selection = {result['resume_id']: False for result in match_results}
    
    # Handle shortlist selection when button is pressed
    if shortlist_button:
        success_msg, info_msg = handle_shortlist_selection(match_results)
        if success_msg:
            st.success(success_msg)
        if info_msg:
            st.info(info_msg)
    
    # Display a note about shortlisting
    st.info("Check the boxes next to candidates you want to shortlist for detailed analysis.")
            
    # Table with checkbox column for shortlisting
    st.subheader("Match Scores")
    
    # Table header with 6 columns (removed PDF column, expanded scores into 3 columns)
    header_col1, header_col2, header_col3, header_col4, header_col5, header_col6 = st.columns([0.5, 2, 1, 1, 1, 1.5])
    with header_col1:
        st.write("**Shortlist**")
    with header_col2:
        st.write("**Resume**")
    with header_col3:
        st.write("**Match**")
    with header_col4:
        st.write("**Semantic**")
    with header_col5:
        st.write("**Keyword**")
    with header_col6:
        st.write("**Matching Skills**")
    
    # Create a list to store results data
    results_data = []
    
    # Create checkboxes for each row
    for i, result in enumerate(match_results):
        # Find the resume path for PDF viewing
        resume_path = None
        resume_filename = None
        for resume in st.session_state.resumes:
            if resume['id'] == result['resume_id']:
                resume_path = resume.get('filepath')
                resume_filename = resume['filename']
                break
                
        # Get common keywords (top 7 - increased from 5 to show more technical skills)
        common_keywords = result['details'].get('common_keywords', [])[:7]
        
        # Create styled tag pills HTML
        common_keywords_html = ""
        for keyword in common_keywords:
            common_keywords_html += f'<span class="tag-pill">{keyword}</span>'
        
        if not common_keywords:
            common_keywords_html = "No matching skills"
        
        # Create unique row ID
        row_id = f"{result['resume_id']}_{i}"
                
        # Create main row data with 6 columns
        col1, col2, col3, col4, col5, col6 = st.columns([0.5, 2, 1, 1, 1, 1.5])
        
        with col1:
            # Create a unique key for each checkbox
            checkbox_key = f"shortlist_{row_id}"
            shortlisted = st.checkbox("", key=checkbox_key, value=st.session_state.shortlist_selection.get(result['resume_id'], False))
            # Update selection in session state
            st.session_state.shortlist_selection[result['resume_id']] = shortlisted
        
        with col2:
            st.write(resume_filename)
        
        # Display individual score columns with colored values
        with col3:
            # Match score with blue color
            st.markdown(f'<span class="match-score">{result["score"]:.2f}</span>', unsafe_allow_html=True)
        
        with col4:
            # Semantic score with green color
            st.markdown(f'<span class="semantic-score">{result["details"]["semantic_similarity"]:.2f}</span>', unsafe_allow_html=True)
        
        with col5:
            # Keyword score with orange color
            st.markdown(f'<span class="keyword-score">{result["details"]["keyword_similarity"]:.2f}</span>', unsafe_allow_html=True)
        
        with col6:
            # Display styled tags
            st.markdown(common_keywords_html, unsafe_allow_html=True)
        
        # Add a separator between rows
        st.markdown("---")
        
        # Store data for reference
        results_data.append({
            'Resume': resume_filename,
            'Resume Path': resume_path,
            'Match Score': result['score'],
            'Semantic Score': result['details']['semantic_similarity'],
            'Keyword Score': result['details']['keyword_similarity'],
            'Resume ID': result['resume_id'],
            'Job ID': result['job_id'],
            'Common Keywords': common_keywords
        })

def handle_shortlist_selection(match_results):
    """Handle the shortlist selection and update session state"""
    # Get selected resume IDs
    selected_ids = [resume_id for resume_id, selected in st.session_state.shortlist_selection.items() if selected]
    
    if not selected_ids:
        return None, "No candidates selected for shortlisting. Please check at least one candidate."
    else:
        # Store shortlisted candidate details in session state
        shortlisted = []
        for result in match_results:
            if result['resume_id'] in selected_ids:
                # Find the full resume data
                for resume in st.session_state.resumes:
                    if resume['id'] == result['resume_id']:
                        # Add match score and details to resume data for analysis page
                        resume_with_score = resume.copy()
                        resume_with_score['match_score'] = result['score']
                        resume_with_score['matched_job_id'] = result['job_id']
                        resume_with_score['match_details'] = result['details']
                        shortlisted.append(resume_with_score)
                        break
        
        # Update session state with shortlisted candidates
        st.session_state.shortlisted_candidates = shortlisted
        
        # Return success message with count
        return f"✅ Successfully shortlisted {len(shortlisted)} candidates for analysis!", "Go to the Analysis tab to view detailed information about shortlisted candidates."

if __name__ == "__main__":
    render()