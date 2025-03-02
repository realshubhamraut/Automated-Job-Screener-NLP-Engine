import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import os
from typing import Dict, List, Any
from streamlit_pdf_viewer import pdf_viewer

from app.components import charts, document_view
from src.document_processor.embedder import TextEmbedder
from src.matching_engine.hybrid_search import HybridSearchEngine
from src.config import DEFAULT_SIMILARITY_THRESHOLD, DEFAULT_HYBRID_WEIGHT
from src.utils.logger import get_logger

logger = get_logger(__name__)

def render():
    st.title("Know best fit candidates against job description")
    st.write("Find the best candidates with ATS score.")
    
    # Check if documents are uploaded
    if not st.session_state.resumes:
        st.warning("Please upload at least one resume first.")
        return
        
    if not st.session_state.job_descriptions:
        st.warning("Please upload at least one job description first.")
        return
    
    # Initialize shortlisted candidates in session state if not exists
    if "shortlisted_candidates" not in st.session_state:
        st.session_state.shortlisted_candidates = []
    
    # Sidebar settings
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
        
        # Submit button (aligned left)
        submitted = st.form_submit_button("Match Resumes")
    
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

def display_match_results(match_results: List[Dict], threshold: float = 0.6):
    """Display match results in a table and visualizations"""
    # Create header with shortlist and export buttons aligned right
    header_col1, header_col2 = st.columns([1, 1])
    
    with header_col1:
        st.header("Matching Results")
    
    with header_col2:
        # Add styles for right-aligned buttons
        st.markdown(
            """
            <style>
            div.stButton {
                text-align: right;
                margin-top: 1.5rem;
            }
            div.stDownloadButton {
                text-align: right;
                margin-top: 0.5rem;
                margin-bottom: 1.5rem;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Create a row for the buttons
        button_col1, button_col2 = st.columns([1, 1])
        
        with button_col1:
            # Export button
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
            

        with button_col2:
            # Shortlist button
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
    
    st.download_button(
        label="Export JSON",
        data=export_json,
        file_name=f"match_results_{time.strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        key="export_json_btn"
    )
            
    # Table with checkbox column for shortlisting
    st.subheader("Match Scores")
    
    # Table header
    header_col1, header_col2, header_col3, header_col4, header_col5, header_col6 = st.columns([0.5, 2, 1, 1, 1, 1])
    with header_col1:
        st.write("**Shortlist**")
    with header_col2:
        st.write("**Resume**")
    with header_col3:
        st.write("**Match Score**")
    with header_col4:
        st.write("**Semantic Score**")
    with header_col5:
        st.write("**Keyword Score**")
    with header_col6:
        st.write("**Common Tags**")
    
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
                
        # Get common keywords (top 3)
        common_keywords = result['details'].get('common_keywords', [])[:3]
        common_keywords_text = ", ".join(common_keywords) if common_keywords else "None"
                
        col1, col2, col3, col4, col5, col6 = st.columns([0.5, 2, 1, 1, 1, 1])
        
        with col1:
            # Create a unique key for each checkbox
            checkbox_key = f"shortlist_{result['resume_id']}_{i}"
            shortlisted = st.checkbox("", key=checkbox_key, value=st.session_state.shortlist_selection.get(result['resume_id'], False))
            # Update selection in session state
            st.session_state.shortlist_selection[result['resume_id']] = shortlisted
        
        with col2:
            st.write(resume_filename)
            # Add PDF viewer button if the resume is a PDF file
            if resume_path and resume_path.lower().endswith('.pdf'):
                with st.expander("View Resume PDF", expanded=False):
                    try:
                        pdf_viewer(resume_path, width=700, height=800)
                    except Exception as e:
                        st.error(f"Could not load PDF: {str(e)}")
        
        with col3:
            st.write(f"{result['score']:.2f}")
        
        with col4:
            st.write(f"{result['details']['semantic_similarity']:.2f}")
        
        with col5:
            st.write(f"{result['details']['keyword_similarity']:.2f}")
        
        with col6:
            st.write(common_keywords_text)
        
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