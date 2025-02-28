import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from typing import Dict, List, Any

from app.components import charts, document_view
from src.document_processor.embedder import TextEmbedder
from src.matching_engine.hybrid_search import HybridSearchEngine
from src.config import DEFAULT_SIMILARITY_THRESHOLD, DEFAULT_HYBRID_WEIGHT
from src.utils.logger import get_logger

logger = get_logger(__name__)

def render():
    st.title("Match Resumes to Job Descriptions")
    st.write("Compare resumes against job descriptions to find the best matches.")
    
    # Check if documents are uploaded
    if not st.session_state.resumes:
        st.warning("Please upload at least one resume first.")
        return
        
    if not st.session_state.job_descriptions:
        st.warning("Please upload at least one job description first.")
        return
    
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
        # Job description selection
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
        
        # Resume selection
        st.subheader("Select Resumes to Match")
        resume_options = {doc['filename']: i for i, doc in enumerate(st.session_state.resumes)}
        
        # Select all option
        all_selected = st.checkbox("Select All Resumes", value=False)
        
        if all_selected:
            selected_resume_indices = list(range(len(st.session_state.resumes)))
        else:
            # Use direct indices as options
            selected_resume_indices = st.multiselect(
                "Choose resumes",
                options=list(resume_options.values()),
                format_func=lambda x: list(resume_options.keys())[list(resume_options.values()).index(x)],
                default=[],
                key="resume_multiselect"
            )
        
        # Submit button
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
    st.header("Matching Results")
    
    # Create DataFrame for results
    results_df = pd.DataFrame([
        {
            'Resume': result['resume_filename'],
            'Job Description': result['job_filename'],
            'Match Score': f"{result['score']:.2f}",
            'Semantic Score': f"{result['details']['semantic_similarity']:.2f}",
            'Keyword Score': f"{result['details']['keyword_similarity']:.2f}"
        } for result in match_results
    ])
    
    # Display results table
    st.subheader("Match Scores")
    st.dataframe(results_df, use_container_width=True)
    
    # Display top matches above threshold
    st.subheader(f"Top Matches (Score > {threshold})")
    
    # Filter matches above threshold
    top_matches = [match for match in match_results if match['score'] >= threshold]
    
    if not top_matches:
        st.info(f"No matches found above the threshold of {threshold}. Try adjusting the threshold in the sidebar.")
        return
    
    # Display each match with details
    for i, match in enumerate(top_matches):
        with st.expander(f"{match['resume_filename']} ↔ {match['job_filename']} (Score: {match['score']:.2f})", expanded=(i==0)):
            col1, col2 = st.columns(2)
            
            with col1:
                # Find the resume by ID
                for resume in st.session_state.resumes:
                    if resume['id'] == match['resume_id']:
                        document_view.render_document_card(
                            resume['filename'], 
                            resume['processed']['clean_text'][:500] + "...", 
                            "Resume"
                        )
                        break
            
            with col2:
                # Find the job by ID
                for job in st.session_state.job_descriptions:
                    if job['id'] == match['job_id']:
                        document_view.render_document_card(
                            job['filename'], 
                            job['processed']['clean_text'][:500] + "...", 
                            "Job Description"
                        )
                        break
            
            # Show match details
            st.subheader("Match Details")
            
            # Create columns for scores
            c1, c2, c3 = st.columns(3)
            
            with c1:
                charts.render_gauge(match['score'], "Overall Score", 0, 1)
            
            with c2:
                charts.render_gauge(
                    match['details']['semantic_similarity'], 
                    "Semantic Similarity", 
                    0, 
                    1
                )
            
            with c3:
                charts.render_gauge(
                    match['details']['keyword_similarity'], 
                    "Keyword Similarity", 
                    0, 
                    1
                )
            
            # Display common keywords
            if 'common_keywords' in match['details'] and match['details']['common_keywords']:
                st.subheader("Common Keywords")
                
                # Create keyword pills
                html_keywords = '<div style="display: flex; flex-wrap: wrap; gap: 8px;">'
                for keyword in match['details']['common_keywords']:
                    html_keywords += f'<div style="background-color: #e0f7fa; border-radius: 16px; padding: 6px 12px; font-size: 14px;">{keyword}</div>'
                html_keywords += '</div>'
                
                st.markdown(html_keywords, unsafe_allow_html=True)
            
            # Display missing keywords
            if 'missing_keywords' in match['details'] and match['details']['missing_keywords']:
                st.subheader("Missing Skills/Keywords")
                
                # Create keyword pills for missing keywords
                html_missing = '<div style="display: flex; flex-wrap: wrap; gap: 8px;">'
                for keyword in match['details']['missing_keywords']:
                    html_missing += f'<div style="background-color: #ffebee; border-radius: 16px; padding: 6px 12px; font-size: 14px;">{keyword}</div>'
                html_missing += '</div>'
                
                st.markdown(html_missing, unsafe_allow_html=True)
    
    # Export options
    st.subheader("Export Results")
    
    if st.button("Export Results as JSON"):
        # Prepare export data
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
        
        # Convert to JSON string
        export_json = json.dumps(export_data, indent=2)
        
        # Create download button
        st.download_button(
            label="Download JSON",
            data=export_json,
            file_name=f"match_results_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )