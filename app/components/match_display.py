import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any, List, Tuple

def render_match_result(resume: Dict[str, Any], job: Dict[str, Any], score: float, 
                       details: Dict[str, Any] = None):
    """
    Display a match result between a resume and job description
    
    Args:
        resume: Resume document dictionary
        job: Job description dictionary  
        score: Overall match score
        details: Optional details about the match (semantic score, keyword score, etc.)
    """
    st.markdown(f"### Match Score: {score:.2f}")
    
    # Create a progress bar for the overall score
    st.progress(score)
    
    # Create columns for the match components
    col1, col2 = st.columns(2)
    
    with col1:
        if details and "semantic_score" in details:
            st.metric("Semantic Score", f"{details['semantic_score']:.2f}")
    
    with col2:
        if details and "keyword_score" in details:
            st.metric("Keyword Score", f"{details['keyword_score']:.2f}")
    
    # Display matching skills
    if details and "matching_skills" in details and details["matching_skills"]:
        with st.expander("Matching Skills", expanded=True):
            matching_skills = details["matching_skills"]
            st.write(", ".join(sorted(matching_skills)))
    
    # Display missing skills
    if details and "missing_skills" in details and details["missing_skills"]:
        with st.expander("Missing Skills", expanded=True):
            missing_skills = details["missing_skills"]
            st.write(", ".join(sorted(missing_skills)))
    
    # Document comparison
    with st.expander("Document Comparison", expanded=False):
        st.subheader("Resume")
        st.write(f"**{resume['filename']}**")
        st.write(resume.get("summary", "No summary available"))
        
        st.subheader("Job Description")
        st.write(f"**{job['filename']}**")
        st.write(job.get("summary", "No summary available"))

def display_match_table(matches: List[Tuple], show_detailed=True):
    """
    Display a table of matches
    
    Args:
        matches: List of (resume, score, details) or (job, score, details) tuples
        show_detailed: Whether to show detailed match information
    """
    if not matches:
        st.info("No matches found above the threshold.")
        return
    
    # Create dataframe for matches
    data = []
    
    for doc, score, details in matches:
        item = {
            "Document": doc['filename'],
            "Match Score": f"{score:.2f}",
            "Skills Match": f"{len(details.get('matching_skills', []))}/{len(details.get('required_skills', []))}",
        }
        
        # Add detailed scores if available
        if show_detailed and details:
            if "semantic_score" in details:
                item["Semantic Score"] = f"{details['semantic_score']:.2f}"
            if "keyword_score" in details:
                item["Keyword Score"] = f"{details['keyword_score']:.2f}"
        
        data.append(item)
    
    # Convert to dataframe and display
    df = pd.DataFrame(data)
    st.dataframe(df, hide_index=True)
    
    # Create bar chart of match scores
    fig = px.bar(
        df, 
        x="Document", 
        y="Match Score", 
        title="Match Scores",
        color="Match Score",
        color_continuous_scale="viridis"
    )
    st.plotly_chart(fig, use_container_width=True)