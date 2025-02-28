import streamlit as st
import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Add project root to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.nlp.summarizer import TextSummarizer
from src.document_processor.chunker import TextChunker
from src.utils.logger import get_logger

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = get_logger(__name__)

def render():
    st.title("Document Analysis")
    
    # Check if documents are uploaded
    if not st.session_state.get("resumes") and not st.session_state.get("job_descriptions"):
        st.warning("Please upload resumes or job descriptions first.")
        st.info("Go to the Upload page to add documents.")
        return
    
    # Create tabs for Resume Analysis and Job Description Analysis
    tab1, tab2 = st.tabs(["Resume Analysis", "Job Description Analysis"])
    
    with tab1:
        if not st.session_state.get("resumes"):
            st.info("No resumes uploaded yet.")
        else:
            analyze_resumes()
    
    with tab2:
        if not st.session_state.get("job_descriptions"):
            st.info("No job descriptions uploaded yet.")
        else:
            analyze_job_descriptions()

def analyze_resumes():
    st.header("Resume Analysis")
    
    # Create a selectbox for resumes
    resume_options = [f"{i+1}. {resume['filename']}" for i, resume in enumerate(st.session_state.resumes)]
    selected_resume_index = st.selectbox("Select Resume to Analyze", range(len(resume_options)), format_func=lambda x: resume_options[x])
    
    # Get the selected resume
    selected_resume = st.session_state.resumes[selected_resume_index]
    
    # Display the summary if available, otherwise generate it
    if "summary" not in selected_resume:
        with st.spinner("Generating summary..."):
            try:
                summarizer = TextSummarizer()
                summary = summarizer.summarize(selected_resume['original_text'])
                selected_resume["summary"] = summary
                # Update the resume in the session state
                st.session_state.resumes[selected_resume_index] = selected_resume
            except Exception as e:
                logger.error(f"Error generating summary: {str(e)}")
                summary = "Failed to generate summary."
                selected_resume["summary"] = summary
    
    # Overview metrics
    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Length", f"{len(selected_resume['original_text'])} chars")
    with col2:
        st.metric("Word Count", len(selected_resume['original_text'].split()))
    with col3:
        st.metric("Processed Tokens", selected_resume['processed'].get('token_count', 0))
    with col4:
        entity_count = sum(len(entities) for entities in selected_resume['processed'].get('entities', {}).values())
        st.metric("Entities Detected", entity_count)
    
    # Display summary
    st.subheader("Summary")
    st.write(selected_resume.get("summary", "No summary available."))
    
    # Entity Analysis
    st.subheader("Entity Analysis")
    
    if 'entities' in selected_resume['processed']:
        # Create expandable sections for each entity type
        entities = selected_resume['processed']['entities']
        
        # Create a bar chart of entity counts
        entity_counts = {ent_type: len(ents) for ent_type, ents in entities.items() if ents}
        if entity_counts:
            df = pd.DataFrame({
                'Entity Type': list(entity_counts.keys()),
                'Count': list(entity_counts.values())
            })
            fig = px.bar(df, x='Entity Type', y='Count', title="Entity Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Display entities by type
        for entity_type, entity_list in entities.items():
            if entity_list:
                with st.expander(f"{entity_type} ({len(entity_list)})"):
                    st.write(", ".join(sorted(set(entity_list))))
    else:
        st.info("No entities detected in this resume.")
    
    # Keyword Analysis
    st.subheader("Keyword Analysis")
    
    # Get processed text tokens
    if 'clean_text' in selected_resume['processed']:
        tokens = selected_resume['processed']['clean_text'].split()
        
        # Remove common stopwords for better keyword analysis
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words and len(token) > 2]
        
        # Get token frequency
        token_freq = Counter(filtered_tokens)
        common_tokens = token_freq.most_common(20)
        
        if common_tokens:
            # Create a dataframe for visualization
            df = pd.DataFrame(common_tokens, columns=['Token', 'Frequency'])
            
            # Create bar chart
            fig = px.bar(df, x='Token', y='Frequency', title="Top Keywords")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No significant keywords found.")
    else:
        st.info("Processed text not available for keyword analysis.")
    
    # Document Content
    st.subheader("Document Content")
    with st.expander("View Full Content", expanded=False):
        st.text_area("Original Text", selected_resume['original_text'], height=400)

def analyze_job_descriptions():
    st.header("Job Description Analysis")
    
    # Create a selectbox for job descriptions
    jd_options = [f"{i+1}. {jd['filename']}" for i, jd in enumerate(st.session_state.job_descriptions)]
    selected_jd_index = st.selectbox("Select Job Description to Analyze", range(len(jd_options)), format_func=lambda x: jd_options[x])
    
    # Get the selected job description
    selected_jd = st.session_state.job_descriptions[selected_jd_index]
    
    # Display the summary if available, otherwise generate it
    if "summary" not in selected_jd:
        with st.spinner("Generating summary..."):
            try:
                summarizer = TextSummarizer()
                summary = summarizer.summarize(selected_jd['original_text'])
                selected_jd["summary"] = summary
                # Update the job description in the session state
                st.session_state.job_descriptions[selected_jd_index] = selected_jd
            except Exception as e:
                logger.error(f"Error generating summary: {str(e)}")
                summary = "Failed to generate summary."
                selected_jd["summary"] = summary
    
    # Overview metrics
    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Length", f"{len(selected_jd['original_text'])} chars")
    with col2:
        st.metric("Word Count", len(selected_jd['original_text'].split()))
    with col3:
        st.metric("Processed Tokens", selected_jd['processed'].get('token_count', 0))
    with col4:
        entity_count = sum(len(entities) for entities in selected_jd['processed'].get('entities', {}).values())
        st.metric("Entities Detected", entity_count)
    
    # Display summary
    st.subheader("Summary")
    st.write(selected_jd.get("summary", "No summary available."))
    
    # Extract sections from job description (if available)
    st.subheader("Job Sections")
    try:
        chunker = TextChunker()
        sections = chunker.chunk_by_section(selected_jd['original_text'])
        
        if sections:
            for i, (section_name, section_text) in enumerate(sections.items()):
                with st.expander(f"{section_name}", expanded=i==0):
                    st.write(section_text)
        else:
            st.info("No clear sections detected. Displaying full content.")
            with st.expander("Full Job Description", expanded=True):
                st.write(selected_jd['original_text'])
    except Exception as e:
        logger.error(f"Error chunking job description: {str(e)}")
        st.error("Could not analyze job sections. Displaying full content.")
        with st.expander("Full Job Description", expanded=True):
            st.write(selected_jd['original_text'])
    
    # Entity Analysis
    st.subheader("Entity Analysis")
    
    if 'entities' in selected_jd['processed']:
        # Create expandable sections for each entity type
        entities = selected_jd['processed']['entities']
        
        # Create a bar chart of entity counts
        entity_counts = {ent_type: len(ents) for ent_type, ents in entities.items() if ents}
        if entity_counts:
            df = pd.DataFrame({
                'Entity Type': list(entity_counts.keys()),
                'Count': list(entity_counts.values())
            })
            fig = px.bar(df, x='Entity Type', y='Count', title="Entity Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Display entities by type
        for entity_type, entity_list in entities.items():
            if entity_list:
                with st.expander(f"{entity_type} ({len(entity_list)})"):
                    st.write(", ".join(sorted(set(entity_list))))
    else:
        st.info("No entities detected in this job description.")
    
    # Keyword Analysis
    st.subheader("Keyword Analysis")
    
    # Get processed text tokens
    if 'clean_text' in selected_jd['processed']:
        tokens = selected_jd['processed']['clean_text'].split()
        
        # Remove common stopwords for better keyword analysis
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words and len(token) > 2]
        
        # Get token frequency
        token_freq = Counter(filtered_tokens)
        common_tokens = token_freq.most_common(20)
        
        if common_tokens:
            # Create a dataframe for visualization
            df = pd.DataFrame(common_tokens, columns=['Token', 'Frequency'])
            
            # Create bar chart
            fig = px.bar(df, x='Token', y='Frequency', title="Top Keywords")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No significant keywords found.")
    else:
        st.info("Processed text not available for keyword analysis.")
    
    # Document Content
    st.subheader("Document Content")
    with st.expander("View Full Content", expanded=False):
        st.text_area("Original Text", selected_jd['original_text'], height=400)