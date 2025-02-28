import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional

def render_document_card(filename: str, content_preview: str, document_type: str = "Document"):
    """
    Display a simple card view of a document with basic information
    
    Args:
        filename: The filename of the document
        content_preview: A preview of the document content
        document_type: The type of document (e.g., "Resume", "Job Description")
    """
    st.markdown(f"**{document_type}**: {filename}")
    st.markdown(
        f"""
        <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; height: 200px; overflow-y: auto;">
        {content_preview}
        </div>
        """, 
        unsafe_allow_html=True
    )

def display_document_summary(document: Dict[str, Any], is_resume: bool = True):
    """
    Display a summary of a document (resume or job description)
    
    Args:
        document: The document dictionary containing processed information
        is_resume: Whether this document is a resume (True) or job description (False)
    """
    # Basic document info
    st.subheader(document.get("filename", "Unnamed Document"))
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Length", f"{len(document['original_text'])} chars")
    
    with col2:
        word_count = len(document['original_text'].split())
        st.metric("Words", f"{word_count}")
    
    with col3:
        if 'entities' in document.get('processed', {}) and 'SKILL' in document['processed']['entities']:
            skills_count = len(document['processed']['entities']['SKILL'])
            st.metric("Skills", f"{skills_count}")
        else:
            st.metric("Skills", "0")
    
    # Summary section
    if "summary" in document:
        with st.expander("Summary", expanded=True):
            st.write(document["summary"])
    
    # Entity display
    if 'entities' in document.get('processed', {}):
        with st.expander("Entities", expanded=False):
            entities = document['processed']['entities']
            
            # Display skills first if they exist
            if 'SKILL' in entities and entities['SKILL']:
                st.write("**Skills:**")
                skills_df = pd.DataFrame({"Skill": sorted(entities['SKILL'])})
                st.dataframe(skills_df, hide_index=True)
            
            # Display other relevant entities based on document type
            if is_resume:
                entity_types = ['ORG', 'EDUCATION', 'EXPERIENCE', 'DATE']
            else:
                entity_types = ['ORG', 'REQUIREMENT', 'JOB_TITLE']
            
            for entity_type in entity_types:
                if entity_type in entities and entities[entity_type]:
                    st.write(f"**{entity_type.replace('_', ' ').title()}:**")
                    st.write(", ".join(sorted(entities[entity_type])))
    
    # Full text 
    with st.expander("Full Text", expanded=False):
        st.text_area("", document['original_text'], height=300)

def display_document_list(documents: List[Dict[str, Any]], 
                        title: str = "Documents",
                        on_select=None,
                        display_count: int = 3):
    """
    Display a list of documents with expandable details
    
    Args:
        documents: List of document dictionaries
        title: Title for the document list
        on_select: Optional callback function when a document is selected
        display_count: Number of documents to display initially
    """
    if not documents:
        st.info(f"No {title.lower()} available.")
        return
    
    st.subheader(title)
    
    for i, doc in enumerate(documents[:display_count]):
        with st.expander(f"{i+1}. {doc['filename']}", expanded=False):
            # Display basic info about document
            cols = st.columns([3, 1, 1])
            with cols[0]:
                st.write(doc.get("summary", "No summary available.")[:200] + "...")
            with cols[1]:
                word_count = len(doc['original_text'].split())
                st.metric("Words", f"{word_count}")
            with cols[2]:
                if 'entities' in doc.get('processed', {}) and 'SKILL' in doc['processed']['entities']:
                    st.metric("Skills", f"{len(doc['processed']['entities']['SKILL'])}")
            
            # Add View Details button if callback provided
            if on_select:
                if st.button("View Details", key=f"view_{doc['id']}"):
                    on_select(doc)
    
    if len(documents) > display_count:
        with st.expander("More...", expanded=False):
            for i, doc in enumerate(documents[display_count:], display_count):
                st.write(f"{i+1}. {doc['filename']}")
                if on_select:
                    if st.button("View", key=f"view_{doc['id']}"):
                        on_select(doc)