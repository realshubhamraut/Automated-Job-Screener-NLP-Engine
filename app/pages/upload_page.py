import streamlit as st
import os
import pandas as pd
from datetime import datetime
import uuid
import json
import shutil
import pickle

from src.document_processor.ingestion import DocumentLoader
from src.document_processor.processor import DocumentProcessor
from src.utils.logger import get_logger
from src.config import ALLOWED_RESUME_TYPES, ALLOWED_JOB_DESC_TYPES, MAX_FILE_SIZE_MB

logger = get_logger(__name__)

# Define paths for persistent storage
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")

# Ensure directories exist
def ensure_dirs_exist():
    for doc_type in ["resumes", "job_descriptions"]:
        os.makedirs(os.path.join(RAW_DIR, doc_type), exist_ok=True)
        os.makedirs(os.path.join(PROCESSED_DIR, doc_type), exist_ok=True)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Load documents from persistent storage
def load_documents():
    # Only load if the session_state is empty
    if not st.session_state.resumes:
        try:
            resumes_dir = os.path.join(PROCESSED_DIR, "resumes")
            if os.path.exists(resumes_dir):
                for filename in os.listdir(resumes_dir):
                    if filename.endswith('.json'):
                        file_path = os.path.join(resumes_dir, filename)
                        with open(file_path, 'r') as f:
                            resume = json.load(f)
                            st.session_state.resumes.append(resume)
                logger.info(f"Loaded {len(st.session_state.resumes)} resumes from storage")
        except Exception as e:
            logger.error(f"Error loading resumes: {str(e)}")
    
    if not st.session_state.job_descriptions:
        try:
            jobs_dir = os.path.join(PROCESSED_DIR, "job_descriptions")
            if os.path.exists(jobs_dir):
                for filename in os.listdir(jobs_dir):
                    if filename.endswith('.json'):
                        file_path = os.path.join(jobs_dir, filename)
                        with open(file_path, 'r') as f:
                            job = json.load(f)
                            st.session_state.job_descriptions.append(job)
                logger.info(f"Loaded {len(st.session_state.job_descriptions)} job descriptions from storage")
        except Exception as e:
            logger.error(f"Error loading job descriptions: {str(e)}")

def render():
    # Ensure directories exist
    ensure_dirs_exist()
    
    # Load documents from persistent storage
    load_documents()
    
    # Create a layout with columns for title and clear button
    title_col, spacer, clear_btn_col = st.columns([5, 1, 2])
    
    # Title in left column
    with title_col:
        st.title("Upload Documents")
    
    # Clear button in right column
    with clear_btn_col:
        # Add some vertical space to align with title
        st.write("")  # Empty space for alignment
        if st.button("Clear All Documents", key="clear_all_btn", type="primary"):
            clear_all_documents()
            st.success("All documents have been cleared.")
            st.rerun()
    
    st.write("Upload resumes and job descriptions in batch to begin the matching process.")
    
    # Set up tabs for different upload types
    resume_tab, job_desc_tab, manage_tab = st.tabs(["*Upload Resumes*", "*Upload Job Descriptions*", "*Manage Uploads*"])
    
    # Resume Upload Tab
    with resume_tab:
        st.subheader("Upload Resume Files")
        st.write("Supported formats: PDF, DOCX, and TXT")
        
        # File uploader for resumes
        resume_files = st.file_uploader(
            "Choose resume files",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt"],
            key="resume_uploader"
        )
        
        if resume_files:
            if st.button("Process Resumes", key="process_resumes_btn"):
                with st.spinner("Processing resumes..."):
                    process_uploads(resume_files, "resume")
    
    # Job Description Upload Tab
    with job_desc_tab:
        st.subheader("Upload Job Description Files")
        st.write("Supported formats: PDF, DOCX, TXT, and Markdown")
        
        # File uploader for job descriptions
        job_desc_files = st.file_uploader(
            "Choose job description files",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt", "md"],
            key="job_desc_uploader"
        )
        
        if job_desc_files:
            if st.button("Process Job Descriptions", key="process_jobs_btn"):
                with st.spinner("Processing job descriptions..."):
                    process_uploads(job_desc_files, "job_description")
    
    # Manage Uploads Tab
    with manage_tab:
        
        # Place tables side by side
        table_col1, table_col2 = st.columns(2)
        
        with table_col1:
            if st.session_state.resumes:
                st.write(f"Resumes: {len(st.session_state.resumes)}")
                display_document_table(st.session_state.resumes)
            else:
                st.info("No resumes uploaded yet.")
        
        with table_col2:
            if st.session_state.job_descriptions:
                st.write(f"Job Descriptions: {len(st.session_state.job_descriptions)}")
                display_document_table(st.session_state.job_descriptions)
            else:
                st.info("No job descriptions uploaded yet.")
        
        # Combine documents for preview
        combined_docs = []
        combined_docs.extend(st.session_state.resumes)
        combined_docs.extend(st.session_state.job_descriptions)
        
        # Then, single preview section for all documents
        if combined_docs:
            display_document_previews(combined_docs)
        
def display_document_table(documents):
    """Display just the table part without previews"""
    # Create a DataFrame for display
    df = pd.DataFrame([{
        'ID': doc['id'],
        'Filename': doc['filename'],
        'Upload Time': doc['upload_time'],
        'Text Length': len(doc['original_text']) if 'original_text' in doc else 0
    } for doc in documents])
    
    # Display the table
    st.dataframe(df)


def display_document_previews(documents):
    """Display side-by-side document preview columns with resume and job description types"""
    
    # Split documents by type
    resumes = [doc for doc in documents if doc['document_type'] == 'resume']
    job_descriptions = [doc for doc in documents if doc['document_type'] == 'job_description']
    
    # Create two columns for side-by-side preview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Resume Previews")
        
        if resumes:
            # Create options dictionary for resumes only
            resume_options = {doc['filename']: i for i, doc in enumerate(resumes)}
            
            selected_resume = st.selectbox(
                "", 
                list(resume_options.keys()), 
                key="select_resume"
            )
            
            if selected_resume:
                document = resumes[resume_options[selected_resume]]
                st.text_area(
                    "Preview", 
                    document.get('original_text', '')[:1000], 
                    height=300,
                    key=f"preview_{document['id']}_resume"
                )
                
                # Button to remove document
                if st.button(f"Remove {selected_resume}", key=f"remove_{document['id']}_resume"):
                    remove_document(document)
                    st.success(f"Removed {selected_resume}")
                    st.rerun()
        else:
            st.info("No resumes available for preview")
            st.text_area(
                "Preview", 
                "Upload resumes to view preview", 
                height=300,
                key="preview_empty_resume"
            )
    
    with col2:
        st.subheader("Job Description Previews")
        
        if job_descriptions:
            # Create options dictionary for job descriptions only
            job_options = {doc['filename']: i for i, doc in enumerate(job_descriptions)}
            
            selected_job = st.selectbox(
                "", 
                list(job_options.keys()), 
                key="select_job"
            )
            
            if selected_job:
                document = job_descriptions[job_options[selected_job]]
                st.text_area(
                    "Preview", 
                    document.get('original_text', '')[:1000], 
                    height=300,
                    key=f"preview_{document['id']}_job"
                )
                
                # Button to remove document
                if st.button(f"Remove {selected_job}", key=f"remove_{document['id']}_job"):
                    remove_document(document)
                    st.success(f"Removed {selected_job}")
                    st.rerun()
        else:
            st.info("No job descriptions available for preview")
            st.text_area(
                "Preview", 
                "Upload job descriptions to view preview", 
                height=300,
                key="preview_empty_job"
            )


def process_uploads(files, doc_type):
    """
    Process uploaded files and save to persistent storage
    
    Args:
        files: List of uploaded files
        doc_type: Type of document ('resume' or 'job_description')
    """
    # Initialize loader and processor
    loader = DocumentLoader()
    processor = DocumentProcessor()
    
    # Process each file
    for file in files:
        try:
            # Check file size
            if file.size > (MAX_FILE_SIZE_MB * 1024 * 1024):
                st.error(f"File {file.name} is too large. Maximum size is {MAX_FILE_SIZE_MB}MB.")
                continue
            
            # Check extension
            _, ext = os.path.splitext(file.name)
            if doc_type == 'resume' and ext.lower() not in ALLOWED_RESUME_TYPES:
                st.error(f"Invalid resume file type: {ext}")
                continue
            elif doc_type == 'job_description' and ext.lower() not in ALLOWED_JOB_DESC_TYPES:
                st.error(f"Invalid job description file type: {ext}")
                continue
            
            # Generate shorter ID (less than 7 chars)
            # First letter of document type + 5 random chars
            doc_id = f"{doc_type[0]}{uuid.uuid4().hex[:5]}"
            
            # Save raw file to disk
            folder_type = "resumes" if doc_type == "resume" else "job_descriptions"
            raw_file_path = os.path.join(RAW_DIR, folder_type, f"{doc_id}{ext}")
            
            with open(raw_file_path, "wb") as f:
                f.write(file.getbuffer())
            
            # Extract text from document
            text = loader.load_document(file)
            
            # Process the document
            processed_doc = processor.process_document(text, doc_type=doc_type)
            
            # Create document dictionary with HH:MM:SS time format
            document = {
                'id': doc_id,
                'filename': file.name,
                'upload_time': datetime.now().strftime('%H:%M:%S'),
                'document_type': doc_type,
                'original_text': text,
                'processed': processed_doc,
                'file_path': raw_file_path
            }
            
            # Save processed document to JSON
            processed_file_path = os.path.join(PROCESSED_DIR, folder_type, f"{doc_id}.json")
            with open(processed_file_path, "w") as f:
                json.dump(document, f, indent=2)
            
            # Save vector representation (assuming processed_doc has vector data)
            vector_file_path = os.path.join(VECTOR_STORE_DIR, f"{doc_id}.pkl")
            with open(vector_file_path, "wb") as f:
                pickle.dump(processed_doc, f)
            
            # Add document to session state
            if doc_type == 'resume':
                st.session_state.resumes.append(document)
            else:
                st.session_state.job_descriptions.append(document)
            
            # Log success
            logger.info(f"Successfully processed {doc_type}: {file.name}")
            st.success(f"Successfully processed {file.name}")
            
        except Exception as e:
            logger.error(f"Error processing {doc_type} {file.name}: {str(e)}")
            st.error(f"Error processing {file.name}: {str(e)}")


def clear_all_documents():
    """
    Clear all documents from both session state and disk
    """
    # Clear session state
    st.session_state.resumes = []
    st.session_state.job_descriptions = []
    st.session_state.match_results = []
    
    # Remove files from disk
    try:
        # Clear raw files
        for folder_type in ["resumes", "job_descriptions"]:
            folder_path = os.path.join(RAW_DIR, folder_type)
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        
        # Clear processed files
        for folder_type in ["resumes", "job_descriptions"]:
            folder_path = os.path.join(PROCESSED_DIR, folder_type)
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
        
        # Clear vector store
        for filename in os.listdir(VECTOR_STORE_DIR):
            file_path = os.path.join(VECTOR_STORE_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                
        logger.info("All documents cleared from storage")
    except Exception as e:
        logger.error(f"Error clearing documents: {str(e)}")
        st.error(f"Error clearing documents: {str(e)}")


def remove_document(document):
    """
    Remove a single document from session state and disk
    
    Args:
        document: Document dictionary to remove
    """
    try:
        # Remove from session state
        if document['document_type'] == 'resume':
            st.session_state.resumes.remove(document)
        else:
            st.session_state.job_descriptions.remove(document)
            
        # Remove raw file if it exists
        if 'file_path' in document and os.path.exists(document['file_path']):
            os.remove(document['file_path'])
            
        # Remove processed JSON file
        folder_type = "resumes" if document['document_type'] == "resume" else "job_descriptions"
        processed_path = os.path.join(PROCESSED_DIR, folder_type, f"{document['id']}.json")
        if os.path.exists(processed_path):
            os.remove(processed_path)
            
        # Remove vector file
        vector_path = os.path.join(VECTOR_STORE_DIR, f"{document['id']}.pkl")
        if os.path.exists(vector_path):
            os.remove(vector_path)
            
        logger.info(f"Removed document {document['filename']}")
    except Exception as e:
        logger.error(f"Error removing document {document['filename']}: {str(e)}")
        st.error(f"Error removing document: {str(e)}")