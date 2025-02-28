"""
Global configuration settings for the application
"""
import os
from pathlib import Path
from typing import Dict, Any

# Application identity
APP_NAME = "AI Job Matcher"
APP_VERSION = "1.0.0"

# Project directory structure
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Raw document storage paths
RESUME_RAW_DIR = DATA_DIR / "resumes" / "raw"
JOB_DESC_RAW_DIR = DATA_DIR / "job_descriptions" / "raw"
RESUME_RAW_DIR.mkdir(exist_ok=True, parents=True)
JOB_DESC_RAW_DIR.mkdir(exist_ok=True, parents=True)

# Processed document storage paths
RESUME_PROCESSED_DIR = DATA_DIR / "resumes" / "processed"
JOB_DESC_PROCESSED_DIR = DATA_DIR / "job_descriptions" / "processed"
RESUME_PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
JOB_DESC_PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

# Vector database configuration
VECTOR_DB_TYPE = "faiss"  # Options: 'faiss', 'chroma', 'milvus'
VECTOR_DB_DIR = DATA_DIR / "vector_db"
VECTOR_STORE_DIR = VECTOR_DB_DIR  # Alternative name used in some modules
VECTOR_DB_DIR.mkdir(exist_ok=True)

# Model configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"

# Document processing settings
DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 50

# Matching settings
DEFAULT_SIMILARITY_THRESHOLD = 0.6
DEFAULT_HYBRID_WEIGHT = 0.7  # Weight for semantic search vs keyword matching

# LLM API configuration
# Using Google API exclusively (no OpenAI or Anthropic)
API_KEYS = {
    "google": os.environ.get("GOOGLE_API_KEY", ""),
    # These are set to empty strings to avoid errors if code tries to access them
    "openai": "",
    "anthropic": "",
    "cohere": ""
}

# Default LLM provider - set to Google
DEFAULT_LLM_PROVIDER = "google"
GOOGLE_MODEL = "gemini-pro"  # Google's generative model

# Application settings
STREAMLIT_THEME = {
    "primary_color": "#0083B8",
    "background_color": "#FFFFFF",
    "secondary_background_color": "#F0F2F6",
    "text_color": "#262730",
    "font": "sans serif",
}

# Default system prompts for LLM interactions
SYSTEM_PROMPTS = {
    "interview_questions": "Generate relevant interview questions based on the job description and resume. Focus on assessing the candidate's skills, experience, and fit for the role.",
    "skill_assessment": "Analyze the match between the resume and job description. Identify strengths, weaknesses, and areas for improvement.",
    "resume_feedback": "Provide constructive feedback on the resume. Suggest improvements for better alignment with the target job description.",
}

# File type settings
ALLOWED_RESUME_TYPES = [".pdf", ".docx", ".txt"]
ALLOWED_JOB_DESC_TYPES = [".pdf", ".docx", ".txt", ".md"]
MAX_FILE_SIZE_MB = 10