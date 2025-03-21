import os
from pathlib import Path
import logging

# Project directory paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Default log file
DEFAULT_LOG_FILE = LOG_DIR / "app.log"

# Document directories
RESUME_DIR = DATA_DIR / "resumes"
JOB_DESC_DIR = DATA_DIR / "job_descriptions"
RESUME_RAW_DIR = RESUME_DIR / "raw"
JOB_DESC_RAW_DIR = JOB_DESC_DIR / "raw"
RESUME_PROCESSED_DIR = DATA_DIR / "processed" / "resumes"
JOB_DESC_PROCESSED_DIR = DATA_DIR / "processed" / "job_descriptions"
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

# Fine-tuned models directory and settings
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Fine-tuned model paths
FINETUNED_EMBEDDING_MODEL = MODELS_DIR / "finetuned_embeddings"
FINETUNED_SKILL_EXTRACTOR = MODELS_DIR / "skill_extractor" 
FINETUNED_VOICE_ANALYZER = MODELS_DIR / "voice_analyzer"

# Fine-tuning settings
FINETUNING_EPOCHS = 3
FINETUNING_BATCH_SIZE = 16
FINETUNING_LEARNING_RATE = 2e-5

# Flag to enable/disable fine-tuned models
USE_FINETUNED_MODELS = True

# LLM API configuration
API_KEYS = {
    "gemini": os.getenv("GEMINI_API_KEY", ""),
    # Add other API keys as needed
}

# Logging level
LOG_LEVEL = logging.INFO