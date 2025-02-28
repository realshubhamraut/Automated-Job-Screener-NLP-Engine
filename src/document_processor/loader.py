import logging
import os
import sys
from datetime import datetime
from typing import Optional

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure default log file
DEFAULT_LOG_FILE = f"logs/app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def setup_logging(log_level: Optional[int] = None, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure logging for the application
    
    Args:
        log_level (int, optional): Logging level (e.g., logging.INFO)
        log_file (str, optional): Path to log file
        
    Returns:
        logging.Logger: Root logger
    """
    if log_level is None:
        # Get log level from environment or use INFO as default
        log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        
    if log_file is None:
        log_file = DEFAULT_LOG_FILE
        
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create and get the root logger
    logger = logging.getLogger()
    logger.info(f"Logging initialized. Log level: {logging.getLevelName(log_level)}. Log file: {log_file}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name
    
    Args:
        name (str): Logger name, typically __name__
        
    Returns:
        logging.Logger: Logger instance
    """
    logger = logging.getLogger(name)
    
    # If root logger is not configured, set up basic configuration
    if not logging.root.handlers:
        setup_logging()
        
    return logger