import logging
import os
import sys
from datetime import datetime
from typing import Optional

def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Path to log file (None for auto-generated filename)
        log_to_console: Whether to log to console
        
    Returns:
        Logger object
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate log filename with timestamp if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"app_log_{timestamp}.log")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Create console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    return root_logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module
    
    Args:
        name: Name of the module (typically __name__)
        
    Returns:
        Logger object for the module
    """
    # If root logger has no handlers, set up logging
    if not logging.getLogger().handlers:
        setup_logging()
    
    return logging.getLogger(name)