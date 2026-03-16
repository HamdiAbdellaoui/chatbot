# Utility for setting up a standardized logger for the application.

import logging
import sys
from app.config import settings

def setup_logger():
    """
    Configures the root logger for the application.
    - Sets the logging level from environment settings.
    - Uses a standardized format for log messages.
    - Outputs logs to the console (stdout).
    """
    log_level = settings.LOG_LEVEL.upper()
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create a handler for stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger
    # Check if handlers are already present to avoid duplication
    if not logger.handlers:
        logger.addHandler(handler)

    logging.info(f"Logger configured with level: {log_level}")
