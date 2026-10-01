# Utility for setting up a standardized logger for the application.

import logging
import re
import sys
from app.config import settings

_TOKEN_QUERY_RE = re.compile(r"([?&]token=)[^&\s]*", re.IGNORECASE)


def redact_url_secrets(value: str) -> str:
    """Replace the value of a `token` query parameter with ***."""
    return _TOKEN_QUERY_RE.sub(r"\1***", value)


class RedactTokenFilter(logging.Filter):
    """Keep the webhook URL token (?token=...) out of access logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple) and record.args:
            record.args = tuple(redact_url_secrets(a) if isinstance(a, str) else a for a in record.args)
        elif isinstance(record.msg, str):
            record.msg = redact_url_secrets(record.msg)
        return True


def install_access_log_redaction() -> None:
    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, RedactTokenFilter) for f in access_logger.filters):
        access_logger.addFilter(RedactTokenFilter())

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

    install_access_log_redaction()

    logging.info(f"Logger configured with level: {log_level}")
