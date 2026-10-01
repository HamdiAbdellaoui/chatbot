# Main entrypoint for the Chatbot Backend FastAPI application.

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.config import settings
from app.routes.chatwoot_webhook import router as chatwoot_router
from app.routes.admin_active_learning import router as admin_active_learning_router
from app.utils.logger import setup_logger

# Setup the logger
setup_logger()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: startup before `yield`, shutdown after.
    - Initializes database connections
    - Loads ML models
    - etc.
    """
    logger.info("Chatbot backend application starting up...")
    logger.info(f"Log level: {settings.LOG_LEVEL}")
    # In a real application, you would initialize your RAG pipeline,
    # database connections, or other resources here.
    # For now, we just log a message.
    logger.info("RAG Pipeline and other services will be initialized here.")
    yield
    logger.info("Chatbot backend application shutting down...")


# Create the FastAPI app instance
app = FastAPI(
    title="Intelligent Omnichannel Chatbot",
    description="Backend service to power the multi-store AI chatbot.",
    version="1.0.0",
    lifespan=lifespan,
)

# Include the webhook router
app.include_router(chatwoot_router, prefix="/api/v1", tags=["Webhooks"])

# Admin endpoints (protected by X-Admin-Token)
app.include_router(admin_active_learning_router, prefix="/api/v1/admin", tags=["Admin"])


@app.get("/", tags=["Health Check"])
async def health_check():
    """
    Health check endpoint to verify that the service is running.
    """
    return {"status": "ok", "message": "Chatbot backend is running."}
