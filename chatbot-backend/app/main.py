# Main entrypoint for the Chatbot Backend FastAPI application.

import logging
from fastapi import FastAPI
from app.config import settings
from app.routes.chatwoot_webhook import router as chatwoot_router
from app.utils.logger import setup_logger


# Setup the logger
setup_logger()
logger = logging.getLogger(__name__)

# Create the FastAPI app instance
app = FastAPI(
    title="Intelligent Omnichannel Chatbot",
    description="Backend service to power the multi-store AI chatbot.",
    version="1.0.0"
)

# Include the webhook router
app.include_router(chatwoot_router, prefix="/api/v1", tags=["Webhooks"])

@app.on_event("startup")
async def startup_event():
    """
    Event handler for application startup.
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


@app.on_event("shutdown")
async def shutdown_event():
    """
    Event handler for application shutdown.
    """
    logger.info("Chatbot backend application shutting down...")


@app.get("/", tags=["Health Check"])
async def health_check():
    """
    Health check endpoint to verify that the service is running.
    """
    return {"status": "ok", "message": "Chatbot backend is running."}
