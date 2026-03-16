# Configuration management for the application.
# Uses Pydantic's BaseSettings to load environment variables.

from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    # --- General Application Settings ---
    APP_NAME: str = "Intelligent Omnichannel Chatbot"
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")

    # --- Chatwoot Settings ---
    # This secret is used to verify that incoming webhooks are from Chatwoot.
    # It must match the one you configure in the Chatwoot Agent Bot settings.
    CHATWOOT_WEBHOOK_SECRET: str = Field("your_chatwoot_webhook_secret", env="CHATWOOT_WEBHOOK_SECRET")

    # --- AI/RAG Pipeline Settings (Placeholders for now) ---
    # These will be used later to connect to the LLM and Qdrant.
    OPENAI_API_KEY: str = Field("your_openai_api_key", env="OPENAI_API_KEY")
    QDRANT_URL: str = Field("http://localhost:6333", env="QDRANT_URL")
    EMBEDDING_MODEL_NAME: str = "multilingual-e5-large"

    # --- Application Settings ---
    ENVIRONMENT: str = "development" # "development" or "production"
    APP_PORT: int = 8000

    class Config:
        # This tells Pydantic to look for a .env file in the project root.
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Create a single instance of the settings to be used throughout the application.
settings = Settings()
