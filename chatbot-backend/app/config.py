# Configuration management for the application.
# Uses Pydantic's BaseSettings to load environment variables.

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator

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

    # Whether to enforce Chatwoot webhook signature validation.
    # Keep this enabled in production.
    CHATWOOT_VALIDATE_WEBHOOK_SIGNATURE: bool = Field(True, env="CHATWOOT_VALIDATE_WEBHOOK_SIGNATURE")

    # Chatwoot API settings (used to send bot replies back to Chatwoot).
    # In Docker Compose, Chatwoot is typically reachable at http://chatwoot:3000
    CHATWOOT_BASE_URL: str = Field("http://localhost:3000", env="CHATWOOT_BASE_URL")
    CHATWOOT_API_TOKEN: str = Field("", env="CHATWOOT_API_TOKEN")
    CHATWOOT_ACCOUNT_ID: int | None = Field(None, env="CHATWOOT_ACCOUNT_ID")

    # Network timeout for Chatwoot API calls (messages, labels, assignments, etc.)
    CHATWOOT_REQUEST_TIMEOUT_S: float = Field(6.0, env="CHATWOOT_REQUEST_TIMEOUT_S")

    @field_validator("CHATWOOT_ACCOUNT_ID", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v):
        if v == "":
            return None
        return v

    # --- AI/RAG Pipeline Settings (Placeholders for now) ---
    # These will be used later to connect to the LLM and Qdrant.
    OPENAI_API_KEY: str = Field("your_openai_api_key", env="OPENAI_API_KEY")
    # LLM model configuration (Phase 1).
    OPENAI_MODEL: str = Field("gpt-4o", env="OPENAI_MODEL")
    OPENAI_REQUEST_TIMEOUT_S: float = Field(20.0, env="OPENAI_REQUEST_TIMEOUT_S")
    OPENAI_MAX_TOKENS: int = Field(400, env="OPENAI_MAX_TOKENS")

    # --- PII masking (LLM privacy) ---
    # If enabled, logs only span metadata (type/start/end/confidence) at DEBUG level.
    # Never logs raw values.
    PII_DEBUG: bool = Field(False, env="PII_DEBUG")

    # Default minimum confidence to mask (applies as an override across types when set).
    # If unset, the PII service uses per-type defaults.
    PII_MIN_CONFIDENCE: float | None = Field(None, env="PII_MIN_CONFIDENCE")

    # Comma-separated allowlist terms to avoid masking known safe words (cities, product names, etc.).
    # Example: PII_ALLOWED_TERMS="Tunis,Sfax,AcmePhone"
    PII_ALLOWED_TERMS: str = Field("", env="PII_ALLOWED_TERMS")
    QDRANT_URL: str = Field("http://localhost:6333", env="QDRANT_URL")
    QDRANT_API_KEY: str = Field("", env="QDRANT_API_KEY")
    QDRANT_COLLECTION: str = Field("shared", env="QDRANT_COLLECTION")

    # --- Multi-store settings (Phase 3) ---
    # JSON mapping defining stores and inbox matching rules.
    # Example:
    # STORES_JSON='{
    #   "default": {"qdrant_collection": "shared"},
    #   "store_a": {
    #     "match": {"inbox_ids": [1], "inbox_names": ["Store A"]},
    #     "qdrant_collection": "store_a",
    #     "woocommerce": {"base_url": "https://store-a.tld", "consumer_key": "ck_...", "consumer_secret": "cs_..."}
    #   }
    # }'
    STORES_JSON: str = Field("", env="STORES_JSON")
    DEFAULT_STORE_KEY: str = Field("default", env="DEFAULT_STORE_KEY")

    # --- WooCommerce (Phase 3 / Step 6) ---
    # Network timeout for WooCommerce API calls.
    WOOCOMMERCE_REQUEST_TIMEOUT_S: float = Field(12.0, env="WOOCOMMERCE_REQUEST_TIMEOUT_S")

    # Developer-only: allow explicit /wc commands to call WooCommerce directly.
    # Keep disabled in production unless you explicitly want this behavior.
    WOOCOMMERCE_COMMANDS_ENABLED: bool = Field(False, env="WOOCOMMERCE_COMMANDS_ENABLED")

    # --- Human escalation (Phase 3) ---
    ESCALATION_ENABLED: bool = Field(True, env="ESCALATION_ENABLED")
    ESCALATION_LABEL: str = Field("human_handoff", env="ESCALATION_LABEL")
    ESCALATION_ACK_LABEL: str = Field("human_handoff_ack", env="ESCALATION_ACK_LABEL")
    ESCALATION_ASSIGNEE_ID: int | None = Field(None, env="ESCALATION_ASSIGNEE_ID")
    ESCALATION_TEAM_ID: int | None = Field(None, env="ESCALATION_TEAM_ID")

    # Upper bound for total Chatwoot API time spent in escalation per webhook.
    ESCALATION_API_BUDGET_S: float = Field(2.5, env="ESCALATION_API_BUDGET_S")

    # Comma-separated keywords. Keep simple; detection runs case-insensitive.
    ESCALATION_KEYWORDS: str = Field(
        "human,agent,representative,talk to someone,talk to a human,customer service,support",
        env="ESCALATION_KEYWORDS",
    )

    # Send one final acknowledgement message when escalation is triggered.
    ESCALATION_SEND_ACK: bool = Field(True, env="ESCALATION_SEND_ACK")
    ESCALATION_ACK_MESSAGE: str = Field(
        "Okay — I’m connecting you to a human agent.",
        env="ESCALATION_ACK_MESSAGE",
    )

    # Optional: escalate automatically when retrieval confidence is low.
    ESCALATION_LOW_CONFIDENCE_ENABLED: bool = Field(False, env="ESCALATION_LOW_CONFIDENCE_ENABLED")
    ESCALATION_MIN_TOP_SCORE: float = Field(0.15, env="ESCALATION_MIN_TOP_SCORE")

    # Basic retrieval tuning (no LlamaIndex yet)
    RAG_TOP_K: int = Field(4, env="RAG_TOP_K")
    RAG_MAX_CONTEXT_CHARS: int = Field(4000, env="RAG_MAX_CONTEXT_CHARS")

    # Embeddings configuration (for basic retrieval).
    # Provider can be:
    # - "openai" (default, light-weight)
    # - "sentence-transformers" (for multilingual-e5-* models)
    # - "local" (deterministic hashing embeddings for offline dev/testing)
    EMBEDDINGS_PROVIDER: str = Field("openai", env="EMBEDDINGS_PROVIDER")
    OPENAI_EMBEDDING_MODEL: str = Field("text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")

    # Used when EMBEDDINGS_PROVIDER="local"
    LOCAL_EMBEDDING_DIM: int = Field(384, env="LOCAL_EMBEDDING_DIM")

    # Optional for sentence-transformers provider; left as your target PRD model.
    EMBEDDING_MODEL_NAME: str = Field("multilingual-e5-large", env="EMBEDDING_MODEL_NAME")

    # --- Application Settings ---
    ENVIRONMENT: str = "development" # "development" or "production"
    APP_PORT: int = 8000

    class Config:
        # This tells Pydantic to look for a .env file in the project root.
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Create a single instance of the settings to be used throughout the application.
settings = Settings()
