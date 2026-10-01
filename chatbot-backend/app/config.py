# Configuration management for the application.
# Uses Pydantic's BaseSettings to load environment variables.

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, model_validator

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    # --- General Application Settings ---
    APP_NAME: str = "Intelligent Omnichannel Chatbot"
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")

    # --- Admin API ---
    # Simple shared-secret token for protecting admin endpoints.
    # Requests must include header: X-Admin-Token: <ADMIN_API_TOKEN>
    ADMIN_API_TOKEN: str = Field("", env="ADMIN_API_TOKEN")

    # --- Chatwoot Settings ---
    # How incoming Agent Bot webhooks are authenticated:
    # - "token" (default): shared secret passed in the Agent Bot URL as
    #   ?token=<CHATWOOT_WEBHOOK_TOKEN>. Works with Chatwoot v4.1.0, which does
    #   not sign Agent Bot webhooks.
    # - "signature": X-Chatwoot-Signature / X-Chatwoot-Timestamp headers
    #   (Chatwoot >= v4.13), verified with CHATWOOT_WEBHOOK_SECRET.
    # - "none": no check at all (development only; warns on every request).
    # Left empty, the mode is resolved by webhook_auth_mode() below.
    CHATWOOT_WEBHOOK_AUTH_MODE: str = Field("", env="CHATWOOT_WEBHOOK_AUTH_MODE")

    # Shared secret for the "token" mode. Empty = every webhook is rejected.
    CHATWOOT_WEBHOOK_TOKEN: str = Field("", env="CHATWOOT_WEBHOOK_TOKEN")

    # HMAC secret for the "signature" mode (the Agent Bot secret shown by Chatwoot >= v4.13).
    CHATWOOT_WEBHOOK_SECRET: str = Field("", env="CHATWOOT_WEBHOOK_SECRET")

    # "signature" mode: maximum accepted gap between X-Chatwoot-Timestamp and now.
    CHATWOOT_WEBHOOK_MAX_SKEW_S: int = Field(300, env="CHATWOOT_WEBHOOK_MAX_SKEW_S")

    # Deprecated: replaced by CHATWOOT_WEBHOOK_AUTH_MODE. Only an explicit
    # "false" is still honoured (mapped to mode "none") when the new mode is unset.
    CHATWOOT_VALIDATE_WEBHOOK_SIGNATURE: bool | None = Field(None, env="CHATWOOT_VALIDATE_WEBHOOK_SIGNATURE")

    # Chatwoot API settings (used to send bot replies back to Chatwoot).
    # In Docker Compose, Chatwoot is typically reachable at http://chatwoot:3000
    CHATWOOT_BASE_URL: str = Field("http://localhost:3000", env="CHATWOOT_BASE_URL")
    CHATWOOT_API_TOKEN: str = Field("", env="CHATWOOT_API_TOKEN")
    CHATWOOT_ACCOUNT_ID: int | None = Field(None, env="CHATWOOT_ACCOUNT_ID")

    # Network timeout for Chatwoot API calls (messages, labels, assignments, etc.)
    CHATWOOT_REQUEST_TIMEOUT_S: float = Field(6.0, env="CHATWOOT_REQUEST_TIMEOUT_S")

    @field_validator("CHATWOOT_ACCOUNT_ID", "CHATWOOT_VALIDATE_WEBHOOK_SIGNATURE", mode="before")
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

    # How orders requested in chat are handled:
    # - "handoff" (default): the model can only call `request_order`; nothing is
    #   created in WooCommerce. A private note + ORDER_VALIDATION_LABEL are added
    #   to the Chatwoot conversation and it is escalated to a human agent.
    # - "direct": legacy behavior, the model calls `create_draft_order` and a
    #   pending WooCommerce order is created. Tests only.
    WOOCOMMERCE_ORDER_MODE: str = Field("handoff", env="WOOCOMMERCE_ORDER_MODE")
    ORDER_VALIDATION_LABEL: str = Field("order_validation", env="ORDER_VALIDATION_LABEL")

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

    # Optional: escalate automatically when confidence is low.
    ESCALATION_LOW_CONFIDENCE_ENABLED: bool = Field(False, env="ESCALATION_LOW_CONFIDENCE_ENABLED")
    # Deprecated: superseded by CONFIDENCE_LOW_THRESHOLD/CONFIDENCE_HIGH_THRESHOLD
    # below (combined 3-signal confidence score). Kept only so old .env files
    # referencing it don't fail to load; it is no longer read by the code.
    ESCALATION_MIN_TOP_SCORE: float = Field(0.15, env="ESCALATION_MIN_TOP_SCORE")

    # --- Confidence scoring (3 signals: RAG score, LLM self-assessment, business-decision rule) ---
    # See app/services/confidence_service.py. Combined score >= HIGH: normal;
    # between LOW and HIGH: uncertainty signal (flagged, not escalated);
    # below LOW: full escalation.
    CONFIDENCE_HIGH_THRESHOLD: float = Field(0.75, env="CONFIDENCE_HIGH_THRESHOLD")
    CONFIDENCE_LOW_THRESHOLD: float = Field(0.50, env="CONFIDENCE_LOW_THRESHOLD")

    # Perf: the LLM self-assessment call (signal #2) is a second, extra LLM
    # round-trip on top of the main reply, so it adds real latency. When True
    # (recommended default), it's only made when the RAG score is genuinely
    # ambiguous (see confidence_service.should_call_llm_confidence_signal) —
    # i.e. clearly-high or clearly-low RAG scores skip it entirely, since the
    # combined score/escalation outcome wouldn't meaningfully change anyway.
    # Set to False to always call it (useful for measuring worst-case latency).
    CONFIDENCE_LLM_SIGNAL_AMBIGUOUS_ONLY: bool = Field(True, env="CONFIDENCE_LLM_SIGNAL_AMBIGUOUS_ONLY")

    # Comma-separated keywords indicating the message involves a business decision
    # (refund, dispute, order cancellation, ...) that deserves extra caution.
    # Same matching model as ESCALATION_KEYWORDS (case-insensitive).
    BUSINESS_DECISION_KEYWORDS: str = Field(
        "réclamation,litige,remboursement,remboursé,exception,annulation,"
        "annuler ma commande,annuler la commande,dispute,plainte,compensation,"
        "dédommagement,geste commercial,dérogation,refund,cancel order,"
        "cancel my order,complaint,chargeback",
        env="BUSINESS_DECISION_KEYWORDS",
    )

    # --- Conversation session memory (multi-turn) ---
    # If set, session history is stored in Redis (fast, native TTL).
    REDIS_URL: str = Field("", env="REDIS_URL")
    # Number of most recent turns (individual user/assistant messages) to keep per conversation.
    SESSION_HISTORY_TURNS: int = Field(6, env="SESSION_HISTORY_TURNS")
    # How long a conversation's history is retained without activity.
    SESSION_TTL_SECONDS: int = Field(3600, env="SESSION_TTL_SECONDS")

    # --- Shared PostgreSQL database (Phase 3) ---
    # Used by conversation_log_service (and, via fallback, active learning) for
    # persisting observability data. Example: postgresql://user:pass@postgres:5432/chatwoot
    APP_DATABASE_URL: str = Field("", env="APP_DATABASE_URL")

    # --- Active learning loop (Phase 3) ---
    # When enabled, the backend logs low-confidence events to PostgreSQL for supervisor review.
    # Keep disabled by default unless you configured ACTIVE_LEARNING_DATABASE_URL.
    ACTIVE_LEARNING_ENABLED: bool = Field(False, env="ACTIVE_LEARNING_ENABLED")
    # Example: postgresql://user:pass@postgres:5432/chatwoot
    ACTIVE_LEARNING_DATABASE_URL: str = Field("", env="ACTIVE_LEARNING_DATABASE_URL")

    @model_validator(mode="after")
    def _fallback_app_database_url(self):
        # Keep ACTIVE_LEARNING_DATABASE_URL for backward compatibility: if
        # APP_DATABASE_URL is unset, reuse it so existing deployments keep working.
        if not self.APP_DATABASE_URL and self.ACTIVE_LEARNING_DATABASE_URL:
            self.APP_DATABASE_URL = self.ACTIVE_LEARNING_DATABASE_URL
        return self

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

    def webhook_auth_mode(self) -> tuple[str, bool]:
        """Return (effective webhook auth mode, used_deprecated_flag).

        Unknown values fall back to "token" (fail closed).
        """
        mode = (self.CHATWOOT_WEBHOOK_AUTH_MODE or "").strip().lower()
        if mode:
            return (mode if mode in {"token", "signature", "none"} else "token"), False
        if self.CHATWOOT_VALIDATE_WEBHOOK_SIGNATURE is False:
            return "none", True
        return "token", False

    class Config:
        # This tells Pydantic to look for a .env file in the project root.
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Create a single instance of the settings to be used throughout the application.
settings = Settings()
