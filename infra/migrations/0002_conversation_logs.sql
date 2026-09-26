-- Reference schema for app/services/conversation_log_service.py
-- See infra/migrations/README.md — this file is NOT executed automatically.

CREATE TABLE IF NOT EXISTS conversation_logs (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    conversation_id BIGINT NULL,
    inbox_id BIGINT NULL,
    store_key TEXT NULL,
    direction TEXT NOT NULL,
    content_masked TEXT NULL,
    pii_types JSONB NULL,
    rag_top_score DOUBLE PRECISION NULL,
    rag_hits_count INTEGER NULL,
    confidence_score DOUBLE PRECISION NULL,
    escalated BOOLEAN NOT NULL DEFAULT false,
    escalation_reason TEXT NULL,
    model TEXT NULL,
    latency_ms INTEGER NULL
);

CREATE INDEX IF NOT EXISTS idx_conversation_logs_created_at
    ON conversation_logs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversation_logs_conversation_id
    ON conversation_logs (conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversation_logs_store_key
    ON conversation_logs (store_key);
