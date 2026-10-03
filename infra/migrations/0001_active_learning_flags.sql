-- Reference schema for app/services/active_learning_service.py
-- See infra/migrations/README.md — this file is NOT executed automatically.

CREATE TABLE IF NOT EXISTS active_learning_flags (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    event_type TEXT NOT NULL,
    store_key TEXT NULL,
    inbox_id BIGINT NULL,
    conversation_id BIGINT NULL,
    reason TEXT NULL,
    top_score DOUBLE PRECISION NULL,
    hits_count INTEGER NULL,
    sources JSONB NULL,
    masked_user_message TEXT NULL,
    model TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_active_learning_flags_created_at
    ON active_learning_flags (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_active_learning_flags_conversation_id
    ON active_learning_flags (conversation_id);
CREATE INDEX IF NOT EXISTS idx_active_learning_flags_store_key
    ON active_learning_flags (store_key);
