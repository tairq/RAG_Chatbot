-- ═══════════════════════════════════════════════════════
-- RAG Chatbot — Chat History Tables (Supabase / pgvector)
-- Run this after init_db.sql in your Supabase SQL Editor.
-- ═══════════════════════════════════════════════════════

-- 1. Chat sessions — groups messages into conversations
CREATE TABLE IF NOT EXISTS chat_sessions (
    id          BIGSERIAL PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'New Chat',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Chat messages — individual turns in a session
CREATE TABLE IF NOT EXISTS chat_messages (
    id          BIGSERIAL PRIMARY KEY,
    session_id  BIGINT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    sources     JSONB,          -- stores the list of retrieved chunks
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast message retrieval by session
CREATE INDEX IF NOT EXISTS idx_chat_messages_session
    ON chat_messages (session_id, created_at);

-- Index for listing sessions newest-first
CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated
    ON chat_sessions (updated_at DESC);

-- Auto-update updated_at on chat_sessions when a new message is added
CREATE OR REPLACE FUNCTION touch_chat_session()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE chat_sessions
    SET updated_at = NOW()
    WHERE id = NEW.session_id;
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_chat_messages_touch_session
    AFTER INSERT ON chat_messages
    FOR EACH ROW
    EXECUTE FUNCTION touch_chat_session();
