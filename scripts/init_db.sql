-- ═══════════════════════════════════════════════════════
-- RAG Chatbot — Supabase / pgvector Setup
-- Run this in your Supabase SQL Editor (one-time setup).
-- ═══════════════════════════════════════════════════════

-- 1. Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the documents table
CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content     TEXT NOT NULL,
    embedding   VECTOR(2048),   -- Nemotron-3-Embed-1B (via OpenRouter)
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Index for approximate nearest-neighbor search
--    NOTE: pgvector caps IVFFlat / HNSW at 2000 dimensions.
--    At 2048-d we skip the index (fine for small datasets).
--    For large datasets, consider dim reduction or a model <= 2000 dims.

-- 4. Index for listing documents by title (faster GET /api/documents)
CREATE INDEX IF NOT EXISTS idx_documents_title
    ON documents (title);

-- 5. RLS: allow anon key to read / write documents
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Enable insert for anon" ON documents
    FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "Enable select for anon" ON documents
    FOR SELECT TO anon USING (true);

CREATE POLICY "Enable delete for anon" ON documents
    FOR DELETE TO anon USING (true);

-- 6. Match function — cosine-similarity search
--    The `<=>` operator computes cosine distance; 1 - distance = similarity.
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding VECTOR(2048),
    match_count INT DEFAULT 5
)
RETURNS TABLE(
    id          BIGINT,
    title       TEXT,
    content     TEXT,
    chunk_index INT,
    similarity  FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.title,
        d.content,
        d.chunk_index,
        1 - (d.embedding <=> query_embedding) AS similarity
    FROM documents d
    WHERE d.embedding IS NOT NULL
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
