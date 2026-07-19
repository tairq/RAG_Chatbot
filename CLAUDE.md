# RAG Chatbot — CLAUDE.md

## Project Overview
A Retrieval-Augmented Generation (RAG) chatbot built with **FastAPI + Streamlit + Google Gemini + Supabase (pgvector)**.

Upload a PDF → it gets chunked, embedded via Gemini, and stored in Supabase's vector DB. Ask questions → relevant chunks are retrieved and Gemini answers based *only* on the source document.

## Commands

| Action | Command |
|--------|---------|
| Start FastAPI server | `uvicorn app.main:app --reload` (port 8000) |
| Start Streamlit UI | `streamlit run streamlit_app/app.py` (port 8501) |
| Install dependencies | `pip install -r requirements.txt` |
| Run tests | `pytest tests/ -v` |

## Project Structure

```
RAG_Chatbot/
├── app/               # FastAPI backend
│   ├── main.py            # Entry point & endpoints
│   ├── config.py          # Env vars & settings
│   ├── database/          # Supabase + pgvector operations
│   ├── ingestion/         # PDF → text → chunks → embeddings
│   ├── retrieval/         # Query → vector search → context
│   └── chatbot/           # Gemini LLM calls
├── streamlit_app/     # Streamlit frontend (upload + chat)
├── scripts/           # SQL init scripts
└── tests/             # Pytest tests
```

## Architecture

```
PDF Upload → Text Extraction → Chunking → Gemini Embedding → Supabase pgvector
                                                                    ↓
User Question → Gemini Embedding → Vector Search → Top-k Chunks → Gemini LLM → Answer
```

## Tech Stack

| Component | Choice |
|-----------|--------|
| Backend | FastAPI (Python) |
| Frontend | Streamlit |
| Embeddings | OpenRouter — `nvidia/nemotron-3-embed-1b:free` (2048-d) |
| LLM | Google Gemini 2.0 Flash (free tier) |
| Vector DB | Supabase + pgvector |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase anon/public key |
| `GEMINI_API_KEY` | Google AI Studio API key |

## Key Rules

- **Answer only from context.** If the retrieved chunks don't contain the answer, respond: "I don't have enough information to answer that."
- **Chunk size:** ~1000 characters with 200-character overlap.
- **Vector dimension:** 2048 (nemotron-3-embed-1b via OpenRouter).
- **Free-tier limits:** Gemini 2.0 Flash → 60 RPM, Embeddings → 30 RPM.

## Supabase Setup

Run `scripts/init_db.sql` in the Supabase SQL Editor to create:
1. The `vector` extension (pgvector)
2. The `documents` table (id, title, chunk_index, content, embedding, created_at)
3. The `match_documents` PostgreSQL function for cosine-similarity search
4. An HNSW/IVFFlat index (skipped — 2048-d exceeds pgvector's 2000-d index limit; fine for small datasets, use dim-reduction or a sub-2000 model for large-scale)
