# RAG Chatbot — CLAUDE.md

## Project Overview
A Retrieval-Augmented Generation (RAG) chatbot built with **FastAPI + Vanilla HTML/CSS/JS + OpenRouter (free models) + Supabase (pgvector) + self-hosted LLM fallback**.

Upload a PDF → it gets chunked, embedded via OpenRouter, and stored in Supabase's vector DB. Ask questions → relevant chunks are retrieved and the LLM answers based *only* on the source document.

**Hybrid mode**: When the user asks a question that has nothing to do with uploaded documents (low vector similarity), the app falls back to a general-purpose chat model hosted at a custom endpoint — so the bot can hold a normal conversation instead of saying "I don't know."

| Scenario | Model Used |
|----------|-----------|
| Document-related question (high similarity) | OpenRouter RAG (strict) |
| General chat / off-topic question | Self-hosted local endpoint |

## Commands

| Action | Command |
|--------|---------|
| Start FastAPI server (serves SPA + API) | `uvicorn app.main:app --reload` (port 8000) |
| Open the app | http://localhost:8000 |
| Legacy Streamlit UI | `streamlit run streamlit_app/app.py` (port 8501) |
| Install dependencies | `pip install -r requirements.txt` |
| Run tests | `pytest tests/ -v` |

## Project Structure

```
RAG_Chatbot/
├── app/                   # FastAPI backend
│   ├── main.py                # Entry point & all endpoints
│   ├── config.py              # Env vars & settings
│   ├── database/
│   │   ├── supabase_client.py # Supabase connection
│   │   ├── vector_store.py    # Documents pgvector CRUD
│   │   └── chat_store.py      # Chat sessions & messages CRUD
│   ├── ingestion/
│   │   ├── pdf_processor.py   # PDF text extraction
│   │   ├── chunker.py         # Text chunking
│   │   └── embedder.py        # OpenRouter embedding API
│   ├── retrieval/
│   │   └── retriever.py       # Query → embed → search
│   └── chatbot/
│       ├── openrouter_client.py  # OpenRouter LLM (RAG / strict mode)
│       ├── local_client.py       # Self-hosted LLM (general chat fallback)
│       └── gemini_client.py      # Gemini LLM (legacy)
├── static/                # Vanilla SPA frontend (served by FastAPI)
│   ├── index.html             # SPA shell
│   ├── css/
│   │   └── style.css          # Dark theme styles
│   └── js/
│       └── app.js             # Chat app logic (no frameworks)
├── streamlit_app/
│   └── app.py             # Legacy Streamlit UI (deprecated)
├── scripts/
│   ├── init_db.sql            # Documents table + match_documents function
│   └── init_chat_history.sql  # Chat sessions/messages tables
└── tests/
    └── ...
```

## API Endpoints

### System
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serves the SPA frontend (`static/index.html`) |
| GET | `/api/health` | Health check (used by frontend on load) |

### Document Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/documents` | List uploaded documents |
| POST | `/api/upload` | Upload & process a PDF |
| DELETE | `/api/documents/{title}` | Delete a document |

### Chat Session Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat/sessions` | Create a new session |
| GET | `/api/chat/sessions` | List sessions (newest first) |
| GET | `/api/chat/sessions/{id}` | Get session details |
| PATCH | `/api/chat/sessions/{id}` | Rename a session |
| DELETE | `/api/chat/sessions/{id}` | Delete session + messages |
| GET | `/api/chat/sessions/{id}/messages` | Get all messages (oldest first) |
| POST | `/api/chat/sessions/{id}/messages` | Send message → RAG → save response |

### Legacy
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat` | Stateless Q&A (no history) |

## Frontend Features

- **Multi-session conversations** — create, switch, rename, delete conversations
- **Persistent history** — messages saved to Supabase, survive page refreshes
- **Auto-titling** — first question becomes the session title
- **Source expanders** — expandable accordion under each assistant answer showing retrieved chunks
- **Document management** — upload/delete PDFs from the sidebar
- **Dark theme** — gray-900/800/700 palette with blue-500 accent
- **Mobile responsive** — sidebar becomes slide-out drawer below 768px
- **Toast notifications** — slide-in error/success/info messages

## Architecture

```
PDF Upload → Text Extraction → Chunking → OpenRouter Embedding → Supabase pgvector
                                                                        ↓
User Question → OpenRouter Embedding → Vector Search → Top-k Chunks → Similarity ≥ 0.5? ──Yes──→ OpenRouter LLM (strict RAG)
                                                                        │
                                                                        └─No (off-topic)──→ Self-hosted LLM (general chat)
```

## Tech Stack

| Component | Choice |
|-----------|--------|
| Backend | FastAPI (Python) |
| Frontend | Vanilla HTML/CSS/JS (served by FastAPI) |
| RAG LLM | OpenRouter — `nvidia/nemotron-3-ultra-550b-a55b:free` |
| General Chat LLM | Self-hosted endpoint (`auto/best-chat`) |
| Embeddings | OpenRouter — `nvidia/nemotron-3-embed-1b:free` (2048-d) |
| Vector DB | Supabase + pgvector |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase anon/public key |
| `OPENROUTER_API_KEY` | OpenRouter API key (for both chat + embeddings) |
| `OPENROUTER_CHAT_MODEL` | OpenRouter model for chat (default: `nvidia/nemotron-3-ultra-550b-a55b:free`) |
| `OPENROUTER_EMBEDDING_MODEL` | OpenRouter model for embeddings (default: `nvidia/nemotron-3-embed-1b:free`) |
| `GEMINI_API_KEY` | Google AI Studio API key (legacy, unused if OpenRouter is active) |
| `LOCAL_CHAT_ENDPOINT` | Self-hosted LLM endpoint URL (hybrid fallback) |
| `LOCAL_CHAT_API_KEY` | API key for the self-hosted endpoint |
| `LOCAL_CHAT_MODEL` | Model name on the self-hosted endpoint (default: `auto/best-chat`) |

## Free-Tier Limits

OpenRouter free models have rate limits (~10-20 RPM depending on the model). The client includes a built-in rate limiter to stay within bounds. If you hit `429` errors, wait a few minutes or switch to a different free model by changing `OPENROUTER_CHAT_MODEL` in `.env`.

## Key Rules

- **Document questions → strict RAG.** If the retrieved chunks contain the answer (similarity ≥ 0.5), answer only from context. If the answer can't be found, say: "I don't have enough information to answer that from the provided document."
- **General questions → hybrid fallback.** If no chunks are found or all are below the similarity threshold (0.5), the query goes to the self-hosted general chat model instead.
- **Chunk size:** ~1000 characters with 200-character overlap.
- **Vector dimension:** 2048 (nemotron-3-embed-1b via OpenRouter).
- **Free-tier limits:** OpenRouter free models are rate-limited (~10-20 RPM). The client has a built-in rate limiter.

