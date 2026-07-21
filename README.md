# Nexus — RAG Chatbot

A Retrieval-Augmented Generation chatbot with multi-session conversations, document management, and a hybrid LLM architecture — all served as a single-page app.

**Upload a document** (PDF, DOCX, XLSX, PPTX) → it's chunked, embedded via OpenRouter, and stored in Supabase's pgvector database. **Ask questions** → relevant chunks are retrieved, and the LLM answers based *only* on the source document.

```
┌────────────────────────────────────────────────────────────┐
│  PDF/DOCX/XLSX/PPTX                                       │
│       ↓                                                    │
│  Text Extraction → Chunking → OpenRouter Embedding         │
│                                         ↓                  │
│                               Supabase pgvector            │
│                                         ↓                  │
│  User Question → OpenRouter Embedding → Vector Search      │
│                                         ↓                  │
│  Similarity ≥ 0.5? ──Yes──→ OpenRouter LLM (strict RAG)   │
│       │                                                    │
│       └─No / rate-limited ──→ Self-hosted LLM (fallback)   │
└────────────────────────────────────────────────────────────┘
```

## Stack

| Layer | Choice |
|-------|--------|
| Backend | [FastAPI](https://fastapi.tiangolo.com/) (Python) |
| Frontend | Vanilla HTML/CSS/JS — served by FastAPI (branded **Nexus**) |
| RAG LLM | [OpenRouter](https://openrouter.ai/) — model-agnostic gateway |
| Fallback LLM | Self-hosted endpoint ([OpenAI-compatible](https://openai.com)) |
| Embeddings | OpenRouter — `nvidia/nemotron-3-embed-1b:free` (2048-d) |
| Vector DB | [Supabase](https://supabase.com/) + pgvector |
| Chunking | Recursive character split via tiktoken (~1000 chars, 200 overlap) |

### Why OpenRouter?

Chat and embedding models are configured via OpenRouter, so the backend can run on free-tier models for development/testing or be pointed at Claude, GPT-4o, Gemini 2.5, or any other OpenRouter-supported model with a single config change — no code changes required.

The current default chat model is `nvidia/nemotron-3-ultra-550b-a55b:free`. Swap it by changing `OPENROUTER_CHAT_MODEL` in `.env`.

## Features

- **Multi-session conversations** — create, switch, rename, and delete conversations. History persists in Supabase across page refreshes.
- **Document management** — upload and delete documents from the sidebar. Supported formats: PDF, DOCX, XLSX/XLS, PPTX.
- **Document @-mention** — type `@` in the chat input to filter by a specific uploaded document.
- **Hybrid answering** — document questions use strict RAG; off-topic questions or explicit document name mentions fall back gracefully.
- **Rate-limit resilience** — if the OpenRouter free tier hits a 429, the backend falls back to the self-hosted chat model without errors.
- **Auto-titling** — sessions are automatically renamed to the first question you ask.
- **Source expanders** — each assistant answer shows the retrieved source chunks in an expandable accordion.
- **Dark theme** — gray-900/800/700 palette with blue-500 accent, responsive layout (sidebar becomes a slide-out drawer on mobile).

## Prerequisites

- Python 3.10+
- A [Supabase](https://supabase.com) project (free tier works)
- An [OpenRouter](https://openrouter.ai/keys) API key (free tier works for development)

### Optional

- A self-hosted LLM endpoint (any OpenAI-compatible server) — the general-chat fallback and OpenRouter rate-limit escape hatch. If omitted, the app falls back gracefully via OpenRouter.

## Setup

### 1. Clone & install

```bash
pip install -r requirements.txt
```

### 2. Environment variables

Copy `.env.example` to `.env` and fill in your credentials:

```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-anon-key
OPENROUTER_API_KEY=sk-or-v1-your-openrouter-key
OPENROUTER_CHAT_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
```

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_KEY` | Yes | Supabase anon/public key |
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key (chat + embeddings) |
| `OPENROUTER_CHAT_MODEL` | No | OpenRouter chat model (default: `nvidia/nemotron-3-ultra-550b-a55b:free`) |
| `OPENROUTER_EMBEDDING_MODEL` | No | OpenRouter embedding model (default: `nvidia/nemotron-3-embed-1b:free`) |
| `GEMINI_API_KEY` | No | Legacy — only needed if switching back to Gemini |
| `LOCAL_CHAT_ENDPOINT` | No | Self-hosted LLM endpoint URL (fallback) |
| `LOCAL_CHAT_API_KEY` | No | API key for the self-hosted endpoint |
| `LOCAL_CHAT_MODEL` | No | Model name on the self-hosted endpoint (default: `auto/best-chat`) |

### 3. Database setup

Open your Supabase project's **SQL Editor** and run these scripts **in order**:

1. `scripts/init_db.sql` — creates the `documents` table with pgvector, the `match_documents` search function, and RLS policies.
2. `scripts/init_chat_history.sql` — creates the `chat_sessions` and `chat_messages` tables with cascading deletes and auto-timestamps.

### 4. Start the app

```bash
uvicorn app.main:app --reload
```

Then open **http://localhost:8000** in your browser.

That's it — one command. The API and frontend are served from the same process.

- App: http://localhost:8000
- API docs: http://localhost:8000/docs

> **Note on the legacy Streamlit UI:** The old Streamlit frontend (`streamlit_app/`) is **deprecated** and no longer maintained. The current frontend is the vanilla SPA served directly by FastAPI on port 8000.

### 5. Free-tier rate limits

OpenRouter free models are limited to roughly 10–20 requests per minute (depending on the model). The client includes a built-in rate limiter. If you hit `429` errors, wait a few minutes or switch to a different free model by changing `OPENROUTER_CHAT_MODEL` in `.env`.

## API Endpoints

### System

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serves the SPA frontend |
| GET | `/api/health` | Health check |

### Documents

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/documents` | List uploaded documents |
| POST | `/api/upload` | Upload & process a document (PDF, DOCX, XLSX, PPTX) |
| DELETE | `/api/documents/{title}` | Delete a document and its chunks |

### Chat Sessions

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
| POST | `/api/chat` | Stateless Q&A (no history saved) |

## Project Structure

```
RAG_Chatbot/
├── app/                      # FastAPI backend
│   ├── main.py                    # Entry point & all endpoints
│   ├── config.py                  # Env vars & settings
│   ├── database/
│   │   ├── supabase_client.py     # Supabase connection singleton
│   │   ├── vector_store.py        # Documents CRUD + vector search
│   │   └── chat_store.py          # Chat sessions & messages CRUD
│   ├── ingestion/
│   │   ├── document_processor.py  # Multi-format text extraction
│   │   ├── chunker.py             # Recursive text chunking (~1000 chars)
│   │   └── embedder.py            # OpenRouter embedding API client
│   └── chatbot/
│       ├── openrouter_client.py   # OpenRouter LLM (strict RAG)
│       └── local_client.py        # Self-hosted LLM (fallback)
├── static/                   # Vanilla SPA frontend
│   ├── index.html                 # App shell (branded "Nexus")
│   ├── css/style.css              # Dark theme styles
│   └── js/app.js                  # Chat app logic (no frameworks)
├── scripts/                  # Database setup
│   ├── init_db.sql                # Documents table + match_documents
│   └── init_chat_history.sql      # Chat sessions/messages tables
├── streamlit_app/            # Legacy frontend (deprecated)
├── tests/                    # Test suite
├── .env.example              # Environment variable template
└── requirements.txt          # Python dependencies
```

## How It Works

1. **Upload a document** — the system extracts text (PDF via pypdf, DOCX via python-docx, XLSX via openpyxl, PPTX via python-pptx), splits it into overlapping chunks of ~1000 characters, generates embeddings via OpenRouter's embedding API, and stores everything in Supabase's pgvector.

2. **Ask a question** — the query is embedded using the same OpenRouter model, the top-5 most similar chunks are retrieved via cosine similarity, and the LLM answers based *only* on those chunks.

3. **Hybrid fallback** — if the retrieved chunks don't meet the similarity threshold (0.5) and no document was explicitly named, the query is answered by the general-chat model instead, so the bot can hold a normal conversation.

4. **OpenRouter fallback** — if the free-tier rate limit is hit (HTTP 429), the backend automatically falls back to the self-hosted LLM with the same document context, so the app never crashes on rate limits.

## Document Support

| Format | Library | Status |
|--------|---------|--------|
| PDF (`.pdf`) | pypdf | ✅ |
| Word (`.docx`) | python-docx | ✅ — paragraphs + tables |
| Excel (`.xlsx`, `.xls`) | openpyxl | ✅ — all sheets, row-by-row |
| PowerPoint (`.pptx`) | python-pptx | ✅ — slides + tables + XML fallback |
