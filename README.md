# RAG Chatbot

A Retrieval-Augmented Generation chatbot that ingests PDF documents and answers questions based on their content.

**Stack:** FastAPI + Streamlit + Google Gemini + Supabase (pgvector)

## How It Works

1. **Upload a PDF** — the system extracts text, splits it into chunks, generates embeddings via Google Gemini, and stores everything in Supabase with pgvector.
2. **Ask questions** — your question is embedded, the most relevant chunks are retrieved from the vector database, and Gemini answers based only on those chunks.

## Prerequisites

- Python 3.10+
- A [Supabase](https://supabase.com) project (free tier works)
- A [Google AI Studio](https://aistudio.google.com/) API key (free tier)

## Setup

### 1. Clone & install

```bash
pip install -r requirements.txt
```

### 2. Environment variables

Copy `.env.example` to `.env` and fill in your credentials:

```
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-anon-key
GEMINI_API_KEY=your-gemini-api-key
```

### 3. Database setup

Open your Supabase project's **SQL Editor** and run:

- `scripts/init_db.sql` — creates the `documents` table, the `match_documents` search function, and the vector index.

### 4. Start the app

```bash
# Terminal 1 — FastAPI backend
uvicorn app.main:app --reload

# Terminal 2 — Streamlit frontend
streamlit run streamlit_app/app.py
```

- Backend API: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

## Usage

1. Go to the **Upload Document** tab
2. Upload a PDF file
3. Switch to the **Chat** tab
4. Ask questions about the document
5. Each answer shows which source chunks were used

## Project Structure

```
├── app/               # FastAPI backend
│   ├── main.py            # API endpoints
│   ├── config.py          # Configuration
│   ├── database/          # Supabase + vector store
│   ├── ingestion/         # PDF processing pipeline
│   ├── retrieval/         # Vector search
│   └── chatbot/           # Gemini LLM integration
├── streamlit_app/     # Frontend
├── scripts/           # Database setup
└── tests/             # Tests
```
