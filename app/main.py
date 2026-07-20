import logging
import os
import shutil
import uuid
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("rag_chatbot")
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.chatbot.local_client import chat_general
from app.chatbot.openrouter_client import ask_openrouter, ask_openrouter_with_history
from app.config import CHUNK_OVERLAP, CHUNK_SIZE, SIMILARITY_THRESHOLD, TOP_K
from app.database.chat_store import (
    add_message,
    create_session,
    delete_session,
    get_messages,
    get_session,
    list_sessions,
    update_session_title,
)
from app.database.vector_store import (
    delete_document,
    list_documents,
    search_similar,
    store_chunks,
)
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import embed_batch
from app.ingestion.pdf_processor import extract_text_from_pdf

# Ensure the uploads directory exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="RAG Chatbot", version="1.1.0")

# CORS — allow Streamlit frontend on port 8501
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global Exception Handler ──────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return a JSON error instead of a bare 500."""
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.error("Unhandled exception on %s %s:\n%s", request.method, request.url, tb)
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )


# ── Schemas ──────────────────────────────────────────────


class ChatRequest(BaseModel):
    query: str
    top_k: int = TOP_K


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int


class DocumentInfo(BaseModel):
    title: str


class DeleteResponse(BaseModel):
    message: str
    chunks_deleted: int


# ── Session Schemas ──────────────────────────────────────


class SessionCreateRequest(BaseModel):
    title: str = "New Chat"


class SessionUpdateRequest(BaseModel):
    title: str


class SessionMessageRequest(BaseModel):
    query: str
    top_k: int = TOP_K


class SessionMessageResponse(BaseModel):
    role: str
    content: str
    sources: list[dict[str, Any]]
    created_at: str | None = None


# ── Endpoints ────────────────────────────────────────────


# ── Serve the Single-Page App ────────────────────────────

# Mount static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    """Serve the SPA frontend."""
    return FileResponse("static/index.html")


@app.get("/api/health")
def health():
    """Health check endpoint (used by the frontend on load)."""
    return {"status": "ok", "service": "RAG Chatbot API", "version": "1.1.0"}


# ── Document Endpoints ───────────────────────────────────


@app.post("/api/upload", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document — extracts text, chunks it, embeds via OpenRouter,
    and stores in Supabase pgvector.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save uploaded PDF temporarily
    unique_name = f"{uuid.uuid4()}_{file.filename}"
    pdf_path = UPLOAD_DIR / unique_name

    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Extract text
        text = extract_text_from_pdf(str(pdf_path))

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract any text from the PDF. The file may be scanned or image-based.",
            )

        # Chunk
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Text was extracted but no chunks were created.",
            )

        # Embed
        embeddings = embed_batch(chunks)

        # Store
        title = file.filename
        count = store_chunks(chunks, embeddings, title)

        return UploadResponse(
            message=f"Document '{title}' processed successfully.",
            filename=title,
            chunks_created=count,
        )

    finally:
        # Clean up uploaded file
        if pdf_path.exists():
            pdf_path.unlink()


@app.get("/api/documents")
def get_documents():
    """List all uploaded documents with chunk counts."""
    docs = list_documents()
    return {"documents": docs}


@app.delete("/api/documents/{title:path}")
def remove_document(title: str):
    """Delete a document and all its chunks."""
    count = delete_document(title)
    if count == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return DeleteResponse(
        message=f"Document '{title}' deleted.",
        chunks_deleted=count,
    )


# ── Stateless Chat Endpoint (legacy) ─────────────────────


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Ask a question (stateless) — retrieves relevant chunks and generates an answer.
    Falls back to the general chat model when no relevant document context is found.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query = request.query.strip()
    results = search_similar(embed_batch([query])[0], top_k=request.top_k)

    if not results:
        # No document results at all → use the general chat model
        answer = chat_general(query)
        return ChatResponse(answer=answer, sources=[])

    # Check if the best-matched chunk is relevant enough
    max_similarity = max(r.get("similarity", 0) for r in results)
    if max_similarity < SIMILARITY_THRESHOLD:
        # Retrieved chunks are below the relevance threshold → use general chat
        answer = chat_general(query)
        return ChatResponse(answer=answer, sources=[])

    # Relevant context found → use strict RAG
    answer = ask_openrouter(query, results)

    return ChatResponse(
        answer=answer,
        sources=[
            {"id": r["id"], "content": r["content"], "similarity": r["similarity"]}
            for r in results
        ],
    )


# ── Chat Session Endpoints ───────────────────────────────


@app.post("/api/chat/sessions")
def api_create_session(body: SessionCreateRequest):
    """Create a new chat session."""
    session = create_session(title=body.title)
    return {"session": session}


@app.get("/api/chat/sessions")
def api_list_sessions():
    """List all chat sessions, newest first."""
    sessions = list_sessions()
    return {"sessions": sessions}


@app.get("/api/chat/sessions/{session_id}")
def api_get_session(session_id: int):
    """Get a single session by id."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": session}


@app.get("/api/chat/sessions/{session_id}/messages")
def api_get_messages(session_id: int):
    """Get all messages for a session, oldest first."""
    msgs = get_messages(session_id)
    return {"messages": msgs}


@app.post("/api/chat/sessions/{session_id}/messages")
def api_send_message(session_id: int, body: SessionMessageRequest):
    """
    Send a message in a session — performs RAG and saves both user message
    and assistant response to the database.
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Verify session exists
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    query = body.query.strip()

    # 1. Save the user message immediately
    add_message(session_id=session_id, role="user", content=query)

    # 2. Retrieve relevant chunks (RAG)
    results = search_similar(embed_batch([query])[0], top_k=body.top_k)

    if not results or max((r.get("similarity", 0) for r in results), default=0) < SIMILARITY_THRESHOLD:
        # No relevant document context → use the general chat model
        answer = chat_general(query)
        sources: list[dict[str, Any]] = []
    else:
        # Build history from previous messages for context
        history = get_messages(session_id)
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in history
            if m["role"] in ("user", "assistant")
        ]

        # 3. Generate answer with conversation history
        answer = ask_openrouter_with_history(query, results, chat_history)

        sources = [
            {"id": r["id"], "content": r["content"], "similarity": r["similarity"]}
            for r in results
        ]

    # 4. Save the assistant response
    saved = add_message(
        session_id=session_id,
        role="assistant",
        content=answer,
        sources=sources,
    )

    # 5. Auto-title: update session title from first exchange
    if session.get("title", "").strip() in ("New Chat", "", None):
        # Use first ~50 chars of the query as the session title
        new_title = query.strip()[:60]
        if len(query.strip()) > 60:
            new_title += "..."
        update_session_title(session_id, new_title)

    return SessionMessageResponse(
        role="assistant",
        content=answer,
        sources=sources,
        created_at=saved.get("created_at"),
    )


@app.patch("/api/chat/sessions/{session_id}")
def api_update_session(session_id: int, body: SessionUpdateRequest):
    """Rename a chat session."""
    session = update_session_title(session_id, body.title)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": session}


@app.delete("/api/chat/sessions/{session_id}")
def api_delete_session(session_id: int):
    """Delete a session and all its messages."""
    success = delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted"}
