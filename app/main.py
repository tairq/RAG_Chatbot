import logging
import os
import re
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

from app.chatbot.local_client import chat_general, chat_rag
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
from app.ingestion.document_processor import extract_text, is_supported
from app.ingestion.embedder import embed_batch

# Ensure the uploads directory exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="RAG Chatbot", version="1.1.0")

# CORS — allow local development frontends
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
    mentioned_docs: list[str] | None = None


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
    mentioned_docs: list[str] | None = None


class SessionMessageResponse(BaseModel):
    role: str
    content: str
    sources: list[dict[str, Any]]
    created_at: str | None = None


# ── Response Formatting ──────────────────────────────────


def to_plain_text(answer: str) -> str:
    """Remove common Markdown syntax from a model response before returning it."""
    answer = re.sub(r"```(?:[a-zA-Z0-9_+-]+)?\n?", "", answer)
    answer = answer.replace("```", "")
    answer = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", answer)
    answer = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", answer)
    answer = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", answer)
    answer = re.sub(r"(?m)^\s*[-*+]\s+", "", answer)
    answer = re.sub(r"(?m)^\s*\d+[.)]\s+", "", answer)
    answer = re.sub(r"(?m)^\s*>\s?", "", answer)
    answer = re.sub(r"(?m)^\s*([-*_])(?:\s*\1){2,}\s*$", "", answer)
    answer = re.sub(r"(\*\*|__)(.*?)\1", r"\2", answer)
    answer = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", answer)
    answer = re.sub(r"(?<!_)_([^_\n]+)_(?!_)", r"\1", answer)
    answer = re.sub(r"`([^`]+)`", r"\1", answer)
    return re.sub(r"\n{3,}", "\n\n", answer).strip()


def resolve_document_references(
    query: str,
    mentioned_docs: list[str] | None,
) -> tuple[list[str] | None, str]:
    """Find uploaded filenames written directly in a query and remove them for retrieval."""
    titles = list(mentioned_docs or [])
    retrieval_query = query

    for document in list_documents():
        title = document.get("title", "")
        if title and re.search(re.escape(title), query, flags=re.IGNORECASE):
            if title not in titles:
                titles.append(title)
            retrieval_query = re.sub(re.escape(title), "", retrieval_query, flags=re.IGNORECASE)

    retrieval_query = re.sub(r"\s{2,}", " ", retrieval_query).strip(" ,:;-.?")
    return (titles or None), (retrieval_query or query)


# ── Helpers ────────────────────────────────────────────────


def _answer_with_fallback(
    query: str,
    results: list[dict[str, Any]],
    chat_history: list[dict[str, str]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Generate an answer using OpenRouter with a fallback to the local LLM.

    If OpenRouter is rate-limited (429), falls back to the local RAG model
    so the chatbot stays responsive.

    Args:
        query: The user's question.
        results: Retrieved chunks from vector search.
        chat_history: Optional conversation history for multi-turn context.

    Returns:
        Tuple of (answer_text, sources_list).
    """
    try:
        if chat_history:
            answer = ask_openrouter_with_history(query, results, chat_history)
        else:
            answer = ask_openrouter(query, results)
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "Rate limit" in err_str or "rate_limit" in err_str:
            logger.warning("OpenRouter rate-limited, falling back to local RAG model")
            answer = chat_rag(query, results)
        else:
            # Re-raise non-rate-limit errors
            raise

    answer = to_plain_text(answer)

    sources = [
        {"id": r["id"], "content": r["content"], "similarity": r["similarity"]}
        for r in results
    ]
    return answer, sources


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
    Upload a document — extracts text, chunks it, embeds via OpenRouter,
    and stores in Supabase pgvector.

    Supported formats: PDF, DOCX, XLSX, XLS, PPTX
    """
    if not file.filename or not is_supported(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Supported formats: PDF, DOCX, XLSX, PPTX",
        )

    # Save uploaded file temporarily
    unique_name = f"{uuid.uuid4()}_{file.filename}"
    tmp_path = UPLOAD_DIR / unique_name

    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Extract text (routes by extension automatically)
        text = extract_text(str(tmp_path), file.filename)

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract any text from the file. It may be scanned, image-only, or empty.",
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
            message=f"'{title}' processed successfully ({count} chunks).",
            filename=title,
            chunks_created=count,
        )

    finally:
        # Clean up uploaded file
        if tmp_path.exists():
            tmp_path.unlink()


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
    filter_titles, retrieval_query = resolve_document_references(query, request.mentioned_docs)
    results = search_similar(
        embed_batch([retrieval_query])[0],
        top_k=request.top_k,
        filter_titles=filter_titles,
    )

    if not results:
        # No document results at all → use the general chat model
        answer = to_plain_text(chat_general(query))
        return ChatResponse(answer=answer, sources=[])

    # Check if the best-matched chunk is relevant enough
    max_similarity = max(r.get("similarity", 0) for r in results)
    if max_similarity < SIMILARITY_THRESHOLD and not filter_titles:
        # Retrieved chunks are below the relevance threshold and no document
        # was explicitly named → use general chat
        answer = to_plain_text(chat_general(query))
        return ChatResponse(answer=answer, sources=[])

    # Relevant context found → use strict RAG (with OpenRouter fallback)
    # Note: if filter_titles is set (user named a specific document), we
    # skip the similarity threshold check — the user explicitly asked about
    # that document so we should answer from it.
    answer, sources = _answer_with_fallback(query, results)

    return ChatResponse(
        answer=answer,
        sources=sources,
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
    filter_titles, retrieval_query = resolve_document_references(query, body.mentioned_docs)

    # 1. Save the user message immediately
    add_message(session_id=session_id, role="user", content=query)

    # 2. Retrieve relevant chunks (RAG)
    results = search_similar(
        embed_batch([retrieval_query])[0],
        top_k=body.top_k,
        filter_titles=filter_titles,
    )

    if not results or (max((r.get("similarity", 0) for r in results), default=0) < SIMILARITY_THRESHOLD and not filter_titles):
        # No relevant document context → use the general chat model
        # (unless the user explicitly named a document — then use results anyway)
        answer = to_plain_text(chat_general(query))
        sources: list[dict[str, Any]] = []
    else:
        # Build history from previous messages for context
        history = get_messages(session_id)
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in history
            if m["role"] in ("user", "assistant")
        ]

        # 3. Generate answer with conversation history (with OpenRouter fallback)
        answer, sources = _answer_with_fallback(query, results, chat_history)

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
