import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.chatbot.gemini_client import ask_gemini, ask_gemini_with_history
from app.config import CHUNK_OVERLAP, CHUNK_SIZE
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

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# CORS — allow Streamlit frontend on port 8501
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5


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


# ── Endpoints ────────────────────────────────────────────


@app.get("/")
def root():
    return {"status": "ok", "service": "RAG Chatbot API"}


@app.post("/api/upload", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document — extracts text, chunks it, embeds via Gemini,
    and stores in Supabase.
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


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Ask a question — retrieves relevant chunks and generates an answer.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Retrieve relevant chunks
    query = request.query.strip()
    results = search_similar(embed_batch([query])[0], top_k=request.top_k)

    if not results:
        return ChatResponse(
            answer="No relevant documents found. Please upload a PDF first.",
            sources=[],
        )

    # Generate answer
    answer = ask_gemini(query, results)

    return ChatResponse(
        answer=answer,
        sources=[
            {"id": r["id"], "content": r["content"], "similarity": r["similarity"]}
            for r in results
        ],
    )


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
