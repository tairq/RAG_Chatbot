"""
RAG Chatbot — Streamlit Frontend

Two-mode UI:
1. Upload Document — upload a PDF to process and store.
2. Chat — ask questions about uploaded documents.

Run with: streamlit run streamlit_app/app.py
"""

import os
from pathlib import Path

import requests
import streamlit as st

# ── Configuration ─────────────────────────────────────────

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📄",
    layout="wide",
)

# ── Session state init ────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "documents" not in st.session_state:
    st.session_state.documents = []


# ── Helpers ───────────────────────────────────────────────


def check_api() -> bool:
    """Check if the FastAPI backend is reachable."""
    try:
        r = requests.get(f"{API_BASE}/", timeout=3)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def refresh_documents():
    """Refresh the document list from the API."""
    try:
        r = requests.get(f"{API_BASE}/api/documents", timeout=5)
        if r.status_code == 200:
            st.session_state.documents = r.json().get("documents", [])
    except requests.ConnectionError:
        st.session_state.documents = []


def upload_pdf(file) -> dict | None:
    """Upload a PDF to the API and return the response."""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        r = requests.post(f"{API_BASE}/api/upload", files=files, timeout=60)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"Upload failed: {r.json().get('detail', 'Unknown error')}")
            return None
    except requests.ConnectionError:
        st.error("Cannot connect to the backend. Is the server running?")
        return None


def send_chat_query(query: str, top_k: int = 5) -> dict | None:
    """Send a chat query to the API and return the response."""
    try:
        r = requests.post(
            f"{API_BASE}/api/chat",
            json={"query": query, "top_k": top_k},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"Chat error: {r.json().get('detail', 'Unknown error')}")
            return None
    except requests.ConnectionError:
        st.error("Cannot connect to the backend. Is the server running?")
        return None


# ── Sidebar ───────────────────────────────────────────────

with st.sidebar:
    st.title("📄 RAG Chatbot")

    api_ok = check_api()
    if api_ok:
        st.success("✅ Backend connected")
    else:
        st.error("❌ Backend unreachable — start the server first")
        st.code("uvicorn app.main:app --reload")
        st.stop()

    st.divider()
    st.subheader("Uploaded Documents")
    refresh_documents()

    if st.session_state.documents:
        for doc in st.session_state.documents:
            st.write(f"- {doc.get('title', 'Unknown')}")
    else:
        st.info("No documents uploaded yet.")

    # Top-k slider
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=5)

    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ── Main tabs ─────────────────────────────────────────────

tab_upload, tab_chat = st.tabs(["📤 Upload Document", "💬 Chat"])


# ── Upload Tab ────────────────────────────────────────────

with tab_upload:
    st.header("Upload a PDF Document")
    st.markdown(
        "Upload a PDF file. The system will extract the text, split it into "
        "chunks, generate embeddings with Google Gemini, and store them in "
        "the Supabase vector database."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        if st.button("🚀 Process Document", type="primary", use_container_width=True):
            with st.spinner("Processing PDF..."):
                result = upload_pdf(uploaded_file)

            if result:
                st.success(
                    f"✅ **{result['filename']}** processed successfully!\n\n"
                    f"Created **{result['chunks_created']}** chunks and stored them "
                    f"in the vector database."
                )
                refresh_documents()
                st.rerun()

    # Show document preview
    if uploaded_file is not None:
        st.divider()
        st.subheader("Document Preview")
        st.info(f"**Filename:** {uploaded_file.name}")
        st.info(f"**File size:** {len(uploaded_file.getvalue()) / 1024:.1f} KB")


# ── Chat Tab ──────────────────────────────────────────────

with tab_chat:
    st.header("Chat with Your Documents")

    st.markdown(
        "Ask questions about your uploaded documents. The system will retrieve "
        "the most relevant chunks and answer based *only* on the source content."
    )

    # Display chat history
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

                # Show sources if available
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 View source chunks", expanded=False):
                        for i, src in enumerate(message["sources"], 1):
                            sim = src.get("similarity", 0)
                            st.markdown(
                                f"**Source {i}** (similarity: {sim:.3f})"
                            )
                            st.text(src.get("content", "")[:500] + "...")
                            st.divider()

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.write(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = send_chat_query(prompt, top_k=top_k)

            if result:
                answer = result["answer"]
                sources = result.get("sources", [])
                st.write(answer)

                if sources:
                    with st.expander("📚 View source chunks", expanded=False):
                        for i, src in enumerate(sources, 1):
                            sim = src.get("similarity", 0)
                            st.markdown(
                                f"**Source {i}** (similarity: {sim:.3f})"
                            )
                            st.text(src.get("content", "")[:500] + "...")
                            st.divider()
                else:
                    st.info("No source chunks were retrieved.")
            else:
                answer = "Sorry, I couldn't get an answer. Check the backend connection."
                st.error(answer)

            # Save assistant message
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources if result else [],
                }
            )
