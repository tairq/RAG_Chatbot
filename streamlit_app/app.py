"""
RAG Chatbot — Streamlit Frontend

Full-featured chat interface with session management:
- Multi-session conversation support
- Persistent chat history via Supabase
- Document upload & management
- Expandable source chunks per message

Run with: streamlit run streamlit_app/app.py (backend must be running on port 8000)
"""

import os
from datetime import datetime

import requests
import streamlit as st

# ── Configuration ─────────────────────────────────────────

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ──────────────────────────────────────────────

st.markdown(
    """
<style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    .chat-message { margin-bottom: 0.5rem; }
    .source-badge {
        background: #f0f2f6;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        color: #555;
        display: inline-block;
        margin-right: 4px;
    }
    div[data-testid="stSidebarNav"] { display: none; }
    .session-btn {
        width: 100%;
        text-align: left;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        margin-bottom: 2px;
        border: none;
        background: transparent;
        cursor: pointer;
        font-size: 0.85rem;
        transition: background 0.15s;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .session-btn:hover { background: #f0f2f6; }
    .session-btn.active { background: #e0e7ff; font-weight: 600; }
    .session-title { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
    .session-time { font-size: 0.7rem; color: #999; white-space: nowrap; margin-left: 8px; }
    .doc-item { font-size: 0.85rem; padding: 0.25rem 0; }
    .doc-item .del-btn {
        color: #ff4b4b;
        cursor: pointer;
        font-size: 0.8rem;
        margin-left: 8px;
    }
    /* Better spacing for messages */
    .stChatMessage { padding: 1rem 0; }
    /* Source expander styling */
    .source-content {
        font-size: 0.8rem;
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 4px;
        border-left: 3px solid #ccc;
        margin: 0.25rem 0;
        white-space: pre-wrap;
        max-height: 200px;
        overflow-y: auto;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state init ────────────────────────────────────

def init_state():
    """Initialize all session state variables."""
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "sessions" not in st.session_state:
        st.session_state.sessions = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "api_ok" not in st.session_state:
        st.session_state.api_ok = False
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5
    if "page" not in st.session_state:
        st.session_state.page = "chat"  # "chat" | "upload"


init_state()


# ── API Helpers ────────────────────────────────────────────

BASE = API_BASE


def api_get(path: str, timeout: int = 10) -> dict | None:
    """Generic GET request to the API."""
    try:
        r = requests.get(f"{BASE}{path}", timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except requests.ConnectionError:
        return None


def api_post(path: str, json: dict | None = None, timeout: int = 30) -> dict | None:
    """Generic POST request to the API."""
    try:
        r = requests.post(f"{BASE}{path}", json=json, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except requests.ConnectionError:
        return None


def api_delete(path: str, timeout: int = 10) -> dict | None:
    """Generic DELETE request to the API."""
    try:
        r = requests.delete(f"{BASE}{path}", timeout=timeout)
        return r.json() if r.status_code in (200, 204) else None
    except requests.ConnectionError:
        return None


def api_patch(path: str, json: dict, timeout: int = 10) -> dict | None:
    """Generic PATCH request to the API."""
    try:
        r = requests.patch(f"{BASE}{path}", json=json, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except requests.ConnectionError:
        return None


def check_api() -> bool:
    """Ping the backend health endpoint."""
    result = api_get("/")
    ok = result is not None and result.get("status") == "ok"
    if ok != st.session_state.api_ok:
        st.session_state.api_ok = ok
    return ok


def load_sessions():
    """Fetch all chat sessions from the backend."""
    result = api_get("/api/chat/sessions")
    if result:
        st.session_state.sessions = result.get("sessions", [])
    else:
        st.session_state.sessions = []


def load_messages(session_id: int | None = None):
    """Fetch messages for the current or given session."""
    sid = session_id or st.session_state.current_session_id
    if sid is None:
        st.session_state.messages = []
        return
    result = api_get(f"/api/chat/sessions/{sid}/messages")
    if result:
        st.session_state.messages = result.get("messages", [])
    else:
        st.session_state.messages = []


def load_documents():
    """Fetch the list of uploaded documents."""
    result = api_get("/api/documents")
    if result:
        st.session_state.documents = result.get("documents", [])


def create_new_session(title: str = "New Chat") -> int | None:
    """Create a new session and return its ID."""
    result = api_post("/api/chat/sessions", json={"title": title})
    if result:
        session = result.get("session", {})
        return session.get("id")
    return None


def delete_session_backend(session_id: int) -> bool:
    """Delete a session via the API."""
    result = api_delete(f"/api/chat/sessions/{session_id}")
    return result is not None


def send_message(query: str, top_k: int = 5) -> dict | None:
    """Send a message in the current session (RAG) and return the response."""
    sid = st.session_state.current_session_id
    if sid is None:
        return None
    result = api_post(
        f"/api/chat/sessions/{sid}/messages",
        json={"query": query, "top_k": top_k},
        timeout=45,
    )
    return result


def upload_pdf(file) -> dict | None:
    """Upload a PDF to the API."""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        r = requests.post(f"{BASE}/api/upload", files=files, timeout=60)
        if r.status_code == 200:
            return r.json()
        else:
            detail = r.json().get("detail", "Unknown error")
            st.error(f"Upload failed: {detail}")
            return None
    except requests.ConnectionError:
        st.error("Cannot connect to the backend. Is the server running?")
        return None


# ── UI Actions ─────────────────────────────────────────────

def switch_session(session_id: int):
    """Switch the active chat session."""
    st.session_state.current_session_id = session_id
    load_messages()
    st.rerun()


def new_chat():
    """Create a new session and switch to it."""
    sid = create_new_session()
    if sid:
        st.session_state.current_session_id = sid
        st.session_state.messages = []
        load_sessions()
        st.rerun()
    else:
        st.error("Failed to create a new session. Check the backend.")


def delete_session_action(session_id: int):
    """Delete a session and refresh."""
    was_current = session_id == st.session_state.current_session_id
    delete_session_backend(session_id)
    load_sessions()
    if was_current:
        # Switch to the first available session or clear
        if st.session_state.sessions:
            st.session_state.current_session_id = st.session_state.sessions[0]["id"]
            load_messages()
        else:
            st.session_state.current_session_id = None
            st.session_state.messages = []
    st.rerun()


def delete_document_action(title: str):
    """Delete a document and refresh."""
    api_delete(f"/api/documents/{title}")
    load_documents()
    st.rerun()


def refresh_all():
    """Refresh all data from the backend."""
    check_api()
    load_sessions()
    if st.session_state.current_session_id:
        load_messages()
    load_documents()


# ── Render functions ──────────────────────────────────────

def render_session_list():
    """Render the sidebar session list with clickable items."""
    st.markdown("### 💬 Conversations")

    # New Chat button
    if st.button(
        "＋ New Chat",
        use_container_width=True,
        type="primary",
        key="new_chat_btn",
    ):
        new_chat()

    st.divider()

    if not st.session_state.sessions:
        st.caption("No conversations yet. Start a new chat!")
        return

    sessions = st.session_state.sessions
    current = st.session_state.current_session_id

    for sess in sessions:
        sid = sess["id"]
        title = sess.get("title", "New Chat")
        updated = sess.get("updated_at", "")

        # Format the timestamp
        time_str = ""
        if updated:
            try:
                dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                now = datetime.now(dt.tzinfo)
                diff = now - dt
                if diff.total_seconds() < 3600:
                    time_str = f"{int(diff.total_seconds() // 60)}m ago"
                elif diff.total_seconds() < 86400:
                    time_str = f"{int(diff.total_seconds() // 3600)}h ago"
                else:
                    time_str = dt.strftime("%b %d")
            except ValueError:
                time_str = ""

        is_active = sid == current
        active_class = "active" if is_active else ""

        cols = st.columns([20, 1])
        with cols[0]:
            if st.button(
                f"{title[:50]}..." if len(title) > 50 else title,
                key=f"session_{sid}",
                use_container_width=True,
                type="secondary" if not is_active else "primary",
                help=f"Switch to this conversation • Last updated: {time_str}" if time_str else "Switch to this conversation",
            ):
                switch_session(sid)
        with cols[1]:
            # Delete button (only show a subtle X)
            if st.button("🗑", key=f"del_session_{sid}", help="Delete this conversation"):
                delete_session_action(sid)


def render_document_list():
    """Render the uploaded documents section in the sidebar."""
    st.divider()
    st.markdown("### 📄 Documents")

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🔄 Refresh docs", use_container_width=True, key="refresh_docs"):
            load_documents()
            st.rerun()
    with col2:
        if st.button("📤 Upload", use_container_width=True, key="goto_upload"):
            st.session_state.page = "upload"
            st.rerun()

    if not st.session_state.documents:
        st.caption("No documents uploaded yet.")
        return

    for doc in st.session_state.documents:
        title = doc.get("title", "Unknown")
        with st.container():
            c1, c2, c3 = st.columns([5, 1, 1])
            with c1:
                st.markdown(f"📄 {title[:40]}..." if len(title) > 40 else f"📄 {title}")
            with c3:
                if st.button("🗑", key=f"del_doc_{title}", help="Delete this document"):
                    delete_document_action(title)


def render_chat_messages():
    """Render the conversation messages."""
    if not st.session_state.messages:
        # Empty state
        st.markdown(
            """
            <div style="text-align: center; padding: 4rem 1rem; color: #888;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">💬</div>
                <h3>Ask questions about your documents</h3>
                <p>Upload a PDF, then start asking questions. The answers come from your documents only.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])

        with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
            st.markdown(content)

            if role == "assistant" and sources:
                with st.expander(f"📚 View sources ({len(sources)} chunks)", expanded=False):
                    for i, src in enumerate(sources, 1):
                        sim = src.get("similarity", 0)
                        st.markdown(
                            f"**Source {i}** — similarity: `{sim:.3f}`"
                        )
                        st.markdown(
                            f'<div class="source-content">{src.get("content", "")}</div>',
                            unsafe_allow_html=True,
                        )
                        if i < len(sources):
                            st.divider()


def render_chat_input():
    """Render the chat input box."""
    if st.session_state.current_session_id is None:
        st.info("💡 Start a new conversation or select one from the sidebar.")
        return

    if prompt := st.chat_input("Ask a question about your documents...", key="chat_input"):
        # Immediately show user message
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching documents and generating answer..."):
                result = send_message(prompt, top_k=st.session_state.top_k)

            if result:
                answer = result["content"]
                sources = result.get("sources", [])
                st.markdown(answer)

                if sources:
                    with st.expander(f"📚 View sources ({len(sources)} chunks)", expanded=False):
                        for i, src in enumerate(sources, 1):
                            sim = src.get("similarity", 0)
                            st.markdown(f"**Source {i}** — similarity: `{sim:.3f}`")
                            st.markdown(
                                f'<div class="source-content">{src.get("content", "")}</div>',
                                unsafe_allow_html=True,
                            )
                            if i < len(sources):
                                st.divider()
            else:
                st.error("Sorry, I couldn't get an answer. Check the backend connection.")

        # Reload messages from backend to keep in sync
        load_messages()
        load_sessions()  # Refresh the session list (title may have auto-updated)
        st.rerun()


def render_chat_page():
    """Main chat page layout."""
    # Header with session info
    if st.session_state.current_session_id:
        current_title = "Chat"
        for s in st.session_state.sessions:
            if s["id"] == st.session_state.current_session_id:
                current_title = s.get("title", "Chat")
                break
        st.markdown(f"## 💬 {current_title}")
    else:
        st.markdown("## 💬 Chat with Your Documents")

    st.markdown(
        '<p style="color: #888; font-size: 0.9rem;">'
        "Answers are generated from your uploaded documents only. "
        "No external knowledge is used.</p>",
        unsafe_allow_html=True,
    )

    # Messages area (scrollable container)
    chat_container = st.container()
    with chat_container:
        render_chat_messages()

    # Chat input (fixed at bottom)
    st.divider()
    render_chat_input()


def render_upload_page():
    """Upload document page."""
    st.markdown("## 📤 Upload Document")

    st.markdown(
        "Upload a PDF file. The system will extract the text, split it into "
        "chunks, generate embeddings, and store them in the vector database."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        accept_multiple_files=False,
        key="pdf_uploader",
    )

    if uploaded_file is not None:
        st.markdown(f"**File:** {uploaded_file.name}")
        st.markdown(f"**Size:** {len(uploaded_file.getvalue()) / 1024:.1f} KB")

        if st.button("🚀 Process Document", type="primary", use_container_width=True):
            with st.spinner("Processing PDF..."):
                result = upload_pdf(uploaded_file)

            if result:
                st.success(
                    f"✅ **{result['filename']}** processed successfully!\n\n"
                    f"Created **{result['chunks_created']}** chunks and stored them "
                    f"in the vector database."
                )
                load_documents()

                # Offer to go back to chat
                if st.button("💬 Go to Chat", type="secondary", use_container_width=True):
                    st.session_state.page = "chat"
                    st.rerun()

    # Show current documents
    st.divider()
    st.markdown("### Stored Documents")
    load_documents()

    if st.session_state.documents:
        for doc in st.session_state.documents:
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown(f"📄 {doc.get('title', 'Unknown')}")
            with cols[1]:
                if st.button("Delete", key=f"del_doc_page_{doc.get('title', '')}"):
                    delete_document_action(doc.get("title", ""))
    else:
        st.info("No documents uploaded yet.")


# ── Main Layout ────────────────────────────────────────────

# Check API connectivity
check_api()

# ── Sidebar ────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<h1 style='font-size: 1.5rem; margin-bottom: 0.25rem;'>🤖 RAG Chatbot</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Powered by Gemini + pgvector")

    # API status
    if st.session_state.api_ok:
        st.markdown(
            '<p style="color: #22c55e; font-size: 0.8rem;">✅ Backend connected</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p style="color: #ef4444; font-size: 0.8rem;">❌ Backend unreachable</p>',
            unsafe_allow_html=True,
        )
        st.code("uvicorn app.main:app --reload")
        if st.button("🔄 Retry connection", use_container_width=True):
            refresh_all()
            st.rerun()
        st.stop()

    # Navigation tabs
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button(
            "💬 Chat",
            use_container_width=True,
            type="primary" if st.session_state.page == "chat" else "secondary",
        ):
            st.session_state.page = "chat"
            st.rerun()
    with nav_col2:
        if st.button(
            "📤 Upload",
            use_container_width=True,
            type="primary" if st.session_state.page == "upload" else "secondary",
        ):
            st.session_state.page = "upload"
            st.rerun()

    if st.session_state.page == "chat":
        render_session_list()

    st.divider()

    # Settings
    with st.expander("⚙️ Settings", expanded=False):
        st.session_state.top_k = st.slider(
            "Chunks to retrieve",
            min_value=1,
            max_value=10,
            value=st.session_state.top_k,
            help="Number of relevant document chunks to retrieve per question",
        )

    render_document_list()

    st.divider()
    st.caption("v1.1.0")

# ── Main Content ───────────────────────────────────────────

# Periodically refresh sessions & docs
refresh_all()

if st.session_state.page == "chat":
    render_chat_page()
elif st.session_state.page == "upload":
    render_upload_page()
