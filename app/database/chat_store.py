from typing import Any

from app.database.supabase_client import get_supabase


def create_session(title: str = "New Chat") -> dict[str, Any]:
    """Create a new chat session and return it."""
    supabase = get_supabase()
    result = (
        supabase.table("chat_sessions")
        .insert({"title": title})
        .execute()
    )
    return result.data[0] if result.data else {}


def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """List chat sessions, newest first. Returns id, title, created_at, updated_at."""
    supabase = get_supabase()
    result = (
        supabase.table("chat_sessions")
        .select("id, title, created_at, updated_at")
        .order("updated_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data if result.data else []


def get_session(session_id: int) -> dict[str, Any] | None:
    """Get a single session by id."""
    supabase = get_supabase()
    result = (
        supabase.table("chat_sessions")
        .select("*")
        .eq("id", session_id)
        .execute()
    )
    return result.data[0] if result.data else None


def update_session_title(session_id: int, title: str) -> dict[str, Any] | None:
    """Update the title of a chat session."""
    supabase = get_supabase()
    result = (
        supabase.table("chat_sessions")
        .update({"title": title})
        .eq("id", session_id)
        .execute()
    )
    return result.data[0] if result.data else None


def delete_session(session_id: int) -> bool:
    """Delete a session and all its messages (CASCADE)."""
    supabase = get_supabase()
    # ON DELETE CASCADE handles messages
    result = (
        supabase.table("chat_sessions")
        .delete()
        .eq("id", session_id)
        .execute()
    )
    return len(result.data) > 0


def add_message(
    session_id: int,
    role: str,
    content: str,
    sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Add a message to a session. Returns the created message."""
    supabase = get_supabase()

    row: dict[str, Any] = {
        "session_id": session_id,
        "role": role,
        "content": content,
    }
    if sources is not None:
        row["sources"] = sources

    result = supabase.table("chat_messages").insert(row).execute()
    return result.data[0] if result.data else {}


def get_messages(session_id: int) -> list[dict[str, Any]]:
    """Get all messages for a session, oldest first."""
    supabase = get_supabase()
    result = (
        supabase.table("chat_messages")
        .select("id, role, content, sources, created_at")
        .eq("session_id", session_id)
        .order("created_at")
        .execute()
    )
    return result.data if result.data else []
