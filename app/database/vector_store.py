from typing import Any

from app.database.supabase_client import get_supabase


def store_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    title: str,
) -> int:
    """
    Insert chunk text + embedding vectors into the documents table.

    Args:
        chunks: List of text chunks.
        embeddings: Corresponding embedding vectors.
        title: Source PDF filename.

    Returns:
        Number of rows inserted.
    """
    supabase = get_supabase()

    rows = [
        {
            "title": title,
            "chunk_index": i,
            "content": chunk,
            "embedding": embedding,
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    result = supabase.table("documents").insert(rows).execute()
    return len(result.data) if result.data else 0


def search_similar(
    query_embedding: list[float],
    top_k: int = 5,
    filter_titles: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Search for the most similar chunks via cosine similarity.

    Args:
        query_embedding: The embedding vector to search with.
        top_k: Number of results to return.
        filter_titles: Optional list of document titles to restrict search to.

    Returns:
        List of dicts with 'id', 'content', 'similarity', 'title'.
    """
    supabase = get_supabase()

    params: dict[str, Any] = {
        "query_embedding": query_embedding,
        "match_count": top_k,
    }
    if filter_titles:
        params["filter_titles"] = filter_titles

    result = supabase.rpc(
        "match_documents",
        params,
    ).execute()

    return result.data if result.data else []


def list_documents() -> list[dict[str, Any]]:
    """Return distinct document titles with chunk counts."""
    supabase = get_supabase()
    result = (
        supabase.table("documents")
        .select("title", count="exact")
        .order("created_at", desc=True)
        .execute()
    )
    # Deduplicate by title — the raw query returns one row per chunk,
    # not one row per document. Without aggregation, we collapse in Python.
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for doc in result.data or []:
        title = doc.get("title")
        if title and title not in seen:
            seen.add(title)
            unique.append(doc)
    return unique


def delete_document(title: str) -> int:
    """Delete all chunks for a given document title."""
    supabase = get_supabase()
    result = supabase.table("documents").delete().eq("title", title).execute()
    return len(result.data) if result.data else 0
