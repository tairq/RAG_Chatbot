from typing import Any

from app.config import TOP_K
from app.database.vector_store import search_similar
from app.ingestion.embedder import embed_text


async def retrieve(query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """
    Retrieve the top-k most relevant chunks for a query.

    1. Embed the query using Gemini.
    2. Perform vector similarity search in Supabase.
    3. Return the chunks with their similarity scores.

    Args:
        query: The user's question.
        top_k: Number of chunks to retrieve.

    Returns:
        List of dicts with keys: id, content, similarity, title.
    """
    query_embedding = embed_text(query)
    results = search_similar(query_embedding, top_k=top_k)
    return results
