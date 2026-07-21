"""
OpenAI-compatible client for the self-hosted local LLM endpoint.

Used as a fallback when the RAG system can't find relevant context —
gives the bot general conversation ability instead of a cold "I don't know."

Also used as a RAG fallback when OpenRouter free tier is rate-limited.
"""

from typing import Any

from openai import OpenAI

from app.config import LOCAL_CHAT_API_KEY, LOCAL_CHAT_ENDPOINT, LOCAL_CHAT_MODEL

_client = OpenAI(base_url=LOCAL_CHAT_ENDPOINT, api_key=LOCAL_CHAT_API_KEY)


def _build_rag_system_prompt(context_chunks: list[dict[str, Any]]) -> str:
    """Build a system prompt with retrieved document context for local LLM."""
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("title", "Document")
        text = chunk["content"]
        context_parts.append(f"[Source {i} — {source}]:\n{text}")

    context_str = "\n\n---\n\n".join(context_parts)

    return f"""You are a helpful assistant that answers questions based ONLY on the provided source document context.

Instructions:
- If the answer is clearly stated in the context, answer concisely and cite the source.
- If the context contains partial information, answer with what you know and note what's missing.
- If the answer CANNOT be found in the context, say: "I don't have enough information to answer that from the provided document."
- Do NOT use any external knowledge or make up information.
- IMPORTANT: Write your response in PLAIN TEXT only. Do NOT use any markdown formatting (no **bold**, no *italic*, no # headings, no --- lines, no bullet lists with - or *). Use emojis where appropriate to make the text friendly and engaging. Use line breaks and indentation for structure instead of markdown.

Context from the document(s):
{context_str}"""


def chat_general(query: str) -> str:
    """
    Send a general (non-RAG) query to the local model.

    Args:
        query: The user's question.

    Returns:
        The model's answer text.
    """
    response = _client.chat.completions.create(
        model=LOCAL_CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful, friendly assistant. Answer the user's "
                    "question naturally and conversationally. If you don't know "
                    "something, say so honestly. "
                    "IMPORTANT: Write your response in PLAIN TEXT only. Do NOT "
                    "use any markdown formatting (no **bold**, no *italic*, no "
                    "# headings, no --- lines, no bullet lists with - or *). "
                    "Use emojis where appropriate to make the text friendly and "
                    "engaging. Use line breaks and indentation for structure "
                    "instead of markdown."
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content or ""


def chat_rag(query: str, context_chunks: list[dict[str, Any]]) -> str:
    """
    Send a RAG query with document context to the local model.

    Used as a fallback when OpenRouter free models are rate-limited.

    Args:
        query: The user's question.
        context_chunks: Relevant chunks from vector search.

    Returns:
        The model's answer text.
    """
    system_prompt = _build_rag_system_prompt(context_chunks)

    response = _client.chat.completions.create(
        model=LOCAL_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {query}"},
        ],
    )
    return response.choices[0].message.content or ""
