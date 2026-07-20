from typing import Any

from google import genai
from google.genai import types as genai_types

from app.config import CHAT_MODEL, GEMINI_API_KEY

_client = genai.Client(api_key=GEMINI_API_KEY)


def _build_system_prompt(context_chunks: list[dict[str, Any]]) -> str:
    """Build the system prompt with retrieved context."""
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

Context from the document(s):
{context_str}"""


def _history_to_contents(
    chat_history: list[dict[str, str]] | None,
) -> list[genai_types.Content]:
    """Convert chat history dicts to Gemini Content objects."""
    if not chat_history:
        return []

    contents = []
    for msg in chat_history:
        role = "user" if msg.get("role") == "user" else "model"
        content = msg.get("content", "")
        contents.append(
            genai_types.Content(
                role=role,
                parts=[genai_types.Part(text=content)],
            )
        )
    return contents


def ask_gemini(
    query: str,
    context_chunks: list[dict[str, Any]],
) -> str:
    """
    Send a query with retrieved context to Gemini and return the answer.
    Stateless — no conversation history.

    Args:
        query: The user's question.
        context_chunks: Relevant chunks from vector search.

    Returns:
        The model's answer text.
    """
    system_prompt = _build_system_prompt(context_chunks)

    response = _client.models.generate_content(
        model=CHAT_MODEL,
        contents=[f"Question: {query}"],
        config=genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
        ),
    )

    return response.text


def ask_gemini_with_history(
    query: str,
    context_chunks: list[dict[str, str]],
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    """
    Send a query with context and conversation history for multi-turn chat.

    Args:
        query: The user's question.
        context_chunks: Relevant chunks from vector search.
        chat_history: Previous messages as [{"role": "user"/"assistant", "content": "..."}]

    Returns:
        The model's answer text.
    """
    system_prompt = _build_system_prompt(context_chunks)

    # Build contents: conversation history + new question
    contents = _history_to_contents(chat_history)
    contents.append(
        genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=f"Question: {query}")],
        )
    )

    response = _client.models.generate_content(
        model=CHAT_MODEL,
        contents=contents,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
        ),
    )

    return response.text
