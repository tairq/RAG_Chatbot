import time
from collections import deque
from typing import Any

from openai import OpenAI

from app.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_CHAT_MODEL, OPENROUTER_RPM

_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)

# ── Rate limiting (free-tier friendly) ─────────────────────
_call_times: deque[float] = deque(maxlen=OPENROUTER_RPM)
_MIN_INTERVAL = 60.0 / max(OPENROUTER_RPM, 1)


def _rate_limit():
    """Throttle requests to avoid 429s on free-tier OpenRouter models."""
    now = time.monotonic()

    if _call_times:
        elapsed = now - _call_times[-1]
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
            now = time.monotonic()

    if len(_call_times) >= _call_times.maxlen:
        earliest = _call_times[0]
        if now - earliest < 60:
            time.sleep(60.0 - (now - earliest))

    _call_times.append(time.monotonic())


# ── Prompt building (same as before, model-agnostic) ───────


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
- IMPORTANT: Write your response in PLAIN TEXT only. Do NOT use any markdown formatting (no **bold**, no *italic*, no # headings, no --- lines, no bullet lists with - or *). Use emojis where appropriate to make the text friendly and engaging. Use line breaks and indentation for structure instead of markdown.

Context from the document(s):
{context_str}"""


# ── Public API ─────────────────────────────────────────────


def ask_openrouter(
    query: str,
    context_chunks: list[dict[str, Any]],
) -> str:
    """
    Send a query with retrieved context to OpenRouter and return the answer.
    Stateless — no conversation history.

    Args:
        query: The user's question.
        context_chunks: Relevant chunks from vector search.

    Returns:
        The model's answer text.
    """
    system_prompt = _build_system_prompt(context_chunks)
    _rate_limit()

    response = _client.chat.completions.create(
        model=OPENROUTER_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {query}"},
        ],
    )

    return response.choices[0].message.content or ""


def ask_openrouter_with_history(
    query: str,
    context_chunks: list[dict[str, Any]],
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

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if chat_history:
        for msg in chat_history:
            role = "user" if msg.get("role") == "user" else "assistant"
            messages.append({"role": role, "content": msg.get("content", "")})

    messages.append({"role": "user", "content": f"Question: {query}"})

    _rate_limit()

    response = _client.chat.completions.create(
        model=OPENROUTER_CHAT_MODEL,
        messages=messages,
    )

    return response.choices[0].message.content or ""
