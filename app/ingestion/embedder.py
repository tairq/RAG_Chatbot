import time
from collections import deque

import httpx

from app.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_EMBEDDING_MODEL,
)

_HTTP_TIMEOUT = 60.0

# ── Rate limiting (free-tier friendly) ─────────────────────
# Stay polite — ~30 RPM is safe for free models.
_embedding_times: deque[float] = deque(maxlen=25)
_MIN_INTERVAL = 60.0 / 25  # ~2.4 seconds between calls


def _rate_limit():
    """Throttle requests to avoid rate limits on free-tier models."""
    now = time.monotonic()

    if _embedding_times:
        elapsed = now - _embedding_times[-1]
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
            now = time.monotonic()

    if len(_embedding_times) >= _embedding_times.maxlen:
        earliest = _embedding_times[0]
        if now - earliest < 60:
            time.sleep(60.0 - (now - earliest))

    _embedding_times.append(time.monotonic())


def _call_embeddings(texts: list[str]) -> list[list[float]]:
    """Raw call to OpenRouter embedding API via httpx."""
    _rate_limit()
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        resp = client.post(
            f"{OPENROUTER_BASE_URL}/embeddings",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_EMBEDDING_MODEL,
                "input": texts if len(texts) > 1 else texts[0],
            },
        )
        resp.raise_for_status()
        data = resp.json()

    # Sort by index to maintain order
    sorted_items = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_items]


# ── Public API ───────────────────────────────────────────


def embed_text(text: str) -> list[float]:
    """
    Generate an embedding vector for a single text string.

    Uses OpenRouter with a free embedding model (nemotron-3-embed-1b).

    Args:
        text: The text to embed.

    Returns:
        A list of floats (2048-dimensional vector).
    """
    return _call_embeddings([text])[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate embedding vectors for a batch of texts.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (each 2048-dim).
    """
    return _call_embeddings(texts)
