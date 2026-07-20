"""
OpenAI-compatible client for the self-hosted local LLM endpoint.

Used as a fallback when the RAG system can't find relevant context —
gives the bot general conversation ability instead of a cold "I don't know."
"""

from openai import OpenAI

from app.config import LOCAL_CHAT_API_KEY, LOCAL_CHAT_ENDPOINT, LOCAL_CHAT_MODEL

_client = OpenAI(base_url=LOCAL_CHAT_ENDPOINT, api_key=LOCAL_CHAT_API_KEY)


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
                    "something, say so honestly."
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content or ""
