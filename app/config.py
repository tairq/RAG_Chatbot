import os
from dotenv import load_dotenv

load_dotenv()

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Google Gemini (legacy — switch to OpenRouter for chat)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

# Embedding model (OpenRouter / nvidia)
OPENROUTER_EMBEDDING_MODEL = os.getenv(
    "OPENROUTER_EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b:free"
)
EMBEDDING_MODEL = OPENROUTER_EMBEDDING_MODEL
EMBEDDING_DIMENSION = 2048

# Chat model (OpenRouter — free tier)
OPENROUTER_CHAT_MODEL = os.getenv(
    "OPENROUTER_CHAT_MODEL", "tencent/hy3:free"
)
CHAT_MODEL = OPENROUTER_CHAT_MODEL

# Self-hosted local chat endpoint (hybrid fallback)
LOCAL_CHAT_ENDPOINT = os.getenv("LOCAL_CHAT_ENDPOINT", "")
LOCAL_CHAT_API_KEY = os.getenv("LOCAL_CHAT_API_KEY", "")
LOCAL_CHAT_MODEL = os.getenv("LOCAL_CHAT_MODEL", "auto/best-chat")

# Ingestion settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
TOP_K = 5
SIMILARITY_THRESHOLD = 0.5

# OpenRouter rate-limits (free-tier friendly)
OPENROUTER_RPM = int(os.getenv("OPENROUTER_RPM", "10"))
