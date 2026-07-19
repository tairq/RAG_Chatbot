import os
from dotenv import load_dotenv

load_dotenv()

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# OpenRouter (fallback for embeddings)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENROUTER_EMBEDDING_MODEL = os.getenv(
    "OPENROUTER_EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-en"
)

# Embedding model (Gemini primary)
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMENSION = 2048

# Chat model
CHAT_MODEL = "models/gemini-2.0-flash"

# Ingestion settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
TOP_K = 5
SIMILARITY_THRESHOLD = 0.5
