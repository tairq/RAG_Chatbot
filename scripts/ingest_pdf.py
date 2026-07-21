"""
One-shot PDF ingestion script.

Usage: python scripts/ingest_pdf.py <path-to-pdf>

Processes the PDF using the app's own pipeline (extract -> chunk -> embed -> store)
so you don't need the FastAPI server running.
"""

import sys
import os
import time
import httpx

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ingestion.pdf_processor import extract_text_from_pdf
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import embed_batch
from app.database.vector_store import store_chunks
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


# ── Helpers ──────────────────────────────────────────────

def _sanitize_text(text: str) -> str:
    """Remove characters that PostgreSQL cannot store (null bytes, etc.)."""
    return text.replace("\x00", "")


# ── Rate-limit handling ──────────────────────────────────
# Free-tier quota: 100 texts per minute per model.
# Each batch of N texts counts as N quota units.

_BATCH_SIZE = 10         # texts per API call
_MIN_TEXTS_PER_SEC = 1.8   # stay under 100/min (100/60 ≈ 1.67, add margin)


def _embed_with_retry(batch: list[str], batch_no: int, total: int) -> list[list[float]]:
    """Call embed_batch with rate-limit awareness and retry on 429."""
    while True:
        try:
            return embed_batch(batch)
        except (httpx.HTTPStatusError, RuntimeError) as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str:
                print(f"      Rate limited! Waiting 30s...")
                time.sleep(30)
                # Continue loop to retry
            else:
                raise


def main(pdf_path: str):
    if not os.path.exists(pdf_path):
        print(f"ERROR: File not found: {pdf_path}")
        sys.exit(1)

    title = os.path.basename(pdf_path)
    print(f"Processing: {title}")
    print(f"   File size: {os.path.getsize(pdf_path) / 1024:.1f} KB")

    # 1. Extract text
    print("   Extracting text...")
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print("ERROR: No text extracted. The PDF may be scanned or image-based.")
        sys.exit(1)
    print(f"   OK - Extracted {len(text):,} characters")

    # 2. Chunk
    print("   Chunking text...")
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    if not chunks:
        print("ERROR: No chunks created.")
        sys.exit(1)
    print(f"   OK - Created {len(chunks)} chunks")

    # 3. Embed with rate-limit awareness
    print("   Generating embeddings (this will take a moment)...")
    total_batches = (len(chunks) + _BATCH_SIZE - 1) // _BATCH_SIZE
    all_embeddings = []
    for i in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[i : i + _BATCH_SIZE]
        batch_no = i // _BATCH_SIZE + 1

        embeddings = _embed_with_retry(batch, batch_no, total_batches)
        all_embeddings.extend(embeddings)
        print(f"      Batch {batch_no}/{total_batches} done ({len(batch)} chunks)")

        # Pace: stay under 100 texts/min (1.67 texts/sec)
        time.sleep(len(batch) / _MIN_TEXTS_PER_SEC)

    print(f"   OK - Generated {len(all_embeddings)} embeddings (768-d each)")

    # 3b. Sanitize chunks (remove chars PG rejects, like null bytes)
    chunks = [_sanitize_text(c) for c in chunks]

    # 4. Store in Supabase
    print("   Storing in Supabase...")
    count = store_chunks(chunks, all_embeddings, title)
    print(f"   OK - Stored {count} rows in the documents table")

    print(f"\nDone! '{title}' is now in Supabase.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_pdf.py <path-to-pdf>")
        sys.exit(1)
    main(sys.argv[1])
