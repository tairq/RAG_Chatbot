import tiktoken


def _token_count(text: str, model: str = "cl100k_base") -> int:
    """Approximate token count for a text string."""
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[str]:
    """
    Split text into overlapping chunks using a recursive character approach.

    Splits on paragraph breaks first, then newlines, then sentences,
    ensuring each chunk stays under chunk_size characters.

    Args:
        text: The full document text.
        chunk_size: Maximum characters per chunk.
        overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    # Normalize whitespace
    text = text.strip()

    # Split into paragraphs first
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph exceeds chunk_size, finalize the current chunk
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from the end of the previous
            if overlap > 0 and chunks:
                current_chunk = _get_overlap_tail(chunks[-1], overlap)
            else:
                current_chunk = ""

        # If a single paragraph is longer than chunk_size, split it further
        if len(para) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            chunks.extend(_split_long_paragraph(para, chunk_size, overlap))
        else:
            separator = "\n\n" if current_chunk else ""
            current_chunk += separator + para

    # Final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _split_long_paragraph(
    para: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split a single long paragraph into chunks by sentence boundaries."""
    import re

    sentences = re.split(r"(?<=[.!?])\s+", para)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) > chunk_size and current:
            chunks.append(current.strip())
            current = _get_overlap_tail(chunks[-1], overlap) if overlap > 0 else ""

        current += (" " if current else "") + sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _get_overlap_tail(text: str, overlap: int) -> str:
    """Return the last `overlap` characters from text for chunk overlap."""
    if len(text) <= overlap:
        return text
    return text[-overlap:]
