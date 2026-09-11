"""Token-agnostic sliding-window chunker with overlap.

Why character windows instead of token windows: keeps the pipeline free of
tokenizer dependencies at ingest time; the downstream embedder truncates
gracefully. Overlap preserves cross-boundary context for retrieval.
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    chunk_index: int
    start_char: int
    end_char: int


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not (0 <= chunk_overlap < chunk_size):
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    chunks: list[Chunk] = []
    step = chunk_size - chunk_overlap
    index = 0
    for start in range(0, len(cleaned), step):
        end = min(start + chunk_size, len(cleaned))
        # Avoid tiny trailing sliver: merge into previous chunk.
        if chunks and len(cleaned) - start < chunk_overlap:
            last = chunks[-1]
            chunks[-1] = Chunk(
                text=cleaned[last.start_char : end],
                chunk_index=last.chunk_index,
                start_char=last.start_char,
                end_char=end,
            )
            break
        chunks.append(Chunk(text=cleaned[start:end], chunk_index=index, start_char=start, end_char=end))
        index += 1
        if end == len(cleaned):
            break
    return chunks
