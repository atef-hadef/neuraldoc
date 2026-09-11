"""FAISS-backed vector store with on-disk persistence.

Stores chunk payloads (doc id, filename, text, offsets) alongside an
IndexFlatIP index over L2-normalized vectors (= cosine similarity).
Persists as <index_dir>/index.faiss + <index_dir>/payloads.json.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from loguru import logger


@dataclass
class StoredChunk:
    id: int
    document_id: int
    filename: str
    text: str
    chunk_index: int


class FaissVectorStore:
    def __init__(self, dim: int, index_dir: str | Path):
        self.dim = dim
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index = None
        self._payloads: list[StoredChunk] = []
        self._load_or_init()

    # -- lifecycle -----------------------------------------------------
    def _paths(self) -> tuple[Path, Path]:
        return self.index_dir / "index.faiss", self.index_dir / "payloads.json"

    def _load_or_init(self) -> None:
        import faiss

        faiss_path, payloads_path = self._paths()
        self._index = faiss.IndexFlatIP(self.dim)
        if faiss_path.exists() and payloads_path.exists():
            try:
                self._index = faiss.read_index(str(faiss_path))
                raw = json.loads(payloads_path.read_text(encoding="utf-8"))
                self._payloads = [StoredChunk(**r) for r in raw]
                logger.info(f"Loaded FAISS index with {len(self._payloads)} vectors")
            except Exception as exc:
                logger.warning(f"Corrupt index, reinitializing: {exc}")
                self._index = faiss.IndexFlatIP(self.dim)
                self._payloads = []

    def persist(self) -> None:
        import faiss

        faiss_path, payloads_path = self._paths()
        faiss.write_index(self._index, str(faiss_path))
        payloads_path.write_text(
            json.dumps([asdict(p) for p in self._payloads], ensure_ascii=False), encoding="utf-8"
        )

    # -- ops -----------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self._payloads)

    def add(self, vectors: np.ndarray, chunks: list[StoredChunk]) -> None:
        assert vectors.shape[0] == len(chunks)
        assert vectors.shape[1] == self.dim, f"dim mismatch {vectors.shape[1]} != {self.dim}"
        self._index.add(vectors.astype(np.float32))
        self._payloads.extend(chunks)
        self.persist()

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[StoredChunk, float]]:
        if self.size == 0:
            return []
        q = query_vector.reshape(1, -1).astype(np.float32)
        scores, idxs = self._index.search(q, min(top_k, self.size))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if 0 <= idx < len(self._payloads):
                results.append((self._payloads[idx], float(score)))
        return results

    def clear(self) -> None:
        import faiss

        self._index = faiss.IndexFlatIP(self.dim)
        self._payloads = []
        self.persist()
