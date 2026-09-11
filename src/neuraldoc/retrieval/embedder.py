"""Local sentence-transformer embeddings with a deterministic fallback.

Production rationale: the service must boot and tests must pass even when the
HF model cache is cold or the host is offline. We therefore lazy-load
sentence-transformers and fall back to a hashed bag-of-words embedding that
preserves the cosine-similarity contract (L2-normalized float32 vectors).
"""

import hashlib
import re

import numpy as np
from loguru import logger

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", dim: int = 384):
        self.model_name = model_name
        self.dim = dim
        self._model = None
        self._fallback = False

    def _ensure_model(self) -> None:
        if self._model is not None or self._fallback:
            return
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
        except Exception as exc:  # offline / no cache / missing lib
            logger.warning(f"Embedding model unavailable, using hashed fallback: {exc}")
            self._fallback = True

    @staticmethod
    def _hash_embed(texts: list[str], dim: int) -> np.ndarray:
        vecs = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                vecs[i, h % dim] += 1.0
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vecs / norms).astype(np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_model()
        if self._fallback or self._model is None:
            return self._hash_embed(texts, self.dim)
        try:
            return np.asarray(
                self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                dtype=np.float32,
            )
        except Exception as exc:
            logger.warning(f"Embedding inference failed, fallback engaged: {exc}")
            self._fallback = True
            return self._hash_embed(texts, self.dim)
