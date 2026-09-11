"""Search service: embed query -> FAISS lookup -> ranked, cited results."""

from dataclasses import dataclass

from loguru import logger

from neuraldoc.retrieval.embedder import Embedder
from neuraldoc.retrieval.vectorstore import FaissVectorStore, StoredChunk


@dataclass
class SearchHit:
    chunk: StoredChunk
    score: float

    @property
    def citation(self) -> str:
        return f"{self.chunk.filename} [chunk {self.chunk.chunk_index}] (score={self.score:.3f})"


class SearchService:
    def __init__(self, embedder: Embedder, store: FaissVectorStore):
        self.embedder = embedder
        self.store = store

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        query = query.strip()
        if not query:
            return []
        qvec = self.embedder.embed([query])[0]
        raw = self.store.search(qvec, top_k=top_k)
        hits = [SearchHit(chunk=c, score=s) for c, s in raw]
        logger.info(f"Search '{query[:60]}' -> {len(hits)} hits")
        return hits
