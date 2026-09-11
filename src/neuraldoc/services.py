"""Application service: orchestrates parsing -> chunking -> embedding -> storage."""

from pathlib import Path

import numpy as np
from loguru import logger
from sqlalchemy.orm import Session

from neuraldoc.anomaly.detector import AnomalyDetector, AnomalyResult
from neuraldoc.config import Settings
from neuraldoc.db.models import ChunkRow, Document
from neuraldoc.exceptions import EmptyDocumentError
from neuraldoc.ingestion.chunker import chunk_text
from neuraldoc.ingestion.parsers import parse_file
from neuraldoc.retrieval.embedder import Embedder
from neuraldoc.retrieval.qa import QAService
from neuraldoc.retrieval.search import SearchService
from neuraldoc.retrieval.vectorstore import FaissVectorStore, StoredChunk


class DocumentService:
    def __init__(
        self,
        settings: Settings,
        session_factory,
        embedder: Embedder,
        store: FaissVectorStore,
    ):
        self.settings = settings
        self.session_factory = session_factory
        self.embedder = embedder
        self.store = store
        self.search_service = SearchService(embedder, store)
        self.qa_service = QAService(self.search_service)
        self.anomaly = AnomalyDetector(contamination=settings.anomaly_contamination)

    # -- ingest --------------------------------------------------------
    def ingest_file(self, path: str | Path, original_filename: str | None = None) -> Document:
        parsed = parse_file(path)
        chunks = chunk_text(
            parsed.text,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            raise EmptyDocumentError(f"No chunks produced for {parsed.filename}")
        vectors = self.embedder.embed([c.text for c in chunks])
        # Ensure store dim matches embedder (first run may differ from config default)
        if vectors.shape[1] != self.store.dim:
            logger.warning("Embedding dim changed; reinitializing vector store")
            self.store.dim = vectors.shape[1]
            self.store.clear()

        with self.session_factory() as session:
            doc = Document(
                filename=original_filename or parsed.filename,
                file_type=parsed.suffix,
                char_count=len(parsed.text),
                chunk_count=len(chunks),
            )
            session.add(doc)
            session.flush()  # assign id
            for chunk in chunks:
                session.add(
                    ChunkRow(document_id=doc.id, chunk_index=chunk.chunk_index, text=chunk.text)
                )
            session.commit()
            doc_id, filename = doc.id, doc.filename

        stored = [
            StoredChunk(
                id=len(self.store._payloads) + i,
                document_id=doc_id,
                filename=filename,
                text=c.text,
                chunk_index=c.chunk_index,
            )
            for i, c in enumerate(chunks)
        ]
        self.store.add(vectors, stored)
        logger.info(f"Ingested {filename}: {len(chunks)} chunks (doc_id={doc_id})")
        with self.session_factory() as session:
            return session.get(Document, doc_id)

    def list_documents(self) -> list[Document]:
        with self.session_factory() as session:
            from sqlalchemy import select

            return list(session.scalars(select(Document).order_by(Document.created_at.desc())))

    def delete_document(self, document_id: int) -> bool:
        with self.session_factory() as session:
            doc = session.get(Document, document_id)
            if doc is None:
                return False
            session.delete(doc)
            session.commit()
        # Rebuild index without deleted doc (simple + correct for portfolio scale)
        self._rebuild_index()
        return True

    def _rebuild_index(self) -> None:
        from sqlalchemy import select

        with self.session_factory() as session:
            rows = list(session.scalars(select(ChunkRow).order_by(ChunkRow.id)))
        self.store.clear()
        if not rows:
            return
        vectors = self.embedder.embed([r.text for r in rows])
        with self.session_factory() as session:
            docs = {d.id: d.filename for d in session.scalars(select(Document))}
        stored = [
            StoredChunk(
                id=i, document_id=r.document_id, filename=docs.get(r.document_id, "?"),
                text=r.text, chunk_index=r.chunk_index,
            )
            for i, r in enumerate(rows)
        ]
        self.store.add(vectors, stored)

    # -- anomalies -----------------------------------------------------
    def detect_anomalies(self) -> list[AnomalyResult]:
        per_doc: dict[int, list[np.ndarray]] = {}
        names: dict[int, str] = {}
        for payload in self.store._payloads:
            q = self.embedder.embed([payload.text])[0]
            per_doc.setdefault(payload.document_id, []).append(q)
            names[payload.document_id] = payload.filename
        doc_vectors = {
            doc_id: (names[doc_id], np.mean(vecs, axis=0)) for doc_id, vecs in per_doc.items()
        }
        return self.anomaly.detect(doc_vectors)
