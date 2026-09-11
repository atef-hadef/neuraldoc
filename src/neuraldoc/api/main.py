"""FastAPI application factory with lifespan-managed singletons."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from loguru import logger

from neuraldoc.api import deps
from neuraldoc.api.routers import anomalies, ask, documents, health, search
from neuraldoc.config import get_settings
from neuraldoc.db.models import init_db, make_engine, make_session_factory
from neuraldoc.logging import configure_logging
from neuraldoc.retrieval.embedder import Embedder
from neuraldoc.retrieval.vectorstore import FaissVectorStore
from neuraldoc.services import DocumentService


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    engine = make_engine(settings.database_url)
    init_db(engine)
    session_factory = make_session_factory(engine)
    embedder = Embedder(model_name=settings.embedding_model, dim=settings.embedding_dim)
    store = FaissVectorStore(dim=embedder.dim, index_dir=settings.index_dir)
    # Reconcile dim once the real model loads (fallback dim may differ).
    try:
        probe = embedder.embed(["probe"])
        if probe.shape[1] != store.dim:
            store.dim = probe.shape[1]
    except Exception as exc:
        logger.warning(f"Embedding probe failed: {exc}")
    service = DocumentService(settings, session_factory, embedder, store)
    deps.set_service(service)
    logger.info("NeuralDoc API ready")
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="NeuralDoc", version="0.1.0", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(search.router)
    app.include_router(ask.router)
    app.include_router(anomalies.router)
    return app


app = create_app()
