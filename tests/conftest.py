"""Shared pytest fixtures: isolated settings, service, API client."""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neuraldoc.api import deps
from neuraldoc.api.main import create_app
from neuraldoc.config import Settings
from neuraldoc.db.models import init_db, make_engine, make_session_factory
from neuraldoc.retrieval.embedder import Embedder
from neuraldoc.retrieval.vectorstore import FaissVectorStore
from neuraldoc.services import DocumentService


@pytest.fixture()
def tmp_dirs():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        (base / "indices").mkdir()
        (base / "uploads").mkdir()
        yield base


@pytest.fixture()
def service(tmp_dirs: Path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_dirs}/test.db",
        index_dir=tmp_dirs / "indices",
        upload_dir=tmp_dirs / "uploads",
        chunk_size=200,
        chunk_overlap=40,
    )
    engine = make_engine(settings.database_url)
    # NullPool-equivalent behavior on Windows: dispose explicitly at teardown
    # so TemporaryDirectory cleanup never hits a locked sqlite file.
    init_db(engine)
    session_factory = make_session_factory(engine)
    embedder = Embedder(dim=64)  # hashed fallback path in CI (offline-safe)
    embedder._fallback = True
    store = FaissVectorStore(dim=64, index_dir=settings.index_dir)
    svc = DocumentService(settings, session_factory, embedder, store)
    yield svc
    engine.dispose()


@pytest.fixture()
def client(service: DocumentService):
    app = create_app()
    # Bypass lifespan: inject test service directly.
    deps.set_service(service)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def sample_txt(tmp_path: Path) -> Path:
    p = tmp_path / "policy.txt"
    p.write_text(
        "Refund Policy. Customers may request a refund within 30 days of purchase. "
        "Refunds are issued to the original payment method. "
        "Shipping fees are non-refundable unless the item arrived damaged. ",
        encoding="utf-8",
    )
    return p
