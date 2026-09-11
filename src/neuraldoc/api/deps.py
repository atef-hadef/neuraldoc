"""Shared dependency accessor for the DocumentService singleton."""

from neuraldoc.services import DocumentService

_service: DocumentService | None = None


def set_service(service: DocumentService) -> None:
    global _service
    _service = service


def get_service() -> DocumentService:
    if _service is None:
        raise RuntimeError("Service not initialized (lifespan did not run)")
    return _service
