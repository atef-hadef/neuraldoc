"""Health router."""

from fastapi import APIRouter, Depends

from neuraldoc.api.deps import get_service
from neuraldoc.schemas import HealthOut
from neuraldoc.services import DocumentService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health(service: DocumentService = Depends(get_service)) -> HealthOut:
    return HealthOut(status="ok", documents=len(service.list_documents()), vectors=service.store.size)
