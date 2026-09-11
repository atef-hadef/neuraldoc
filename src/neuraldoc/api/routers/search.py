"""Semantic search endpoint."""

from fastapi import APIRouter, Depends, Query

from neuraldoc.api.deps import get_service
from neuraldoc.schemas import SearchHitOut, SearchResponse
from neuraldoc.services import DocumentService

router = APIRouter(tags=["search"])


@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(min_length=2, max_length=500),
    top_k: int = Query(default=5, ge=1, le=20),
    service: DocumentService = Depends(get_service),
) -> SearchResponse:
    hits = service.search_service.search(q, top_k=top_k)
    return SearchResponse(
        query=q,
        hits=[
            SearchHitOut(
                filename=h.chunk.filename,
                chunk_index=h.chunk.chunk_index,
                text=h.chunk.text,
                score=h.score,
                citation=h.citation,
            )
            for h in hits
        ],
    )
