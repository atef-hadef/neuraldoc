"""Cited QA endpoint (LangGraph agent)."""

from fastapi import APIRouter, Depends

from neuraldoc.api.deps import get_service
from neuraldoc.schemas import AskRequest, AskResponse
from neuraldoc.services import DocumentService

router = APIRouter(tags=["qa"])


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, service: DocumentService = Depends(get_service)) -> AskResponse:
    result = service.qa_service.ask(body.question)
    return AskResponse(question=result.question, answer=result.answer, citations=result.citations)
