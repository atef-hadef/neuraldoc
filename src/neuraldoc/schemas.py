"""Pydantic v2 schemas for API I/O."""

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: int
    filename: str
    file_type: str
    char_count: int
    chunk_count: int
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class SearchHitOut(BaseModel):
    filename: str
    chunk_index: int
    text: str
    score: float = Field(ge=-1.0, le=2.0)
    citation: str


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHitOut]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[str]


class AnomalyOut(BaseModel):
    document_id: int
    filename: str
    score: float
    is_anomaly: bool


class HealthOut(BaseModel):
    status: str
    documents: int
    vectors: int
