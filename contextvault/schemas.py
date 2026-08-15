"""Public HTTP response contracts."""

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComponentStatus(StrictModel):
    ready: bool
    detail: str


class ModelComponentStatus(ComponentStatus):
    embedding_ready: bool
    generation_ready: bool
    installed_models: list[str]


class StatusResponse(StrictModel):
    ready: bool
    application: ComponentStatus
    database: ComponentStatus
    ollama: ModelComponentStatus


class DocumentResponse(StrictModel):
    id: int
    filename: str
    media_type: str
    chunk_count: int
    created_at: str


class DocumentListResponse(StrictModel):
    documents: list[DocumentResponse]


class QueryRequest(StrictModel):
    question: str


class RetrievalHitResponse(StrictModel):
    chunk_id: int
    document_id: int
    filename: str
    page_number: int | None
    modality: str
    excerpt: str
    lexical_rank: int | None
    lexical_score: float | None
    semantic_rank: int | None
    semantic_score: float | None
    fused_score: float


class EvidenceResponse(RetrievalHitResponse):
    label: str


class QueryTimingResponse(StrictModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class QueryResponse(StrictModel):
    answer: str
    sufficient: bool
    citations: list[str]
    evidence: list[EvidenceResponse]
    retrieval_trace: list[RetrievalHitResponse]
    timing: QueryTimingResponse
