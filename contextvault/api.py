"""FastAPI routes and HTTP-level translation."""

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile, status

from contextvault.errors import (
    CapacityError,
    DocumentNotFoundError,
    DuplicateDocumentError,
    ExtractionError,
    ModelUnavailableError,
    UploadValidationError,
)
from contextvault.retrieval import RetrievalHit
from contextvault.schemas import (
    ComponentStatus,
    DocumentListResponse,
    DocumentResponse,
    EvidenceResponse,
    ModelComponentStatus,
    QueryRequest,
    QueryResponse,
    QueryTimingResponse,
    RetrievalHitResponse,
    StatusResponse,
)

router = APIRouter(prefix="/api")


@router.get("/status", response_model=StatusResponse)
def get_status(request: Request) -> StatusResponse:
    database_ready, database_detail = request.app.state.database.health()
    model_status = request.app.state.model_gateway.status()
    models_ready = model_status.embedding_ready and model_status.generation_ready

    return StatusResponse(
        ready=database_ready and model_status.reachable and models_ready,
        application=ComponentStatus(ready=True, detail="ContextVault API is running"),
        database=ComponentStatus(ready=database_ready, detail=database_detail),
        ollama=ModelComponentStatus(
            ready=model_status.reachable and models_ready,
            detail=model_status.detail,
            embedding_ready=model_status.embedding_ready,
            generation_ready=model_status.generation_ready,
            installed_models=list(model_status.installed_models),
        ),
    )


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(request: Request) -> DocumentListResponse:
    documents = request.app.state.database.list_documents()
    return DocumentListResponse(
        documents=[
            DocumentResponse.model_validate(document, from_attributes=True)
            for document in documents
        ]
    )


@router.post("/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def upload_document(request: Request, file: Annotated[UploadFile, File()]) -> DocumentResponse:
    try:
        document = request.app.state.ingestion_service.ingest(file.filename or "", file.file)
    except DuplicateDocumentError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except CapacityError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except (UploadValidationError, ExtractionError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return DocumentResponse.model_validate(document, from_attributes=True)


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(request: Request, document_id: int) -> Response:
    try:
        request.app.state.ingestion_service.delete(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _hit_response(hit: RetrievalHit, *, include_full_content: bool = False) -> RetrievalHitResponse:
    return RetrievalHitResponse(
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        filename=hit.filename,
        page_number=hit.page_number,
        modality=hit.modality,
        excerpt=hit.content if include_full_content else hit.content[:600],
        lexical_rank=hit.lexical_rank,
        lexical_score=hit.lexical_score,
        semantic_rank=hit.semantic_rank,
        semantic_score=hit.semantic_score,
        fused_score=hit.fused_score,
    )


@router.post("/query", response_model=QueryResponse)
def query_vault(request: Request, payload: QueryRequest) -> QueryResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Question cannot be empty",
        )
    if len(question) > 500:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Question must be 500 characters or fewer",
        )
    try:
        result = request.app.state.rag_service.query(question)
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    evidence = []
    for item in result.evidence:
        hit_response = _hit_response(item.hit, include_full_content=True)
        evidence.append(EvidenceResponse(label=item.label, **hit_response.model_dump()))
    total_ms = result.retrieval_ms + result.generation_ms
    return QueryResponse(
        answer=result.answer,
        sufficient=result.sufficient,
        citations=list(result.citations),
        evidence=evidence,
        retrieval_trace=[_hit_response(hit) for hit in result.trace],
        timing=QueryTimingResponse(
            retrieval_ms=result.retrieval_ms,
            generation_ms=result.generation_ms,
            total_ms=total_ms,
        ),
    )
