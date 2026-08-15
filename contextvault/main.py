"""ContextVault FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from contextvault import __version__
from contextvault.api import router
from contextvault.config import Settings
from contextvault.database import Database
from contextvault.errors import DatabaseError, ModelResponseError
from contextvault.ingestion import IngestionService
from contextvault.model_gateway import ModelGateway
from contextvault.ollama_service import OllamaService
from contextvault.rag import RagService
from contextvault.retrieval import RetrievalService

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    settings: Settings | None = None,
    model_gateway: ModelGateway | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    database = Database(resolved_settings.database_path)
    gateway = model_gateway or OllamaService(resolved_settings)
    ingestion_service = IngestionService(resolved_settings, database, gateway)
    retrieval_service = RetrievalService(resolved_settings, database, gateway)
    rag_service = RagService(resolved_settings, retrieval_service, gateway)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved_settings.ensure_runtime_directories()
        database.initialize()
        yield

    app = FastAPI(
        title="ContextVault API",
        version=__version__,
        description="Offline evidence-first multimodal retrieval and RAG",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.database = database
    app.state.model_gateway = gateway
    app.state.ingestion_service = ingestion_service
    app.state.retrieval_service = retrieval_service
    app.state.rag_service = rag_service
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.exception_handler(ModelResponseError)
    async def model_response_error_handler(
        _request: object, exc: ModelResponseError
    ) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(DatabaseError)
    async def database_error_handler(_request: object, exc: DatabaseError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
