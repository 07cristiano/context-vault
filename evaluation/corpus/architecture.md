# API architecture

FastAPI exposes the HTTP endpoints and Pydantic validates incoming request and outgoing response structures. SQLite stores documents and chunks. The API layer translates errors but does not implement retrieval or model logic.
