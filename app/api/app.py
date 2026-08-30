"""FastAPI factory."""

from __future__ import annotations

from fastapi import FastAPI

def create_app() -> FastAPI:
    app = FastAPI(title="Personal Intelligence Platform", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
