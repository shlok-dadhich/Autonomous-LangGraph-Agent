"""FastAPI factory — wires all product routes."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse


def create_app() -> FastAPI:
    app = FastAPI(title="Personal Intelligence Platform", version="0.1.0", docs_url="/docs", redoc_url="/redoc")

    @app.get("/health")
    def health():
        return {"status": "ok", "phase": "5-product"}

    # Wire routers (lazy import to avoid circular)
    from app.api.routes import admin, ask, digests, entities, feedback, search, stories, topics, users

    app.include_router(digests.router, prefix="/digests", tags=["digests"])
    app.include_router(stories.router, prefix="/stories", tags=["stories"])
    app.include_router(search.router, prefix="/search", tags=["search"])
    app.include_router(ask.router, prefix="/ask", tags=["ask"])
    app.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
    app.include_router(users.router, prefix="/users", tags=["users"])
    app.include_router(entities.router, prefix="/entities", tags=["entities"])
    app.include_router(topics.router, prefix="/topics", tags=["topics"])
    app.include_router(admin.router, prefix="/admin", tags=["admin"])

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        # Minimal SSR dashboard (Phase 5)
        return """
        <html><head><title>Your Brief</title></head><body style="font-family:system-ui;max-width:800px;margin:auto;padding:24px;">
        <h1>Your Brief</h1>
        <nav><a href="/docs">API Docs</a> | <a href="/stories">Stories</a> | <a href="/digests">Digests</a></nav>
        <h2>Trending in Your Topics</h2><p>API at /stories?topic=AI+Agents</p>
        <h2>New Since Yesterday</h2><p>GET /digests?since=yesterday</p>
        <h2>Following / Saved</h2><p>POST /feedback with action FOLLOW/SAVE</p>
        <h2>Ask Your Intelligence</h2><p>POST /ask {question: "What changed in AI agents this week?"}</p>
        </body></html>
        """

    return app
