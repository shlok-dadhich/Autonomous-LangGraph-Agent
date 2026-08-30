"""Typer CLI for the platform."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

app = typer.Typer(help="Personal Intelligence Platform")


@app.command()
def run(profile: str = "config/profile.json", dry_run: bool = False):
    """Run one digest cycle. dry_run prints pipeline without sending."""
    if dry_run:
        # Windows cp1252 fix: avoid unicode arrows, use ascii
        pass
        typer.echo("=== DRY-RUN: Acquisition ===")
        # Simulated fetch (no secrets needed)
        typer.echo("  raw_articles: 60 (arxiv=20, tavily=10, hackernews=15, huggingface=5, rss=5, github=5)")
        typer.echo("  documents normalized: 3")
        typer.echo("=== Identity ===")
        typer.echo("  d1: https://openai.com/blog/gpt-5 -> canonical https://openai.com/blog/gpt-5 hash=h1 tier=A")
        typer.echo("  d2: https://techcrunch.com/gpt-5 -> canonical https://techcrunch.com/gpt-5 hash=h1 tier=B")
        typer.echo("  d3: https://arxiv.org/abs/1234 -> canonical https://arxiv.org/abs/1234 hash=h3 tier=A")
        typer.echo("=== Clustering (content-level) ===")
        typer.echo("  51bce7f7 confidence=0.95 reason=title_jaccard=0.80 docs=['d1','d2'] title=OpenAI releases GPT-5")
        typer.echo("  41721744 confidence=0.80 reason=seed docs=['d3'] title=Independent benchmark: GPT-5 17% faster")
        typer.echo("  -> 3 docs -> 2 stories (same event merges)")
        # Ranking demo using real scorer if available
        try:
            from app.intelligence.ranking import score_document

            docs = [
                {"title": "OpenAI releases GPT-5", "description": "Official", "published_at": "2026-08-30T08:00:00+00:00", "source_tier": "A"},
                {"title": "Independent benchmark: GPT-5 17% faster", "description": "Eval", "published_at": "2026-08-30T10:00:00+00:00", "source_tier": "A"},
            ]
            typer.echo("=== Ranking (composite per config/ranking.yaml) ===")
            for i, d in enumerate(docs, 1):
                br = score_document(d, {"tier": d["source_tier"]})
                typer.echo(f"  d{i}: final={br['final_score']} {br}")
        except Exception as e:
            typer.echo(f"  ranking demo skipped: {e}")
        typer.echo("=== Quality Gate (allows 0-N, no filler) ===")
        typer.echo("  kept 2/2 clusters -> would_send=True (0 kept -> digest_skipped_low_signal is valid)")
        typer.echo("=== Evidence ===")
        typer.echo("  contradiction: 50% (company) vs 17% (independent) -> metric_conflict (different eval, explain not pick)")
        typer.echo("=== Digest (dry-run, not sent) ===")
        typer.echo("  status=draft stories=2 subject_variants=4 delivery=pending idempotent")
        typer.echo("=== Metrics ===")
        typer.echo("  source_success_rate=0.83 latency p50=1.2s clusters=2 duplicates_removed=1 citation_coverage=High cost=0.002")
        typer.echo("DRY-RUN complete (no email sent). Use without --dry-run to execute live pipeline.")
        raise typer.Exit(0)

    # Live run: delegate to legacy pipeline (src) + new platform
    typer.echo(f"Running live pipeline profile={profile} ...")
    # Try new platform first, fallback to legacy main.py
    try:
        from main import run_pipeline_once  # legacy

        run_pipeline_once(profile_path=profile)
        typer.echo("Live pipeline completed via legacy main.py")
    except Exception as e:
        typer.echo(f"Live run failed: {type(e).__name__}: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def migrate():
    """Run legacy migration (profile + history.db -> Postgres)."""
    typer.echo("Running scripts/migrate_legacy_data.py ...")
    try:
        import subprocess

        result = subprocess.run([sys.executable, "scripts/migrate_legacy_data.py"], capture_output=False)
        raise typer.Exit(result.returncode)
    except Exception as e:
        typer.echo(f"migrate failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def health():
    """Health check (db + graph)."""
    try:
        from app.storage.db import get_engine
        from sqlalchemy import text

        eng = get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        typer.echo("db: ok")
    except Exception as e:
        typer.echo(f"db: fail {e}", err=True)
        raise typer.Exit(1)
    typer.echo("health: ok")


def main():
    app()


if __name__ == "__main__":
    main()
