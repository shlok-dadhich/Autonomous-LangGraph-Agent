"""Typer CLI for the platform."""

from __future__ import annotations

import typer

app = typer.Typer(help="Personal Intelligence Platform")

@app.command()
def run(profile: str = "config/profile.json", dry_run: bool = False):
    """Run one digest cycle."""
    typer.echo(f"run profile={profile} dry_run={dry_run} — TODO Phase 1")

@app.command()
def migrate():
    """Run legacy migration."""
    typer.echo("migrate — TODO Phase 1")

def main():
    app()

if __name__ == "__main__":
    main()
