# Newsletter Agent

An autonomous, production-style AI newsletter worker that discovers high-signal AI updates, ranks them against your interests, writes concise technical summaries, and delivers a polished HTML digest by email.

## What It Does

- Runs a weekly pipeline using a persistent scheduler.
- Collects AI content from multiple intelligence layers:
  - Arxiv (academic papers)
  - Tavily web search (general web/news)
  - Social signals via Tavily (Reddit/Hugging Face/Hacker News domains)
  - Hacker News top stories
  - Hugging Face Daily Papers
- Merges and deduplicates results against previously sent URLs.
- Semantically ranks items with sentence-transformers.
- Applies a fallback search when content volume is too low.
- Generates concise, personalized article analysis with Groq.
- Renders a premium HTML newsletter via Jinja2.
- Sends email through SMTP and emits Telegram success/error alerts.
- Persists state in SQLite (`history.db`, `checkpoints.db`) under `data/`.

## High-Level Architecture

1. `main.py` bootstraps logging, env checks, and the scheduler.
2. `src/graph/blueprint.py` builds a LangGraph flow with parallel fan-out.
3. `src/graph/nodes.py` orchestrates source fetch -> merge -> dedupe -> rank -> write -> deliver.
4. `src/core/database.py` stores sent URL history and checkpoint housekeeping.
5. `src/core/ranker.py` computes semantic relevance scores.
6. `src/core/writer.py` calls Groq to generate article insights.
7. `src/services/template_service.py` + `src/templates/email_body.html` render newsletter HTML.
8. `src/services/email_service.py` and `src/services/telegram_bot.py` handle delivery and alerts.

## Project Structure

```text
newsletter_agent/
  main.py
  requirements.txt
  config/profile.json
  src/
    core/
    graph/
    services/
    tools/
    templates/
    utils/
```

## Requirements

- Python 3.11+ (3.12 recommended)
- Existing virtual environment (recommended)
- API keys and delivery credentials (see environment section)

## Environment Variables

Use `.env.example` as your template.

Required for startup checks:

- `TAVILY_API_KEY`
- `GROQ_API_KEY`
- `SMTP_APP_PASS`
- `TELEGRAM_BOT_TOKEN`

Also commonly needed:

- `SMTP_USER`
- `SMTP_TO` (optional; defaults to `SMTP_USER`)
- `TELEGRAM_CHAT_ID`
- `GROQ_MODEL` (optional)
- `NEWSLETTER_TIMEZONE` (optional; default `UTC`)
- `NEWSLETTER_DATA_DIR` (optional; default `/data`)

### Local Windows Tip

For local runs on Windows, set:

```env
NEWSLETTER_DATA_DIR=./data
```

This keeps SQLite files in the project folder instead of `/data`.

## Quick Start

1. Activate your existing virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from `.env.example` and fill real values.
4. Adjust your interests in `config/profile.json`.
5. Start worker:

```bash
python main.py
```

The scheduler starts and waits for the weekly run (Monday 08:00 in configured timezone).

## Scheduling Behavior

- Weekly job: Monday at 08:00 (`NEWSLETTER_TIMEZONE` or UTC).
- Monthly housekeeping: Day 1 at 00:00 (checkpoint cleanup).
- Signal handlers support graceful shutdown.

## Deployment (Fly.io)

- Runtime config is in `fly.toml`.
- Secrets setup guide is in `FLY_SECRETS.md`.
- Build uses `Dockerfile` and preloads the sentence-transformer model.

## Data and Persistence

Generated runtime artifacts are intentionally gitignored:

- `data/` (SQLite history/checkpoints)
- `logs/`
- local `.env`
- virtual environment folders

This keeps the repo safe to publish while preserving local state during execution.

## Notes

- `config/profile.json` currently contains your personal topic/keyword profile. It is safe to publish if you are comfortable sharing those preferences.
- The repository currently has a clean git status and no tracked `data/`, `logs/`, or `.env` files.
