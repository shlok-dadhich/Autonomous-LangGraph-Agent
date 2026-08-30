# Source Catalog

**Status:** Phase 1 — 6 legacy sources wrapped behind `SourceConnector`; P1/P2 sources stubbed for expansion.
**Config:** `config/sources.yaml` is source of truth for enabled/adapter/rate limits/trusted domains.

## 1. Protocol

```python
# app/connectors/base.py
@dataclass class SourceQuery: topics, keywords, max_results, extra
@dataclass class RawDocument: title, url, description, source, published_at, author, external_id, metadata
class SourceConnector(Protocol):
    async def fetch(self, query: SourceQuery) -> list[RawDocument]: ...
```

Registry: `app/connectors/__init__.py` — `register("arxiv")(ArxivConnector)` and `get_connector(name)`.

All adapters return `RawDocument` and preserve `published_at/fetched_at`; business code never imports `tavily`/`requests` directly.

## 2. P0 — Active (wrapped legacy clients)

| Name | Adapter Class | Legacy Client | Registers As | Notes |
|------|---------------|---------------|--------------|-------|
| arxiv | `ArxivConnector` | `src/tools/arxiv_client.fetch_arxiv_papers` | `arxiv` | cs.AI/cs.LG, 7d lookback |
| tavily/news | `TavilyConnector` | `src/tools/tavily_client.fetch_tavily_results` | `tavily`, `news` | Tavily basic→advanced fallback |
| hackernews | `HackerNewsConnector` | `src/tools/hn_client.fetch_hn_stories` | `hackernews` | score≥50, threaded |
| huggingface | `HFConnector` | `src/tools/hf_client.fetch_hf_daily_papers` | `huggingface` | Daily Papers API |
| rss | `RSSConnector` | `src/tools/rss_client.fetch_rss_sources` | `rss` | feed_specs from `extra` |
| reddit/social | `RedditConnector` | `src/tools/social_signal_client.fetch_social_signals` | `reddit`, `social_signals` | via Tavily domains |

Adapters are idempotent and independently disableable via `sources.yaml`.

## 3. P0 Stubs — Ready for Phase 5

| Name | Status |
|------|--------|
| openalex | stub — returns `[]`, Phase 5 will call OpenAlex API |
| semantic_scholar | stub — Phase 5 |
| crossref | stub — Phase 5 |
| github | stub — releases/trending/velocity, Phase 5 |
| regulation | stub — NIST/EU/CISA/CVE, Phase 6 |

## 4. Configuration

```yaml
# config/sources.yaml (excerpt)
sources:
  arxiv: {enabled: true, categories: [cs.AI, cs.LG], days_back: 7, max_results: 20}
  tavily: {enabled: true, max_results: 10}
  rss: {enabled: true, feeds: [anthropic_news, huggingface_blog]}
```

## 5. Source Quality Tiers (Phase 2 will score)

- **A** primary (papers, official docs, benchmarks)
- **B** reputable journalism
- **C** community (GH/HN/Reddit)
- **D** unknown/SEO/slop

`Source.tier` and `Source.reliability_score` track health; `SourceConnector` does not decide quality — intelligence layer does.

## 6. Migration Note

Legacy `src/tools/*` remain operational; no behavior change in Phase 1. Phase 2 will route `app/graph/nodes/acquisition.py` through the registry instead of direct `src.graph.nodes` calls.
