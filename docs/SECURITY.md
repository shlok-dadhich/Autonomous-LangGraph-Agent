# Security — External Content as Untrusted

**Status:** Phase 4 (baseline); hardening continues Phase 7.

## 1. Principles

- All article `title/description/url` is `UNTRUSTED_SOURCE_CONTENT` — never treated as instruction.
- Secrets via env (`pydantic-settings`), never DB.
- SSRF guard, HTML sanitize, input validation on every ingress.

## 2. Controls Implemented (Phase 4)

| Control | Location | Notes |
|---------|----------|-------|
| SSRF guard (private IP block) | `app/core/security.py:is_private_url()` + `is_private_url()` check before fetch | Blocks 127/10/172.16/192.168/::1 |
| HTML sanitize | `sanitize_html()` | Strips tags before LLM prompt |
| Prompt injection framing | Writer must frame article as `UNTRUSTED_SOURCE_CONTENT` | LLM treats as evidence, not instruction |
| Input validation | `UserInteraction` action allowlist | `VALID_ACTIONS` |
| Secrets | `app/core/config.py` `SecretStr` | `require()` checks |

## 3. Email Template Safety

- `digest.html` uses Jinja2 auto-escaping; feedback URLs are first-party (`feedback_base`) not open redirects.
- `tracked_url` preserves canonical URL with UTM appended via `urlunparse` (no JS).

## 4. Future (Phase 7)

- Rate limiting per route, auth/authz, encrypted secrets at rest, audit logs
- Tests: SSRF, HTML injection, prompt injection, poisoned docs, oversized content, auth bypass
- `data/export` + `data/delete` endpoints for privacy

## 5. Checklist (Open)

- [ ] Rate limit on `POST /feedback`
- [ ] Auth on `app/api/deps.py`
- [ ] HTML sanitization test suite
- [ ] SSRF test with private IP
