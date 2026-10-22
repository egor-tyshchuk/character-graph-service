# Configuration

Every setting is an environment variable, read by `Settings` in [`app/config.py`](../app/config.py). [`.env.example`](../.env.example) documents the same list with one-line comments. [`docker-compose.yml`](../docker-compose.yml) passes every variable to `api` and `worker`, with the defaults below. The README's [Configuration](../README.md#configuration) section lists the settings you are most likely to change.

The dedup thresholds are explained in the README's [AI pipeline](../README.md#ai-pipeline); for providers other than OpenAI, see [Choosing an LLM provider and models](llm-providers.md).

## Auth

| Variable | Default | Meaning |
|---|---|---|
| `JWT_SECRET` | `dev-insecure-jwt-secret-change-me-please` (compose only; required by `Settings`) | HMAC secret for access and refresh JWTs, at least 32 characters. Set a long random value outside local demos. |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm: `HS256`, `HS384` or `HS512` |
| `ACCESS_TOKEN_TTL_SECONDS` | `900` | Access token lifetime |
| `REFRESH_TOKEN_TTL_SECONDS` | `86400` | Refresh token lifetime |
| `REFRESH_FAMILY_MAX_LIFETIME_SECONDS` | `604800` | Absolute lifetime of a refresh-token family (7 days); after it, refresh fails and the app token must be exchanged again |
| `APP_TOKENS` | `demo:05c7cd5c…25767` (compose only, accepts `demo-app-token`) | Comma-separated `app_id:sha256(app_token)` pairs. Dev only, replace outside local demos. |

See [Getting tokens](../README.md#getting-tokens) for how to generate an app token. While `JWT_SECRET` is the compose demo secret, or `APP_TOKENS` accepts the demo token `demo-app-token`, the api and the worker log a structured `insecure_demo_secret` warning at startup naming the setting (and, for `APP_TOKENS`, the app ids). The demo values are defined once, in [`app/auth/demo.py`](../app/auth/demo.py), and a unit test checks that they match `docker-compose.yml` and `.env.example`.

## Infrastructure

| Variable | Default | Meaning |
|---|---|---|
| `NEO4J_URI` | `bolt://neo4j:7687` | Neo4j Bolt URI (fixed by compose) |
| `NEO4J_USER` | `neo4j` | Neo4j user (fixed to `neo4j` by compose: the `neo4j` container only creates that user) |
| `NEO4J_PASSWORD` | `password` | Neo4j password. It also initialises the `neo4j` container on first start; after a change, run `docker-compose down -v`. |
| `REDIS_URL` | `redis://redis:6379/0` | Queue, locks and refresh-token state (fixed by compose) |
| `JOB_QUEUE` | `arq` | `arq` (worker) or `inline` (process in the API request; used by tests). Fixed to `arq` by compose. |
| `UPLOAD_DIR` | `/data/uploads` | Shared volume for raw uploads (fixed by compose) |
| `MAX_UPLOAD_BYTES` | `20971520` | Maximum article size (20 MiB) |

## LLM

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | empty | Required for extraction; the stack starts without it |
| `OPENAI_BASE_URL` | empty (official endpoint) | Base URL of any OpenAI-compatible API |
| `LLM_MODEL` | `gpt-4.1-mini` | Chat model for extraction (and the judge unless `JUDGE_MODEL` is set), with Structured Outputs |
| `VISION_MODEL` | `gpt-4.1-mini` | Vision model for PDF images and rendered pages |
| `JUDGE_MODEL` | empty | Chat model for the dedup judge only; empty means `LLM_MODEL` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for characters and relation types |
| `EMBEDDING_DIM` | `1536` | Embedding dimension; defines the vector indexes. Changing it needs a fresh database. |
| `LLM_MAX_CONCURRENCY` | `4` | Concurrent LLM requests per process |
| `LLM_TIMEOUT_SECONDS` | `60` | Timeout of one LLM request |
| `LLM_MAX_RETRIES` | `3` | Retries inside the OpenAI SDK (with backoff) |

## Ingestion

| Variable | Default | Meaning |
|---|---|---|
| `CHUNK_MAX_CHARS` | `6000` | Maximum characters per extraction chunk |
| `CHUNK_OVERLAP_CHARS` | `500` | Overlap between neighbouring chunks |
| `MAX_CHUNKS_PER_ARTICLE` | `200` | Articles with more chunks fail before any extraction call |
| `PDF_MAX_IMAGES` | `20` | Maximum images described per PDF |
| `PDF_MIN_IMAGE_SIDE` | `100` | Embedded images whose shorter side is below this many pixels are skipped |
| `PDF_SCANNED_PAGE_MIN_CHARS` | `50` | Pages with less text are treated as scans and rendered whole, if they have visual content |
| `PDF_RENDER_DPI` | `150` | DPI for whole-page rendering |
| `PDF_MAX_RENDER_PIXELS` | `25000000` | Maximum pixels of a rendered page or embedded image; larger pages render at a lower DPI (at least 36, else skipped), larger images are skipped |
| `VISION_MODE` | `auto` | `auto` renders scanned or graphics-heavy pages; `all_pages` renders every non-blank page |
| `PDF_DRAWINGS_MIN_COUNT` | `20` | Significant vector drawings needed for a page to count as graphical |
| `PDF_DRAWINGS_MIN_AREA_RATIO` | `0.12` | Share of the page those drawings must cover |

## Dedup and locking

| Variable | Default | Meaning |
|---|---|---|
| `DEDUP_CANDIDATES` | `5` | Nearest existing characters considered |
| `DEDUP_AUTO_MERGE_THRESHOLD` | `0.95` | Cosine at or above which a candidate is merged without the judge |
| `DEDUP_JUDGE_THRESHOLD` | `0.65` | Cosine at or above which the LLM judge decides; must not exceed the auto-merge threshold |
| `DEDUP_MAX_JUDGE_CALLS_PER_ARTICLE` | `10` | Judge calls per article |
| `RELATION_TYPE_QUERY_THRESHOLD` | `0.7` | Minimum cosine for a free-form `?type=` to resolve to a known type |
| `RESOLVE_LOCK_TIMEOUT_SECONDS` | `900` | Resolve lock TTL; must exceed judge calls × LLM timeout |
| `RESOLVE_LOCK_ACQUIRE_TIMEOUT_SECONDS` | `600` | How long article processing waits for the lock |
| `MERGE_LOCK_ACQUIRE_TIMEOUT_SECONDS` | `10` | How long `POST /characters/merge` waits before `409` |

## Background jobs

| Variable | Default | Meaning |
|---|---|---|
| `RECONCILE_ENABLED` | `false` | Run periodic reconciliation in the worker, see [Reconciliation](robustness.md#reconciliation-opt-in) |
| `RECONCILE_INTERVAL_MINUTES` | `60` | Must divide 60, or be whole hours dividing 24 |
| `RECONCILE_BATCH_SIZE` | `200` | Characters per page of a run |
| `RECONCILE_MAX_JUDGE_CALLS` | `50` | Judge calls per run |
| `STUCK_SWEEP_ENABLED` | `true` | Periodically fail articles stuck in `pending`/`processing` |
| `STUCK_SWEEP_INTERVAL_MINUTES` | `10` | Same grid rule as above |
| `STUCK_ARTICLE_AFTER_SECONDS` | `4200` | Age after which an unfinished article counts as stuck; must exceed `JOB_TIMEOUT_SECONDS` |
| `JOB_TIMEOUT_SECONDS` | `3600` | ARQ job timeout of the worker; must exceed `RESOLVE_LOCK_ACQUIRE_TIMEOUT_SECONDS + RESOLVE_LOCK_TIMEOUT_SECONDS` |
| `JOB_MAX_TRIES` | `3` | Attempts per article job; only Neo4j/Redis outages are retried, see [Retries on infrastructure errors](robustness.md#retries-on-infrastructure-errors) |
| `JOB_RETRY_DELAY_SECONDS` | `30` | Delay before an article job is retried after a Neo4j/Redis outage; must be less than `STUCK_ARTICLE_AFTER_SECONDS - JOB_TIMEOUT_SECONDS` |

## Logging

| Variable | Default | Meaning |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Log level |
| `LOG_JSON` | `true` | JSON logs (structlog), or human-readable console output |

Logs carry a request id and the article/job id. An incoming `X-Request-ID` is echoed only if it matches `[A-Za-z0-9._-]{1,128}`; anything else is replaced by a generated id, so a client cannot inject arbitrary text into logs. Requests to `/health/*` are logged at `debug`, so orchestrator probes do not flood the `INFO` log.

## docker-compose only

| Variable | Default | Meaning |
|---|---|---|
| `API_PORT` | `8000` | Host port of the API |
| `NEO4J_HTTP_PORT` | `7474` | Host port of the Neo4j Browser, bound to `127.0.0.1` |
| `NEO4J_BOLT_PORT` | `7687` | Host port of Neo4j Bolt, bound to `127.0.0.1` |
| `NEO4J_HEAP_MAX_SIZE` | `512m` | Neo4j JVM heap limit |
| `NEO4J_PAGECACHE_SIZE` | `256m` | Neo4j page cache size |

Two compose behaviours are worth knowing:

- Neo4j applies `NEO4J_PASSWORD` only when it first initialises its volume. To change the password later, run `docker-compose down -v` (this deletes the data).
- The legacy `docker-compose` 1.29 rejects `--env-file /dev/null`. To run without `.env`, point `--env-file` at an empty file instead.
