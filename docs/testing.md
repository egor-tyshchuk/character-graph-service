# Tests

The README's [Tests](../README.md#tests) section has the commands and the fast local recipe. This page has the details.

```bash
make test        # = uv run pytest
make lint        # ruff check + ruff format --check
make typecheck   # mypy --strict on app and tests
```

- **Requirements.** Only Docker is needed: no OpenAI key and no network access to OpenAI. The suite starts `neo4j:5.26-community` and `redis:7-alpine` with testcontainers, once per session and lazily. The LLM is replaced by a deterministic fake ([`tests/fakes.py`](../tests/fakes.py)) that returns scripted extractions, hash-based 8-dimensional embeddings and scripted judge verdicts. `Settings` is always built explicitly, so the tests never read `.env`, and an autouse fixture removes every `Settings` variable from the environment for each test, so a shell export such as `LLM_MODEL` cannot leak in (the `TEST_*` overrides below are not `Settings` variables and keep working).
- **CI.** [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs `uv sync --locked` (a stale `uv.lock` fails the build), then ruff, the format check, mypy and pytest through `uv run --frozen` on `ubuntu-latest`, where Docker is available for testcontainers. A second job, `docker`, runs `docker compose config -q`, installs the legacy `docker-compose` 1.29.2 standalone binary and runs `docker-compose config -q`, then builds the image through the compose file with `docker compose build`.

| Directory | What it covers |
|---|---|
| `tests/unit` | No I/O: chunking, Markdown/PDF parsing (PDFs generated in the test, including embedded images, vector graphics, the pixel cap and the image-loss counters), the sample PDF (byte-identical to its generator, with a vector-only diagram page), within-article merge, profile hashing, Pydantic schemas of LLM output, the OpenAI client against a mocked transport, settings validation, cron derivation, the error contract (status codes, shape, request id on `500`), the middleware (request-id format, health log level), the layering rules (imports walked from the source), and the prompts-in-README check. |
| `tests/integration` | Real Neo4j and Redis: the repository and schema, the token service (rotation, reuse detection), the full pipeline (dedup across articles, judge band, `NOT_SAME`, judge outage, PDF image-loss counters, blob kept on infrastructure errors and retried runs), the resolve lock (concurrent articles give one node), manual merge, reconciliation, the sweep. |
| `tests/api` | The FastAPI app through `httpx` with the inline queue: auth and `401` on every protected route (the routes are enumerated from the app, so a new unprotected route fails the suite); body and field limits; uploads (JSON, multipart text, raw `text/markdown` body, `.md` file, PDF; size and type errors; `wait`; idempotent re-upload); all four read endpoints after the data is persisted; merge and reconcile endpoints; health. |
| `tests/e2e` | Opt-in, against the real LLM: extraction, cross-language dedup, the PDF image and vector-diagram relations and the type-query threshold (see below). Skipped unless `RUN_E2E=1`. |

## End-to-end tests (opt-in, real LLM)

`RUN_E2E=1 uv run pytest tests/e2e -v` runs [`tests/e2e/test_real_llm.py`](../tests/e2e/test_real_llm.py) against the real provider. It is the only test module that reads `.env`, and only with `RUN_E2E=1`. From `.env` it takes just the LLM keys: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, the models, `EMBEDDING_DIM`, the concurrency, retry and timeout settings, the dedup and relation-type thresholds, `DEDUP_CANDIDATES`, `DEDUP_MAX_JUDGE_CALLS_PER_ARTICLE` and `VECTOR_SEARCH_K`, plus the budgets a longer judge timeout must validate with: `RESOLVE_LOCK_TIMEOUT_SECONDS`, `RESOLVE_LOCK_ACQUIRE_TIMEOUT_SECONDS`, `RECONCILE_MAX_JUDGE_CALLS` and `JOB_TIMEOUT_SECONDS` (see [Slow judge models](llm-providers.md#slow-judge-models)). So it works with OpenAI as well as with the [Gemini profile](llm-providers.md#gemini-profile). Everything else is test infrastructure: its own module-scoped Neo4j and Redis containers and the inline pipeline. Its vector indexes use `EMBEDDING_DIM`, so they never collide with the 8-dimensional suite. Without `RUN_E2E=1`, or without a key in `.env`, the module is skipped, so plain `uv run pytest` reports it as skipped.

The module ingests `examples/sample_en.md`, `sample_ru.md` and `sample.pdf` once, in this order. Each file goes through `file_article` from [`app/api/uploads.py`](../app/api/uploads.py), so its format, title and content hash (normalized text fingerprint for Markdown, raw SHA-256 for the PDF) are exactly those of a production upload. It then checks on the shared graph:

- **Markdown extraction:** ≥ 3 characters, ≥ 2 relations, UPPER_SNAKE_CASE types.
- **Cross-language dedup:** one Anna node and one Oleg node, each with Latin and Cyrillic surfaces.
- **The image-only relation from the PDF:** a Mark–Lena relation with evidence from the PDF article.
- **The vector-diagram relation from the PDF:** page 3 of `sample.pdf` is a diagram drawn as vector shapes, with its labels outlined, so it has no text layer. It shows Igor Smirnov as a student of Lena Ivanova, a relation that no sample text states, and gives him the RU article's context (16, helps Anna Petrova at the bakery) so that the judge can link him to «Игорь Смирнов». The test needs at least 2 images described and an Igor–Lena relation of any type and direction with PDF evidence. The relation may attach to the Russian article's «Игорь Смирнов» node or to a new «Igor Smirnov» node; both are accepted, and a failure message lists the Igor nodes and Igor–Lena relations found.
- **The type-query threshold on real embeddings:** `sibling` and `brother` resolve to SIBLING_OF, `друг` to FRIEND_OF, and the `enemy` ~ FRIEND_OF similarity is below `RELATION_TYPE_QUERY_THRESHOLD`. When the graph has a type containing `ENEMY`, `enemy` must resolve to such a type; otherwise only the negative check (not FRIEND_OF) applies, and its failure message says so.

**Budget.** One run makes about 3 extraction calls, 2 vision calls, about 20 embedding calls and about 10 judge calls (the 0.65 band sends more pairs to the judge, and the diagram page adds Igor), plus SDK retries. On a free tier with a small daily judge quota, run it once per quota window. When the judge is out of quota, the articles still finish. The Markdown and dedup tests then fail with `judge unavailable or capped (quota or DEDUP_MAX_JUDGE_CALLS_PER_ARTICLE)`, which tells a judge shortage apart from a logic failure. The two PDF tests do not depend on judge coverage: they need the PDF article `done` and look for the relation over all matching nodes, merged or not, and their failure messages report the PDF's unjudged count.

**Measured results** (2026-10-23, current version; Gemini through the OpenAI-compatible endpoint: `gemini-3.5-flash-lite` for extraction and vision, `gemini-3.8-flash` as judge, `gemini-embedding-2` at 1536 dimensions): the module passed **9/9 in 334 s (about 5.5 minutes)**, with the [slow-judge settings](llm-providers.md#slow-judge-models) `JUDGE_TIMEOUT_SECONDS=60`, `RESOLVE_LOCK_TIMEOUT_SECONDS=2500` and `RECONCILE_MAX_JUDGE_CALLS=14`. Earlier runs on this version failed because one `gemini-3.8-flash` judge call took longer than the default 20 s `JUDGE_TIMEOUT_SECONDS`; as designed, the failed call left the rest of that article unjudged. The run covers both PDF relations: the embedded image on page 2 (Mark → Lena) and the vector diagram on page 3, which has no text layer and was read by the vision model (Igor → Lena). Earlier runs were sometimes limited by the free-tier judge quota or by HTTP 503 responses; the judge-coverage guard then names the cause instead of failing as a logic error.

## Faster local runs

On a slow Docker host, starting containers dominates the run time. You can point the suite at already-running throwaway services instead. They must be **dedicated containers on other ports**, never the compose stack:

```bash
docker run -d --rm --name cgs-test-neo4j -p 17687:7687 -e NEO4J_AUTH=neo4j/testpassword neo4j:5.26-community
docker run -d --rm --name cgs-test-redis -p 16379:6379 redis:7-alpine
until docker exec cgs-test-neo4j cypher-shell -u neo4j -p testpassword 'RETURN 1' >/dev/null 2>&1; do sleep 2; done

TEST_ALLOW_WIPE=1 TEST_NEO4J_URI=bolt://localhost:17687 TEST_REDIS_URL=redis://localhost:16379 uv run pytest
```

- **The suite deletes all data in these services.** It wipes Neo4j before each test and flushes the Redis DB; when the URL names no DB, `/15` is used. It also creates 8-dimensional vector indexes. Pointed at the compose Neo4j, it would delete your data, and the api would then refuse to start with `SchemaMismatchError` (1536 expected).
- **The wipe guard.** The suite refuses to run when `TEST_NEO4J_URI` or `TEST_REDIS_URL` is set without `TEST_ALLOW_WIPE=1`.
- **Other overrides.** `TEST_NEO4J_USER` and `TEST_NEO4J_PASSWORD` default to `neo4j` and `testpassword`.

**Ryuk.** testcontainers starts its Ryuk reaper container to clean up after crashed runs. On some Docker Desktop versions Ryuk can't connect and only adds about 40 s of startup. There you can set `TESTCONTAINERS_RYUK_DISABLED=true`: the suite's own session teardown removes its containers anyway. CI keeps Ryuk enabled.
