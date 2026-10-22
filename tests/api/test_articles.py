import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
import structlog
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from structlog.testing import capture_logs

from app.config import Settings
from app.container import Container
from app.llm.client import LLMClient
from app.llm.schemas import ExtractedCharacter, ExtractedRelation, ExtractionResult
from app.main import create_app
from app.queue import JobQueue
from tests.conftest import TEST_APP_TOKEN
from tests.fakes import FakeLLMClient
from tests.pdf_factory import PageSpec, make_pdf, make_png, set_title

LIGHTHOUSE = "# The Lighthouse\n\nAnna Petrova and Mark Petrov kept the lighthouse together.\n"
PDF_TEXT = "Anna Petrova and Mark Petrov kept the lighthouse together."
CHUNK = b"x" * 64 * 1024
MARKDOWN_BODY = {"Content-Type": "text/markdown; charset=utf-8"}
CHUNK_COUNT = 32


class NeverProcessingQueue:
    def __init__(self) -> None:
        self.enqueued: list[str] = []

    async def enqueue_article(self, article_id: str) -> None:
        self.enqueued.append(article_id)

    async def enqueue_reconcile(self) -> bool:
        return True


class ChunkedBody:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.sent = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            self.sent += 1
            yield chunk


class BrokenQueue:
    async def enqueue_article(self, article_id: str) -> None:
        raise ConnectionError("redis is down")

    async def enqueue_reconcile(self) -> bool:
        return True


def siblings() -> ExtractionResult:
    return ExtractionResult(
        characters=[
            ExtractedCharacter(name="Anna Petrova", aliases=[], description="Lighthouse keeper."),
            ExtractedCharacter(name="Mark Petrov", aliases=[], description="Her brother."),
        ],
        relations=[
            ExtractedRelation(
                source="Anna Petrova",
                target="Mark Petrov",
                type="SIBLING_OF",
                description="Anna is Mark's sister.",
                evidence="Anna Petrova and Mark Petrov kept the lighthouse together.",
            )
        ],
    )


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient(dim=8, extractions={"Anna Petrova": siblings()})


def container_of(app: FastAPI) -> Container:
    container: Container = app.state.container
    return container


@asynccontextmanager
async def serve(
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    queue: JobQueue | None = None,
) -> AsyncIterator[tuple[httpx.AsyncClient, FastAPI]]:
    application = create_app(settings, llm=llm, queue=queue)
    async with LifespanManager(application):
        transport = httpx.ASGITransport(app=application, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            token = await c.post("/auth/token", json={"app_token": TEST_APP_TOKEN})
            token.raise_for_status()
            c.headers["Authorization"] = f"Bearer {token.json()['access_token']}"
            yield c, application


Upload = Callable[..., Awaitable[httpx.Response]]


@pytest.fixture
def upload(client: httpx.AsyncClient, auth_headers: dict[str, str]) -> Upload:
    async def send(*, headers: dict[str, str] | None = None, **kwargs: Any) -> httpx.Response:
        return await client.post(
            "/articles/upload", headers={**auth_headers, **(headers or {})}, **kwargs
        )

    return send


async def fetch(client: httpx.AsyncClient, article_id: str, **kwargs: Any) -> dict[str, Any]:
    response = await client.get(f"/articles/{article_id}", **kwargs)
    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    return body


def uploaded_files(settings: Settings) -> list[str]:
    if not settings.upload_dir.exists():
        return []
    return sorted(path.name for path in settings.upload_dir.iterdir())


@pytest.mark.parametrize(
    ("method", "path"), [("POST", "/articles/upload"), ("GET", "/articles/some-id")]
)
async def test_article_endpoints_require_auth(
    client: httpx.AsyncClient, method: str, path: str
) -> None:
    response = await client.request(method, path)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


async def test_upload_markdown_text_processes_and_persists(
    app: FastAPI,
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    upload: Upload,
    settings: Settings,
) -> None:
    response = await upload(data={"text": LIGHTHOUSE})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] in ("pending", "done")
    article = await fetch(client, body["id"], headers=auth_headers)
    assert article["status"] == "done"
    assert article["title"] == "The Lighthouse"
    assert article["format"] == "markdown"
    assert article["stats"] == {
        "characters": 2,
        "relations": 1,
        "images_described": 0,
        "images_failed": 0,
        "images_skipped": 0,
        "unjudged_characters": 0,
    }
    records, _, _ = await container_of(app).driver.execute_query(
        "MATCH (c:Character)-[:APPEARS_IN]->(:Article {id: $id}) RETURN c.name AS name "
        "ORDER BY name",
        id=body["id"],
    )
    assert [record["name"] for record in records] == ["Anna Petrova", "Mark Petrov"]
    assert uploaded_files(settings) == []


async def test_upload_multipart_text_field(
    client: httpx.AsyncClient, auth_headers: dict[str, str], upload: Upload
) -> None:
    response = await upload(files={"text": (None, LIGHTHOUSE), "title": (None, "Keepers")})

    assert response.status_code == 202
    assert response.json()["title"] == "Keepers"
    article = await fetch(client, response.json()["id"], headers=auth_headers)
    assert article["status"] == "done"
    assert article["stats"]["characters"] == 2


async def test_upload_json_text(
    client: httpx.AsyncClient, auth_headers: dict[str, str], upload: Upload
) -> None:
    response = await upload(json={"text": LIGHTHOUSE, "title": "Keepers"})

    assert response.status_code == 202
    article = await fetch(client, response.json()["id"], headers=auth_headers)
    assert article["status"] == "done"
    assert article["title"] == "Keepers"
    assert article["stats"]["characters"] == 2


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"title": "Keepers"}, id="missing-text"),
        pytest.param({"text": 42}, id="non-string-text"),
        pytest.param(["not", "an", "object"], id="not-an-object"),
    ],
)
async def test_upload_json_without_text_422(upload: Upload, payload: Any) -> None:
    response = await upload(json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_upload_invalid_json_422(upload: Upload) -> None:
    response = await upload(content=b"{not json", headers={"Content-Type": "application/json"})

    assert response.status_code == 422


async def test_upload_markdown_file(
    client: httpx.AsyncClient, auth_headers: dict[str, str], upload: Upload
) -> None:
    response = await upload(files={"file": ("tale.md", LIGHTHOUSE.encode(), "text/markdown")})

    assert response.status_code == 202
    article = await fetch(client, response.json()["id"], headers=auth_headers)
    assert article["status"] == "done"
    assert article["title"] == "The Lighthouse"
    assert article["format"] == "markdown"
    assert article["stats"]["characters"] == 2


async def test_upload_raw_markdown_body(
    client: httpx.AsyncClient, auth_headers: dict[str, str], upload: Upload
) -> None:
    response = await upload(content=LIGHTHOUSE.encode(), headers=MARKDOWN_BODY)

    assert response.status_code == 202
    article = await fetch(client, response.json()["id"], headers=auth_headers)
    assert article["status"] == "done"
    assert article["title"] == "The Lighthouse"
    assert article["format"] == "markdown"
    assert article["stats"]["characters"] == 2


async def test_upload_raw_plain_text_body_with_title_query(upload: Upload) -> None:
    response = await upload(
        content=b"Anna Petrova met Mark Petrov.",
        headers={"Content-Type": "text/plain"},
        params={"title": "  Keepers  "},
    )

    assert response.status_code == 202
    assert response.json()["title"] == "Keepers"
    assert response.json()["format"] == "markdown"


async def test_raw_markdown_body_and_json_text_are_duplicates(
    upload: Upload, fake_llm: FakeLLMClient
) -> None:
    first = await upload(json={"text": LIGHTHOUSE})
    calls = len(fake_llm.extract_calls)

    second = await upload(
        content=LIGHTHOUSE.replace("\n", "\r\n").encode("utf-8-sig"), headers=MARKDOWN_BODY
    )

    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert len(fake_llm.extract_calls) == calls


async def test_raw_markdown_body_must_be_utf8(upload: Upload, settings: Settings) -> None:
    response = await upload(content=b"# Title\n\n\xff\xfe broken", headers=MARKDOWN_BODY)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert uploaded_files(settings) == []


@pytest.mark.parametrize(
    ("kwargs", "expected_title"),
    [
        pytest.param(
            {"data": {"text": LIGHTHOUSE, "title": "  Keepers  "}}, "Keepers", id="form-title"
        ),
        pytest.param(
            {"data": {"text": "Anna Petrova met Mark Petrov.", "title": "  "}},
            "Untitled",
            id="blank-title-ignored",
        ),
        pytest.param(
            {"files": {"file": ("keepers-log.txt", b"Anna Petrova met Mark Petrov.")}},
            "keepers-log",
            id="filename-stem",
        ),
        pytest.param({"json": {"text": "Anna Petrova met Mark Petrov."}}, "Untitled", id="none"),
    ],
)
async def test_title_precedence(
    upload: Upload, kwargs: dict[str, Any], expected_title: str
) -> None:
    response = await upload(**kwargs)

    assert response.status_code == 202
    assert response.json()["title"] == expected_title


async def test_upload_pdf_file_with_image(
    client: httpx.AsyncClient, auth_headers: dict[str, str], upload: Upload
) -> None:
    pdf = make_pdf([PageSpec(text=PDF_TEXT, images=[make_png(300, 300, "siblings")])])

    response = await upload(files={"file": ("lighthouse.pdf", pdf, "application/pdf")})

    assert response.status_code == 202
    article = await fetch(client, response.json()["id"], headers=auth_headers)
    assert article["status"] == "done"
    assert article["format"] == "pdf"
    assert article["title"] == "lighthouse"
    assert article["stats"] == {
        "characters": 2,
        "relations": 1,
        "images_described": 1,
        "images_failed": 0,
        "images_skipped": 0,
        "unjudged_characters": 0,
    }


async def test_pdf_detected_by_content_and_titled_from_metadata(upload: Upload) -> None:
    pdf = set_title(make_pdf([PageSpec(text=PDF_TEXT)]), "Keeper Diaries")

    response = await upload(files={"file": ("notes.md", pdf, "text/markdown")})

    assert response.status_code == 202
    assert response.json()["format"] == "pdf"
    assert response.json()["title"] == "Keeper Diaries"


@pytest.mark.parametrize("title", [None, "Explicit title"])
async def test_unreadable_pdf_rejected(
    upload: Upload, settings: Settings, title: str | None
) -> None:
    response = await upload(
        files={"file": ("broken.pdf", b"%PDF-1.7 garbage", "application/pdf")},
        data={"title": title} if title else None,
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert uploaded_files(settings) == []


async def test_reupload_same_content_returns_existing(
    upload: Upload, fake_llm: FakeLLMClient
) -> None:
    first = await upload(data={"text": LIGHTHOUSE})
    calls = len(fake_llm.extract_calls)

    second = await upload(data={"text": LIGHTHOUSE})

    assert first.status_code == 202
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert second.json()["status"] == "done"
    assert len(fake_llm.extract_calls) == calls


async def test_reupload_pdf_same_bytes_returns_existing(upload: Upload) -> None:
    pdf = make_pdf([PageSpec(text=PDF_TEXT)])
    first = await upload(files={"file": ("a.pdf", pdf)})

    second = await upload(files={"file": ("b.pdf", pdf)})

    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]


async def test_reupload_after_failure_requeues(
    settings: Settings, fake_llm: FakeLLMClient, clean_stores: None
) -> None:
    async with serve(settings) as (unconfigured, _):
        failed = await unconfigured.post("/articles/upload", data={"text": LIGHTHOUSE})
        assert failed.status_code == 202
        article_id = failed.json()["id"]
        assert (await fetch(unconfigured, article_id))["status"] == "failed"

    async with serve(settings, llm=fake_llm) as (configured, _):
        retried = await configured.post("/articles/upload", data={"text": LIGHTHOUSE})

        assert retried.status_code == 202
        assert retried.json()["id"] == article_id
        article = await fetch(configured, article_id)
        assert article["status"] == "done"
        assert article["error"] is None
        assert article["stats"]["characters"] == 2
    assert uploaded_files(settings) == []


async def test_enqueue_failure_marks_article_failed(
    settings: Settings, fake_llm: FakeLLMClient, clean_stores: None
) -> None:
    async with serve(settings, llm=fake_llm, queue=BrokenQueue()) as (broken, app):
        response = await broken.post("/articles/upload", data={"text": LIGHTHOUSE})
        assert response.status_code == 500
        records, _, _ = await container_of(app).driver.execute_query(
            "MATCH (a:Article) RETURN a.id AS id, a.status AS status, a.error AS error"
        )
        assert len(records) == 1
        assert records[0]["status"] == "failed"
        assert "ConnectionError" in records[0]["error"]
        assert uploaded_files(settings) == []

    async with serve(settings, llm=fake_llm) as (healthy, _):
        retried = await healthy.post("/articles/upload", data={"text": LIGHTHOUSE})

        assert retried.status_code == 202
        assert retried.json()["id"] == records[0]["id"]
        assert (await fetch(healthy, records[0]["id"]))["status"] == "done"


async def test_duplicate_race_returns_existing_and_deletes_new_blob(
    app: FastAPI, upload: Upload, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = await upload(data={"text": LIGHTHOUSE})
    repo = container_of(app).repo
    lookup = repo.get_article_by_hash
    lookups: list[str] = []

    async def racing_lookup(content_hash: str) -> Any:
        lookups.append(content_hash)
        return None if len(lookups) == 1 else await lookup(content_hash)

    monkeypatch.setattr(repo, "get_article_by_hash", racing_lookup)

    second = await upload(data={"text": LIGHTHOUSE})

    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert len(lookups) == 2
    assert uploaded_files(settings) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"data": {"title": "Only a title"}}, id="neither"),
        pytest.param(
            {"data": {"text": LIGHTHOUSE}, "files": {"file": ("a.md", LIGHTHOUSE.encode())}},
            id="both",
        ),
        pytest.param({"data": {"file": "not a file"}}, id="file-as-plain-field"),
    ],
)
async def test_upload_rejects_both_and_neither(upload: Upload, kwargs: dict[str, Any]) -> None:
    response = await upload(**kwargs)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"data": {"text": ""}}, id="empty-text"),
        pytest.param({"data": {"text": " \r\n\t\n"}}, id="whitespace-text"),
        pytest.param({"json": {"text": "\ufeff  "}}, id="json-bom-only"),
        pytest.param({"files": {"file": ("empty.md", b"")}}, id="empty-file"),
        pytest.param({"content": b"", "headers": MARKDOWN_BODY}, id="empty-markdown-body"),
        pytest.param(
            {"content": b"\xef\xbb\xbf \r\n", "headers": MARKDOWN_BODY}, id="blank-markdown-body"
        ),
    ],
)
async def test_upload_empty_content_422(upload: Upload, kwargs: dict[str, Any]) -> None:
    response = await upload(**kwargs)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"files": {"file": ("a.md", b"x" * 11)}}, id="file"),
        pytest.param({"files": {"file": ("a.pdf", b"%PDF-" + b"x" * 6)}}, id="pdf"),
        pytest.param({"data": {"text": "x" * 11}}, id="form-text"),
        pytest.param({"data": {"text": "é" * 6}}, id="form-text-utf8-bytes"),
        pytest.param({"json": {"text": "x" * 11}}, id="json-text"),
        pytest.param({"files": {"file": ("big.md", b"x" * 200_000)}}, id="multipart-body"),
        pytest.param({"json": {"text": "x" * 200_000}}, id="json-body"),
        pytest.param({"content": b"x" * 11, "headers": MARKDOWN_BODY}, id="markdown-body"),
        pytest.param(
            {"content": "é".encode() * 6, "headers": {"Content-Type": "text/plain"}},
            id="plain-body-utf8-bytes",
        ),
        pytest.param(
            {"content": b"x" * 200_000, "headers": MARKDOWN_BODY}, id="markdown-body-over-allowance"
        ),
    ],
)
async def test_upload_too_large(
    settings: Settings, fake_llm: FakeLLMClient, clean_stores: None, kwargs: dict[str, Any]
) -> None:
    limited = settings.model_copy(update={"max_upload_bytes": 10})
    async with serve(limited, llm=fake_llm) as (c, _):
        response = await c.post("/articles/upload", **kwargs)

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    assert fake_llm.extract_calls == []


async def test_upload_at_size_limit_accepted(
    settings: Settings, fake_llm: FakeLLMClient, clean_stores: None
) -> None:
    limited = settings.model_copy(update={"max_upload_bytes": 10})
    async with serve(limited, llm=fake_llm) as (c, _):
        response = await c.post("/articles/upload", files={"file": ("a.md", b"x" * 10)})

    assert response.status_code == 202


async def test_upload_binary_non_pdf_unsupported(upload: Upload, settings: Settings) -> None:
    png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff\xfe"

    response = await upload(files={"file": ("image.md", png, "text/markdown")})

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"
    assert uploaded_files(settings) == []


async def test_upload_unsupported_content_type(upload: Upload) -> None:
    response = await upload(
        content=LIGHTHOUSE.encode(), headers={"Content-Type": "application/xml"}
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


async def test_get_article_unknown_404(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get("/articles/does-not-exist", headers=auth_headers)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_upload_wait_returns_finished_article(upload: Upload) -> None:
    response = await upload(json={"text": LIGHTHOUSE}, params={"wait": "true"})

    assert response.status_code == 200
    assert response.json()["status"] == "done"
    assert response.json()["stats"]["characters"] == 2


async def test_upload_wait_timeout_returns_202(
    settings: Settings, fake_llm: FakeLLMClient, clean_stores: None
) -> None:
    queue = NeverProcessingQueue()
    async with serve(settings, llm=fake_llm, queue=queue) as (c, _):
        started = time.monotonic()
        response = await c.post(
            "/articles/upload",
            json={"text": LIGHTHOUSE},
            params={"wait": "true", "wait_timeout": 1},
        )
        elapsed = time.monotonic() - started

    assert response.status_code == 202
    assert response.json()["status"] == "pending"
    assert queue.enqueued == [response.json()["id"]]
    assert 1 <= elapsed < 3


async def test_upload_wait_returns_failed_article(settings: Settings, clean_stores: None) -> None:
    async with serve(settings) as (c, _):
        response = await c.post(
            "/articles/upload", data={"text": LIGHTHOUSE}, params={"wait": "true"}
        )

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert "LLMNotConfiguredError" in response.json()["error"]


@pytest.mark.parametrize("wait_timeout", [0, 121])
async def test_upload_wait_timeout_bounds(upload: Upload, wait_timeout: int) -> None:
    response = await upload(
        json={"text": LIGHTHOUSE}, params={"wait": "true", "wait_timeout": wait_timeout}
    )

    assert response.status_code == 422


async def test_reupload_with_crlf_and_extra_spaces_is_duplicate(
    upload: Upload, fake_llm: FakeLLMClient
) -> None:
    first = await upload(data={"text": LIGHTHOUSE})
    calls = len(fake_llm.extract_calls)
    noisy = "\ufeff# The  Lighthouse  \r\n\r\n\r\n\r\nAnna Petrova and\tMark Petrov kept the "
    noisy += "lighthouse together.   \r\n\r\n"

    second = await upload(data={"text": noisy})

    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert len(fake_llm.extract_calls) == calls


async def test_json_text_and_md_file_with_same_text_are_duplicates(
    upload: Upload, fake_llm: FakeLLMClient
) -> None:
    first = await upload(json={"text": LIGHTHOUSE})
    calls = len(fake_llm.extract_calls)

    second = await upload(
        files={"file": ("tale.md", LIGHTHOUSE.replace("\n", "\r\n").encode("utf-8-sig"))}
    )

    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert len(fake_llm.extract_calls) == calls


async def test_meaningful_text_change_is_new_article(upload: Upload) -> None:
    first = await upload(data={"text": LIGHTHOUSE})

    second = await upload(data={"text": LIGHTHOUSE.replace("together", "alone")})

    assert second.status_code == 202
    assert second.json()["id"] != first.json()["id"]


@pytest.mark.parametrize(
    ("content_type", "chunks"),
    [
        pytest.param(
            "multipart/form-data; boundary=BOUND",
            [
                b'--BOUND\r\nContent-Disposition: form-data; name="file"; filename="a.md"\r\n\r\n',
                *[CHUNK] * CHUNK_COUNT,
                b"\r\n--BOUND--\r\n",
            ],
            id="multipart-file",
        ),
        pytest.param(
            "application/x-www-form-urlencoded",
            [b"pad%d=%s&" % (index, CHUNK) for index in range(CHUNK_COUNT)] + [b"text=hi"],
            id="urlencoded-many-fields",
        ),
        pytest.param(
            "application/x-www-form-urlencoded",
            [b"text=", *[CHUNK] * CHUNK_COUNT],
            id="urlencoded-one-field",
        ),
        pytest.param(
            "application/json",
            [b'{"text": "', *[CHUNK] * CHUNK_COUNT, b'"}'],
            id="json",
        ),
        pytest.param("text/markdown", [CHUNK] * CHUNK_COUNT, id="markdown"),
    ],
)
async def test_chunked_upload_over_limit_stops_reading(
    settings: Settings,
    fake_llm: FakeLLMClient,
    clean_stores: None,
    content_type: str,
    chunks: list[bytes],
) -> None:
    body = ChunkedBody(chunks)
    limited = settings.model_copy(update={"max_upload_bytes": 10})
    async with serve(limited, llm=fake_llm) as (c, _):
        response = await c.post(
            "/articles/upload", content=body, headers={"Content-Type": content_type}
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    assert body.sent <= 3 < len(chunks)
    assert uploaded_files(limited) == []


async def test_upload_wait_on_unfinished_duplicate_returns_202(
    settings: Settings, fake_llm: FakeLLMClient, clean_stores: None
) -> None:
    queue = NeverProcessingQueue()
    async with serve(settings, llm=fake_llm, queue=queue) as (c, _):
        first = await c.post("/articles/upload", json={"text": LIGHTHOUSE})
        second = await c.post(
            "/articles/upload",
            json={"text": LIGHTHOUSE},
            params={"wait": "true", "wait_timeout": 1},
        )

    assert first.status_code == 202
    assert second.status_code == 202
    assert second.json()["id"] == first.json()["id"]
    assert second.json()["status"] == "pending"
    assert queue.enqueued == [first.json()["id"]]


async def test_upload_wait_on_finished_duplicate_returns_200(upload: Upload) -> None:
    first = await upload(json={"text": LIGHTHOUSE})

    second = await upload(json={"text": LIGHTHOUSE}, params={"wait": "true"})

    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert second.json()["status"] == "done"


async def test_failed_compensation_keeps_original_error_and_removes_blob(
    settings: Settings,
    fake_llm: FakeLLMClient,
    clean_stores: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unavailable(article_id: str, error: str) -> bool:
        raise RuntimeError("neo4j is down")

    async with serve(settings, llm=fake_llm, queue=BrokenQueue()) as (c, app):
        monkeypatch.setattr(container_of(app).repo, "mark_article_failed", unavailable)
        with capture_logs(processors=[structlog.processors.format_exc_info]) as logs:
            response = await c.post("/articles/upload", data={"text": LIGHTHOUSE})

    assert response.status_code == 500
    surfaced = next(entry for entry in logs if entry["event"] == "unhandled_exception")
    assert surfaced["exception"].rstrip().endswith("ConnectionError: redis is down")
    assert "article_mark_failed_failed" in {entry["event"] for entry in logs}
    assert uploaded_files(settings) == []
