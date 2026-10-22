# API walk-through

A full curl session against every endpoint, with the status codes and response shapes. The README has the short version in [curl examples for every endpoint](../README.md#curl-examples-for-every-endpoint) and the [endpoint table](../README.md#endpoints); tokens are explained in [Getting tokens](../README.md#getting-tokens).

The examples use `jq` to capture values, and they run from the repository root so the `examples/` files resolve. The responses shown are abridged. Ids are shown as `<…>`, and model-generated text is shown as `…`, because it depends on the model.

```bash
export API=http://localhost:8000
```

## Authentication

```bash
# exchange the app token for a pair
TOKENS=$(curl -s -X POST "$API/auth/token" \
  -H 'Content-Type: application/json' -d '{"app_token": "demo-app-token"}')
echo "$TOKENS" | jq
# {"access_token": "<jwt>", "refresh_token": "<jwt>", "token_type": "bearer", "expires_in": 900}

ACCESS=$(echo "$TOKENS" | jq -r .access_token)
REFRESH=$(echo "$TOKENS" | jq -r .refresh_token)
AUTH="Authorization: Bearer $ACCESS"

# rotate the refresh token: a new pair; the old refresh token is now spent
NEW_TOKENS=$(curl -s -X POST "$API/auth/refresh" \
  -H 'Content-Type: application/json' -d "{\"refresh_token\": \"$REFRESH\"}")
echo "$NEW_TOKENS" | jq 'keys'
# ["access_token", "expires_in", "refresh_token", "token_type"]

# presenting the spent refresh token again is reuse: 401, and the whole family is revoked
curl -s -X POST "$API/auth/refresh" \
  -H 'Content-Type: application/json' -d "{\"refresh_token\": \"$REFRESH\"}"
# {"error":{"code":"unauthorized","message":"Refresh token revoked or expired"}}

# a protected endpoint without a token
curl -s "$API/characters"
# {"error":{"code":"unauthorized","message":"Missing bearer token"}}
```

The reuse above revoked the whole family, so the refresh token in `$NEW_TOKENS` is dead as well. Access tokens are stateless, so `$AUTH` keeps working until it expires after 15 minutes. After that, exchange the app token again.

## Upload articles

`POST /articles/upload` accepts `application/json`, `multipart/form-data`, `application/x-www-form-urlencoded`, `text/markdown` or `text/plain`. A raw `text/markdown` or `text/plain` body is a Markdown article too: it must be UTF-8 (an explicit `charset=` other than `utf-8`/`utf8` is rejected with `415`), is limited like a multipart upload (1× `MAX_UPLOAD_BYTES`, checked while it streams in), is hashed like JSON text, and takes its title from `?title=` (at most 300 characters), else its first heading. The status code tells you what happened:

| Status | Meaning |
|---|---|
| `202` | The article was accepted and queued, or a previously `failed` article with the same content was re-queued. |
| `200` | Identical content already exists, and its article is returned. With `?wait=true`, `200` also means processing finished (`done` or `failed`) within `wait_timeout` seconds (1–120, default 60). |
| `413` | The upload exceeds `MAX_UPLOAD_BYTES` (20 MiB). |
| `415` | Unsupported content type, a raw text body that declares a non-UTF-8 `charset=`, or a file that is neither a PDF nor UTF-8 text. |
| `422` | Both `text` and `file` were sent, or neither; or the content is empty; or a raw `text/markdown`/`text/plain` body is not UTF-8; or `title` (field or `?title=`) is longer than 300 characters; or the PDF can't be opened (corrupt or password-protected). |

```bash
# multipart file upload, Markdown; wait for the result
EN=$(curl -s -X POST "$API/articles/upload?wait=true&wait_timeout=120" -H "$AUTH" \
  -F file=@examples/sample_en.md | jq -r .id)

# JSON upload of Markdown text (here: the Russian sample, which shares Анна Петрова with the English one)
RU=$(jq -n --rawfile text examples/sample_ru.md '{text: $text}' |
  curl -s -X POST "$API/articles/upload?wait=true&wait_timeout=120" -H "$AUTH" \
    -H 'Content-Type: application/json' --data-binary @- | jq -r .id)

# multipart file upload, PDF; page 2 is an embedded image and page 3 a vector diagram without a text layer,
# so the vision model reads both
PDF=$(curl -s -X POST "$API/articles/upload?wait=true&wait_timeout=120" -H "$AUTH" \
  -F file=@examples/sample.pdf | jq -r .id)

# a raw Markdown body (for a file: --data-binary @examples/sample_en.md) with an explicit title
printf 'Lena Ivanova teaches chemistry. Anna Petrova is her friend.\n' |
  curl -s -X POST "$API/articles/upload?title=Friends" -H "$AUTH" \
    -H 'Content-Type: text/markdown' --data-binary @- | jq '{id, title, status}'
# {"id": "<uuid>", "title": "Friends", "status": "pending"}

# multipart text field with an explicit title; without ?wait it returns 202 at once
curl -s -X POST "$API/articles/upload" -H "$AUTH" \
  -F 'text=Mark Petrov is the younger brother of Anna Petrova.' -F title=Siblings | jq '{id, status}'
# {"id": "<uuid>", "status": "pending"}
```

## Article status

```bash
curl -s "$API/articles/$PDF" -H "$AUTH" | jq
```

```text
{
  "id": "<uuid>",
  "title": "Linden Street Bakery Newsletter",
  "format": "pdf",
  "content_hash": "<sha256>",
  "status": "done",
  "error": null,
  "stats": {"characters": …, "relations": …, "images_described": 2, "images_failed": 0, "images_skipped": 0, "unjudged_characters": 0},
  "created_at": "2026-10-21T14:05:25.402915Z",
  "updated_at": "2026-10-21T14:09:37.492000Z"
}
```

`status` moves from `pending` to `processing` and ends as `done` or `failed`. A failed article carries an `error` such as `LLMError: …`. For PDFs, `images_described` counts the images and rendered pages the vision model described, `images_failed` the ones whose vision call failed, and `images_skipped` the ones never sent: over the `PDF_MAX_IMAGES` cap, over the `PDF_MAX_RENDER_PIXELS` cap, or unreadable. A non-zero `images_failed` or `images_skipped` means graphics content may be missing from the graph. Articles stored before these counters existed report `0`. `unjudged_characters` counts the borderline characters the judge did not decide, because of the judge cap or a judge outage. They are stored as new nodes flagged for [reconciliation](robustness.md#reconciliation-opt-in). Uploading the same content again re-queues it.

## Read the graph

```bash
# all characters, ordered by name (limit 1..200, default 50)
curl -s "$API/characters?limit=20&offset=0" -H "$AUTH" | jq
# {"items": [{"id": "<uuid>", "name": "Anna Petrova", "aliases": ["Anna", "Анна", "Анна Петрова"], "description": "…", "article_count": 2}, …],
#  "total": 7, "limit": 20, "offset": 0}

# characters found in one article (404 if the article does not exist)
curl -s "$API/characters/by_article/$EN" -H "$AUTH" | jq '.[] | {id, name, article_count}'

# every relation type in use, most used first
curl -s "$API/relations/types" -H "$AUTH" | jq
# [{"name": "WORKS_FOR", "description": "works for", "usage_count": 3}, {"name": "ENEMY_OF", "description": "enemy of", "usage_count": 1}, …]

# who is connected to Anna by a sibling relation: free-form type, resolved semantically
ANNA=$(curl -s "$API/characters/by_article/$EN" -H "$AUTH" | jq -r '.[] | select(.name == "Anna Petrova") | .id')
curl -s "$API/relations/connected/$ANNA?type=sibling" -H "$AUTH" | jq

# any wording or language works; non-ASCII values need URL encoding
curl -s -G "$API/relations/connected/$ANNA" -H "$AUTH" --data-urlencode 'type=друг' | jq '{requested_type, resolved_type}'
# {"requested_type": "друг", "resolved_type": "FRIEND_OF"}

# the type resolves globally, so a character without such an edge gets empty items
curl -s "$API/relations/connected/$ANNA?type=enemy" -H "$AUTH"
# {"character_id":"<anna-id>","requested_type":"enemy","resolved_type":"ENEMY_OF","items":[]}
```

`GET /relations/connected` returns one item per neighbour and direction. Each item carries every piece of supporting evidence across articles:

```json
{
  "character_id": "<anna-id>",
  "requested_type": "sibling",
  "resolved_type": "SIBLING_OF",
  "items": [
    {
      "character": {"id": "<mark-id>", "name": "Mark Petrov", "aliases": ["…"], "description": "…", "article_count": 2},
      "relation_type": "SIBLING_OF",
      "direction": "outgoing",
      "evidence": [
        {"article_id": "<en-id>", "description": "…", "evidence": "…"}
      ]
    }
  ]
}
```

- `direction` is `outgoing` for `Anna -[SIBLING_OF]-> Mark` and `incoming` for `Mark -[SIBLING_OF]-> Anna`.
- When no known type is similar enough, the response is `200` with `"resolved_type": null` and `"items": []`.
- An unknown character returns `404`, and a blank `type` or one longer than 100 characters returns `422`.

## Fix duplicates

```bash
# manual merge: Анна Петрова from the Russian article into Anna, if dedup kept them apart
SOURCE=$(curl -s "$API/characters/by_article/$RU" -H "$AUTH" |
  jq -r --arg anna "$ANNA" '.[] | select(.id != $anna and .name == "Анна Петрова") | .id')
if [ -z "$SOURCE" ]; then
  echo "dedup already merged the two Annas into $ANNA, nothing to merge"
else
  jq -n --arg t "$ANNA" --arg s "$SOURCE" '{target_id: $t, source_ids: [$s]}' |
    curl -s -X POST "$API/characters/merge" -H "$AUTH" \
      -H 'Content-Type: application/json' --data-binary @- | jq
fi
# 200 -> the merged character (illustrative, not measured): {"id": "<anna-id>", "name": "Anna Petrova", "aliases": ["Anna", "Анна", "Анна Петрова"], …}
```

The sources' `APPEARS_IN` and `RELATED_TO` edges move to the target, the sources are deleted, and the target keeps its id and name and is re-embedded. If dedup already merged the two Annas, the surviving node keeps the name "Anna Petrova", so the selection is empty; the guard matters, because `source_ids: [""]` would be a `404`.

The merge endpoint can also fail:

| Status | Response | When |
|---|---|---|
| `404` | `{"error":{"code":"not_found","message":"Characters not found: <ids>"}}` | A target or source id does not exist. |
| `422` | `{"error":{"code":"validation_error","message":"body: Value error, target_id must not be among source_ids"}}` | The target is also listed as a source. |
| `422` | `…"message":"body: Value error, source_ids must be distinct"}}` | A source id is listed twice. |
| `422` | `…"message":"body.source_ids: List should have at least 1 item after validation, not 0"}}` | `source_ids` is empty. |
| `422` | `…"message":"body.source_ids: List should have at most 50 items after validation, not 51"}}` | More than 50 source ids in one request. |
| `409` | `{"error":{"code":"resource_busy","message":"Another graph update is in progress; retry later"}}`, with `Retry-After: 5` | Another graph update (an article being resolved, another merge, a reconciliation merge) held the lock for longer than `MERGE_LOCK_ACQUIRE_TIMEOUT_SECONDS` (10 s). |
| `503` | `{"error":{"code":"llm_unavailable","message":"LLM is not configured"}}` (or `"LLM provider is unavailable"`) | The merged profile is re-embedded, which needs the embeddings API. |

```bash
# queue a reconciliation run (it works even when the periodic run is disabled)
curl -s -X POST "$API/characters/reconcile" -H "$AUTH"
# {"status":"queued"}   (HTTP 202)
# a second trigger while that run is still waiting in the queue adds nothing:
# {"status":"already_queued"}   (HTTP 202)
```

## Exploring the graph in the Neo4j Browser

The Neo4j Browser is at <http://localhost:7474>. The node and edge types are described in the README's [Graph schema](../README.md#graph-schema).

```cypher
MATCH (s:Character)-[r:RELATED_TO]->(t:Character)
RETURN s.name, r.type, t.name, r.evidence LIMIT 50;

MATCH (c:Character)-[m:APPEARS_IN]->(a:Article)
RETURN c.name, m.match, m.similarity, a.title;
```

## Health

```bash
curl -s "$API/health/live"    # {"status":"ok"}
curl -s "$API/health/ready"   # {"status":"ok","neo4j":"ok","redis":"ok"}, or 503 with "error" for the failing dependency
```
