# Choosing an LLM provider and models

The service speaks the OpenAI API; the assignment allows "OpenAI or any compatible API". OpenAI is the default (see [Configuration](configuration.md) for every model setting). This page covers the tested Gemini profile, the embedding-model comparison, and the full results of the live run on real embeddings. The dedup thresholds themselves are explained in the README's [AI pipeline](../README.md#ai-pipeline).

- [Gemini profile](#gemini-profile)
- [Choosing the embedding model](#choosing-the-embedding-model)
- [Thresholds are provider-specific](#thresholds-are-provider-specific)
- [Gemini free tier notes](#gemini-free-tier-notes)
- [Measured on real embeddings](#measured-on-real-embeddings)

## Gemini profile

Google Gemini works through its OpenAI-compatible endpoint with no code changes. This is the `.env` profile we tested on Gemini's OpenAI-compatible endpoint, with probe calls and live runs on the free tier. The free tier has limits and caveats, listed in the [Gemini free tier notes](#gemini-free-tier-notes) below:

```
OPENAI_API_KEY=<Gemini API key from Google AI Studio>
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-3.5-flash-lite
VISION_MODEL=gemini-3.5-flash-lite
JUDGE_MODEL=gemini-3.6-flash
EMBEDDING_MODEL=gemini-embedding-2
EMBEDDING_DIM=1536
RELATION_TYPE_QUERY_THRESHOLD=0.68
LLM_MAX_CONCURRENCY=2
LLM_MAX_RETRIES=6
```

With a slow or thinking judge model such as `gemini-3.8-flash`, add the three lines from [Slow judge models](#slow-judge-models).

`DEDUP_AUTO_MERGE_THRESHOLD` and `DEDUP_JUDGE_THRESHOLD` keep their defaults of 0.95 and 0.65; the measured run below set `DEDUP_JUDGE_THRESHOLD=0.65` explicitly, which now equals the default. This exact profile was measured end to end on the three sample files; see [Measured on real embeddings](#measured-on-real-embeddings). Since that run, `LLM_MAX_RETRIES=6` applies to extraction, vision and embeddings only; the judge keeps its own `JUDGE_TIMEOUT_SECONDS=20` and `JUDGE_MAX_RETRIES=1`, so a full judge budget still fits the resolve lock. More judge retries need a longer `RESOLVE_LOCK_TIMEOUT_SECONDS`, and then a longer `JOB_TIMEOUT_SECONDS` and `STUCK_ARTICLE_AFTER_SECONDS` too ([why](robustness.md#serialized-resolve-under-a-self-extending-lock)). They also need a lower `RECONCILE_MAX_JUDGE_CALLS` or a longer `JOB_TIMEOUT_SECONDS`, because a reconciliation run's judge budget must fit the job timeout ([why](robustness.md#reconciliation-opt-in)). For example, `JUDGE_MAX_RETRIES=2` gives 40 × 20 × 3 × 2 = 4800 s ≥ 3600 s, so startup validation rejects it.

**Why the judge gets its own model.** Extraction and vision run on every chunk and every image, so a fast, cheap "lite" model is enough. In our probe, `gemini-3.5-flash-lite` extracted correctly in about 2 s and read the image-only diagram correctly. The dedup judge is called rarely, only for candidates in the similarity band, and its mistakes corrupt the graph. `JUDGE_MODEL` therefore lets a stronger model decide those few cases. If `JUDGE_MODEL` is unset, the judge falls back to `LLM_MODEL`.

**Compatibility handled in code** ([`app/llm/client.py`](../app/llm/client.py)):

- Gemini returns `index: null` on the first embedding item. The client then falls back to the response order, which is the input order.
- A blank `OPENAI_BASE_URL` (the compose default) falls back to the official OpenAI endpoint.

## Choosing the embedding model

Both Gemini embedding models were measured on the same profile texts and queries, at 1536 dimensions.

Relation-type query resolution:

| | weakest correct match | strongest wrong match | gap |
|---|---|---|---|
| `gemini-embedding-001` | brother→`SIBLING_OF` 0.652 | enemy→`RIVAL_OF` 0.641 | +0.011 |
| `gemini-embedding-2` | brother→`SIBLING_OF` 0.702 | enemy→`RIVAL_OF` 0.647 | +0.055 |

More correct matches on `gemini-embedding-2`: friend→`FRIEND_OF` 0.790, друг→`FRIEND_OF` 0.725, mentor→`MENTOR_OF` 0.804, works for→`WORKS_FOR` 0.844, работает на→`WORKS_FOR` 0.705. Hence `RELATION_TYPE_QUERY_THRESHOLD=0.68`, inside the gap.

Character dedup:

| | same person, EN/RU | closest different people |
|---|---|---|
| `gemini-embedding-001` | 0.865–0.866 | 0.870 |
| `gemini-embedding-2` | 0.881–0.884 | 0.867 |

- **Embeddings alone cannot decide dedup.** A profile embeds the name together with the description, and people from the same story share context ("the bakery on Linden Street"). So, whichever the model, the cosine cannot reliably separate "the same person in another language" from "a different person in the same story". The live run with extracted profiles shows the same: 0.801 for the same person against 0.772 for different people ([measured](#measured-on-real-embeddings)). That is why the judge band is wide (`DEDUP_JUDGE_THRESHOLD`, default 0.65) and the judge decides both kinds of pair. This matches the [bias against wrong merges](../README.md#ai-pipeline); the per-article judge cap (`DEDUP_MAX_JUDGE_CALLS_PER_ARTICLE`) still bounds the cost.
- **Antonyms embed close together.** `PARENT_OF` vs `CHILD_OF` is 0.878 on `gemini-embedding-2` and 0.886 on `gemini-embedding-001`. This is why relation types are never merged semantically at write time, only by exact normalized name (see [AI pipeline](../README.md#ai-pipeline)).

**Switching the embedding model needs a fresh database.** Vectors from different models are not comparable, even at the same dimension. Run `docker-compose down -v` (this deletes the data). The startup check only catches a dimension change, not a model change. The client always sends `dimensions=EMBEDDING_DIM`, so the model must support that output size.

## Slow judge models

Thinking or otherwise slow models, such as `gemini-3.8-flash`, can take more than the default 20 s per judge call. The call then fails with a timeout, and the rest of that article stays unjudged. Raise `JUDGE_TIMEOUT_SECONDS` together with the budgets that depend on it. This profile is validated, and the e2e tests passed 9/9 with it ([results](testing.md#end-to-end-tests-opt-in-real-llm)):

```
JUDGE_TIMEOUT_SECONDS=60
RESOLVE_LOCK_TIMEOUT_SECONDS=2500
RECONCILE_MAX_JUDGE_CALLS=14
```

With the default `JUDGE_MAX_RETRIES=1`, the numbers fit as follows:

| Rule | Arithmetic |
|---|---|
| Worst case of one judge call: timeout × (retries + 1) × 2 structured-output attempts | 60 × 2 × 2 = 240 s |
| Judge budget of one article, below the resolve lock TTL | 10 × 240 = 2400 s < 2500 s |
| Lock TTL plus acquire wait, below the job timeout | 2500 + 600 = 3100 s < 3600 s |
| Judge budget of one reconciliation run, below the job timeout | 14 × 240 = 3360 s < 3600 s |

If a combination does not fit, startup validation fails and names the rule it breaks.

## Thresholds are provider-specific

The code defaults (0.95 auto-merge, 0.65 judge band, 0.7 type query) target OpenAI's `text-embedding-3-small` and have not been measured on OpenAI. The 0.65 judge band comes from the Gemini measurement and errs towards asking the judge, which costs judge calls but not correctness. The Gemini profile values above were measured; see [Measured on real embeddings](#measured-on-real-embeddings). For any other embedding model, re-measure the thresholds with the same checks and set them in `.env`.

## Gemini free tier notes

- Limits are per model and per day; check them in Google AI Studio.
- `LLM_MAX_CONCURRENCY=2` reduces 429 responses, and the SDK retries 429 and 503 with backoff (`LLM_MAX_RETRIES` for extraction, vision and embeddings; `JUDGE_MAX_RETRIES`, default 1, for the judge, so that the judge budget fits the resolve lock). Once a daily limit is used up, retries cannot help and the article fails.
- A judge 429 does not fail the article. The pair is left unmerged, the node is flagged `needs_reconcile = true`, and the judge is skipped for the rest of the article. Run reconciliation once the quota is back. Only extraction and embedding failures still fail the article.
- On the free tier, Google may use prompts to improve its products. That is fine for the sample data, but not for private documents.

## Measured on real embeddings

The code defaults target OpenAI's `text-embedding-3-small`, but they have not been measured on OpenAI. The judge band default (0.65) was widened after the measurement below. The numbers come from one live run of the [Gemini profile](#gemini-profile):

- **Provider.** Google Gemini through its OpenAI-compatible endpoint.
- **Models.** Extraction and vision on `gemini-3.5-flash-lite` (`LLM_MODEL`, `VISION_MODEL`), the judge on `gemini-3.6-flash` (`JUDGE_MODEL`), and embeddings on `gemini-embedding-2` at 1536 dimensions.
- **Thresholds.** Judge band `DEDUP_JUDGE_THRESHOLD=0.65`, type query `RELATION_TYPE_QUERY_THRESHOLD=0.68`, auto-merge `DEDUP_AUTO_MERGE_THRESHOLD=0.95`.
- **Input.** The three sample files (`examples/sample_en.md`, `examples/sample_ru.md`, `examples/sample.pdf`), uploaded once each, in one pass, to a fresh stack (`docker-compose down -v`, then `docker-compose up -d --build`). The PDF then had two pages; page 3, the vector diagram, was added later and is not part of this run.

| Check | Result |
|---|---|
| `sample_en.md` | `done`: 6 characters, 5 relations |
| `sample_ru.md` | `done`: 4 characters, 2 relations |
| `sample.pdf` | `done`: 2 characters, 1 relation, 1 image described |
| Anna Petrova ↔ Анна Петрова | one node, merged by the judge at cosine 0.801; aliases `Anna`, `Анна`, `Анна Петрова` |
| Oleg Sorokin ↔ Олег Сорокин | one node, merged by the judge at cosine 0.807 |
| Different people | 7 distinct characters stay separate: Anna, Biscuit, Lena, Mark, Oleg, Victor, Игорь. The judge ruled Игорь different from Mark and from Lena, and both `NOT_SAME` verdicts were recorded. |
| Image-only relation (page 2 of `sample.pdf`) | Mark Petrov `MENTOR_OF` Lena Ivanova is present, stored only under the PDF article |
| Vector-diagram relation (page 3 of `sample.pdf`) | not part of this run, since the page was added later; the [end-to-end tests](testing.md#end-to-end-tests-opt-in-real-llm) check it, and their latest run found the Igor–Lena relation |
| Judge | 7 calls in total, 0 unjudged characters, no node flagged `needs_reconcile` |

**Processing time per article.** The expected times are an estimate for normal conditions (a paid key, no 503 responses), not a measurement. The measured times are from the free-tier run above. There, most of the time was spent waiting on the provider: Gemini returned 503 "high demand" three times, the client retried five times, and two of those retries hit the 60 s timeout that the judge then shared with extraction.

| File | Expected (estimate) | Measured on the free tier |
|---|---|---|
| `sample_en.md` | about 10 s | 40 s |
| `sample_ru.md` | about 30–60 s | 202 s, mostly 503 retries and timeouts |
| `sample.pdf` | about 30–60 s | 253 s, mostly 503 retries and timeouts |

The estimate adds up the stages:

- parse and chunk: under a second;
- vision (`gemini-3.5-flash-lite`): about 1–6 s per image or rendered page;
- extraction (`gemini-3.5-flash-lite`): about 2–9 s per chunk;
- embeddings: about 1 s;
- judge (`gemini-3.8-flash`, a slower thinking model): about 5–25 s per call. The calls run one after another under the resolve lock, a deliberate trade-off for duplicate-free writes ([why](robustness.md#serialized-resolve-under-a-self-extending-lock));
- Neo4j write: under a second.

The English article is uploaded first and needs no judge calls, because the graph is still empty. The RU article and the PDF spend most of their expected time in the judge.

Cosines between character profiles:

| Pair | Cosine |
|---|---|
| Anna ~ Анна (same person, at match time) | 0.801 |
| Oleg ~ Олег (same person, at match time) | 0.807 |
| Anna ~ Oleg (closest different people) | 0.772 |
| Biscuit ~ Oleg | 0.738 |
| Lena ~ Mark | 0.715 |
| Oleg ~ Victor | 0.707 |

The different-people cosines are between the stored node embeddings after the merges. The highest similarity seen in the run was 0.842, so auto-merge at 0.95 never triggered, and every merge went through the judge.

Free-form `?type=` queries, resolved through the embedding path:

| `?type=` | Resolved type | Cosine |
|---|---|---|
| `sibling` | `SIBLING_OF` | 0.815 |
| `brother` | `SIBLING_OF` | 0.702 |
| `friend` | `FRIEND_OF` | 0.790 |
| `друг` | `FRIEND_OF` | 0.725 |
| `mentor` | `MENTOR_OF` | 0.804 |
| `works for` | `WORKS_FOR` | 0.844 |
| `работает на` | `WORKS_FOR` | 0.705 |
| `enemy` | `ENEMY_OF` (not `FRIEND_OF`, which scored 0.642) | 0.802 |

The closest wrong candidate was `friend` → `ENEMY_OF` at 0.662, 0.018 under the threshold.

**Takeaways.**

- **The judge does the real work in dedup.** The margin between the same person across languages (0.801) and the closest different people (0.772) is only about 0.03. No cosine threshold separates them reliably, which is why the judge band is wide.
- **The old default band edge was too tight.** The same-person pairs scored 0.801 and 0.807, just above the earlier 0.80 default; a slightly different profile would have fallen below it and silently become a second node. The default is therefore now 0.65, the value of this run.
- **The type-query margin is thin.** The weakest correct match (`brother`, 0.702) and the closest wrong one (0.662) are 0.04 apart. Exact normalized matching is therefore the primary path, and the embedding path is the fallback for free wording and other languages.

**Observations.** These are known behaviours of the lite extraction model, not pipeline errors:

- **A generic word became an alias.** The RU article mentions Anna's «подруга» ("her friend, a chemistry teacher") without a name. The extraction model listed it as a character, and the judge merged it into Lena Ivanova at 0.796. The merge is correct from the context, but the generic word «подруга» is now one of Lena's aliases, although the extraction prompt excludes generic descriptions.
- **A relation came out in the wrong direction.** Игорь is Anna's apprentice. The model labelled this Игорь `WORKS_FOR` Anna rather than Anna `MENTOR_OF` Игорь.
- **A resolved type can return no items.** `enemy` on Anna resolves to `ENEMY_OF` with empty `items`: the type is resolved globally, and Anna has no such edge.
- **A relation in an image description was not extracted.** In a later live run, the vision model described page 3 of `sample.pdf`, but the lite extraction model returned no "student of" relation from the `[Image on page 3: …]` block. The extraction prompt now says to extract characters and relationships from image blocks just like from the prose, and the vision prompt asks for each diagram arrow as a sentence ("<A> is <label> <B>"). This helps, but on lite models extraction from image descriptions can still be unstable; a stronger `LLM_MODEL` is the reliable fix.
- **Two Igors without shared context.** In a later live run, page 3 of `sample.pdf` still showed only "Igor Smirnov, student of Lena Ivanova". The RU article's «Игорь Смирнов» is a 16-year-old who helps Anna at the bakery's open lesson. With nothing in common but the name, the judge called them different people ("different mentors") and kept two nodes. That is the intended bias: when in doubt, same-named people stay apart, because a missed merge is fixed with one [`POST /characters/merge`](api.md#fix-duplicates) while a wrong merge corrupts the graph. The diagram now gives Igor the article's context ("Igor Smirnov, 16, helps Anna Petrova at the bakery", under "Chemistry club at the Linden Street Bakery"), so the judge has evidence to merge.
