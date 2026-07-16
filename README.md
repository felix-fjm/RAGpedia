# RAGpedia — Wikipedia-Grounded RAG System

> Ask any question. Get a sourced, factual answer drawn directly from English Wikipedia - with clickable citations for every claim.

RAGpedia is a fully self-hosted, Dockerised RAG (Retrieval-Augmented Generation) system that turns the English Wikipedia into a continuously updated knowledge base. It grounds any LLM in verified, sourced context - eliminating hallucination on factual questions. You bring your own API key; RAGpedia handles ingestion, embedding, retrieval, and prompt assembly.

**Status:** All five phases (infrastructure, ingestion, query API, weekly-update worker, UI) are implemented and validated end-to-end against a live deployment. See [Validated At This Scale](#validated-at-this-scale) for exactly what was tested and at what size.

---

## How It Works

```
Your Question
     │
     ▼
 Embed query          ← nomic-embed-text-v1.5 (self-hosted)
     │
     ▼
 Vector search        ← Qdrant HNSW index
     │
     ▼
 Top-5 chunks         ← title · section · url · text
     │
     ▼
 Prompt assembly      ← "Answer ONLY using the context below. Cite sources."
     │
     ▼
 LLM API call         ← OpenAI · Anthropic · Ollama (your key)
     │
     ▼
 Answer + Sources     ← grounded, cited, verifiable
```

On a **cache hit** (Redis), steps 2–6 are skipped entirely - response in ~2ms.

---

## Key Design Decisions

| Component | Choice |
|-----------|--------|
| Dump format | Wikimedia Cirrus JSON |
| Scope | Top 15% by pageview (design target: ~1M articles; see [scale note](#validated-at-this-scale)) |
| Embedding model | `nomic-embed-text-v1.5` · 768d · self-hosted via Ollama |
| Vector DB | Qdrant · HNSW index · cosine similarity |
| Retrieval | Dense cosine · k=10 retrieve, top 5 to LLM |
| Chunking | Section-aware · 400–600 tokens · 50-token overlap on splits |
| Section extraction | Two-path: wikitext `== Heading ==` parsing (PATH 1) or Introduction + Body fallback (PATH 2) |
| Metadata per chunk | `title` · `section` · `url` · `last_modified` · `pageview_rank` |
| Update cadence | Weekly (design) · timestamp diff · re-embed changed articles only. **Implemented and tested; not deployed as a live schedule in this repo** — see [Weekly Update Worker](#weekly-update-worker) |
| Cache | Redis · chunks TTL 24h · answers TTL 6h |
| LLM providers | OpenAI · Anthropic · Ollama (BYO key) |
| Backend | FastAPI · async · separate `api` + `worker` containers |
| Auth | None for MVP · API key lives client-side only, never persisted server-side |

---

## System Architecture

Five Docker containers. Two custom services (`api` + `worker`) share three official infrastructure containers.

```
┌──────────────────────────────────────────────────────┐
│                  Shared Infrastructure               │
│  ┌─────────────┐   ┌─────────────┐   ┌────────────┐  │
│  │   qdrant    │   │    redis    │   │  embedder  │  │
│  │ HNSW index  │   │ chunk cache │   │   ollama   │  │
│  │             │   │ answer cache│   │ nomic-embed│  │
│  └──────┬──────┘   └──────┬──────┘   └─────┬──────┘  │
└─────────┼────────────────┼────────────────┼──────────┘
          │                │                │
    ┌─────▼─────┐    ┌─────▼───────────────▼──────┐
    │  worker   │    │            api             │
    │ ingestion │    │  FastAPI · /query · GET /  │
    │ + update  │    │  LLM connector · prompt    │
    └───────────┘    └────────────────────────────┘
```

**Why separate `worker` and `api`?**
The worker is CPU/GPU-bound and runs for hours during ingestion. The API is I/O-bound and must stay responsive 24/7. Separating them means a re-index job never blocks user queries. Both share the same `embedder` container - critical because query vectors and chunk vectors must live in the same embedding space.

---

## Project Structure

```
docker-compose.yml          ← wires all 5 containers
.env                        ← API keys & config (never commit)
.gitignore
CLAUDE.md                   ← project context for Claude Code

api/                        ← FastAPI query endpoint + UI server
 ├── Dockerfile
 ├── main.py                 ← GET /health · POST /query · GET /
 ├── embedder.py             ← embed query via nomic container
 ├── cache.py                ← Redis SHA-256 cache logic
 ├── llm.py                  ← OpenAI / Anthropic / Ollama connector + model validation
 ├── prompt.py               ← top-5 chunk selection + prompt assembly
 ├── requirements.txt
 └── static/
      └── index.html          ← single-file UI (question input, model selector, sources)

worker/                     ← ingestion pipeline + weekly update logic
 ├── Dockerfile
 ├── ingest.py               ← full pipeline orchestrator (supports --limit for scoped runs)
 ├── download.py             ← streaming Cirrus JSON downloader
 ├── parse.py                ← filter · parse · clean · chunk
 ├── embed.py                ← batch embed + upsert to Qdrant, with chunk-level crash resilience
 ├── update.py               ← weekly diff + upsert/delete (+ --force-update test mode)
 ├── scheduler.py             ← cron entry point (Monday 03:00) — NOT deployed, see note below
 └── requirements.txt

qdrant/
 └── config.yaml             ← optional HNSW params
redis/
 └── redis.conf              ← optional maxmemory / eviction policy
embedder/
 └── pull_model.sh           ← pulls nomic-embed-text-v1.5 on start
```

---

## Prerequisites

- **Docker** + **Docker Compose** (v2)
- **8 GB RAM minimum** (16 GB recommended for a full-scale ~1M article index)
- **Disk space** for the Qdrant vector index — scales with how much you ingest (see [scale table](#verifying-your-index))
- An API key from **OpenAI**, **Anthropic**, or a local **Ollama** model
- A `get-docker.sh` script is included for convenience on fresh Linux servers

> **Windows users:** Run all commands inside WSL2 Ubuntu. Open a WSL shell with `wsl -d Ubuntu`.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/felix-fjm/RAGpedia.git
cd RAGpedia
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your values:

```env
# Wikimedia Cirrus dump URL (check for latest at dumps.wikimedia.org)
WIKI_DUMP_URL=https://dumps.wikimedia.org/other/cirrussearch/current/enwiki-20240101-cirrussearch-content.json.gz

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION=wikipedia

# Embedder (Ollama)
EMBEDDER_HOST=embedder
EMBEDDER_PORT=11434
EMBED_MODEL=nomic-embed-text:v1.5

# Ingestion settings
PAGEVIEW_TOP_FRACTION=0.15
MIN_ARTICLE_TOKENS=300
EMBED_BATCH_SIZE=64
DUMP_PATH=/data/wiki_dump.json.gz

# Cache TTLs
CHUNK_CACHE_TTL=86400
ANSWER_CACHE_TTL=21600
```

### 3. Start the infrastructure

```bash
docker compose up -d qdrant redis embedder
```

Wait ~30 seconds for the embedder to pull and load `nomic-embed-text-v1.5`, then verify:

```bash
docker compose exec embedder ollama list
# Should show: nomic-embed-text:v1.5
```

### 4. Run the ingestion pipeline

**Smoke test first (5 articles, ~2 minutes on CPU):**
```bash
docker compose run --rm -e PYTHONUNBUFFERED=1 worker python ingest.py --limit 5
```

**Demo-scale run (~1,500 articles, ~2 hours on CPU)** — this is the scope this repo was actually validated at:
```bash
docker compose run --rm -e PYTHONUNBUFFERED=1 worker python ingest.py --limit 1500
```

**Full run (no `--limit` flag, design target ~1M articles):**
```bash
docker compose run --rm -e PYTHONUNBUFFERED=1 worker python ingest.py
```
> ⚠️ The full unbounded run is untested at production scale in this repo (estimated 4–6 hours on GPU / 2–3 days on CPU based on observed per-article throughput). Start with `--limit` and scale up.

> The API is usable while the worker is still indexing - partial results are returned from whatever is indexed so far.

### 5. Start the API

```bash
docker compose up -d api
```

Verify it's running:
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### 6. Open the UI

```
http://localhost:8000
```

Paste your API key into the settings panel, pick a model, and ask a question. Sources render as clickable Wikipedia links below each answer.

### 7. Or query directly via curl

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR-API-KEY" \
  -d '{"question": "What caused the French Revolution?", "model": "gpt-4o-mini"}'
```

**Supported model strings:**
| Provider | Model strings |
|----------|----------------------|
| OpenAI | `gpt-4o` · `gpt-4o-mini` |
| Anthropic | `claude-sonnet-5` · `claude-haiku-4-5-20251001` |
| Ollama (local) | `llama3.2` (must be pulled separately from the embedding model — see note below) |

`api/llm.py` validates the `model` field against this exact list and returns `422 Unprocessable Entity` with a clear message for anything else, before any embedding or LLM API work happens.

> **Note on Ollama as an LLM provider:** the `embedder` container only pulls `nomic-embed-text-v1.5` (the embedding model) via `pull_model.sh`. If you select `llama3.2` for chat completions, make sure it's pulled separately (`docker compose exec embedder ollama pull llama3.2`) — the embedder and LLM models are independent even though they share a container.

**Example response:**
```json
{
  "answer": "The French Revolution was caused by a combination of financial crisis, social inequality, and Enlightenment ideals [1 - Causes]. The French state was effectively bankrupt by 1788 following costly wars including support for the American Revolution [2 - Financial crisis].",
  "sources": [
    {
      "title": "French Revolution",
      "section": "Causes",
      "url": "https://en.wikipedia.org/wiki/French_Revolution"
    },
    {
      "title": "French Revolution",
      "section": "Financial crisis",
      "url": "https://en.wikipedia.org/wiki/French_Revolution"
    }
  ],
  "cached": false
}
```

---

## API Reference

### `GET /health`
Returns `{"status": "ok"}` when the API is running.

### `GET /`
Serves the single-page UI (`api/static/index.html`).

### `POST /query`

**Headers:**
```
Content-Type: application/json
Authorization: Bearer YOUR-API-KEY
```

**Body:**
```json
{
  "question": "Your question here",
  "model": "gpt-4o-mini"
}
```

**Response:**
```json
{
  "answer": "...",
  "sources": [
    { "title": "...", "section": "...", "url": "..." }
  ],
  "cached": false
}
```

Returns `422` if `model` isn't one of the supported strings listed above.

**Latency budget (cache miss):**

| Step | Typical |
|------|---------|
| Redis cache check | ~2 ms |
| Embed query | ~20 ms |
| Qdrant HNSW search | ~10 ms |
| Prompt assembly | ~1 ms |
| LLM API call | ~1–3 s |
| **Total** | **~1.1–3.1 s** |

On a **cache hit**: ~2 ms flat.

---

## Ingestion Pipeline

The worker processes each Wikipedia article through 8 steps:

| # | Step | Detail |
|---|------|--------|
| 1 | Download | HTTP stream from `dumps.wikimedia.org` · line-by-line, constant memory |
| 2 | Filter | Keep top 15% by pageview rank · discard stubs < 300 tokens |
| 3 | Parse | Extract `title` · sections · `opening_text` · `last_modified` timestamp |
| 4 | Clean | Strip citation markers `[N]` · wikitext markup · HTML tags · normalise whitespace |
| 5 | Section extraction | PATH 1: parse `source_text` wikitext `== Heading ==` markers · PATH 2: Introduction + Body fallback |
| 6 | Chunk | ≤600 tok → 1 chunk · >600 tok → split at paragraph boundary with 50-tok overlap · <50 tok → merge with previous |
| 7 | Embed | `nomic-embed-text-v1.5` · batch 64 · mean-pool + L2-norm → `float32[768]` |
| 8 | Upsert | `PointStruct(id=uuid5(title+section+idx), vector, payload)` → Qdrant |

**Deterministic chunk IDs** (`uuid5(title + section + chunk_index)`) make upserts idempotent - re-running ingestion on a changed article overwrites vectors in place without creating duplicates.

**Crash resilience (embed.py):** individual chunks that persistently fail embedding (e.g. malformed content that survives cleaning) are logged with full diagnostic detail (title, section, chunk index, text repr) and **skipped**, rather than crashing the entire multi-hour run. This was validated against real production data — see [Known Issues](#known-issues--edge-cases) below.

---

## Weekly Update Worker

`worker/update.py` implements incremental updates: it diffs a fresh Cirrus dump against what's currently in Qdrant (by `last_modified` timestamp), re-embeds only changed or new articles, deletes articles no longer present in the dump, and flushes the Redis answer cache so stale answers aren't served.

**`worker/scheduler.py`** wires this up to run every Monday at 03:00 via the `schedule` library — but **this scheduler is not deployed or running anywhere in this repo's demo environment.** It exists to show the intended production entry point. Running it as a live cron job requires standing up a long-lived worker process (e.g. via `docker compose up -d` on the worker service with `scheduler.py` as the entrypoint), which is outside the scope of this portfolio deployment.

**Validation:** rather than run a full weekly diff against ~1M articles (multi-hour, requires a full dump scan to compute the popularity threshold), `update.py` also supports a `--force-update TITLE [TITLE ...]` mode that skips the full-dump scan and streams directly from the dump URL, stopping as soon as the named titles are found. This was used to validate the core diff/upsert/delete logic against the live demo index:

- Corrupted one real article's (`Albert Einstein`) stored timestamp, ran `--force-update`, confirmed it was re-embedded with the **same deterministic point IDs** (pure overwrite, zero duplicates) and its timestamp restored from the real dump.
- Deleted another real article's (`Benin`) points entirely, ran `--force-update` again, confirmed it was correctly re-added.
- Confirmed the Redis cache flush fired (`FLUSHDB`) after the update completed.

```bash
docker compose run --rm worker python update.py --force-update "Article Title" "Another Title"
```

For a full production-style run against the whole dump:
```bash
docker compose run --rm worker python update.py --dump-path /data/wiki_dump.json.gz
```

---

## Verifying Your Index

Check point count:
```bash
curl http://localhost:6333/collections/wikipedia | python3 -m json.tool | grep points_count
```

Browse stored chunks:
```bash
curl -X POST http://localhost:6333/collections/wikipedia/points/scroll \
  -H "Content-Type: application/json" \
  -d '{"limit": 5, "with_payload": true}' | python3 -m json.tool
```

Expected scale:
| Scope | Articles | Approx. chunks | Index size |
|-------|----------|----------------|------------|
| Smoke test | 5 | ~120 | negligible |
| Demo scale (validated) | ~1,479 | ~20,700 | ~300 MB |
| Full (design target, untested) | ~1,000,000 | ~20,000,000 | ~15–20 GB |

---

## Validated At This Scale

To be transparent about what's actually been run and verified, as opposed to designed-for:

- **Ingestion (Phase 2):** run to completion with `--limit 1500` → **1,479 articles, 20,726 points** upserted cleanly into Qdrant. The full unbounded (~1M article) run has not been executed end-to-end in this repo.
- **Query API (Phase 3):** tested live via curl and the browser UI against the demo-scale index above — confirmed correct grounded answers with citations, correct cache-hit behavior, and correct refusal ("I cannot answer") when the index has no relevant content, rather than hallucinating.
- **Weekly update worker (Phase 4):** implemented per spec and validated via `--force-update` against two real articles in the live demo index (see [Weekly Update Worker](#weekly-update-worker) above). Not deployed as a live schedule.
- **UI (Phase 5):** tested live in-browser against the demo-scale index, including model selection, API key entry, source rendering, and the cached-answer badge.

---

## Known Issues & Edge Cases

**Persistent embedder 400 on specific chunks.** During ingestion, the Wikipedia article *"Benin"* consistently triggers a `400 Bad Request` from the Ollama embedder on 2 of its chunks, even after word-count-based truncation and retry. The root cause wasn't fully isolated (candidates: an encoding quirk or unusual whitespace surviving cleanup) but the pipeline handles it correctly by design: `embed.py` logs full diagnostic detail (title, section, chunk index, text repr) and skips just that chunk, letting the rest of the article and all subsequent articles process normally. This was reproduced and confirmed handled correctly on two independent runs (`ingest.py` and `update.py --force-update`), so it's a known, contained edge case rather than a pipeline-ending bug.

**Full-scale (~1M article) ingestion is untested.** The pipeline is designed for it (two-pass streaming keeps memory flat regardless of dump size — see `ingest.py`), but this repo's validation was intentionally scoped to ~1,500 articles for demo purposes. Expect to encounter more edge cases like the Benin case above at full scale; the skip-and-log resilience should handle them without crashing, but this hasn't been proven beyond the demo scale.

---

## Troubleshooting

**`400 Bad Request` from embedder**
Handled automatically — see [Known Issues](#known-issues--edge-cases) above. If you see a `WARNING` or `ERROR` log line mentioning a specific title, that chunk was skipped and ingestion continued; check the end-of-run summary for a full list of skipped chunks.

**WSL2 `SIGBUS` crash during Docker build**
Your system ran out of RAM. Fix: `wsl --shutdown` from PowerShell, then add a memory cap:
```ini
# %USERPROFILE%\.wslconfig
[wsl2]
memory=4GB
swap=4GB
```
Then rebuild with `DOCKER_BUILDKIT=0 docker compose build --no-cache worker`.

**`points_count: 0` after ingestion, or UI shows old/placeholder content**
The container ran with a stale image. Always rebuild after pulling new code: `docker compose build --no-cache worker` / `docker compose build --no-cache api`. Also double-check `.dockerignore` isn't excluding directories your Dockerfile's `COPY` step needs (e.g. `api/static/`) — an overly broad `.dockerignore` entry is an easy way to silently ship an image missing files that exist fine on the host.

**Retrieval returns unrelated articles**
Your index is too small - with very few articles, cosine similarity has little to work with and returns the least-dissimilar chunks regardless of relevance. This is also the *correct* behavior when you ask about something genuinely outside your index — the LLM should say it can't answer rather than hallucinate. Run a larger `--limit` for broader topical coverage.

**`cached: true` returning stale answers**
Redis answer TTL is 6 hours. To flush immediately:
```bash
docker compose exec redis redis-cli FLUSHALL
```

**GitHub push fails with "Password authentication is not supported"**
GitHub requires a Personal Access Token (classic, with the `repo` scope) in place of your password for HTTPS git operations. Generate one at github.com → Settings → Developer settings → Personal access tokens → Tokens (classic), then use it as the password when prompted. Run `git config --global credential.helper store` to avoid re-entering it every push.

---

## Roadmap

- [x] Phase 1 — Infrastructure (Qdrant · Redis · Ollama embedder)
- [x] Phase 2 — Ingestion worker (download · filter · parse · clean · chunk · embed · upsert), with chunk-level crash resilience
- [x] Phase 3 — Query API (FastAPI · Redis cache · LLM connector · model validation)
- [x] Phase 4 — Weekly update worker (timestamp diff · incremental upsert/delete · cache flush) — implemented and validated, not deployed as a live schedule
- [x] Phase 5 — UI (single-page HTML served by API container)

---

## Technical Notes

**Embeddings:** `nomic-embed-text-v1.5` is a BERT-style transformer (12 layers, 768d). Input text → BPE tokens → 12 layers of multi-head self-attention → mean-pool → L2-normalise → `float32[768]`. Semantically similar text lands geometrically close in this space; cosine similarity measures the angle between vectors.

**HNSW search:** Qdrant's Hierarchical Navigable Small World index navigates a small set of candidate vectors from millions without brute-force comparison — returning top-k results in ~5–20ms with ~99% recall vs exact search.

**RAG grounding:** The LLM never accesses Qdrant directly. The API retrieves top-5 chunks, injects them as context with the instruction `"answer ONLY using the context below"`, then calls the LLM. The model synthesises an answer from provided paragraphs - it cannot invent facts that contradict the context, and correctly declines to answer when the context doesn't cover the question.

---

## Security Notes

- API keys are never stored server-side — passed per-request as a Bearer token, held only in browser memory client-side.
- `.env` (containing infrastructure config, never LLM API keys) is gitignored.
- No secrets are committed to this repository's git history.

---

## License

MIT
