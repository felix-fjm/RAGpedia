# WikiRAG MVP — Project Context for Claude Code

> Place this file at the root of your workspace. Claude Code will automatically use it as shared project context across all sessions.
>
> Source documents: `Claude_WikiRAG_MVP_Design.pdf` · `Claude_Implementation_Roadmap.pdf`

---

## What Is WikiRAG?

WikiRAG is a Dockerised RAG (Retrieval-Augmented Generation) system that turns English Wikipedia
into a continuously updated knowledge base, grounding any LLM in sourced, factual context.
The user supplies their own API key; the system handles ingestion, retrieval, and prompt assembly.

---

## Key Design Decisions

| Component | Choice |
|-----------|--------|
| Dump format | Wikimedia Cirrus JSON |
| Scope | Top 15% by pageview (~1M articles) |
| Embedding model | `nomic-embed-text-v1.5` · 768d · self-hosted via Ollama |
| Vector DB | Qdrant (Docker) · HNSW index · cosine similarity |
| Search | Dense cosine · k=10 retrieve, top 5 to LLM |
| Chunking | Section boundary · 400–600 tok · 50-tok overlap on splits |
| Metadata per chunk | `title` · `section` · `url` · `last_modified` · `pageview_rank` |
| Update cadence | Weekly · timestamp diff · re-embed changed articles only |
| Cache | Redis · chunks TTL 24h · answers TTL 6h |
| LLM providers | OpenAI + Anthropic + Ollama (user BYO key) |
| Backend | FastAPI · async · separate `api` + `worker` containers |
| UI | Single-page JS file served by `api` container |
| Auth | None for MVP · API key client-side only, never persisted server-side |

---

## System Architecture — Five Docker Containers

```
┌─────────────────────────────────────────────────────┐
│                   Shared Infrastructure              │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │   qdrant     │  │    redis     │  │ embedder  │  │
│  │ qdrant/qdrant│  │ redis:alpine │  │  ollama/  │  │
│  │ HNSW index   │  │ chunk cache  │  │  nomic-   │  │
│  │ ~15–20 GB    │  │ answer cache │  │  embed    │  │
│  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  │
│         │ ★               │                │        │
└─────────┼─────────────────┼────────────────┼────────┘
          │                 │                │
    ┌─────▼─────┐     ┌─────▼───────────────▼─────┐
    │  worker   │     │            api             │
    │ ingestion │     │  FastAPI · /query · /ui    │
    │ cron job  │     │  LLM connector · prompt    │
    └───────────┘     └────────────────────────────┘
```

★ = shared state accessed by both `worker` and `api`. Qdrant handles concurrent reads/writes
internally. Redis is written by `api` (cache) and flushed by `worker` (post-update).

**Service separation rationale:**
- `worker` is GPU-bound and runs for hours; `api` is I/O-bound and must stay responsive 24/7.
  Separating them means a re-index job cannot block user queries.
- Both share `embedder` — using the same `nomic-embed` container ensures query vectors and chunk
  vectors live in the same embedding space. Different models = meaningless cosine distances.

**docker-compose.yml structure:**
```yaml
services:
  qdrant:
    image: qdrant/qdrant
    volumes: [qdrant_storage:/qdrant/storage]
  redis:
    image: redis:alpine
  embedder:
    image: ollama/ollama        # pulls nomic-embed-text-v1.5 on first start
  worker:
    build: ./worker
    depends_on: [qdrant, embedder]
  api:
    build: ./api
    depends_on: [qdrant, redis, embedder]
    ports: ['8000:8000']
volumes:
  qdrant_storage:               # persists 15–20 GB vector index across restarts
```

**Startup behaviour:**
```
docker compose up
  → qdrant, redis, embedder start first (healthcheck)
  → worker starts: checks if Qdrant collection exists
      if empty  → run full ingestion pipeline (4–6 h GPU)
      if exists → skip, register weekly cron, sleep
  → api starts: serves /query and /ui immediately
    (api is usable while worker indexes — partial results returned)
```

---

## Project Folder Structure

Everything lives inside a single `veritas_app/` root folder.
Only `api/` and `worker/` need custom Dockerfiles — the other three containers use official images.

```
veritas_app/
├── CLAUDE.md                        ← project context (this file)
├── docker-compose.yml               ← wires all 5 containers together
├── .env                             ← API keys, ports, TTL values (never commit)
├── .gitignore
│
├── api/                             ← FastAPI query endpoint + UI server
│   ├── Dockerfile
│   ├── main.py                      ← /query endpoint, /ui static serve
│   ├── embedder.py                  ← calls nomic embedder container
│   ├── cache.py                     ← Redis read/write logic
│   ├── llm.py                       ← OpenAI / Anthropic / Ollama connector
│   ├── prompt.py                    ← prompt assembly, chunk selection
│   ├── requirements.txt
│   └── static/
│       └── index.js                 ← the single-file UI
│
├── worker/                          ← ingestion pipeline + weekly cron
│   ├── Dockerfile
│   ├── ingest.py                    ← full pipeline (steps 1–7)
│   ├── update.py                    ← weekly diff + upsert/delete
│   ├── download.py                  ← streaming Cirrus JSON downloader
│   ├── parse.py                     ← filter, parse, clean, chunk
│   ├── embed.py                     ← batch embed + upsert to Qdrant
│   ├── scheduler.py                 ← cron entry point (Monday 03:00)
│   └── requirements.txt
│
├── qdrant/                          ← no Dockerfile (official image)
│   └── config.yaml                  ← optional: HNSW params, collection config
│
├── redis/                           ← no Dockerfile (official image)
│   └── redis.conf                   ← optional: maxmemory, eviction policy
│
└── embedder/                        ← no Dockerfile (ollama/ollama)
    └── pull_model.sh                ← entrypoint: ollama pull nomic-embed-text-v1.5
```

---

## Data Flow 1 — Wiki Ingestion Pipeline

Runs once on first boot to build the full Qdrant index from scratch.
Each step is a discrete stage inside the `worker` container.

| # | Step | Detail |
|---|------|--------|
| 1 | Download Cirrus JSON dump | HTTP stream from `dumps.wikimedia.org` · ~22 GB compressed · line-by-line |
| 2 | Filter articles | Keep top 15% by pageview rank · discard stubs < 300 tokens · ~1M articles remain |
| 3 | Parse article fields | Extract: `title` · section texts · `opening_text` · `timestamp` (last_modified) |
| 4 | Clean text | Strip `[N]` citation markers, table artefacts, HTML entities · normalise whitespace · drop sections < 50 tok |
| 5 | Chunk by section | ≤600 tok → 1 chunk as-is · >600 tok → split at paragraph boundaries with 50-tok overlap · <50 tok → merge with previous |
| 6 | Attach metadata | Stamp each chunk: `title` · `section` · `url` · `last_modified` · `pageview_rank` · `chunk_index` |
| 7 | Batch embed chunks | `nomic-embed-text-v1.5` · batch size 64 · mean-pool + L2-norm → `float32[768]` |
| 8 | Upsert into Qdrant | `PointStruct(id=uuid5(title+section+idx), vector=float32[768], payload=metadata)` · deterministic IDs enable clean re-upsert · ~15–20 GB final index |

**Output chunk schema:**
```json
{
  "text": "Einstein was born on 14 March 1879...",
  "title": "Albert Einstein",
  "section": "Early life",
  "url": "https://en.wikipedia.org/wiki/Albert_Einstein",
  "last_modified": "2024-11-03",
  "pageview_rank": 38,
  "chunk_index": 2
}
```

**Timing estimate:**
- GPU (e.g. RTX 3080): ~4–6 h total
- CPU only: ~2–3 days
- Embedding (step 7) dominates: ~78k batches at 15ms/batch on GPU

---

## Data Flow 2 — Weekly Update Worker

Triggered by cron every Monday 03:00. Re-downloads the latest Cirrus dump and applies only the
delta to Qdrant. Typical run: 30–60 min vs 4–6 h for a full re-index.

| # | Step | Detail |
|---|------|--------|
| 1 | Download new Cirrus dump | Same source · fresh weekly snapshot published by Wikimedia |
| 2 | Compute timestamp diff | Compare `last_modified` in new dump vs stored in Qdrant payload · batch-scroll Qdrant by title |
| 2a | Changed/new → pre-process | Run steps 3–6 from ingestion pipeline (parse · clean · chunk · metadata) |
| 2b | Changed/new → embed & upsert | Re-embed all chunks · upsert to Qdrant (same deterministic ID overwrites old vector) |
| 2c | Deleted → remove | Query Qdrant for all points where `payload.title == deleted_title` · issue `delete_points` call |
| 3 | Updated Qdrant index | ~50–150k articles changed per week (~5–15% of index) · flush Redis answer cache after update |

**Why timestamp diff, not content hash:** timestamps are already in the Cirrus data at zero extra
cost. Content hashing would require downloading and hashing all 1M articles.

**Cache flush:** after index update, flush Redis answer cache (or reduce TTL to 0) to prevent
stale answers from being served for recently-changed articles.

---

## Data Flow 3 — User Request Flow

Every incoming question follows this path inside the `api` container.
On a cache hit at step 2, steps 3–10 are skipped entirely (~2ms response).

| # | Step | Detail |
|---|------|--------|
| 1 | User request | `HTTP POST { question, model, api_key }` · UI sends to FastAPI `/query` endpoint |
| 2 | Redis cache check | Key = `SHA256(normalised_question + model)` · HIT → return cached answer · MISS → continue |
| 3 | Tokenise query | BPE tokeniser · microseconds · auto-handled by model library |
| 4 | Embed query | `nomic-embed-text-v1.5` forward pass → mean-pool → L2-norm → `float32[768]` · ~10–30ms CPU |
| 5 | Qdrant HNSW search | Cosine similarity · filter `last_modified > cutoff` · k=10 · ~5–20ms |
| 6 | Cache chunk results | Write top-10 results JSON to Redis · key = `chunks:{question_hash}` · TTL 24h |
| 7 | Select top 5 chunks | Trim k=10 → 5 · each chunk carries: `text` · `title` · `section` · `url` · cosine score |
| 8 | Assemble prompt | `"Answer using ONLY the context below. Cite title + section per claim."` + 5 chunks + question · ~2500 tok |
| 9 | LLM API call | POST to OpenAI / Anthropic / Ollama with user API key in `Authorization` header · ~500ms–4s |
| 10 | Cache answer | Write `{ answer, sources }` to Redis · key = `answer:{question_hash}:{model}` · TTL 6h |
| 11 | Response to user | `JSON { answer, sources: [{ title, section, url }] }` · UI renders answer + clickable source links |

**End-to-end latency budget (cache miss):**

| Step | Typical latency |
|------|----------------|
| Redis check | ~2 ms |
| Tokenise + embed query | ~20 ms |
| Qdrant HNSW search | ~10 ms |
| Prompt assembly | ~1 ms |
| LLM API call | ~1–3 s ← dominates |
| Cache write | ~2 ms |
| **Total** | **~1.1–3.1 s** |

---

## UI Specification

- Single JS file served as a static asset by the `api` container at `GET /`
- **Components:** question input field · model selector dropdown · API key input (settings panel)
  · answer display area · source list (title + section + Wikipedia URL per chunk)
- No multi-turn / chat history for MVP — single Q&A per submit
- API key lives in browser session memory only — never sent to server storage, passed as a header
  per request

---

## Build Order & Implementation Roadmap

Phases must be completed in dependency order. Each phase is independently testable.

```
Phase 1: Infrastructure  →  Phase 2: Ingestion Worker  →  Phase 3: Query API
                                                                ↓
                                            Phase 5: UI  ←  Phase 4: Weekly Update Worker
```

### Phase 1 — Infrastructure (docker-compose)

**Goal:** All three shared services running and reachable before writing a line of app code.

1. Add `qdrant/qdrant`, `redis:alpine`, and `ollama/ollama` services to `docker-compose.yml`
2. Configure healthcheck startup order: Qdrant → Redis → Embedder
3. Create `qdrant_storage` named volume (~15–20 GB, persists across restarts)
4. On embedder start: pull `nomic-embed-text-v1.5` model automatically
5. Smoke-test each service: Qdrant REST health endpoint, Redis `PING`, embedder `/api/tags`

### Phase 2 — Ingestion Worker

**Goal:** Build the full Qdrant index from scratch inside the `worker` container.

1. Streaming Cirrus JSON downloader — line-by-line to keep memory flat (~22 GB compressed)
2. Article filter: keep top 15% by pageview rank, discard stubs < 300 tokens
3. Parse fields: `title` · section texts · `opening_text` · `last_modified` timestamp
4. Text cleaner: strip `[N]` citation markers, HTML entities, table artefacts, normalise whitespace
5. Section chunker: ≤600 tok as-is · >600 tok split at paragraph boundary with 50-tok overlap · <50 tok merge with previous
6. Attach metadata + generate deterministic `uuid5(title+section+chunk_index)` point IDs
7. Batch embed (batch=64) via nomic embedder → upsert `PointStruct`s to Qdrant · verify with point count

=> PROMPT FOR CLAUDE CODE:
"Read CLAUDE.md and implement Phase 2 — the ingestion worker. Use the exact parameters from CLAUDE.md throughout:
1. worker/download.py — streaming Cirrus JSON downloader from dumps.wikimedia.org, line-by-line to keep memory flat, ~22 GB compressed input
2. worker/parse.py — filter: keep top 15% by pageview rank, discard stubs < 300 tokens. Parse: extract title, section texts, opening_text, last_modified timestamp. Clean: strip [N] citation markers, HTML entities, table artefacts, normalise whitespace, drop sections < 50 tokens. Chunk: ≤600 tok → 1 chunk as-is · >600 tok → split at paragraph boundaries with 50-tok overlap · <50 tok → merge with previous section
3. worker/embed.py — batch embed via nomic embedder container at http://embedder:11434, batch size 64, mean-pool + L2-norm → float32[768]. Generate deterministic point IDs using uuid5(title + section + chunk_index). Upsert PointStruct(id, vector, payload) to Qdrant. Payload must include: title, section, url, last_modified, pageview_rank, chunk_index
4. worker/ingest.py — wire all steps together, support a --limit N flag for testing on a small article slice (e.g. 10k articles) before the full run. Log progress at each stage.
5. worker/Dockerfile — Python 3.11 slim base, install from requirements.txt, set working directory to /app
6. Read all config (Qdrant host/port, embedder host/port, collection name) from environment variables defined in .env. Do not hardcode any values."

> **Tip:** Run on a 10k-article slice first to validate quality before the full ~4–6 h GPU run.

### Phase 3 — Query API (FastAPI)

**Goal:** Working `/query` endpoint with caching and LLM connector.

1. `POST /query` endpoint: accept `{ question, model, api_key }` — `api_key` passed as header, never stored
2. Redis cache check: key = `SHA256(normalised_question + model)` → HIT returns answer in ~2ms
3. Embed query via the shared nomic embedder container (same model = same vector space as chunks)
4. Qdrant HNSW search: k=10, optional `last_modified` filter; ~5–20ms at 5M+ vectors
5. Prompt assembly: top-5 chunks + `'answer ONLY using context below, cite title+section'` + question (~2500 tok)
6. LLM connector: OpenAI / Anthropic / Ollama — user BYO key passed in `Authorization` header
7. Cache chunk results (TTL 24h) + answer (TTL 6h) · return `JSON { answer, sources: [title, section, url] }`

> **Note:** The API is usable even while the worker is still indexing — partial results are
> returned from whatever is indexed so far.

### Phase 4 — Weekly Update Worker (cron)

**Goal:** Incremental index updates without a full re-index.

1. Re-download latest Cirrus dump every Monday 03:00
2. Compute timestamp diff: compare `last_modified` in new dump vs stored Qdrant payload (batch-scroll by title)
3. Changed/new articles: re-run ingestion steps 3–6, re-embed, upsert — deterministic IDs overwrite old vectors cleanly
4. Deleted articles: query Qdrant for all points where `payload.title == deleted_title`, issue `delete_points` call
5. Flush Redis answer cache immediately after index update to prevent stale answers being served

### Phase 5 — UI (single JS file, served by api container)

**Goal:** Minimal chat-style interface, no extra container.

1. Single static JS file served at `GET /` by the FastAPI `api` container
2. Components: question input · model selector dropdown · API key settings panel · answer display area
3. Source list: `title` + `section` + clickable Wikipedia URL per chunk returned with each answer
4. API key lives in browser session memory only — never sent to server storage, passed as header per request

---

## Key Technical Concepts

### Embeddings — nomic-embed-text-v1.5

BERT-style transformer: 12 layers, 12 attention heads × 64d = 768d.
Input text → BPE tokens → 12 layers of multi-head self-attention → mean-pool all token vectors
→ L2-normalise → `float32[768]`.

The 768-dimensional output vector encodes meaning as a direction in high-dimensional space.
Semantically similar text lands geometrically close. The `√64 = 8` scaling factor in
`softmax(QKᵀ/√d_k)` keeps dot-product variance at 1, preventing gradient vanishing.
`d_model=768`, `h=12 heads`, `d_k=64` follows the BERT-base configuration.

### Qdrant HNSW Search

HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest-neighbour index.
At query time: embed query → cosine similarity search → HNSW navigates ~300–500 candidate vectors
from 5M+ without brute-force comparison → returns top-k in ~5–20ms.
Accuracy: ~99% recall vs exact search. Each stored point = vector + payload (metadata dict).

### RAG Grounding Mechanism

The LLM never accesses Qdrant directly. The API retrieves the top-5 chunks, injects them as
context into the prompt with the instruction `'answer ONLY using the context below'`, then calls
the LLM. The LLM reads the provided paragraphs and synthesises an answer — it cannot invent facts
that contradict the context. Sources are returned alongside the answer for user verification.

### Deterministic Chunk IDs

Each chunk's Qdrant point ID = `uuid5(title + section + chunk_index)`. This makes upsert
idempotent: re-running ingestion on a changed article overwrites existing vectors in place rather
than creating duplicates. Critical for the weekly update worker.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Embedding phase takes days on CPU | Delays initial launch | Document GPU path clearly; allow partial index to serve queries while worker runs |
| Retrieval quality poor on keyword-heavy queries | Answers miss obvious chunks | Add hybrid search (BM25 + dense) in Qdrant v1.1 — one config change, no re-index |
| Wikipedia parsing noise | Garbage chunks pollute index | Robust cleaning regex in step 4; log chunks dropped for QA review |
| LLM call dominates latency (1–3s) | UI feels slow | Redis answer cache absorbs repeated queries; stream LLM response tokens to UI |
| Stale answers after weekly update | User sees outdated information | Flush Redis answer cache immediately after Qdrant update completes |
| Context window overflow on small local models | LLM truncates context silently | Dynamic context budget: detect model, cap chunks (3 chunks for 8k-context models) |
