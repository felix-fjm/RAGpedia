# RAGpedia — Wikipedia-Grounded RAG System

> Ask any question. Get a sourced, factual answer drawn directly from English Wikipedia - with clickable citations for every claim.

RAGpedia is a fully self-hosted, Dockerised RAG (Retrieval-Augmented Generation) system that turns the English Wikipedia into a continuously updated knowledge base. It grounds any LLM in verified, sourced context - eliminating hallucination on factual questions. You bring your own API key; RAGpedia handles ingestion, embedding, retrieval, and prompt assembly.

---

## How It Works

```
Your Question
     │
     ▼
 Embed query          ← nomic-embed-text-v1.5 (self-hosted)
     │
     ▼
 Vector search        ← Qdrant HNSW index (~1M Wikipedia articles)
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
| Scope | Top 15% by pageview (~1M articles) |
| Embedding model | `nomic-embed-text-v1.5` · 768d · self-hosted via Ollama |
| Vector DB | Qdrant · HNSW index · cosine similarity |
| Retrieval | Dense cosine · k=10 retrieve, top 5 to LLM |
| Chunking | Section-aware · 400–600 tokens · 50-token overlap on splits |
| Section extraction | Two-path: wikitext `== Heading ==` parsing (PATH 1) or Introduction + Body fallback (PATH 2) |
| Metadata per chunk | `title` · `section` · `url` · `last_modified` · `pageview_rank` |
| Update cadence | Weekly · timestamp diff · re-embed changed articles only |
| Cache | Redis · chunks TTL 24h · answers TTL 6h |
| LLM providers | OpenAI · Anthropic · Ollama (BYO key) |
| Backend | FastAPI · async · separate `api` + `worker` containers |
| Auth | None for MVP · API key lives client-side only, never persisted |

---

## System Architecture

Five Docker containers. Two custom services (`api` + `worker`) share three official infrastructure containers.

```
┌──────────────────────────────────────────────────────┐
│                  Shared Infrastructure               │
│  ┌─────────────┐   ┌─────────────┐   ┌────────────┐  │
│  │   qdrant    │   │    redis    │   │  embedder  │  │
│  │ HNSW index  │   │ chunk cache │   │   ollama   │  │
│  │ ~15–20 GB   │   │ answer cache│   │ nomic-embed│  │
│  └──────┬──────┘   └──────┬──────┘   └─────┬──────┘  │
└─────────┼────────────────┼────────────────┼──────────┘
          │                │                │
    ┌─────▼─────┐    ┌─────▼───────────────▼──────┐
    │  worker   │    │            api             │
    │ ingestion │    │  FastAPI · /query · GET /  │
    │ cron job  │    │  LLM connector · prompt    │
    └───────────┘    └────────────────────────────┘
```

**Why separate `worker` and `api`?**
The worker is GPU-bound and runs for hours during ingestion. The API is I/O-bound and must stay responsive 24/7. Separating them means a re-index job never blocks user queries. Both share the same `embedder` container - critical because query vectors and chunk vectors must live in the same embedding space.

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
 ├── llm.py                  ← OpenAI / Anthropic / Ollama connector
 ├── prompt.py               ← top-5 chunk selection + prompt assembly
 ├── requirements.txt
 └── static/
      └── index.js            ← single-file UI

worker/                     ← ingestion pipeline + weekly cron
 ├── Dockerfile
 ├── ingest.py               ← full pipeline orchestrator
 ├── download.py             ← streaming Cirrus JSON downloader
 ├── parse.py                ← filter · parse · clean · chunk
 ├── embed.py                ← batch embed + upsert to Qdrant
 ├── update.py               ← weekly diff + upsert/delete
 ├── scheduler.py            ← cron entry point (Monday 03:00)
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
- **8 GB RAM minimum** (16 GB recommended for full ~1M article index)
- **20 GB free disk space** for the Qdrant vector index
- An API key from **OpenAI**, **Anthropic**, or a local **Ollama** model
- **GPU strongly recommended** for initial ingestion (CPU-only: ~2–3 days; GPU: ~4–6 hours)
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

**Medium validation (1,000 articles, ~3–4 hours on CPU):**
```bash
docker compose run --rm -e PYTHONUNBUFFERED=1 worker python ingest.py --limit 1000
```

**Full run (~1M articles, 4–6 hours on GPU / 2–3 days on CPU):**
```bash
docker compose run --rm -e PYTHONUNBUFFERED=1 worker python ingest.py
```

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

### 6. Ask your first question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR-API-KEY" \
  -d '{"question": "What caused the French Revolution?", "model": "gpt-4o-mini"}'
```

**Supported model strings:**
| Provider | Example model strings |
|----------|----------------------|
| OpenAI | `gpt-4o` · `gpt-4o-mini` · `o1` · `o3-mini` |
| Anthropic | `claude-opus-4-6` · `claude-sonnet-4-6` · `claude-haiku-4-5` |
| Ollama (local) | `llama3.2` · `mistral` · any model you have pulled |

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
| 1 | Download | HTTP stream from `dumps.wikimedia.org` · ~22 GB compressed · line-by-line |
| 2 | Filter | Keep top 15% by pageview rank · discard stubs < 300 tokens · ~1M articles |
| 3 | Parse | Extract `title` · sections · `opening_text` · `last_modified` timestamp |
| 4 | Clean | Strip citation markers `[N]` · wikitext markup · HTML tags · normalise whitespace |
| 5 | Section extraction | PATH 1: parse `source_text` wikitext `== Heading ==` markers · PATH 2: Introduction + Body fallback |
| 6 | Chunk | ≤600 tok → 1 chunk · >600 tok → split at paragraph boundary with 50-tok overlap · <50 tok → merge with previous |
| 7 | Embed | `nomic-embed-text-v1.5` · batch 64 · mean-pool + L2-norm → `float32[768]` |
| 8 | Upsert | `PointStruct(id=uuid5(title+section+idx), vector, payload)` → Qdrant |

**Deterministic chunk IDs** (`uuid5(title + section + chunk_index)`) make upserts idempotent - re-running ingestion on a changed article overwrites vectors in place without creating duplicates.

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
| Medium | 1,000 | ~24,000 | ~300 MB |
| Full | ~1,000,000 | ~20,000,000 | ~15–20 GB |

---

## Troubleshooting

**`400 Bad Request` from embedder**
A chunk exceeded the 8,192 BPE token context window. This is handled by `_truncate()` in `embed.py` (caps at 2,000 words) and the 600-token chunk limit in `parse.py`. If it still occurs, ensure `OLLAMA_NUM_CTX=8192` is set in your `docker-compose.yml` embedder environment.

**WSL2 `SIGBUS` crash during Docker build**
Your system ran out of RAM. Fix: `wsl --shutdown` from PowerShell, then add a memory cap:
```ini
# %USERPROFILE%\.wslconfig
[wsl2]
memory=4GB
swap=4GB
```
Then rebuild with `DOCKER_BUILDKIT=0 docker compose build --no-cache worker`.

**`points_count: 0` after ingestion**
The worker container ran with a stale image. Always rebuild before ingesting: `docker compose build --no-cache worker`.

**Retrieval returns unrelated articles**
Your index is too small - with fewer than ~100 articles, cosine similarity has little to work with and returns the least-dissimilar chunks regardless of relevance. Run `--limit 1000` or higher for meaningful retrieval.

**`cached: true` returning stale answers**
Redis answer TTL is 6 hours. To flush immediately:
```bash
docker compose exec redis redis-cli FLUSHALL
```

---

## Weekly Update Worker

*(Phase 4 - coming soon)*

The update worker re-downloads the Cirrus dump every Monday at 03:00 and applies only the delta to Qdrant — typically 50–150k changed articles per week (~5–15% of the index). Full re-index: 4–6 hours. Weekly update: 30–60 minutes.

---

## Roadmap

- [x] Phase 1 — Infrastructure (Qdrant · Redis · Ollama embedder)
- [x] Phase 2 — Ingestion worker (download · filter · parse · clean · chunk · embed · upsert)
- [x] Phase 3 — Query API (FastAPI · Redis cache · LLM connector)
- [ ] Phase 4 — Weekly update worker (cron · timestamp diff · incremental upsert)
- [ ] Phase 5 — UI (single-page JS served by API container)

---

## Technical Notes

**Embeddings:** `nomic-embed-text-v1.5` is a BERT-style transformer (12 layers, 768d). Input text → BPE tokens → 12 layers of multi-head self-attention → mean-pool → L2-normalise → `float32[768]`. Semantically similar text lands geometrically close in this space; cosine similarity measures the angle between vectors.

**HNSW search:** Qdrant's Hierarchical Navigable Small World index navigates ~300–500 candidate vectors from millions without brute-force comparison — returning top-k results in ~5–20ms with ~99% recall vs exact search.

**RAG grounding:** The LLM never accesses Qdrant directly. The API retrieves top-5 chunks, injects them as context with the instruction `"answer ONLY using the context below"`, then calls the LLM. The model synthesises an answer from provided paragraphs - it cannot invent facts that contradict the context.

---

## License

MIT
