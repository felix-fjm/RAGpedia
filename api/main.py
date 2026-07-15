"""
WikiRAG Query API.

Endpoints:
  GET  /health  → liveness check
  GET  /        → serve the single-page UI (static/index.js)
  POST /query   → full RAG pipeline (embed → search → prompt → LLM → cache)

Request flow for POST /query (cache miss):
  1. SHA-256 cache key from normalised question + model
  2. Redis answer-cache check  → HIT: return immediately
  3. Embed query via nomic-embed-text (shared embedder container)
  4. Qdrant HNSW search, k=10
  5. Cache raw chunk results in Redis (TTL 24 h)
  6. Assemble prompt from top-5 chunks
  7. LLM call (OpenAI / Anthropic / Ollama) with user's API key
  8. Cache answer in Redis (TTL 6 h)
  9. Return { answer, sources }
"""

import logging
import os
from pathlib import Path

import redis as redis_lib
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient

import cache as cache_mod
import embedder as embedder_mod
import llm as llm_mod
import prompt as prompt_mod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

QDRANT_HOST       = os.environ["QDRANT_HOST"]
QDRANT_PORT       = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.environ["QDRANT_COLLECTION"]

REDIS_HOST        = os.environ["REDIS_HOST"]
REDIS_PORT        = int(os.environ.get("REDIS_PORT", 6379))
CHUNK_CACHE_TTL   = int(os.environ.get("CHUNK_CACHE_TTL", 86400))
ANSWER_CACHE_TTL  = int(os.environ.get("ANSWER_CACHE_TTL", 21600))

EMBEDDER_HOST     = os.environ["EMBEDDER_HOST"]
EMBEDDER_PORT     = int(os.environ.get("EMBEDDER_PORT", 11434))
EMBED_MODEL       = os.environ["EMBED_MODEL"]

EMBEDDER_URL      = f"http://{EMBEDDER_HOST}:{EMBEDDER_PORT}"
RETRIEVE_K        = 10  # candidates retrieved from Qdrant
STATIC_DIR        = Path(__file__).parent / "static"

# ── App & shared clients ──────────────────────────────────────────────────────

app    = FastAPI(title="WikiRAG")
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
redis  = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    model: str


class Source(BaseModel):
    title: str
    section: str
    url: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    cached: bool = False

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def ui():
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    authorization: str | None = Header(default=None),
) -> QueryResponse:
    # Extract Bearer token if present
    api_key: str | None = None
    if authorization and authorization.lower().startswith("bearer "):
        api_key = authorization[7:].strip()

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")

    model = req.model.strip()
    if not model:
        raise HTTPException(status_code=422, detail="model must not be empty")

    try:
        llm_mod.validate_model(model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Step 1 — cache key
    q_hash = cache_mod.question_hash(question, model)

    # Step 2 — answer cache check
    cached = cache_mod.get_answer(redis, q_hash, model)
    if cached:
        logger.info("Cache HIT  [%s...]", q_hash[:8])
        return QueryResponse(answer=cached["answer"], sources=cached["sources"], cached=True)

    logger.info("Cache MISS [%s...] — running pipeline", q_hash[:8])

    # Step 3 — embed query
    try:
        query_vector = embedder_mod.embed_query(question, EMBEDDER_URL, EMBED_MODEL)
    except Exception as exc:
        logger.error("Embedder error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Embedder error: {exc}") from exc

    # Step 4 — Qdrant HNSW search
    hits = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=RETRIEVE_K,
    ).points

    if not hits:
        return QueryResponse(
            answer="No relevant articles were found in the index for your question.",
            sources=[],
        )

    chunks = [
        {
            "text":          hit.payload.get("text", ""),
            "title":         hit.payload.get("title", ""),
            "section":       hit.payload.get("section", ""),
            "url":           hit.payload.get("url", ""),
            "last_modified": hit.payload.get("last_modified", ""),
            "score":         hit.score,
        }
        for hit in hits
    ]

    # Step 5 — cache chunk results
    cache_mod.set_chunks(redis, q_hash, chunks, CHUNK_CACHE_TTL)

    # Step 6 — assemble prompt (trims to top-5 internally)
    final_prompt, sources = prompt_mod.assemble_prompt(question, chunks)

    # Step 7 — LLM call
    try:
        answer = llm_mod.complete(
            prompt=final_prompt,
            model=model,
            api_key=api_key,
            ollama_url=EMBEDDER_URL,
        )
    except Exception as exc:
        logger.error("LLM error: %s", exc)
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from exc

    # Step 8 — cache answer
    answer_data = {"answer": answer, "sources": sources}
    cache_mod.set_answer(redis, q_hash, model, answer_data, ANSWER_CACHE_TTL)

    # Step 9 — respond
    return QueryResponse(answer=answer, sources=sources)
