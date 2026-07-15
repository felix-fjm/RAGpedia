"""
Batch embedding via the Ollama nomic-embed-text container and upsert to Qdrant.

Point IDs are deterministic: uuid5(NAMESPACE_DNS, "title|section|chunk_index")
so re-running ingestion on a changed article overwrites existing vectors in
place rather than creating duplicates.
"""

import logging
import uuid

import httpx
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

logger = logging.getLogger(__name__)

VECTOR_SIZE = 768


# ── Qdrant collection ─────────────────────────────────────────────────────────

def get_or_create_collection(client: QdrantClient, collection_name: str) -> None:
    """Create the Qdrant collection if it does not already exist."""
    try:
        client.get_collection(collection_name)
        logger.info("Collection '%s' already exists — skipping creation.", collection_name)
    except (UnexpectedResponse, Exception):
        logger.info("Creating collection '%s'.", collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


# ── Point ID ──────────────────────────────────────────────────────────────────

def make_point_id(title: str, section: str, chunk_index: int) -> str:
    """Deterministic UUID5 from title + section + chunk_index."""
    key = f"{title}|{section}|{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


# ── Embedding ─────────────────────────────────────────────────────────────────

_MAX_WORDS = 1200   # nomic context is 8192 BPE tokens; wikitext markup can push BPE/word ratio well above 2×,
                    # so 1200 words ≈ ~2400 BPE tokens — comfortably under the limit even for markup-heavy articles
_RETRY_WORDS = 600  # emergency fallback if a 1200-word chunk still triggers a 400


def _truncate(text: str, max_words: int, title: str | None = None) -> str:
    """Truncate text to max_words whitespace-separated words, logging a WARNING if applied."""
    words = text.split()
    if len(words) <= max_words:
        return text
    label = f"'{title}'" if title else "(no title)"
    logger.warning(
        "Truncating chunk %s from %d words to %d words before embedding.",
        label, len(words), max_words,
    )
    return " ".join(words[:max_words])


def embed_texts(
    texts: list[str],
    embedder_url: str,
    model: str,
    titles: list[str] | None = None,
    sections: list[str] | None = None,
    chunk_indices: list[int] | None = None,
) -> tuple[np.ndarray, set[int]]:
    """
    Embed a list of texts via Ollama POST /api/embed, one request per text.

    On a 400 response, logs a WARNING with full chunk diagnostics and retries
    once at _RETRY_WORDS.  If that also returns 400, logs an ERROR and records
    the position in failed_indices (a zero-vector placeholder is inserted so
    the array shape stays consistent — callers must skip these positions).

    Returns:
        vectors        — float32 array shape (len(texts), 768), L2-normalised
        failed_indices — set of positions that could not be embedded
    """
    embeddings: list = []
    failed_indices: set[int] = set()

    for i, text in enumerate(texts):
        title        = titles[i]        if titles        else None
        section      = sections[i]      if sections      else None
        chunk_index  = chunk_indices[i] if chunk_indices else None

        payload = {"model": model, "input": _truncate(text, _MAX_WORDS, title)}
        if i == 0:
            logger.debug("embed_texts payload (first text): %s", payload)

        try:
            response = httpx.post(
                f"{embedder_url}/api/embed",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 400:
                raise

            # First 400 — log diagnostic, then retry at _RETRY_WORDS
            logger.warning(
                "400 from embedder — retrying at %d words. "
                "title=%r section=%r chunk_index=%r len(text)=%d "
                "text[:200]=%r … text[-200:]=%r",
                _RETRY_WORDS,
                title, section, chunk_index, len(text),
                text[:200], text[-200:],
            )
            payload["input"] = _truncate(text, _RETRY_WORDS, title)

            try:
                response = httpx.post(
                    f"{embedder_url}/api/embed",
                    json=payload,
                    timeout=120.0,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as retry_exc:
                if retry_exc.response.status_code != 400:
                    raise
                # Persistent 400 after retry — skip this chunk, do not crash
                logger.error(
                    "Persistent 400 after retry — skipping chunk. "
                    "title=%r section=%r chunk_index=%r len(text)=%d "
                    "text[:200]=%r … text[-200:]=%r",
                    title, section, chunk_index, len(text),
                    text[:200], text[-200:],
                )
                failed_indices.add(i)
                embeddings.append(np.zeros(VECTOR_SIZE, dtype=np.float32))
                continue

        embeddings.append(response.json()["embeddings"][0])

    vectors = np.array(embeddings, dtype=np.float32)

    # L2-normalise each vector (Ollama may already do this, but it's idempotent)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
    vectors = vectors / norms

    return vectors, failed_indices


# ── Upsert pipeline ───────────────────────────────────────────────────────────

def upsert_chunks(
    chunks: list[dict],
    client: QdrantClient,
    collection_name: str,
    embedder_url: str,
    model: str,
    batch_size: int = 64,
) -> tuple[int, list[dict]]:
    """
    Embed `chunks` in batches and upsert PointStructs to Qdrant.

    Each chunk dict must contain: text, title, section, url,
    last_modified, pageview_rank, chunk_index.

    Returns:
        (total_upserted, skipped_chunks)
        total_upserted — number of points successfully upserted
        skipped_chunks — list of {title, section, chunk_index} dicts for
                         chunks that could not be embedded after retrying
    """
    total_upserted = 0
    all_skipped: list[dict] = []

    for batch_start in tqdm(
        range(0, len(chunks), batch_size),
        desc="Embedding & upserting",
        unit="batch",
        leave=False,
    ):
        batch = chunks[batch_start : batch_start + batch_size]

        # Guard: log and skip any chunk whose text is empty or whitespace-only.
        # An empty string sent to /api/embed causes a 400 from Ollama.
        valid_batch = []
        for c in batch:
            if not c.get("text", "").strip():
                logger.warning(
                    "Skipping empty-text chunk — title=%r section=%r chunk_index=%d",
                    c.get("title", ""),
                    c.get("section", ""),
                    c.get("chunk_index", -1),
                )
            else:
                valid_batch.append(c)

        if not valid_batch:
            continue

        batch = valid_batch
        texts         = [c["text"]        for c in batch]
        titles        = [c["title"]       for c in batch]
        sections      = [c["section"]     for c in batch]
        chunk_idx_list = [c["chunk_index"] for c in batch]

        vectors, failed_indices = embed_texts(
            texts, embedder_url, model,
            titles=titles,
            sections=sections,
            chunk_indices=chunk_idx_list,
        )

        points = []
        for i, c in enumerate(batch):
            if i in failed_indices:
                all_skipped.append({
                    "title":       c["title"],
                    "section":     c["section"],
                    "chunk_index": c["chunk_index"],
                })
                continue
            points.append(
                PointStruct(
                    id=make_point_id(c["title"], c["section"], c["chunk_index"]),
                    vector=vectors[i].tolist(),
                    payload={
                        "title":        c["title"],
                        "section":      c["section"],
                        "url":          c["url"],
                        "last_modified": c["last_modified"],
                        "pageview_rank": c["pageview_rank"],
                        "chunk_index":  c["chunk_index"],
                        "text":         c["text"],
                    },
                )
            )

        if points:
            client.upsert(collection_name=collection_name, points=points)
        total_upserted += len(points)

    return total_upserted, all_skipped
