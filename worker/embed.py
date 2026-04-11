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

def embed_texts(
    texts: list[str],
    embedder_url: str,
    model: str,
) -> np.ndarray:
    """
    Embed a batch of texts via Ollama /api/embed.

    Returns float32 array of shape (len(texts), 768), L2-normalised.
    Raises httpx.HTTPStatusError on API errors.
    """
    response = httpx.post(
        f"{embedder_url}/api/embed",
        json={"model": model, "input": texts},
        timeout=120.0,
    )
    response.raise_for_status()

    embeddings = response.json()["embeddings"]  # list[list[float]]
    vectors = np.array(embeddings, dtype=np.float32)

    # L2-normalise each vector (Ollama may already do this, but it's idempotent)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
    vectors = vectors / norms

    return vectors


# ── Upsert pipeline ───────────────────────────────────────────────────────────

def upsert_chunks(
    chunks: list[dict],
    client: QdrantClient,
    collection_name: str,
    embedder_url: str,
    model: str,
    batch_size: int = 64,
) -> int:
    """
    Embed `chunks` in batches and upsert PointStructs to Qdrant.

    Each chunk dict must contain: text, title, section, url,
    last_modified, pageview_rank, chunk_index.

    Returns the total number of points upserted.
    """
    total_upserted = 0

    for batch_start in tqdm(
        range(0, len(chunks), batch_size),
        desc="Embedding & upserting",
        unit="batch",
        leave=False,
    ):
        batch = chunks[batch_start : batch_start + batch_size]
        texts = [c["text"] for c in batch]

        vectors = embed_texts(texts, embedder_url, model)

        points = [
            PointStruct(
                id=make_point_id(c["title"], c["section"], c["chunk_index"]),
                vector=vectors[i].tolist(),
                payload={
                    "title": c["title"],
                    "section": c["section"],
                    "url": c["url"],
                    "last_modified": c["last_modified"],
                    "pageview_rank": c["pageview_rank"],
                    "chunk_index": c["chunk_index"],
                    "text": c["text"],
                },
            )
            for i, c in enumerate(batch)
        ]

        client.upsert(collection_name=collection_name, points=points)
        total_upserted += len(points)

    return total_upserted
