"""
Full ingestion pipeline entry point.

Steps:
  1. Download Cirrus JSON dump to local disk (skipped if file already exists)
  2. First streaming pass: collect all articles into memory for popularity ranking
  3. Filter: top 15% by pageview rank, discard stubs < 300 tokens
  4. Process each article: extract sections → clean → chunk → attach metadata
  5. Batch embed chunks (batch=64) and upsert PointStructs to Qdrant

Usage:
  python ingest.py              # full run (~1M articles)
  python ingest.py --limit 10000  # smoke-test on first 10k articles
"""

import argparse
import logging
import os
import sys

from qdrant_client import QdrantClient
from tqdm import tqdm

from download import download_dump, stream_articles
from embed import get_or_create_collection, upsert_chunks
from parse import filter_articles, process_article

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest")


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        logger.error("Required environment variable '%s' is not set.", name)
        sys.exit(1)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="WikiRAG ingestion pipeline")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Stop after processing N articles (useful for smoke-testing).",
    )
    args = parser.parse_args()

    # ── Config from environment ───────────────────────────────────────────────
    dump_url        = _require_env("WIKI_DUMP_URL")
    qdrant_host     = _require_env("QDRANT_HOST")
    qdrant_port     = int(_require_env("QDRANT_PORT"))
    collection_name = _require_env("QDRANT_COLLECTION")
    embedder_host   = _require_env("EMBEDDER_HOST")
    embedder_port   = int(_require_env("EMBEDDER_PORT"))
    embed_model     = _require_env("EMBED_MODEL")
    top_fraction    = float(os.environ.get("PAGEVIEW_TOP_FRACTION", "0.15"))
    min_tokens      = int(os.environ.get("MIN_ARTICLE_TOKENS", "300"))
    batch_size      = int(os.environ.get("EMBED_BATCH_SIZE", "64"))
    dump_path       = os.environ.get("DUMP_PATH", "/data/wiki_dump.json.gz")

    embedder_url = f"http://{embedder_host}:{embedder_port}"

    logger.info("=== WikiRAG Ingestion Pipeline ===")
    logger.info("Qdrant:   %s:%d  collection=%s", qdrant_host, qdrant_port, collection_name)
    logger.info("Embedder: %s  model=%s", embedder_url, embed_model)
    logger.info("Dump:     %s → %s", dump_url, dump_path)
    if args.limit:
        logger.info("Mode:     SMOKE-TEST (limit=%d articles)", args.limit)
    else:
        logger.info("Mode:     FULL RUN (top %.0f%%)", top_fraction * 100)

    # ── Step 1: Download dump ─────────────────────────────────────────────────
    if os.path.exists(dump_path):
        logger.info("Dump already present at %s — skipping download.", dump_path)
    else:
        logger.info("Step 1/5  Downloading Cirrus dump...")
        download_dump(dump_url, dump_path)

    # ── Step 2: Load articles for ranking ────────────────────────────────────
    # For --limit runs we skip popularity ranking to keep smoke-tests fast:
    # just assign rank=0 to all articles and apply only the stub filter.
    logger.info("Step 2/5  Loading articles from dump...")

    if args.limit:
        # Fast path: stream directly, no popularity pre-scan needed
        raw_articles = list(stream_articles(dump_path, limit=args.limit))
        logger.info("Loaded %d articles (limit mode).", len(raw_articles))

        # In limit mode: keep all articles that pass the stub filter; rank by
        # popularity_score position (higher score = lower rank number).
        raw_articles.sort(key=lambda a: a.get("popularity_score") or 0.0, reverse=True)
        article_rank_pairs = [
            (article, rank + 1)
            for rank, article in enumerate(raw_articles)
        ]
    else:
        # Full run: load all articles (lightweight objects) for popularity ranking.
        # ~6.7M articles × ~200 bytes avg ≈ ~1.3 GB RAM for the index scan.
        logger.info(
            "Loading all articles for popularity ranking "
            "(this reads the full dump once; ~1–2 GB RAM)..."
        )
        raw_articles = list(
            tqdm(stream_articles(dump_path), desc="Loading", unit="articles")
        )
        logger.info("Loaded %d raw articles.", len(raw_articles))

        logger.info(
            "Step 3/5  Filtering: top %.0f%% by pageview rank, min %d tokens...",
            top_fraction * 100,
            min_tokens,
        )
        article_rank_pairs = list(
            filter_articles(raw_articles, top_fraction, min_tokens)
        )
        logger.info("Kept %d articles after filtering.", len(article_rank_pairs))

    # ── Step 3 (limit mode) / Step 4: Connect to Qdrant, create collection ───
    logger.info("Step 4/5  Connecting to Qdrant and ensuring collection exists...")
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    get_or_create_collection(client, collection_name)

    # ── Step 4/5: Process articles → embed → upsert ───────────────────────────
    logger.info("Step 5/5  Processing, embedding, and upserting chunks...")

    total_chunks = 0
    total_upserted = 0

    for article, rank in tqdm(article_rank_pairs, desc="Articles", unit="article"):
        chunks = process_article(article, pageview_rank=rank)
        if not chunks:
            continue

        total_chunks += len(chunks)
        upserted = upsert_chunks(
            chunks=chunks,
            client=client,
            collection_name=collection_name,
            embedder_url=embedder_url,
            model=embed_model,
            batch_size=batch_size,
        )
        total_upserted += upserted

    # ── Verification ──────────────────────────────────────────────────────────
    collection_info = client.get_collection(collection_name)
    point_count = collection_info.points_count

    logger.info("=== Ingestion Complete ===")
    logger.info("Articles processed:  %d", len(article_rank_pairs))
    logger.info("Chunks generated:    %d", total_chunks)
    logger.info("Points upserted:     %d", total_upserted)
    logger.info("Qdrant point count:  %d", point_count)


if __name__ == "__main__":
    main()
