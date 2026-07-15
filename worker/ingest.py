"""
Full ingestion pipeline entry point.

Steps (full run):
  1. Download Cirrus JSON dump to local disk (skipped if file already exists)
  2. Pass 1 — stream dump once, collect only (popularity_score, title) pairs
  3. Compute popularity threshold; build title→rank mapping
  4. Connect to Qdrant; create collection if absent
  5. Pass 2 — stream dump again; for each article in rank_map: parse → chunk → embed → upsert

Steps (--limit smoke-test):
  1. Skipped (stream directly from URL)
  2. Stream N articles into memory, sort by score, assign ranks
  3. Skipped (all N articles kept)
  4. Connect to Qdrant
  5. Process article list

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

from download import download_dump, stream_articles, stream_articles_from_url
from embed import get_or_create_collection, upsert_chunks
from parse import compute_popularity_threshold_from_scores, count_tokens, process_article

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

    total_chunks = 0
    total_upserted = 0
    articles_processed = 0
    skipped_chunks: list[dict] = []

    if args.limit:
        # ── Limit mode: fast smoke-test, stream N articles directly from URL ──
        logger.info("Step 1/5  Skipped (limit mode — streaming directly from URL).")
        logger.info("Step 2/5  Streaming %d articles from URL...", args.limit)
        raw_articles = list(stream_articles_from_url(dump_url, limit=args.limit))
        logger.info("Loaded %d articles (limit mode).", len(raw_articles))

        # Rank within this sample by popularity_score; keep all (no top-% cut)
        raw_articles.sort(key=lambda a: a.get("popularity_score") or 0.0, reverse=True)
        article_rank_pairs = [
            (article, rank + 1)
            for rank, article in enumerate(raw_articles)
        ]

        logger.info(
            "Step 3/5  Skipped (limit mode — all %d articles ranked by score).",
            len(article_rank_pairs),
        )

        logger.info("Step 4/5  Connecting to Qdrant and ensuring collection exists...")
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
        get_or_create_collection(client, collection_name)

        logger.info("Step 5/5  Processing, embedding, and upserting chunks...")
        for article, rank in tqdm(article_rank_pairs, desc="Articles", unit="article"):
            chunks = process_article(article, pageview_rank=rank)
            if not chunks:
                continue
            total_chunks += len(chunks)
            upserted, skipped = upsert_chunks(
                chunks=chunks,
                client=client,
                collection_name=collection_name,
                embedder_url=embedder_url,
                model=embed_model,
                batch_size=batch_size,
            )
            total_upserted += upserted
            skipped_chunks.extend(skipped)
            articles_processed += 1

    else:
        # ── Full run: two-pass streaming — peak RAM stays under ~2 GB ─────────

        # Step 1: Download dump
        if os.path.exists(dump_path):
            logger.info("Dump already present at %s — skipping download.", dump_path)
        else:
            logger.info("Step 1/5  Downloading Cirrus dump...")
            download_dump(dump_url, dump_path)

        # Step 2: Pass 1 — collect (score, title) pairs only.
        # ~1 GB for 6.7M articles, versus ~15 GB when loading full article dicts.
        logger.info(
            "Step 2/5  Pass 1/2: scanning dump for popularity scores "
            "(title + score only)..."
        )
        score_pairs: list[tuple[float, str]] = []
        for article in tqdm(
            stream_articles(dump_path), desc="Pass 1 scoring", unit="articles"
        ):
            title = article.get("title") or ""
            score = article.get("popularity_score") or 0.0
            if title:
                score_pairs.append((score, title))
        logger.info("Pass 1 complete: %d articles scanned.", len(score_pairs))

        # Step 3: Compute threshold and build title→rank mapping.
        logger.info(
            "Step 3/5  Computing popularity threshold (top %.0f%%)...",
            top_fraction * 100,
        )
        _, rank_map = compute_popularity_threshold_from_scores(score_pairs, top_fraction)
        del score_pairs  # release ~1 GB before Pass 2

        # Step 4: Connect to Qdrant
        logger.info("Step 4/5  Connecting to Qdrant and ensuring collection exists...")
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
        get_or_create_collection(client, collection_name)

        # Step 5: Pass 2 — stream dump again; one article in memory at a time.
        logger.info(
            "Step 5/5  Pass 2/2: processing, embedding, and upserting filtered articles..."
        )
        articles_dropped_pop = 0
        articles_dropped_stub = 0

        for article in tqdm(
            stream_articles(dump_path), desc="Pass 2 processing", unit="articles"
        ):
            title = article.get("title") or ""
            if title not in rank_map:
                articles_dropped_pop += 1
                continue

            full_text = (
                (article.get("opening_text") or "")
                + " "
                + (article.get("text") or "")
            )
            if count_tokens(full_text) < min_tokens:
                articles_dropped_stub += 1
                continue

            rank = rank_map[title]
            chunks = process_article(article, pageview_rank=rank)
            if not chunks:
                continue

            total_chunks += len(chunks)
            upserted, skipped = upsert_chunks(
                chunks=chunks,
                client=client,
                collection_name=collection_name,
                embedder_url=embedder_url,
                model=embed_model,
                batch_size=batch_size,
            )
            total_upserted += upserted
            skipped_chunks.extend(skipped)
            articles_processed += 1

        logger.info(
            "Pass 2 results: %d processed, %d dropped (popularity), %d dropped (stub)",
            articles_processed,
            articles_dropped_pop,
            articles_dropped_stub,
        )

    # ── Verification ──────────────────────────────────────────────────────────
    collection_info = client.get_collection(collection_name)
    point_count = collection_info.points_count

    logger.info("=== Ingestion Complete ===")
    logger.info("Articles processed:  %d", articles_processed)
    logger.info("Chunks generated:    %d", total_chunks)
    logger.info("Points upserted:     %d", total_upserted)
    if skipped_chunks:
        skipped_titles = sorted({c["title"] for c in skipped_chunks})
        logger.warning(
            "%d chunk(s) skipped across %d article(s) — see WARNING/ERROR lines above for details. "
            "Affected articles: %s",
            len(skipped_chunks),
            len(skipped_titles),
            ", ".join(skipped_titles),
        )
    else:
        logger.info("Chunks skipped:      0")
    logger.info("Qdrant point count:  %d", point_count)


if __name__ == "__main__":
    main()
