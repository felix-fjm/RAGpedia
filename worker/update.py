"""
Incremental update worker: diff new Cirrus dump against current Qdrant index,
re-embed changed/new articles, delete removed articles, flush Redis.

Steps (normal run)
------------------
1. Load current Qdrant state  →  {title: last_modified} for all stored articles
2. Ensure local dump exists (download if absent); Pass 1 scans (score, title,
   last_modified) for every article to compute the top-15% set
3. Compute diff: changed / added / deleted article sets
4. Pass 2 → re-parse, re-embed, upsert changed + added articles;
   deterministic uuid5 IDs overwrite existing vectors — no duplicates
5. Delete all Qdrant points for removed articles
6. Flush the Redis answer + chunk caches

Steps (--force-update TITLE [TITLE ...])
-----------------------------------------
Targeted update for specific articles — no local dump required.
Streams from URL, finds named titles, diffs against Qdrant state, upserts
only what has changed.  No Pass 1 popularity scan, no deletions.
Designed for testing and manual corrections.

Usage
-----
  python update.py                                         # uses DUMP_PATH env var
  python update.py --dump-path /data/wiki_dump.json.gz    # explicit local file
  python update.py --force-update "Albert Einstein" "Python (programming language)"

NOTE: This is a portfolio repository.  This script is not deployed or scheduled
anywhere.  It is designed as a correct, manually invokable one-shot command.
See worker/scheduler.py for the intended production cron entry point.
"""

import argparse
import logging
import os
import sys

import redis as redis_lib
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue
from tqdm import tqdm

from download import download_dump, stream_articles, stream_articles_from_url
from embed import upsert_chunks
from parse import (
    compute_popularity_threshold_from_scores,
    count_tokens,
    process_article,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("update")


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        logger.error("Required environment variable '%s' is not set.", name)
        sys.exit(1)
    return value


# ── Step 1: load current Qdrant state ────────────────────────────────────────

def load_qdrant_state(
    client: QdrantClient,
    collection_name: str,
    title_filter: list[str] | None = None,
) -> dict[str, str]:
    """
    Return {title: last_modified} for stored articles.

    If title_filter is provided, only fetch state for those specific titles
    (one filtered scroll per title) instead of scanning the whole collection.
    """
    state: dict[str, str] = {}

    if title_filter:
        logger.info(
            "Loading Qdrant state for %d specific title(s)…", len(title_filter)
        )
        for title in title_filter:
            filt = Filter(
                must=[FieldCondition(key="title", match=MatchValue(value=title))]
            )
            offset = None
            while True:
                points, offset = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filt,
                    limit=100,
                    offset=offset,
                    with_payload=["title", "last_modified"],
                    with_vectors=False,
                )
                for point in points:
                    t  = point.payload.get("title") or ""
                    lm = point.payload.get("last_modified") or ""
                    if t and t not in state:
                        state[t] = lm
                if offset is None:
                    break
        logger.info("Qdrant state loaded: %d title(s) found.", len(state))
        return state

    logger.info("Loading current Qdrant state (title → last_modified)…")
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_payload=["title", "last_modified"],
            with_vectors=False,
        )
        for point in points:
            title = point.payload.get("title") or ""
            lm    = point.payload.get("last_modified") or ""
            if title and title not in state:
                state[title] = lm
        if offset is None:
            break

    logger.info("Qdrant state loaded: %d distinct articles.", len(state))
    return state


# ── Step 2: build dump state (Pass 1) ────────────────────────────────────────

def build_dump_state(
    dump_path: str,
    top_fraction: float,
) -> tuple[dict[str, str], dict[str, int]]:
    """
    Pass 1 of the local dump: collect (popularity_score, title, last_modified)
    for every article with a non-zero score.  Peak RAM ≈ 1 GB for the full
    6.7 M-article dump (only scalar fields are held, not full article bodies).

    Returns:
        last_modified_map — {title: last_modified} for the top-fraction set
        rank_map          — {title: rank (1 = most popular)} for the same set
    """
    logger.info(
        "Pass 1/2: scanning dump for popularity scores and timestamps "
        "(top %.0f%% filter will be applied)…",
        top_fraction * 100,
    )

    score_triples: list[tuple[float, str, str]] = []
    for article in tqdm(stream_articles(dump_path), desc="Pass 1 scan", unit="art"):
        title = article.get("title") or ""
        score = article.get("popularity_score") or 0.0
        lm    = (article.get("timestamp") or "")[:10]
        if title and score > 0:
            score_triples.append((score, title, lm))

    logger.info("Pass 1 complete: %d articles with non-zero scores.", len(score_triples))

    score_pairs = [(s, t) for s, t, _ in score_triples]
    _, rank_map = compute_popularity_threshold_from_scores(score_pairs, top_fraction)

    last_modified_map: dict[str, str] = {
        t: lm for _, t, lm in score_triples if t in rank_map
    }
    del score_triples

    logger.info(
        "Dump state built: %d articles in top %.0f%%.",
        len(rank_map), top_fraction * 100,
    )
    return last_modified_map, rank_map


# ── Step 3: compute diff ──────────────────────────────────────────────────────

def compute_diff(
    qdrant_state: dict[str, str],
    dump_last_modified: dict[str, str],
) -> tuple[set[str], set[str], set[str]]:
    """
    Returns (changed, added, deleted):

      changed — title in both; dump last_modified differs from Qdrant value
      added   — title in top-15% dump selection; absent from Qdrant
      deleted — title in Qdrant; absent from new dump top-15% selection
                (covers both Wikipedia deletions and popularity rank drops)
    """
    dump_titles   = set(dump_last_modified)
    qdrant_titles = set(qdrant_state)

    added   = dump_titles - qdrant_titles
    deleted = qdrant_titles - dump_titles
    changed = {
        t for t in dump_titles & qdrant_titles
        if dump_last_modified[t] != qdrant_state[t]
    }

    logger.info(
        "Diff:  %d changed  |  %d added  |  %d deleted",
        len(changed), len(added), len(deleted),
    )
    return changed, added, deleted


# ── Step 4: re-embed and upsert changed + added articles ─────────────────────

def process_upserts(
    to_update: set[str],
    dump_path: str,
    rank_map: dict[str, int],
    client: QdrantClient,
    collection_name: str,
    embedder_url: str,
    embed_model: str,
    batch_size: int,
    min_tokens: int,
) -> tuple[int, list[dict]]:
    """
    Pass 2: stream the dump and process only articles in to_update.

    Deterministic uuid5 point IDs (title|section|chunk_index) mean that
    upserting a changed article overwrites its existing vectors cleanly;
    articles with a different chunk count after re-chunking will have stale
    extra points leftover (acceptable for a weekly cadence where counts rarely
    shrink significantly).

    Returns (total_points_upserted, skipped_chunks).
    """
    if not to_update:
        logger.info("No articles to upsert — skipping Pass 2.")
        return 0, []

    logger.info("Pass 2/2: re-embedding %d changed/new articles…", len(to_update))
    total_upserted = 0
    all_skipped: list[dict] = []
    processed = dropped_stub = 0

    for article in tqdm(
        stream_articles(dump_path), desc="Pass 2 upsert", unit="art"
    ):
        title = article.get("title") or ""
        if title not in to_update:
            continue

        full_text = (
            (article.get("opening_text") or "")
            + " "
            + (article.get("text") or "")
        )
        if count_tokens(full_text) < min_tokens:
            dropped_stub += 1
            continue

        chunks = process_article(article, pageview_rank=rank_map[title])
        if not chunks:
            continue

        upserted, skipped = upsert_chunks(
            chunks=chunks,
            client=client,
            collection_name=collection_name,
            embedder_url=embedder_url,
            model=embed_model,
            batch_size=batch_size,
        )
        total_upserted += upserted
        all_skipped.extend(skipped)
        processed += 1

    logger.info(
        "Pass 2 complete: %d articles upserted  |  %d dropped (stub).",
        processed, dropped_stub,
    )
    return total_upserted, all_skipped


# ── Step 5: delete points for removed articles ────────────────────────────────

def process_deletions(
    deleted_titles: set[str],
    client: QdrantClient,
    collection_name: str,
) -> int:
    """
    Delete every Qdrant point whose payload.title matches a title in
    deleted_titles.  Counts before deleting so the summary can report
    exact point counts rather than just article counts.

    Returns total points deleted.
    """
    if not deleted_titles:
        logger.info("No articles to delete.")
        return 0

    logger.info("Deleting %d articles from Qdrant…", len(deleted_titles))
    total_deleted = 0

    for title in tqdm(deleted_titles, desc="Deleting", unit="article"):
        filt = Filter(
            must=[FieldCondition(key="title", match=MatchValue(value=title))]
        )
        n = client.count(
            collection_name=collection_name,
            count_filter=filt,
            exact=True,
        ).count
        if n > 0:
            client.delete(
                collection_name=collection_name,
                points_selector=FilterSelector(filter=filt),
            )
            total_deleted += n

    logger.info(
        "Deleted %d points across %d articles.", total_deleted, len(deleted_titles)
    )
    return total_deleted


# ── Step 6: flush Redis caches ────────────────────────────────────────────────

def flush_redis_cache(redis_host: str, redis_port: int) -> None:
    """
    Flush the entire Redis DB (answer cache + chunk cache).
    Warns rather than crashing if Redis is unreachable — index updates should
    not be rolled back just because the cache server is temporarily down.
    """
    try:
        r = redis_lib.Redis(
            host=redis_host, port=redis_port, socket_connect_timeout=5
        )
        r.ping()
        r.flushdb()
        logger.info("Redis cache flushed (FLUSHDB).")
    except redis_lib.exceptions.ConnectionError as exc:
        logger.warning("Redis unreachable — cache not flushed: %s", exc)


# ── Force-update: stream from URL, no local dump required ────────────────────

def run_force_update(
    titles: list[str],
    dump_url: str,
    client: QdrantClient,
    collection_name: str,
    embedder_url: str,
    embed_model: str,
    batch_size: int,
    min_tokens: int,
    redis_host: str,
    redis_port: int,
) -> None:
    """
    Targeted update for specific articles — streams from URL, no local dump.

    Algorithm:
      1. Load Qdrant state for named titles only (filtered scroll, fast).
      2. Stream the Cirrus dump URL; collect article dicts for named titles,
         stopping as soon as all are found.
      3. For each found article: compare dump last_modified vs Qdrant value.
         - If absent or changed → upsert (rank=1 for all force-updated titles).
         - If identical → skip (already up to date).
      4. Flush Redis cache.

    No Pass 1 popularity scan, no deletions — this is a targeted correction,
    not a full diff.
    """
    logger.info("=== Force-Update Mode: %d title(s) ===", len(titles))
    for t in titles:
        logger.info("  · %s", t)

    # Step 1 — load Qdrant state for named titles only
    qdrant_state = load_qdrant_state(client, collection_name, title_filter=titles)

    # Step 2 — stream URL to find named articles; stop early when all found
    remaining = set(titles)
    found_articles: list[dict] = []

    logger.info("Streaming Cirrus dump from URL to locate named titles…")
    for article in stream_articles_from_url(dump_url):
        title = article.get("title") or ""
        if title in remaining:
            found_articles.append(article)
            remaining.discard(title)
            logger.info("  Found [%d/%d]: %s", len(titles) - len(remaining), len(titles), title)
        if not remaining:
            logger.info("All %d title(s) located — stopping stream early.", len(titles))
            break

    if remaining:
        logger.warning(
            "%d title(s) not found in dump (may be redirects or non-article pages): %s",
            len(remaining), sorted(remaining),
        )

    # Step 3 — diff and upsert
    total_upserted = 0
    all_skipped: list[dict] = []
    skipped_uptodate = skipped_stub = processed = 0

    for article in found_articles:
        title = article.get("title") or ""
        dump_lm = (article.get("timestamp") or "")[:10]
        qdrant_lm = qdrant_state.get(title)

        if qdrant_lm == dump_lm:
            logger.info("SKIP (up to date): %s  last_modified=%s", title, dump_lm)
            skipped_uptodate += 1
            continue

        action = "UPDATE" if qdrant_lm else "ADD"
        logger.info(
            "%s: %s  dump=%s  qdrant=%s",
            action, title, dump_lm, qdrant_lm or "absent",
        )

        full_text = (
            (article.get("opening_text") or "")
            + " "
            + (article.get("text") or "")
        )
        if count_tokens(full_text) < min_tokens:
            logger.warning("SKIP (stub — too short): %s", title)
            skipped_stub += 1
            continue

        chunks = process_article(article, pageview_rank=1)
        if not chunks:
            continue

        upserted, skipped = upsert_chunks(
            chunks=chunks,
            client=client,
            collection_name=collection_name,
            embedder_url=embedder_url,
            model=embed_model,
            batch_size=batch_size,
        )
        total_upserted += upserted
        all_skipped.extend(skipped)
        processed += 1

    # Step 4 — flush Redis
    flush_redis_cache(redis_host, redis_port)

    # Summary
    final_count = client.get_collection(collection_name).points_count
    logger.info("=== Force-Update Complete ===")
    logger.info("Titles requested:    %d", len(titles))
    logger.info("Titles found:        %d", len(found_articles))
    logger.info("Articles upserted:   %d", processed)
    logger.info("Articles up to date: %d", skipped_uptodate)
    logger.info("Articles skipped (stub): %d", skipped_stub)
    logger.info("Points upserted:     %d", total_upserted)
    logger.info("Qdrant point count:  %d", final_count)

    if all_skipped:
        skipped_titles = sorted({c["title"] for c in all_skipped})
        logger.warning(
            "%d chunk(s) skipped (persistent 400) across %d article(s): %s",
            len(all_skipped), len(skipped_titles), ", ".join(skipped_titles),
        )
    else:
        logger.info("Chunks skipped:      0")


# ── Orchestrator (callable from scheduler.py) ─────────────────────────────────

def run_update(
    dump_path_override: str | None = None,
    force_update_titles: list[str] | None = None,
) -> None:
    """Run the full incremental update pipeline (or a targeted force-update)."""

    # ── Config ────────────────────────────────────────────────────────────────
    dump_url        = _require_env("WIKI_DUMP_URL")
    qdrant_host     = _require_env("QDRANT_HOST")
    qdrant_port     = int(_require_env("QDRANT_PORT"))
    collection_name = _require_env("QDRANT_COLLECTION")
    embedder_host   = _require_env("EMBEDDER_HOST")
    embedder_port   = int(_require_env("EMBEDDER_PORT"))
    embed_model     = _require_env("EMBED_MODEL")
    redis_host      = _require_env("REDIS_HOST")
    redis_port      = int(os.environ.get("REDIS_PORT", "6379"))
    top_fraction    = float(os.environ.get("PAGEVIEW_TOP_FRACTION", "0.15"))
    min_tokens      = int(os.environ.get("MIN_ARTICLE_TOKENS", "300"))
    batch_size      = int(os.environ.get("EMBED_BATCH_SIZE", "64"))
    default_dump    = os.environ.get("DUMP_PATH", "/data/wiki_dump.json.gz")

    embedder_url = f"http://{embedder_host}:{embedder_port}"

    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    try:
        info = client.get_collection(collection_name)
    except Exception:
        logger.error(
            "Collection '%s' not found — run ingest.py to build the initial index.",
            collection_name,
        )
        sys.exit(1)

    if info.points_count == 0:
        logger.warning(
            "Collection '%s' is empty — run ingest.py first.", collection_name
        )

    # ── Fast path: --force-update ─────────────────────────────────────────────
    if force_update_titles:
        run_force_update(
            titles=force_update_titles,
            dump_url=dump_url,
            client=client,
            collection_name=collection_name,
            embedder_url=embedder_url,
            embed_model=embed_model,
            batch_size=batch_size,
            min_tokens=min_tokens,
            redis_host=redis_host,
            redis_port=redis_port,
        )
        return

    # ── Normal path: full diff against local dump ─────────────────────────────
    dump_path = dump_path_override or default_dump
    if not os.path.exists(dump_path):
        logger.info("No local dump at %s — downloading from URL…", dump_path)
        download_dump(dump_url, dump_path)

    logger.info("=== WikiRAG Incremental Update ===")
    logger.info("Qdrant:   %s:%d  collection=%s", qdrant_host, qdrant_port, collection_name)
    logger.info("Embedder: %s  model=%s", embedder_url, embed_model)
    logger.info("Dump:     %s", dump_path)

    # Steps 1–5
    qdrant_state             = load_qdrant_state(client, collection_name)
    dump_lm_map, rank_map    = build_dump_state(dump_path, top_fraction)
    changed, added, deleted  = compute_diff(qdrant_state, dump_lm_map)

    if not changed and not added and not deleted:
        logger.info("Index is already up to date — nothing to do.")
        return

    total_upserted, skipped_chunks = process_upserts(
        to_update=changed | added,
        dump_path=dump_path,
        rank_map=rank_map,
        client=client,
        collection_name=collection_name,
        embedder_url=embedder_url,
        embed_model=embed_model,
        batch_size=batch_size,
        min_tokens=min_tokens,
    )

    total_deleted = process_deletions(deleted, client, collection_name)

    flush_redis_cache(redis_host, redis_port)

    # ── Summary ───────────────────────────────────────────────────────────────
    final_count = client.get_collection(collection_name).points_count

    logger.info("=== Update Complete ===")
    logger.info("Articles changed:    %d", len(changed))
    logger.info("Articles added:      %d", len(added))
    logger.info("Articles deleted:    %d", len(deleted))
    logger.info("Points upserted:     %d", total_upserted)
    logger.info("Points deleted:      %d", total_deleted)
    logger.info("Qdrant point count:  %d", final_count)

    if skipped_chunks:
        skipped_titles = sorted({c["title"] for c in skipped_chunks})
        logger.warning(
            "%d chunk(s) skipped (persistent 400) across %d article(s): %s",
            len(skipped_chunks), len(skipped_titles), ", ".join(skipped_titles),
        )
    else:
        logger.info("Chunks skipped:      0")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="WikiRAG incremental update")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dump-path",
        default=None,
        metavar="PATH",
        help=(
            "Path to a local Cirrus dump .json.gz.  "
            "Falls back to DUMP_PATH env var; downloads from WIKI_DUMP_URL "
            "if neither points to an existing file."
        ),
    )
    mode.add_argument(
        "--force-update",
        nargs="+",
        metavar="TITLE",
        help=(
            "Force-update specific articles by title without a local dump.  "
            "Streams from WIKI_DUMP_URL, stops as soon as all named titles are "
            "found, then upserts any that are missing or have a changed "
            "last_modified timestamp.  No Pass 1 scan, no deletions.  "
            "Intended for testing and manual corrections.  "
            'Example: --force-update "Albert Einstein" "Python (programming language)"'
        ),
    )

    args = parser.parse_args()
    run_update(
        dump_path_override=args.dump_path,
        force_update_titles=args.force_update,
    )


if __name__ == "__main__":
    main()
