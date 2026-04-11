"""
Streaming Cirrus JSON dump downloader.

The Wikimedia Cirrus dump is in Elasticsearch bulk format:
  line 1: {"index": {"_type": "page", "_id": "12"}}
  line 2: {"namespace": 0, "title": "...", "text": "...", ...}
  (repeating pairs)

We stream the gzip-compressed file line-by-line, yielding only namespace-0
(main article) data lines. Memory usage stays flat regardless of dump size.
"""

import gzip
import json
import logging
import os
import urllib.request
from typing import Generator

from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_dump(url: str, dest_path: str) -> None:
    """Stream-download the Cirrus dump to dest_path, showing a progress bar."""
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    logger.info("Downloading dump from %s → %s", url, dest_path)

    req = urllib.request.Request(url, headers={"User-Agent": "WikiRAG/1.0"})
    with urllib.request.urlopen(req) as response:
        total = int(response.headers.get("Content-Length", 0)) or None
        chunk_size = 1024 * 1024  # 1 MB

        with open(dest_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading",
        ) as bar:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))

    logger.info("Download complete: %s", dest_path)


def stream_articles(path: str, limit: int | None = None) -> Generator[dict, None, None]:
    """
    Yield article dicts (namespace 0 only) from a local Cirrus gz dump.

    Skips index lines (those containing an "index" key) and any article
    whose namespace is not 0 (talk pages, templates, etc.).

    Args:
        path:  Path to the local .json.gz dump file.
        limit: If set, stop after yielding this many articles.
    """
    yielded = 0

    with gzip.open(path, "rt", encoding="utf-8") as gz:
        for raw_line in gz:
            line = raw_line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line: %s", line[:120])
                continue

            # Skip Elasticsearch index lines
            if "index" in obj:
                continue

            # Keep only main-namespace articles
            if obj.get("namespace", -1) != 0:
                continue

            yield obj
            yielded += 1

            if limit is not None and yielded >= limit:
                logger.info("Reached article limit (%d). Stopping stream.", limit)
                break

    logger.info("Streamed %d articles from %s", yielded, path)
