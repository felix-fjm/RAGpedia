"""
Article filtering, parsing, cleaning, and chunking.

Pipeline per article:
  extract_sections → clean_text → merge short sections → chunk_section

Token counting uses a whitespace-split approximation (1 word ≈ 1 token),
which is accurate enough for the 50/300/600-token thresholds used here.
"""

import html
import logging
import re
from typing import Generator

logger = logging.getLogger(__name__)

# ── Token counting ────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Approximate token count: split on whitespace."""
    return len(text.split())


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Strip citation markers, HTML entities, table artefacts; normalise whitespace.
    """
    if not text:
        return ""

    # Strip [N] citation / footnote markers
    text = re.sub(r"\[\d+\]", "", text)

    # Decode HTML entities (&amp; &lt; &nbsp; etc.)
    text = html.unescape(text)

    # Remove wiki table markup remnants ({| ... |})
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)

    # Remove table cell / header lines (lines starting with | or !)
    text = re.sub(r"(?m)^\s*[|!].*$", "", text)

    # Remove bare wiki markup characters left over (= signs, * bullets)
    text = re.sub(r"(?m)^={2,}.*={2,}$", "", text)   # == Section == headers
    text = re.sub(r"(?m)^\*+\s*$", "", text)           # stray bullet lines

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse runs of spaces/tabs within a line
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


# ── Section extraction ────────────────────────────────────────────────────────

def extract_sections(article: dict) -> list[tuple[str, str]]:
    """
    Return a list of (section_name, raw_text) tuples.

    Strategy:
    - Split article text into paragraph blocks (double-newline boundaries).
    - A block whose first line exactly matches a known heading title starts
      a new section; subsequent blocks belong to that section.
    - The first run of blocks before any heading becomes the "Introduction".
    """
    text: str = article.get("text") or ""
    headings: list[str] = article.get("heading") or []
    opening_text: str = article.get("opening_text") or ""

    heading_set = {h.strip() for h in headings}

    blocks = [b.strip() for b in re.split(r"\n\n+", text) if b.strip()]

    sections: list[tuple[str, str]] = []
    current_name = "Introduction"
    current_blocks: list[str] = []

    for block in blocks:
        first_line = block.split("\n")[0].strip()

        if first_line in heading_set:
            # Save accumulated blocks under the current section name
            if current_blocks:
                sections.append((current_name, "\n\n".join(current_blocks)))
            current_name = first_line
            # The rest of the block after the heading line is section content
            rest = "\n".join(block.split("\n")[1:]).strip()
            current_blocks = [rest] if rest else []
        else:
            current_blocks.append(block)

    if current_blocks:
        sections.append((current_name, "\n\n".join(current_blocks)))

    # Fallback: if nothing was parsed but opening_text exists, use it
    if not sections and opening_text:
        sections.append(("Introduction", opening_text))

    return sections


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_section(
    text: str,
    max_tokens: int = 600,
    overlap_tokens: int = 50,
) -> list[str]:
    """
    Split section text into chunks according to the spec:
      < 50 tok  → return [] (caller merges with previous section)
      ≤ 600 tok → return [text] as a single chunk
      > 600 tok → split at paragraph boundaries with 50-tok overlap

    When splitting, overlap is achieved by carrying the last N tokens of the
    previous chunk into the start of the next chunk.
    """
    n = count_tokens(text)

    if n < 50:
        return []

    if n <= max_tokens:
        return [text]

    # Split at paragraph boundaries
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current_paras: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if current_tokens + para_tokens > max_tokens and current_paras:
            # Emit current chunk
            chunks.append("\n\n".join(current_paras))

            # Build overlap tail from end of the just-emitted chunk
            overlap_paras: list[str] = []
            overlap_count = 0
            for p in reversed(current_paras):
                pt = count_tokens(p)
                if overlap_count + pt <= overlap_tokens:
                    overlap_paras.insert(0, p)
                    overlap_count += pt
                else:
                    break

            current_paras = overlap_paras
            current_tokens = overlap_count

        current_paras.append(para)
        current_tokens += para_tokens

    if current_paras:
        chunks.append("\n\n".join(current_paras))

    return chunks


# ── Article processing ────────────────────────────────────────────────────────

def process_article(article: dict, pageview_rank: int) -> list[dict]:
    """
    Full per-article pipeline: extract → clean → merge short sections → chunk.

    Returns a list of chunk dicts with all required metadata fields:
      text, title, section, url, last_modified, pageview_rank, chunk_index
    """
    title: str = article.get("title") or ""
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    # Cirrus timestamp is ISO-8601; store date portion only
    raw_ts: str = article.get("timestamp") or ""
    last_modified = raw_ts[:10] if raw_ts else ""

    sections = extract_sections(article)

    # Clean each section
    cleaned: list[tuple[str, str]] = []
    for section_name, section_text in sections:
        c = clean_text(section_text)
        if c:
            cleaned.append((section_name, c))

    # Merge sections with < 50 tokens into the preceding section
    merged: list[tuple[str, str]] = []
    for section_name, section_text in cleaned:
        if count_tokens(section_text) < 50 and merged:
            prev_name, prev_text = merged[-1]
            merged[-1] = (prev_name, prev_text + "\n\n" + section_text)
        else:
            merged.append((section_name, section_text))

    # Chunk and build output dicts
    result: list[dict] = []
    chunk_index = 0

    for section_name, section_text in merged:
        chunk_texts = chunk_section(section_text)
        for chunk_text in chunk_texts:
            result.append({
                "text": chunk_text,
                "title": title,
                "section": section_name,
                "url": url,
                "last_modified": last_modified,
                "pageview_rank": pageview_rank,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

    return result


# ── Popularity filtering ──────────────────────────────────────────────────────

def compute_popularity_threshold(
    articles: list[dict],
    top_fraction: float,
) -> tuple[float, dict[str, int]]:
    """
    Given a list of articles (already loaded from a first streaming pass),
    compute the popularity_score threshold that retains the top `top_fraction`
    of articles, and return a title→rank mapping (1 = most popular).

    Only articles with a non-zero popularity_score are considered.
    """
    scored = [
        (a.get("popularity_score") or 0.0, a.get("title") or "")
        for a in articles
        if (a.get("popularity_score") or 0.0) > 0
    ]

    scored.sort(key=lambda x: x[0], reverse=True)

    keep_n = max(1, int(len(scored) * top_fraction))
    threshold = scored[keep_n - 1][0] if scored else 0.0

    rank_map: dict[str, int] = {
        title: rank + 1
        for rank, (_, title) in enumerate(scored[:keep_n])
    }

    logger.info(
        "Popularity threshold: %.6f (keeping top %d / %d articles)",
        threshold,
        keep_n,
        len(scored),
    )
    return threshold, rank_map


def filter_articles(
    articles: list[dict],
    top_fraction: float,
    min_tokens: int,
) -> Generator[tuple[dict, int], None, None]:
    """
    Yield (article, pageview_rank) for articles that pass both filters:
      1. popularity_score in the top `top_fraction`
      2. opening_text + text has at least `min_tokens` tokens

    Uses compute_popularity_threshold to determine the score cutoff.
    """
    threshold, rank_map = compute_popularity_threshold(articles, top_fraction)

    passed = dropped_pop = dropped_stub = 0

    for article in articles:
        score = article.get("popularity_score") or 0.0
        title = article.get("title") or ""

        if score < threshold or title not in rank_map:
            dropped_pop += 1
            continue

        full_text = (article.get("opening_text") or "") + " " + (article.get("text") or "")
        if count_tokens(full_text) < min_tokens:
            dropped_stub += 1
            continue

        passed += 1
        yield article, rank_map[title]

    logger.info(
        "Filter results: %d passed, %d dropped (popularity), %d dropped (stub)",
        passed,
        dropped_pop,
        dropped_stub,
    )
