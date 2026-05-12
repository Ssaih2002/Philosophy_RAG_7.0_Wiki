from __future__ import annotations

import os
import re
from typing import Any, Dict, List


def _clean_text(s: str) -> str:
    t = (s or "").replace("\u00a0", " ").strip()
    t = re.sub(r"[ \t]+", " ", t)
    # keep paragraphs; remove repeated blank lines
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def load_epub(file_path: str) -> List[Dict[str, Any]]:
    """
    Load an .epub into pseudo-pages:
      {"text": "...", "page": "chapter-3:para-12", "source": "<filename>"}

    Notes:
    - EPUB has no stable PDF-like pagination; we expose a deterministic pseudo locator.
    - Each paragraph becomes one "page" unit (para index within chapter).
    """
    try:
        from ebooklib import epub  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Missing dependency for .epub support. Install with: pip install ebooklib") from e

    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Missing dependency for .epub support. Install with: pip install beautifulsoup4") from e

    book = epub.read_epub(file_path)
    source = os.path.basename(file_path)

    pages: List[Dict[str, Any]] = []
    chapter_idx = 0
    for item in book.get_items():
        # only parse XHTML/HTML documents
        if item.get_type() != epub.ITEM_DOCUMENT:
            continue
        chapter_idx += 1
        try:
            html = item.get_content()
        except Exception:
            continue
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            continue

        # collect paragraph-like blocks; fallback to all text if none
        blocks: List[str] = []
        for tag in soup.find_all(["p", "li", "blockquote"]):
            txt = tag.get_text("\n", strip=True)
            txt = _clean_text(txt)
            if txt:
                blocks.append(txt)

        if not blocks:
            txt = soup.get_text("\n", strip=True)
            txt = _clean_text(txt)
            if txt:
                blocks = [txt]

        for para_idx, txt in enumerate(blocks, start=1):
            pages.append(
                {
                    "text": txt,
                    "page": f"chapter-{chapter_idx}:para-{para_idx}",
                    "source": source,
                }
            )
    return pages

