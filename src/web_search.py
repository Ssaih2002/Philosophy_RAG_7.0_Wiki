from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
import re

logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str


def _normalize_wikipedia_query(query: str) -> str:
    """
    Wikipedia API 对异常长/像 prompt 的查询更容易触发风控；这里把用户输入收敛成“可检索短词”。
    优先提取括号中的拉丁字母人名（如 Henri Lefebvre），其次取第一行并截断。
    """
    q = (query or "").strip()
    if not q:
        return ""
    # Prefer Latin-name inside parentheses: （Henri Lefebvre, ...） or (Henri Lefebvre, ...)
    m = re.search(r"[（(]\s*([A-Za-z][A-Za-z .,'’-]{2,80})", q)
    if m:
        cand = m.group(1).strip()
        cand = re.sub(r"\s+", " ", cand)
        if 3 <= len(cand) <= 90:
            return cand
    # Otherwise, take first line only (often the actual topic)
    first = q.splitlines()[0].strip()
    # Remove common instruction-like prefixes
    first = re.sub(r"^(帮我|请|麻烦|能否|我要|我想|请你)\s*", "", first)
    # Hard cap length
    if len(first) > 120:
        first = first[:120].rstrip()
    return first


def _wiki_headers() -> dict:
    # https://meta.wikimedia.org/wiki/User-Agent_policy
    # 需包含“应用名/版本 + 可联系信息”；禁止伪装常见浏览器 UA。
    return {
        "User-Agent": "PhilosophyRAG/1.0 (+https://github.com/local/Philosophy_UP; local-dev)",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en,de;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    }


def _wiki_get(
    url: str,
    params: Dict[str, Any],
    *,
    timeout: float,
    trust_env: bool,
) -> httpx.Response:
    with httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        trust_env=trust_env,
    ) as client:
        return client.get(url, params=params, headers=_wiki_headers())


def _wiki_get_with_fallback(url: str, params: Dict[str, Any], *, timeout: float) -> Tuple[Optional[dict], Optional[str]]:
    """
    先直连（trust_env=False）：Wikipedia 对许多代理出口会统一 403；
    直连失败再尝试遵循环境变量代理（便于“必须走代理才能上网”的环境）。
    """
    last_detail = ""
    for trust_env, label in ((False, "direct"), (True, "proxy-env")):
        try:
            r = _wiki_get(url, params, timeout=timeout, trust_env=trust_env)
            if r.status_code == 200:
                return r.json(), None
            if r.status_code == 403:
                last_detail = f"{label} 403"
                logger.warning("Wikipedia API 403 (%s): %s", label, url)
                continue
            last_detail = f"{label} HTTP {r.status_code}"
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            last_detail = f"{label} {e!r}"
        except httpx.RequestError as e:
            last_detail = f"{label} {e!r}"
    return None, last_detail or "unknown"


def web_search_wikipedia(
    query: str,
    *,
    max_results: int = 6,
    lang: str = "en",
) -> List[WebSearchResult]:
    """
    Wikipedia-only search via MediaWiki API (no API key).
    Returns title/url/snippet (extract) for each matched page.
    """
    q = _normalize_wikipedia_query(query)
    if not q:
        return []

    wiki_lang = (lang or "en").strip().lower()
    if wiki_lang not in ("en", "de", "zh"):
        wiki_lang = "en"
    base = f"https://{wiki_lang}.wikipedia.org"

    # Step 1: search page titles
    search_url = f"{base}/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": q,
        "srlimit": max(1, int(max_results)),
        "utf8": 1,
        "format": "json",
    }
    data, err = _wiki_get_with_fallback(search_url, params, timeout=20.0)
    if data is None:
        logger.warning("Wikipedia search unreachable: %s", err)
        return []

    titles: List[str] = []
    for it in (data.get("query") or {}).get("search") or []:
        t = str(it.get("title") or "").strip()
        if t:
            titles.append(t)
    if not titles:
        return []

    # Step 2: fetch extracts + canonical URLs (batch)
    params2 = {
        "action": "query",
        "prop": "extracts|info",
        "inprop": "url",
        "exintro": 1,
        "explaintext": 1,
        "exchars": 420,
        "titles": "|".join(titles),
        "format": "json",
        "redirects": 1,
    }
    data2, err2 = _wiki_get_with_fallback(search_url, params2, timeout=25.0)
    if data2 is None:
        logger.warning("Wikipedia extracts unreachable: %s", err2)
        return []

    pages = (data2.get("query") or {}).get("pages") or {}
    out: List[WebSearchResult] = []
    for _, p in pages.items():
        title = str(p.get("title") or "").strip()
        url = str(p.get("fullurl") or "").strip()
        snippet = str(p.get("extract") or "").strip()
        if not url:
            # Fallback to a stable URL if fullurl is missing
            if title:
                url = f"{base}/wiki/{title.replace(' ', '_')}"
        if url:
            out.append(WebSearchResult(title=title or url, url=url, snippet=snippet))

    # Keep original search order where possible
    by_title = {r.title: r for r in out if r.title}
    ordered: List[WebSearchResult] = []
    for t in titles:
        if t in by_title:
            ordered.append(by_title[t])
    if not ordered:
        ordered = out
    return ordered[: max(1, int(max_results))]


def web_search_wikipedia_multi(
    query: str,
    *,
    max_results: int = 6,
    langs: List[str] | None = None,
) -> List[WebSearchResult]:
    """
    Search multiple Wikipedia languages and merge results.
    Dedup by URL; keep stable ordering (lang order, then each lang's ranking).
    """
    q = (query or "").strip()
    if not q:
        return []
    ls = [s.strip().lower() for s in (langs or ["en", "de"]) if s and s.strip()]
    # Keep only supported langs and preserve order
    seen_lang = set()
    langs2: List[str] = []
    for l in ls:
        if l in ("en", "de", "zh") and l not in seen_lang:
            seen_lang.add(l)
            langs2.append(l)
    if not langs2:
        langs2 = ["en", "de", "zh"]

    out: List[WebSearchResult] = []
    seen_url: set[str] = set()
    per_lang = max(1, int(max_results))
    for l in langs2:
        for r in web_search_wikipedia(q, max_results=per_lang, lang=l):
            u = (r.url or "").strip()
            if not u or u in seen_url:
                continue
            seen_url.add(u)
            out.append(r)
            if len(out) >= max(1, int(max_results)):
                return out
    return out

