from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from google import genai

from .config import (
    AUTO_B2_AUX_TIMEOUT_SECONDS,
    AUTO_B2_CACHE_DIR,
    GEMINI_API_KEY,
    GEMINI_AUX_FALLBACK_MODEL,
    GEMINI_AUX_MODEL,
    GEMINI_RETRY_BASE_SECONDS,
    GEMINI_RETRY_JITTER_SECONDS,
    GEMINI_RETRY_MAX_ATTEMPTS,
)


_client = genai.Client(api_key=GEMINI_API_KEY)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _cache_key(
    *,
    question: str,
    target_langs: List[str],
) -> str:
    blob = json.dumps(
        {
            "q": (question or "").strip(),
            "langs": [str(x).strip().lower() for x in (target_langs or [])],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _cache_path(key: str) -> Path:
    root = Path(AUTO_B2_CACHE_DIR)
    _safe_mkdir(root)
    return root / f"{key}.json"


def _load_cache(key: str) -> Optional[Dict[str, Any]]:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _save_cache(key: str, obj: Dict[str, Any]) -> None:
    p = _cache_path(key)
    try:
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _sleep_backoff(attempt: int) -> None:
    import random

    backoff = float(GEMINI_RETRY_BASE_SECONDS) * (2 ** (attempt - 1))
    jitter = random.uniform(0.0, float(GEMINI_RETRY_JITTER_SECONDS))
    time.sleep(backoff + jitter)


def _is_retryable(exc: Exception) -> bool:
    s = str(exc).lower()
    marks = [
        "503",
        "unavailable",
        "high demand",
        "resource_exhausted",
        "429",
        "deadline exceeded",
        "timeout",
        "timed out",
        "connection reset",
        "server disconnected",
    ]
    return any(m in s for m in marks)


def _extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    if not t:
        return None
    # Try strict json first
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Loose: find first {...} block
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


@dataclass(frozen=True)
class AutoB2Result:
    target_langs: List[str]
    dense_queries: List[str]
    sparse_terms: List[str]
    glossary: Dict[str, Dict[str, str]]
    model_used: str
    cached: bool


def generate_b2(
    *,
    question: str,
    target_langs: List[str],
) -> Optional[AutoB2Result]:
    q = (question or "").strip()
    langs = [str(x).strip().lower() for x in (target_langs or []) if str(x).strip()]
    langs = [x for x in langs if x in ("de", "en", "zh")]
    if not q or not langs:
        return None

    # Cache is keyed only by (question, target_langs) so multi-library selection
    # can reuse the same B2 output across libraries with the same dominant language.
    key = _cache_key(question=q, target_langs=langs)
    cached = _load_cache(key)
    if cached:
        try:
            return AutoB2Result(
                target_langs=list(cached.get("target_langs") or langs),
                dense_queries=list(cached.get("dense_queries") or []),
                sparse_terms=list(cached.get("sparse_terms") or []),
                glossary=dict(cached.get("glossary") or {}),
                model_used=str(cached.get("model_used") or ""),
                cached=True,
            )
        except Exception:
            pass

    prompt = f"""You are a bilingual/multilingual philosophy research assistant.

Task:
- Convert the user's question into multilingual retrieval queries and term anchors for a library whose dominant languages are: {", ".join(langs)}.
- Output JSON only.

Constraints:
- dense_queries: 4-8 short search queries total (across all target languages), each <= 180 chars.
- sparse_terms: 10-24 keyword/term anchors total (across all target languages), prefer technical terms, names, and canonical phrases.
- glossary: a mapping of key concepts, each with translations in the requested languages when possible.
- Do NOT include explanations, markdown, or extra text outside JSON.

User question:
{q}

Output JSON schema:
{{
  "target_langs": ["de","en"],
  "dense_queries": ["..."],
  "sparse_terms": ["..."],
  "glossary": {{
    "concept_1": {{"de":"...","en":"...","zh":"..."}},
    "concept_2": {{"de":"...","en":"...","zh":"..."}}
  }}
}}
"""

    candidates = [GEMINI_AUX_MODEL]
    if GEMINI_AUX_FALLBACK_MODEL and GEMINI_AUX_FALLBACK_MODEL != GEMINI_AUX_MODEL:
        candidates.append(GEMINI_AUX_FALLBACK_MODEL)

    deadline = time.time() + float(AUTO_B2_AUX_TIMEOUT_SECONDS)
    last_err: Optional[Exception] = None
    for model_name in candidates:
        attempts = max(1, int(GEMINI_RETRY_MAX_ATTEMPTS))
        for attempt in range(1, attempts + 1):
            if time.time() > deadline:
                break
            try:
                resp = _client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "temperature": 0.2,
                        "max_output_tokens": 1400,
                    },
                )
                text = getattr(resp, "text", "") or ""
                obj = _extract_first_json(text)
                if not obj:
                    raise RuntimeError("AutoB2 returned non-JSON output.")
                dense = [str(x).strip() for x in (obj.get("dense_queries") or []) if str(x).strip()]
                terms = [str(x).strip() for x in (obj.get("sparse_terms") or []) if str(x).strip()]
                gl = obj.get("glossary") or {}
                if not isinstance(gl, dict):
                    gl = {}
                # Keep sizes sane
                dense = dense[:10]
                terms = terms[:40]
                out = {
                    "target_langs": langs,
                    "dense_queries": dense,
                    "sparse_terms": terms,
                    "glossary": gl,
                    "model_used": model_name,
                    "created_at": _now_iso(),
                }
                _save_cache(key, out)
                return AutoB2Result(
                    target_langs=langs,
                    dense_queries=dense,
                    sparse_terms=terms,
                    glossary=gl,  # type: ignore[arg-type]
                    model_used=model_name,
                    cached=False,
                )
            except Exception as e:
                last_err = e
                if attempt >= attempts or (not _is_retryable(e)):
                    break
                _sleep_backoff(attempt)
        if time.time() > deadline:
            break
    _ = last_err
    return None

