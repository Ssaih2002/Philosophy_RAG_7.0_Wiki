import re
from typing import Dict, List


def _normalize_term(term: str) -> str:
    cleaned = re.sub(r"\s+", " ", term.strip())
    return cleaned.lower()


def merge_terms(
    user_terms: List[str],
    auto_terms: List[str],
    max_terms: int = 12,
) -> Dict[str, List[str]]:
    user_clean = [t.strip() for t in user_terms if t and t.strip()]
    auto_clean = [t.strip() for t in auto_terms if t and t.strip()]

    merged: List[str] = []
    dropped: List[str] = []
    seen = set()

    for t in user_clean + auto_clean:
        key = _normalize_term(t)
        if not key or len(key) < 2:
            dropped.append(t)
            continue
        if key in seen:
            dropped.append(t)
            continue
        if len(merged) >= max_terms:
            dropped.append(t)
            continue
        seen.add(key)
        merged.append(t)

    return {
        "user_terms": user_clean,
        "auto_terms": auto_clean,
        "merged_terms": merged,
        "dropped_terms": dropped,
    }
