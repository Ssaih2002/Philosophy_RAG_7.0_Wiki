from __future__ import annotations

from typing import Iterable, List, Optional, Sequence


def hit_at_k(
    ranked_ids: Sequence[str],
    gold_ids: Iterable[str],
    k: int = 5,
) -> float:
    """1.0 if any gold appears in the first k positions, else 0.0."""
    gold_set = {str(g) for g in gold_ids if g}
    if not gold_set:
        return 0.0
    top = list(ranked_ids[:k])
    return 1.0 if any(cid in gold_set for cid in top) else 0.0


def mean_reciprocal_rank(
    ranked_ids: Sequence[str],
    gold_ids: Iterable[str],
) -> float:
    """Reciprocal rank of the first matching gold id; 0 if none."""
    gold_set = {str(g) for g in gold_ids if g}
    if not gold_set:
        return 0.0
    for i, cid in enumerate(ranked_ids):
        if cid in gold_set:
            return 1.0 / float(i + 1)
    return 0.0


def ndcg_at_k(
    ranked_ids: Sequence[str],
    gold_ids: Sequence[str],
    k: int = 10,
    *,
    grades: Optional[dict[str, float]] = None,
) -> float:
    """
    nDCG@k with binary relevance (gold ids relevant) unless grades maps id->gain.
    """
    if k <= 0:
        return 0.0
    top = list(ranked_ids[:k])
    if grades:
        rel = [float(grades.get(cid, 0.0)) for cid in top]
    else:
        gold = {str(g) for g in gold_ids if g}
        rel = [1.0 if cid in gold else 0.0 for cid in top]
    dcg = _dcg(rel)
    ideal = sorted(rel, reverse=True)
    idcg = _dcg(ideal)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def _dcg(rels: List[float]) -> float:
    import math

    s = 0.0
    for i, r in enumerate(rels):
        s += (2**r - 1.0) / math.log2(i + 2.0)
    return s
