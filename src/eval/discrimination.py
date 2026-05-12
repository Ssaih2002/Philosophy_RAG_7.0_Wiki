"""
从同一问题的多模式检索结果中，量化「Hybrid / RRF」相对单路的互补性（不依赖 gold 也可报告）。
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List, Set


def _top_set(ids: List[str], k: int) -> Set[str]:
    out: Set[str] = set()
    for x in (ids or [])[:k]:
        if x:
            out.add(x)
    return out


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def ablation_discrimination_report(rows: List[Dict[str, Any]], *, k: int = 10) -> Dict[str, Any]:
    """
    输入为 eval CLI 输出的多行 JSON 对象列表（已解析），按 qid 分组。
    期望 mode: full_rrf, dense_only, sparse_only, merge_no_rrf。
    """
    by_q: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    for r in rows:
        qid = str(r.get("qid", ""))
        mode = str(r.get("mode", ""))
        ids = r.get("retrieved_ids") or []
        by_q[qid][mode] = list(ids)

    per_q: List[Dict[str, Any]] = []
    for qid, modes in by_q.items():
        full = _top_set(modes.get("full_rrf", []), k)
        den = _top_set(modes.get("dense_only", []), k)
        sp = _top_set(modes.get("sparse_only", []), k)
        mer = _top_set(modes.get("merge_no_rrf", []), k)

        full_not_dense = full - den
        dense_not_full = den - full
        full_uses_sparse_signal = len(full & sp) / float(max(len(full), 1))

        per_q.append(
            {
                "qid": qid,
                "jaccard_full_vs_dense": round(jaccard(full, den), 4),
                "jaccard_full_vs_sparse": round(jaccard(full, sp), 4),
                "fraction_full_intersects_sparse_top": round(full_uses_sparse_signal, 4),
                "count_full_not_in_dense_top": len(full_not_dense),
                "count_dense_not_in_full_top": len(dense_not_full),
                "jaccard_merge_no_rrf_vs_full_rrf": round(jaccard(mer, full), 4),
            }
        )

    def mean(key: str) -> float:
        vals = [float(p[key]) for p in per_q if key in p]
        return round(sum(vals) / float(len(vals)), 6) if vals else 0.0

    return {
        "k": k,
        "n_questions": len(per_q),
        "mean_jaccard_full_vs_dense": mean("jaccard_full_vs_dense"),
        "interpretation_low_jaccard": "full 与 dense Top-k 差异越大，说明融合/重排改变排序越多（需结合 gold 才可知是否变好）。",
        "mean_fraction_full_intersects_sparse_top": mean("fraction_full_intersects_sparse_top"),
        "interpretation_sparse_fraction": "Full RRF 结果中与「纯稀疏 Top-k」重叠比例越高，说明稀疏通道对最终排序贡献越大。",
        "mean_count_full_not_in_dense_top": mean("count_full_not_in_dense_top"),
        "interpretation_hybrid_lift_proxy": "平均每条问题：Full 的 Top-k 中有多少条不在 Dense-only 的 Top-k（越大越说明两路信号不冗余）。",
        "mean_jaccard_merge_vs_full_rrf": mean("jaccard_merge_no_rrf_vs_full_rrf"),
        "interpretation_rrf_vs_concat": "merge_no_rrf 与 full_rrf 的 Jaccard 越低，说明 RRF 相对简单拼接/截断改变了排序。",
        "per_question": per_q,
    }


def gold_lift_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """若各 mode 有 hit@5 / mrr，则计算 Full RRF 相对 Dense-only 的增益（需已标注 gold）。"""
    hit_by_q: Dict[str, Dict[str, float]] = defaultdict(dict)
    mrr_by_q: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        m = r.get("metrics") or {}
        qid = str(r.get("qid", ""))
        mode = str(r.get("mode", ""))
        if "hit@5" in m:
            hit_by_q[qid][mode] = float(m["hit@5"])
        if "mrr" in m:
            mrr_by_q[qid][mode] = float(m["mrr"])

    hit_lifts: List[float] = []
    for modes in hit_by_q.values():
        if "full_rrf" in modes and "dense_only" in modes:
            hit_lifts.append(modes["full_rrf"] - modes["dense_only"])

    mrr_lifts: List[float] = []
    for modes in mrr_by_q.values():
        if "full_rrf" in modes and "dense_only" in modes:
            mrr_lifts.append(modes["full_rrf"] - modes["dense_only"])

    out: Dict[str, Any] = {
        "n_questions_with_hit5": len(hit_by_q),
        "n_questions_with_mrr": len(mrr_by_q),
    }
    if hit_lifts:
        out["mean_hit5_lift_full_rrf_minus_dense"] = round(mean(hit_lifts), 6)
    if mrr_lifts:
        out["mean_mrr_lift_full_rrf_minus_dense"] = round(mean(mrr_lifts), 6)
    return out
