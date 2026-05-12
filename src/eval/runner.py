from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.config import MAX_FINAL_K

from .metrics import hit_at_k, mean_reciprocal_rank, ndcg_at_k

# 与 CLI 文档一致：Baseline1=稠密，Baseline2=稀疏，Hybrid=双路合并无 RRF，Full=RRF 融合
_MODE_ALIASES: Dict[str, str] = {
    "baseline1": "dense_only",
    "b1": "dense_only",
    "baseline2": "sparse_only",
    "b2": "sparse_only",
    "hybrid": "merge_no_rrf",
    "full": "full_rrf",
}

_MODE_LABEL: Dict[str, str] = {
    "dense_only": "Baseline1",
    "sparse_only": "Baseline2",
    "merge_no_rrf": "Hybrid",
    "full_rrf": "Full",
}


_KNOWN_MODES = frozenset({"full_rrf", "dense_only", "sparse_only", "merge_no_rrf"})


def normalize_eval_mode(mode: str) -> str:
    key = (mode or "").strip().lower()
    if key in _MODE_ALIASES:
        return _MODE_ALIASES[key]
    if key in _KNOWN_MODES:
        return key
    return (mode or "").strip()


@dataclass
class EvalItem:
    qid: str
    question: str
    gold_chunk_ids: List[str]
    experiment: str = "ablation"  # ablation | terminology
    query_group: Optional[str] = None  # for terminology: same group id across paraphrases


def _slice_docs(docs: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    return docs[: max(1, int(top_n))]


def _doc_preview_text(raw: str, doc_preview_chars: int) -> str:
    t = raw or ""
    if doc_preview_chars <= 0:
        return t
    return t[: doc_preview_chars]


def _norm_source_filters(source_filters: Optional[List[str]]) -> Optional[List[str]]:
    out = [s.strip() for s in (source_filters or []) if s and str(s).strip()]
    return out if out else None


def _norm_keyword_terms(keyword_terms: Optional[List[str]]) -> Optional[List[str]]:
    out = [s.strip() for s in (keyword_terms or []) if s and str(s).strip()]
    return out if out else None


def run_retrieval_only(
    engine: Any,
    *,
    item: EvalItem,
    mode: str,
    top_n: int = 10,
    use_hybrid_for_sparse: bool = True,
    doc_preview_chars: int = 400,
    keyword_terms: Optional[List[str]] = None,
    source_filters: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    mode: full_rrf | dense_only | sparse_only | merge_no_rrf
    亦接受别名：full、hybrid、baseline1、baseline2（见 normalize_eval_mode）。
    doc_preview_chars: 每条片段写入 JSON 的正文长度；<=0 表示写入全文（人工审阅时可用）。
    """
    mode = normalize_eval_mode(mode)
    ablation = None
    use_hybrid = True
    if mode == "full_rrf":
        use_hybrid = use_hybrid_for_sparse
    elif mode == "dense_only":
        ablation = "dense_only"
        use_hybrid = False
    elif mode == "sparse_only":
        ablation = "sparse_only"
        use_hybrid = use_hybrid_for_sparse
    elif mode == "merge_no_rrf":
        ablation = "merge_no_rrf"
        use_hybrid = use_hybrid_for_sparse
    else:
        raise ValueError(f"unknown mode: {mode}")

    sf = _norm_source_filters(source_filters)
    kw_user = _norm_keyword_terms(keyword_terms)
    # 用户限定关键词：仅在 full_rrf（Full RRF）时传入检索层；其它消融模式忽略
    kw_effective = kw_user if mode == "full_rrf" else None

    eval_final_floor = max(1, min(int(top_n), int(MAX_FINAL_K)))
    docs, meta = engine.retrieve(
        item.question,
        keyword_terms=kw_effective,
        source_filters=sf,
        auto_extract_keywords=True,
        use_hybrid=use_hybrid,
        use_rerank=False,
        use_sep_reference=False,
        answer_style="哲学论述",
        retrieval_ablation=ablation,
        retrieval_final_k_override=eval_final_floor,
    )
    docs = _slice_docs(docs, top_n)
    ids = [str(d.get("chunk_id") or "") for d in docs]
    metrics: Dict[str, Any] = {}
    if item.gold_chunk_ids and ids:
        metrics["hit@5"] = hit_at_k(ids, item.gold_chunk_ids, k=5)
        metrics["mrr"] = mean_reciprocal_rank(ids, item.gold_chunk_ids)
        k_ndcg = min(10, max(len(ids), 1))
        metrics["ndcg@10"] = ndcg_at_k(ids, item.gold_chunk_ids, k=k_ndcg)

    return {
        "qid": item.qid,
        "experiment": item.experiment,
        "mode": mode,
        "mode_label": _MODE_LABEL.get(mode, mode),
        "question": item.question,
        "retrieved_ids": ids,
        "docs_preview": [
            {
                "chunk_id": d.get("chunk_id"),
                "source": d.get("source"),
                "page": d.get("page"),
                "text_preview": _doc_preview_text(d.get("text") or "", doc_preview_chars),
            }
            for d in docs
        ],
        "metrics": metrics,
        "meta": {
            "profile": meta.get("profile"),
            "hybrid": meta.get("hybrid"),
            "retrieval_ablation": meta.get("retrieval_ablation"),
            "keywords_used": meta.get("keywords_used"),
            "source_filters_used": meta.get("source_filters_used"),
            "eval_source_filters_applied": sf,
            "eval_keyword_terms_applied": kw_effective,
            "eval_keyword_terms_ignored_non_full_rrf": kw_user if (kw_user and mode != "full_rrf") else None,
        },
    }


def load_items(path: str) -> List[EvalItem]:
    out: List[EvalItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gold = row.get("gold_chunk_ids") or row.get("gold") or []
            if isinstance(gold, str):
                gold = [gold]
            out.append(
                EvalItem(
                    qid=str(row.get("id", row.get("qid", ""))),
                    question=str(row.get("question", "")),
                    gold_chunk_ids=[str(x) for x in gold],
                    experiment=str(row.get("experiment", "ablation")),
                    query_group=row.get("query_group"),
                )
            )
    return out


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
