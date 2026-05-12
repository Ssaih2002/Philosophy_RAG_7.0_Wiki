"""Lightweight heuristic: flag possible tension between memory/wiki and retrieved Sources."""
from __future__ import annotations

import re
from typing import Any, Dict, List


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def compute_conflict_hint(
    *,
    history_block: str,
    user_wiki_block: str,
    docs: List[Dict[str, Any]],
) -> str:
    """
    Rule-based only (no extra LLM call).
    If memory/wiki contains strong negation/critique markers but retrieved excerpts look uniformly
    positive/neutral, suggest explicit tension handling.
    """
    mem = _norm(history_block + "\n" + user_wiki_block)
    if len(mem) < 8:
        return ""

    neg_markers = (
        "不是",
        "没有",
        "并非",
        "否认",
        "反对",
        "错误",
        "不存在",
        "不可能",
        "矛盾",
        "反驳",
        "批判",
        "not ",
        "no ",
        "deny",
        "false",
        "contradict",
    )
    mem_neg = any(m in mem for m in neg_markers)
    if not mem_neg:
        return ""

    blob = ""
    for d in (docs or [])[:10]:
        blob += " " + _norm(str(d.get("text") or ""))
    if len(blob) < 40:
        return ""

    src_neg = any(m in blob for m in neg_markers)
    if mem_neg and not src_neg:
        return (
            "提示：对话或用户画像中含否定/批评性表述，而当前检索到的文献片段可能未呈现对立立场。"
            "写作时凡需脚注与引用，一律以 Sources 为准；若与记忆不一致，请用一两句话显式说明张力，勿暗中调和。"
        )
    return ""
