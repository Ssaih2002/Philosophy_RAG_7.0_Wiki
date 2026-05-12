"""User-scoped concept relation graph (NetworkX). Non-evidence context for prompts."""
from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from .memory_store import _safe_id

_LOCKS: Dict[str, threading.Lock] = {}
_GUARD = threading.Lock()


def _lock(key: str) -> threading.Lock:
    with _GUARD:
        lk = _LOCKS.get(key)
        if lk is None:
            lk = threading.Lock()
            _LOCKS[key] = lk
        return lk


def graph_path(*, user_id: str, data_dir: str = "data") -> Path:
    uid = _safe_id(user_id, default="default")
    return Path(data_dir) / "memory" / "graph" / uid / "graph.json"


def load_graph(*, user_id: str, data_dir: str = "data") -> nx.DiGraph:
    p = graph_path(user_id=user_id, data_dir=data_dir)
    lk = _lock(str(p))
    with lk:
        if not p.exists():
            return nx.DiGraph()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return nx.DiGraph()
    try:
        return nx.node_link_graph(data, edges="links", directed=True)
    except Exception:
        return nx.DiGraph()


def save_graph(g: nx.DiGraph, *, user_id: str, data_dir: str = "data") -> None:
    p = graph_path(user_id=user_id, data_dir=data_dir)
    lk = _lock(str(p))
    with lk:
        p.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(g, edges="links")
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _tokens(q: str) -> Set[str]:
    q = (q or "").lower()
    parts = re.split(r"[^\w\u4e00-\u9fff]+", q)
    return {p for p in parts if len(p) >= 2}


def _seed_nodes(g: nx.DiGraph, question: str, *, max_seeds: int = 6) -> List[str]:
    toks = _tokens(question)
    scored: List[tuple] = []
    for nid, data in g.nodes(data=True):
        label = str(data.get("label") or nid).lower()
        s = sum(1 for t in toks if t and t in label)
        if s:
            scored.append((-s, str(nid)))
    scored.sort()
    return [x[1] for x in scored[:max_seeds]]


def add_relation(
    *,
    user_id: str,
    src_label: str,
    dst_label: str,
    relation: str,
    data_dir: str = "data",
) -> None:
    """Create or update a directed edge with typed relation (philosophy-oriented)."""
    g = load_graph(user_id=user_id, data_dir=data_dir)
    sid = f"n_{hashlib.md5((src_label or '').encode('utf-8')).hexdigest()[:12]}"
    did = f"n_{hashlib.md5((dst_label or '').encode('utf-8')).hexdigest()[:12]}"
    g.add_node(sid, label=(src_label or "").strip() or sid)
    g.add_node(did, label=(dst_label or "").strip() or did)
    rel = (relation or "RELATES_TO").strip() or "RELATES_TO"
    g.add_edge(sid, did, relation=rel, type=rel)
    save_graph(g, user_id=user_id, data_dir=data_dir)


def format_subgraph_for_prompt(
    user_id: str,
    question: str,
    *,
    data_dir: str = "data",
    max_hops: int = 2,
    max_lines: int = 24,
) -> str:
    g = load_graph(user_id=user_id, data_dir=data_dir)
    if g.number_of_nodes() == 0:
        return ""

    seeds = _seed_nodes(g, question)
    if not seeds:
        seeds = list(g.nodes())[:3]

    seen: Set[str] = set()
    sub = nx.DiGraph()
    for s in seeds:
        if s not in g:
            continue
        lengths = nx.single_source_shortest_path_length(g.to_undirected(), s, cutoff=max_hops)
        for n in lengths:
            if n not in seen:
                seen.add(n)
                sub.add_node(n, **dict(g.nodes[n]))
        for u, v, ed in g.edges(data=True):
            if u in seen and v in seen:
                sub.add_edge(u, v, **ed)

    lines: List[str] = []
    for u, v, data in list(sub.edges(data=True))[:max_lines]:
        rel = str(data.get("relation") or data.get("type") or "REL")
        lu = sub.nodes[u].get("label", u)
        lv = sub.nodes[v].get("label", v)
        lines.append(f"- ({lu}) --[{rel}]--> ({lv})")
    if not lines:
        return ""
    return "Concept graph (relations only; not citable evidence):\n" + "\n".join(lines)
