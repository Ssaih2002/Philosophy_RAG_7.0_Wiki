"""Separate Chroma collection for concept-layer snippets (not mixed with corpus chunks)."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

from .memory_store import _safe_id


def _concept_chroma_dir(user_id: str) -> str:
    uid = _safe_id(user_id, default="default")
    return str(Path("data") / f"chroma_db_concepts__{uid}")


def _collection_name(user_id: str) -> str:
    return f"philosophy_concepts_{_safe_id(user_id, default='default')}"


def _get_collection(user_id: str):
    path = _concept_chroma_dir(user_id)
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(
        name=_collection_name(user_id),
        metadata={"hnsw:space": "cosine"},
    )


def concept_search_block(
    embedder,
    user_id: str,
    question: str,
    *,
    top_k: int = 6,
) -> str:
    col = _get_collection(user_id)
    try:
        n = int(col.count())
    except Exception:
        n = 0
    if n <= 0 or not (question or "").strip():
        return ""
    k = max(1, min(int(top_k), n))
    emb = embedder.encode([(question or "").strip()])[0]
    raw = col.query(query_embeddings=[emb], n_results=k)
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    if not docs:
        return ""
    parts: List[str] = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        doc = doc or ""
        anchor = ""
        if isinstance(meta, dict):
            anchor = str(meta.get("anchor") or meta.get("title") or "")
        head = f"[{i + 1}]"
        if anchor:
            head += f" anchor={anchor}"
        parts.append(head + "\n" + doc[:700])
    return "Concept vector index (separate Chroma domain; not citable evidence):\n" + "\n\n".join(
        parts
    )


def add_concept_documents(
    *,
    user_id: str,
    embedder,
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    anchor_prefix: str = "",
) -> int:
    """Embed and upsert concept texts. Each item gets a stable id."""
    col = _get_collection(user_id)
    rows: List[tuple] = []
    for i, t in enumerate(texts):
        if not (t or "").strip():
            continue
        m: Dict[str, Any] = {}
        if metadatas and i < len(metadatas) and isinstance(metadatas[i], dict):
            m.update(metadatas[i])
        if anchor_prefix:
            m.setdefault("anchor", f"{anchor_prefix}:{i}")
        rows.append((t.strip(), m))
    if not rows:
        return 0
    embs = embedder.encode([r[0] for r in rows])
    ids = [f"c_{uuid.uuid4().hex[:16]}" for _ in rows]
    docs = [r[0] for r in rows]
    metas_out = [r[1] for r in rows]
    col.add(ids=ids, documents=docs, embeddings=list(embs), metadatas=metas_out)
    return len(ids)
