"""将单个文件增量写入已有 library（不 reset 全库），用于回收站恢复等场景。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .config import PROFILE_SETTINGS
from .document_loader import load_single_document
from .embedder import Embedder
from .library_manager import normalize_library_id
from .semantic_chunker import semantic_chunk
from .sparse_retriever import SparseRetriever
from .vector_store import VectorStore


def append_file_to_library(
    profile: str,
    library_id: str,
    file_path: str | Path,
    *,
    embed_batch_size: int = 64,
) -> Dict[str, Any]:
    """
    将磁盘上的单个文档切分、向量化并追加到指定 profile/library。
    chunk_id 从当前 sparse 库尾部顺延，避免与已有 chunk 冲突。
    """
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(str(p))

    lid = normalize_library_id(library_id)
    prof = (profile or "").strip() or "quality"
    params = PROFILE_SETTINGS.get(prof) or PROFILE_SETTINGS["quality"]
    embed_model = params["EMBEDDING_MODEL"]

    pages = load_single_document(str(p))
    if not pages:
        return {"profile": prof, "library_id": lid, "path": str(p), "chunks_added": 0, "message": "文件无有效页面"}

    chunks = semantic_chunk(pages, show_progress=False)
    if not chunks:
        return {"profile": prof, "library_id": lid, "path": str(p), "chunks_added": 0, "message": "切分后无片段"}

    sparse = SparseRetriever(prof, library_id=lid)
    id_start = sparse.next_chunk_id_start()

    embedder = Embedder(embed_model)
    texts = [c["text"] for c in chunks]
    embeddings: list = []
    n_txt = len(texts)
    for start in range(0, n_txt, embed_batch_size):
        end = min(start + embed_batch_size, n_txt)
        part = embedder.encode(texts[start:end], show_progress_bar=False)
        embeddings.extend(part)

    vs = VectorStore(prof, library_id=lid)
    vs.add(chunks, embeddings, show_progress=False, id_offset=id_start)
    sparse.append_chunks(chunks, id_start=id_start, show_progress=False)

    return {
        "profile": prof,
        "library_id": lid,
        "path": str(p.resolve()),
        "chunks_added": len(chunks),
        "chunk_id_start": id_start,
    }
