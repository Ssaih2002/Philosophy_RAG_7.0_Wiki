"""
可复用的 ingest 流水线：供 CLI（ingest.py）与 Web API 调用，并支持进度事件流。
"""
from typing import Any, Callable, Dict, Generator, List, Optional

from .document_loader import load_all_documents_with_errors
from .semantic_chunker import semantic_chunk
from .embedder import Embedder
from .vector_store import VectorStore
from .sparse_retriever import SparseRetriever

_P_LOAD_END = 8.0
_P_CHUNK_END = 14.0
_P_EMBED_END = 82.0
_P_CHROMA_END = 93.0
_P_FTS_END = 99.0


def ingest_event_stream(
    profile: str,
    embedding_model: str,
    *,
    library_id: str = "default",
    data_dir: str = "data",
    embed_batch_size: int = 64,
    reload_sparse_cb: Optional[Callable[[], None]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    逐步 yield 进度事件，最后一条为 type=done。
    """
    yield {"type": "progress", "stage": "load", "percent": 0.0, "message": "加载文档…"}
    pages, load_errors = load_all_documents_with_errors(data_dir, show_progress=False)
    if load_errors and not pages:
        # 全部失败：直接抛出，让前端看到明确原因（而不是后续 0 页导致的“成功但无内容”）
        first = load_errors[0]
        raise RuntimeError(f"所有文档加载失败（errors={len(load_errors)}），示例：{first.get('path')}: {first.get('error')}")
    yield {
        "type": "progress",
        "stage": "load",
        "percent": _P_LOAD_END,
        "message": (
            f"已加载 {len(pages)} 页"
            + (f"（跳过 {len(load_errors)} 个不可读取文件）" if load_errors else "")
        ),
        "total_pages": len(pages),
        "skipped_files": len(load_errors),
    }

    yield {"type": "progress", "stage": "chunk", "percent": _P_LOAD_END, "message": "语义切分…"}
    chunks = semantic_chunk(pages, show_progress=False)
    yield {
        "type": "progress",
        "stage": "chunk",
        "percent": _P_CHUNK_END,
        "message": f"共 {len(chunks)} 个语义片段",
        "total_chunks": len(chunks),
    }

    yield {"type": "progress", "stage": "embed", "percent": _P_CHUNK_END, "message": "加载向量模型…"}
    embedder = Embedder(embedding_model)
    texts = [c["text"] for c in chunks]
    n_txt = len(texts)
    embeddings: List[List[float]] = []
    span = _P_EMBED_END - _P_CHUNK_END
    if n_txt == 0:
        yield {"type": "progress", "stage": "embed", "percent": _P_EMBED_END, "message": "无片段需编码"}
    else:
        for start in range(0, n_txt, embed_batch_size):
            end = min(start + embed_batch_size, n_txt)
            batch = texts[start:end]
            part = embedder.encode(batch, show_progress_bar=False)
            embeddings.extend(part)
            frac = end / n_txt
            pct = _P_CHUNK_END + span * frac
            yield {
                "type": "progress",
                "stage": "embed",
                "percent": round(pct, 1),
                "message": f"向量化 {end}/{n_txt}",
                "current": end,
                "total": n_txt,
            }

    yield {"type": "progress", "stage": "chroma", "percent": _P_EMBED_END, "message": "写入 Chroma…"}
    db = VectorStore(profile, library_id=library_id)
    db.reset_collection()
    db.add(chunks, embeddings, show_progress=False)
    yield {"type": "progress", "stage": "chroma", "percent": _P_CHROMA_END, "message": "向量库写入完成"}

    yield {"type": "progress", "stage": "fts", "percent": _P_CHROMA_END, "message": "构建 FTS5 索引…"}
    sparse = SparseRetriever(profile, library_id=library_id)
    sparse.rebuild(chunks, show_progress=False)
    yield {"type": "progress", "stage": "fts", "percent": _P_FTS_END, "message": "稀疏索引完成"}

    if reload_sparse_cb:
        reload_sparse_cb()

    total_pages = len(pages)
    total_chunks = len(chunks)
    yield {
        "type": "done",
        "percent": 100.0,
        "stage": "done",
        "message": "Ingest 完成",
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "skipped_files": len(load_errors),
        "skipped_examples": [e.get("path") for e in (load_errors[:5] if load_errors else [])],
    }


def run_ingest_pipeline(
    profile: str,
    embedding_model: str,
    *,
    library_id: str = "default",
    data_dir: str = "data",
    embed_batch_size: int = 64,
    emit: Optional[Callable[[Dict[str, Any]], None]] = None,
    reload_sparse_cb: Optional[Callable[[], None]] = None,
) -> Dict[str, int]:
    """同步执行 ingest，可选 emit 回调每个事件；返回 {total_pages, total_chunks}。"""
    result: Dict[str, int] = {"total_pages": 0, "total_chunks": 0}
    for ev in ingest_event_stream(
        profile,
        embedding_model,
        library_id=library_id,
        data_dir=data_dir,
        embed_batch_size=embed_batch_size,
        reload_sparse_cb=reload_sparse_cb,
    ):
        if emit:
            emit(ev)
        if ev.get("type") == "done":
            result = {
                "total_pages": int(ev["total_pages"]),
                "total_chunks": int(ev["total_chunks"]),
            }
    return result
