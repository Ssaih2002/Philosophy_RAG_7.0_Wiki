import json
import uuid
from pathlib import Path
import os
import time
import shutil
import sqlite3
import secrets
import threading

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Any, Dict, Optional

from src.rag_engine import RAGEngine
from src.ingest_pipeline import ingest_event_stream, run_ingest_pipeline
from src.single_doc_ingest import append_file_to_library
from src.library_manager import (
    list_libraries,
    normalize_library_id,
    sparse_db_for,
    chroma_dir_for,
    chroma_collection_for,
    validate_library_id,
)
from src.source_sort_key import sort_library_sources_by_reading
from src.library_language import compute_and_cache_language_profile, load_cached_language_profile
from src.memory_store import conversation_paths, load_recent_turns
from src import config
from src.wiki_manager import wiki_paths, compact_user_wiki_sync
from src.llm_router import generate_answer
from src.trace import new_trace_id, log_stage
from src.translation_pipeline import (
    export_project,
    list_translation_projects,
    load_glossary,
    load_global_glossary,
    load_project,
    prepare_translation_project,
    save_glossary,
    save_global_glossary,
    translate_project,
)
from pathlib import Path


app = FastAPI(title="Philosophy RAG Web API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # 前端通常是直接打开 frontend.html（Origin 可能为 "null"），且不需要 cookie；
    # 关闭 credentials 可避免某些浏览器对 "*" + credentials 的 CORS 拒绝。
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine = RAGEngine()


@app.on_event("startup")
def _startup_warmup():
    """
    Optional warmup to avoid first-request latency spikes.
    Enable with PRELOAD_RERANKER=1 (default) / PRELOAD_RERANKER=0.
    """
    try:
        v = (os.getenv("PRELOAD_RERANKER") or "").strip()
        preload = True if v == "" else bool(int(v))
    except Exception:
        preload = True
    if not preload:
        return
    try:
        # Load CrossEncoder once; stays in memory as long as the process lives.
        rag_engine.reranker._get_model()  # type: ignore[attr-defined]
    except Exception as e:
        # Warmup failure should never prevent server start.
        print(f"[startup] reranker warmup skipped: {e}")

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
LIBRARY_DOCS_DIR = DATA_DIR / "library_docs"
TRANSLATION_UPLOAD_DIR = DATA_DIR / "translation_uploads"
ALLOWED_UPLOAD_EXT = {".pdf", ".docx", ".json", ".epub"}
MAX_UPLOAD_BYTES = 120 * 1024 * 1024  # 单文件上限 120MB


class QuestionRequest(BaseModel):
    question: str
    # --- conversation memory (optional; backward compatible) ---
    user_id: str = "default"
    conversation_id: Optional[str] = None
    memory: bool = True
    history_max_turns: int = 10
    history_max_chars: int = 12000
    wiki_max_chars: int = 3500
    use_concept_graph: bool = True
    use_concept_index: bool = False
    keyword_terms: Optional[List[str]] = None
    source_filters: Optional[List[str]] = None
    auto_extract_keywords: bool = True
    use_hybrid: bool = True
    use_rerank: bool = True
    # 是否同时检索 SEP（斯坦福哲学百科）向量库作为“弱参照”
    use_sep_reference: bool = False
    # 与 src/academic_prompt.py 中 STYLE_* 一致；旧英文 value 仍由 normalize_answer_style 兼容
    answer_style: str = "哲学论述"
    # 供应商：gemini | openai | deepseek
    llm_provider: str = "gemini"
    # 模型 id：如 gemini-3.1-pro-preview / gemini-2.5-pro / gemini-2.5-flash / gpt-5.1 / deepseek-reasoner
    llm_model: str = "gemini-3.1-pro-preview"
    # 是否开启“超长回答”（更高 max_output_tokens，可能更慢且在高峰期更易失败）
    ultra_long_answer: bool = False
    # 多库融合检索：每次请求显式指定
    library_ids: Optional[List[str]] = None
    library_weights: Optional[List[float]] = None


class DocItem(BaseModel):
    text: str
    page: Any
    source: str
    chunk_id: Optional[str] = None


class AnswerResponse(BaseModel):
    answer: str
    docs: List[DocItem]
    conversation_id: str = ""
    keyword_hit_docs: List[DocItem] = Field(default_factory=list)
    keyword_source_stats: List[Dict[str, Any]] = Field(default_factory=list)
    profile: str = "quality"
    keywords_used: List[str] = Field(default_factory=list)
    source_filters_used: List[str] = Field(default_factory=list)
    user_terms_used: List[str] = Field(default_factory=list)
    auto_terms_used: List[str] = Field(default_factory=list)
    dropped_terms: List[str] = Field(default_factory=list)
    keyword_query: str = ""
    term_source: str = "question"
    hybrid: bool = False
    reranked: bool = False
    # 与 src/academic_prompt.py 中 STYLE_* 一致；旧英文 value 仍由 normalize_answer_style 兼容
    answer_style: str = "哲学论述"
    answer_model: str = ""
    answer_max_output_tokens: int = 0
    debug: Dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    library_id: str = "default"


class IngestResponse(BaseModel):
    total_pages: int
    total_chunks: int


class ProfileRequest(BaseModel):
    profile: str


class ProfileResponse(BaseModel):
    profile: str
    available_profiles: List[str]
    needs_reingest: bool = True


class CompactWikiRequest(BaseModel):
    user_id: str = "default"
    llm_provider: str = "gemini"
    llm_model: str = ""


class SuggestSessionTitleRequest(BaseModel):
    question: str = ""
    answer: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    max_output_tokens: int = 0


class TranslationGlossaryRequest(BaseModel):
    glossary: Dict[str, Any] = Field(default_factory=dict)


class TranslationGlobalGlossaryRequest(BaseModel):
    glossary: Dict[str, Any] = Field(default_factory=dict)


class TranslationRunRequest(BaseModel):
    llm_provider: str = ""
    llm_model: str = ""
    max_chunks: int = 0
    concurrency: int = 1
    resume: bool = True


class TranslationExportRequest(BaseModel):
    format: str = "txt"
    output_path: str = ""


@app.get("/api/conversations/list")
def api_conversations_list(user_id: str = "default") -> Dict[str, Any]:
    """
    List conversation ids that already exist on disk for this user.
    The UI uses this to seed localStorage on first load / new browser.
    """
    uid = (user_id or "default").strip() or "default"
    root = conversation_paths(user_id=uid, conversation_id="conv", data_dir="data").root
    out = []
    try:
        if root.exists():
            items = []
            for p in root.glob("*.jsonl"):
                try:
                    st = p.stat()
                    items.append((st.st_mtime, p))
                except Exception:
                    items.append((0.0, p))
            items.sort(key=lambda x: x[0], reverse=True)
            for _, p in items[:200]:
                cid = p.stem
                if cid:
                    out.append(cid)
    except Exception:
        out = []
    return {"ok": True, "user_id": uid, "conversation_ids": out, "root": str(root)}


_DELETE_TOKENS: Dict[str, Dict[str, Any]] = {}
_DELETE_TOKEN_TTL_SECONDS = 120
_LIB_LOCKS: Dict[str, threading.Lock] = {}


def _lib_lock_key(profile: str, library_id: str) -> str:
    return f"{profile}__{normalize_library_id(library_id)}"


def _get_lib_lock(profile: str, library_id: str) -> threading.Lock:
    k = _lib_lock_key(profile, library_id)
    lock = _LIB_LOCKS.get(k)
    if lock is None:
        lock = threading.Lock()
        _LIB_LOCKS[k] = lock
    return lock


def _trash_dir() -> Path:
    p = DATA_DIR / "trash"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_source_filename(source: str) -> str:
    # source 通常就是文件名；禁止路径穿越
    return Path(source).name


def _find_source_file(source: str) -> Optional[Path]:
    name = _safe_source_filename(source)
    if not name:
        return None
    # 优先在 pdf/ 与 uploads/ 找
    for base in [DATA_DIR / "pdf", UPLOAD_DIR]:
        cand = base / name
        if cand.exists() and cand.is_file():
            return cand
    return None


def _sparse_stats(db_path: str) -> Dict[str, int]:
    if not db_path or not Path(db_path).exists():
        return {"chunks": 0, "sources": 0}
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        # chunks 表为本项目 schema 固定存在
        chunks = int(cur.execute("SELECT COUNT(1) FROM chunks").fetchone()[0])
        sources = int(cur.execute("SELECT COUNT(1) FROM (SELECT DISTINCT source FROM chunks)").fetchone()[0])
        return {"chunks": chunks, "sources": sources}
    except Exception:
        return {"chunks": 0, "sources": 0}
    finally:
        con.close()


@app.get("/api/libraries")
def api_list_libraries() -> Dict[str, Any]:
    libs = list_libraries("data")
    out: List[Dict[str, Any]] = []
    for k in libs:
        sparse_path = sparse_db_for(k.profile, k.library_id)
        stats = _sparse_stats(sparse_path)
        chroma_dir = chroma_dir_for(k.profile, k.library_id)
        chroma_collection_expected = chroma_collection_for(k.profile, k.library_id)
        chroma_collection_used = chroma_collection_expected
        chroma_count = None
        try:
            from src.vector_store import VectorStore

            vs = VectorStore(k.profile, library_id=k.library_id)
            chroma_collection_used = getattr(vs, "collection_name", chroma_collection_expected)
            try:
                chroma_count = int(vs.collection.count())
            except Exception:
                chroma_count = None
        except Exception:
            # If Chroma is missing/corrupted, keep expected names and omit counts.
            pass
        lp = load_cached_language_profile(k.profile, k.library_id, data_dir="data")
        out.append(
            {
                "profile": k.profile,
                "library_id": k.library_id,
                "key": k.key,
                "chroma_path": chroma_dir,
                "chroma_collection": chroma_collection_expected,
                "chroma_collection_used": chroma_collection_used,
                "chroma_count": chroma_count,
                "sparse_db_path": sparse_path,
                "chunks_count": stats["chunks"],
                "sources_count": stats["sources"],
                # language profile (cached; computed on-demand by a dedicated endpoint)
                "language_profile": (lp.dist if lp else None),
            }
        )
    return {"libraries": out}


@app.get("/api/libraries/{profile}/{library_id}/language_profile")
def api_library_language_profile(profile: str, library_id: str, force: int = 0) -> Dict[str, Any]:
    """
    Compute (or load cached) language profile for a library.
    This is designed for legacy libraries that were ingested before language profiling existed.
    """
    lid = normalize_library_id(library_id)
    lp = compute_and_cache_language_profile(
        profile,
        lid,
        data_dir="data",
        force_recompute=bool(int(force or 0)),
    )
    if not lp:
        return {"profile": profile, "library_id": lid, "language_profile": None, "cached": False}
    return {
        "profile": profile,
        "library_id": lid,
        "language_profile": lp.dist,
        "sampled_chunks": lp.sampled_chunks,
        "updated_at_ts": lp.updated_at_ts,
        "cached": True,
    }


@app.get("/api/memory/view")
def api_memory_view(user_id: str = "default", conversation_id: str = "") -> Dict[str, Any]:
    """
    Read-only viewer for:
    - recent conversation turns (JSONL)
    - user wiki (user.md)
    """
    uid = (user_id or "default").strip() or "default"
    cid = (conversation_id or "").strip() or "default"
    # history
    msgs = load_recent_turns(
        user_id=uid,
        conversation_id=cid,
        max_turns=12,
        max_chars=16000,
        data_dir="data",
    )
    history = [{"role": r, "content": c} for (r, c) in msgs]
    # paths
    cpaths = conversation_paths(user_id=uid, conversation_id=cid, data_dir="data")
    wpaths = wiki_paths(user_id=uid, data_dir="data")
    wiki_md = ""
    try:
        if wpaths.user.exists():
            wiki_md = wpaths.user.read_text(encoding="utf-8")
    except Exception:
        wiki_md = ""
    # 与注入 prompt 一致：全文返回，便于侧栏核对是否读全
    return {
        "user_id": uid,
        "conversation_id": cid,
        "history": history,
        "wiki_md": wiki_md,
        "wiki_chars": len(wiki_md),
        "paths": {
            "conversation_jsonl": str(cpaths.jsonl_path),
            "wiki_user_md": str(wpaths.user),
            "wiki_index_md": str(wpaths.index),
            "wiki_log_md": str(wpaths.log),
        },
    }


@app.post("/api/wiki/compact")
def api_wiki_compact(req: CompactWikiRequest = CompactWikiRequest()) -> Dict[str, Any]:
    uid = (req.user_id or "default").strip() or "default"
    prov = (req.llm_provider or getattr(config, "WIKI_LLM_PROVIDER", "gemini")).strip() or "gemini"
    model = (req.llm_model or "").strip() or ""
    if not model:
        # default to the same model as wiki updates (fast/stable)
        model = getattr(config, "WIKI_LLM_MODEL", "gemini-2.5-flash")
    out = compact_user_wiki_sync(
        user_id=uid,
        llm_provider=prov,
        llm_model=model,
        data_dir="data",
        reason="manual_api",
    )
    return {"ok": bool(out.get("ok")), "result": out}


@app.post("/api/session/suggest_title")
def api_session_suggest_title(req: SuggestSessionTitleRequest = SuggestSessionTitleRequest()) -> Dict[str, Any]:
    q = (req.question or "").strip()
    a = (req.answer or "").strip()
    if not q and not a:
        return {"ok": False, "title": "", "used_model": ""}

    prov = (req.llm_provider or getattr(config, "SESSION_TITLE_LLM_PROVIDER", "gemini")).strip() or "gemini"
    model = (req.llm_model or getattr(config, "SESSION_TITLE_LLM_MODEL", "gemini-2.5-flash")).strip()
    max_out = int(req.max_output_tokens or getattr(config, "SESSION_TITLE_MAX_OUTPUT_TOKENS", 48) or 48)
    trace_id = new_trace_id()

    prompt = (
        "你要为一次对话会话生成一个短标题。\n"
        "要求：中文；5-18 个字；不加引号；不要句号；不要前缀（如“会话：”）；不要提及模型。\n"
        "只输出标题本身。\n\n"
        f"用户问题：{q[:600]}\n\n"
        f"助手回答摘要（可能截断）：{a[:900]}\n"
    )
    try:
        title, used = generate_answer(
            prompt=prompt,
            provider=prov,
            model=model,
            temperature=0.3,
            max_output_tokens=max(16, min(96, max_out)),
            trace_id=trace_id,
            stage="session.title",
        )
        t = (title or "").strip().replace("\n", " ")
        # sanitize quotes / bullets
        for ch in ['"', "“", "”", "「", "」", "『", "』"]:
            t = t.replace(ch, "")
        t = t.strip(" -—–·•\t")
        if len(t) > 32:
            t = t[:32].rstrip()
        return {"ok": bool(t), "title": t, "used_model": used}
    except Exception as e:
        try:
            log_stage(trace_id=trace_id, stage="session.title", event="error", extra={"err": str(e)[:200]})
        except Exception:
            pass
        return {"ok": False, "title": "", "used_model": ""}


@app.get("/api/libraries/{profile}/{library_id}/sources")
def api_library_sources(profile: str, library_id: str) -> Dict[str, Any]:
    lid = normalize_library_id(library_id)
    db_path = sparse_db_for(profile, lid)
    if not Path(db_path).exists():
        return {"profile": profile, "library_id": lid, "sources": []}
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        rows = cur.execute(
            "SELECT source, COUNT(1) AS n FROM chunks GROUP BY source"
        ).fetchall()
        sources = [{"source": r[0], "chunks": int(r[1])} for r in rows if r and r[0]]
        sources = sort_library_sources_by_reading(sources)
        return {
            "profile": profile,
            "library_id": lid,
            "sources": sources,
        }
    finally:
        con.close()


@app.post("/api/libraries/{profile}/{library_id}/sources/{source}/prepare_delete")
def api_prepare_delete_source(profile: str, library_id: str, source: str) -> Dict[str, Any]:
    lid = normalize_library_id(library_id)
    tok = secrets.token_urlsafe(18)
    _DELETE_TOKENS[tok] = {
        "exp": time.time() + _DELETE_TOKEN_TTL_SECONDS,
        "profile": profile,
        "library_id": lid,
        "source": source,
    }
    return {"confirm_token": tok, "expires_in_seconds": _DELETE_TOKEN_TTL_SECONDS}


@app.delete("/api/libraries/{profile}/{library_id}/sources/{source}")
def api_delete_source(
    profile: str,
    library_id: str,
    source: str,
    confirm_token: str,
) -> Dict[str, Any]:
    lid = normalize_library_id(library_id)
    lock = _get_lib_lock(profile, lid)
    meta = _DELETE_TOKENS.get(confirm_token)
    if not meta or meta.get("exp", 0) < time.time():
        raise HTTPException(status_code=400, detail="confirm_token invalid/expired")
    if meta.get("profile") != profile or meta.get("library_id") != lid or meta.get("source") != source:
        raise HTTPException(status_code=400, detail="confirm_token mismatch")
    _DELETE_TOKENS.pop(confirm_token, None)

    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="library busy (ingest/merge/delete in progress)")

    chunk_ids: List[str] = []
    deleted_from_chroma = 0
    try:
        # 1) 从 sparse DB 找 chunk_ids 并删除
        sparse_path = sparse_db_for(profile, lid)
        if Path(sparse_path).exists():
            con = sqlite3.connect(sparse_path)
            try:
                cur = con.cursor()
                chunk_ids = [
                    r[0]
                    for r in cur.execute(
                        "SELECT chunk_id FROM chunks WHERE source = ?", (source,)
                    ).fetchall()
                ]
                if chunk_ids:
                    cur.execute(
                        "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE source = ?)",
                        (source,),
                    )
                    cur.execute("DELETE FROM chunks WHERE source = ?", (source,))
                    con.commit()
            finally:
                con.close()

        # 2) 从 Chroma 删除对应 ids（ids == chunk_id）
        try:
            from src.vector_store import VectorStore

            vs = VectorStore(profile, library_id=lid)
            if chunk_ids:
                vs.collection.delete(ids=chunk_ids)
                deleted_from_chroma = len(chunk_ids)
        except Exception:
            deleted_from_chroma = 0
    finally:
        try:
            lock.release()
        except Exception:
            pass

    # 3) 磁盘文件移动到 trash（软删除）
    moved = False
    moved_from = None
    moved_to = None
    f = _find_source_file(source)
    if f:
        tdir = _trash_dir() / _now_ts()
        tdir.mkdir(parents=True, exist_ok=True)
        dest = tdir / f.name
        try:
            shutil.move(str(f), str(dest))
            moved = True
            moved_from = str(f)
            moved_to = str(dest)
            (tdir / "manifest.json").write_text(
                json.dumps(
                    {
                        "profile": profile,
                        "library_id": lid,
                        "source": source,
                        "moved_from": moved_from,
                        "moved_to": moved_to,
                        "chunk_ids_count": len(chunk_ids),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            moved = False

    return {
        "profile": profile,
        "library_id": lid,
        "source": source,
        "chunk_ids_deleted": len(chunk_ids),
        "deleted_from_chroma": deleted_from_chroma,
        "file_moved_to_trash": moved,
        "file_from": moved_from,
        "file_to": moved_to,
    }


def _safe_trash_session(name: str) -> str:
    n = (name or "").strip()
    if not n or "/" in n or "\\" in n or n in (".", ".."):
        raise HTTPException(status_code=400, detail="invalid trash session")
    return n


@app.get("/api/trash/items")
def api_trash_list() -> Dict[str, Any]:
    root = _trash_dir()
    items: List[Dict[str, Any]] = []
    if not root.exists():
        return {"items": items}
    for d in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
        if not d.is_dir():
            continue
        mf = d / "manifest.json"
        if not mf.exists():
            continue
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        items.append(
            {
                "trash_session": d.name,
                "profile": data.get("profile"),
                "library_id": data.get("library_id"),
                "source": data.get("source"),
                "file_in_trash": data.get("moved_to"),
                "original_path": data.get("moved_from"),
            }
        )
    return {"items": items}


@app.post("/api/trash/{trash_session}/restore")
def api_trash_restore(trash_session: str) -> Dict[str, Any]:
    """
    将回收站中的文件移回原始路径（manifest.moved_from），并对该库做增量重建索引。
    """
    ts = _safe_trash_session(trash_session)
    tdir = _trash_dir() / ts
    mf = tdir / "manifest.json"
    if not mf.exists():
        raise HTTPException(status_code=404, detail="manifest not found")
    try:
        data = json.loads(mf.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"manifest invalid: {e}") from e

    profile = str(data.get("profile") or "").strip() or "quality"
    lid = normalize_library_id(str(data.get("library_id") or "default"))
    moved_from = data.get("moved_from")
    moved_to = data.get("moved_to")
    if not moved_from or not moved_to:
        raise HTTPException(status_code=400, detail="manifest missing moved_from/moved_to")

    src = Path(moved_to)
    dest = Path(moved_from)
    if not src.is_file():
        raise HTTPException(status_code=400, detail="回收站内文件已不存在，可能已被移动或删除")

    lock = _get_lib_lock(profile, lid)
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="library busy (ingest/merge/delete/restore in progress)")

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            raise HTTPException(
                status_code=409,
                detail=f"目标路径已存在文件，为避免覆盖已中止：{dest}",
            )
        shutil.move(str(src), str(dest))
        try:
            out = append_file_to_library(profile, lid, dest)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"文件已恢复到 {dest}，但增量索引失败：{e}。可稍后对该库重新运行 Ingest。",
            ) from e
        try:
            shutil.rmtree(tdir, ignore_errors=True)
        except Exception:
            pass
        return {
            "ok": True,
            "trash_session": ts,
            "restored_to": str(dest.resolve()),
            **out,
        }
    finally:
        try:
            lock.release()
        except Exception:
            pass


class MergeLibrariesRequest(BaseModel):
    profile: str = "quality"
    source_library_ids: List[str]
    target_library_id: str


@app.post("/api/libraries/merge")
def api_merge_libraries(req: MergeLibrariesRequest) -> Dict[str, Any]:
    profile = (req.profile or "").strip() or rag_engine.get_profile()
    target = normalize_library_id(req.target_library_id)
    srcs = [
        normalize_library_id(x)
        for x in (req.source_library_ids or [])
        if x and str(x).strip()
    ]
    if not srcs:
        raise HTTPException(status_code=400, detail="source_library_ids is required")
    if target in srcs:
        raise HTTPException(status_code=400, detail="target_library_id must not be in sources")

    lock = _get_lib_lock(profile, target)
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="target library busy")
    try:
        from src.vector_store import VectorStore
        from src.sparse_retriever import SparseRetriever

        tgt_vs = VectorStore(profile, library_id=target)
        tgt_sparse = SparseRetriever(profile, library_id=target)  # ensure schema
        con = sqlite3.connect(tgt_sparse.path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        merged_chunks = 0
        for src_lid in srcs:
            src_vs = VectorStore(profile, library_id=src_lid)
            src = src_vs.collection.get(include=["documents", "embeddings", "metadatas"])
            docs = src.get("documents") or []
            metas = src.get("metadatas") or []
            embs = src.get("embeddings") or []
            n = min(len(docs), len(metas), len(embs))
            for i in range(n):
                doc = docs[i] or ""
                meta = metas[i] or {}
                page = meta.get("page", "")
                source = meta.get("source", "Unknown")
                new_id = f"{src_lid}_extra_{merged_chunks}"
                tgt_vs.collection.add(
                    ids=[new_id],
                    documents=[doc],
                    embeddings=[embs[i]],
                    metadatas=[{"page": page, "source": source, "chunk_id": new_id}],
                )
                cur.execute(
                    "INSERT OR REPLACE INTO chunks(chunk_id, text, page, source) VALUES (?, ?, ?, ?)",
                    (new_id, doc, str(page), source or "Unknown"),
                )
                cur.execute(
                    "INSERT INTO chunks_fts(text, chunk_id) VALUES (?, ?)",
                    (doc, new_id),
                )
                merged_chunks += 1

        con.commit()
        con.close()
        return {
            "profile": profile,
            "target_library_id": target,
            "source_library_ids": srcs,
            "merged_chunks": merged_chunks,
        }
    finally:
        try:
            lock.release()
        except Exception:
            pass


def _safe_unique_upload_path(original_name: str) -> Path:
    name = Path(original_name).name
    if not name or name.strip() != name or ".." in name:
        raise HTTPException(status_code=400, detail="非法文件名")
    suf = Path(name).suffix.lower()
    if suf not in ALLOWED_UPLOAD_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"仅支持扩展名：{', '.join(sorted(ALLOWED_UPLOAD_EXT))}",
        )
    stem = Path(name).stem
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    candidate = UPLOAD_DIR / name
    if not candidate.exists():
        return candidate
    for i in range(1, 10000):
        alt = UPLOAD_DIR / f"{stem}_{i}{suf}"
        if not alt.exists():
            return alt
    return UPLOAD_DIR / f"{stem}_{uuid.uuid4().hex[:10]}{suf}"


def _library_docs_dir(*, profile: str, library_id: str) -> Path:
    """
    Per-library document root (upload destination & ingest input).
    Example: data/library_docs/quality__graduation_dissertation/
    """
    lid = normalize_library_id(library_id)
    # keep compatibility: default library still uses global data/ scan (legacy behavior)
    if lid == "default":
        return DATA_DIR
    validate_library_id(lid)
    LIBRARY_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    d = LIBRARY_DOCS_DIR / f"{profile}__{lid}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_unique_path_in_dir(root: Path, original_name: str) -> Path:
    name = Path(original_name).name
    if not name or name.strip() != name or ".." in name:
        raise HTTPException(status_code=400, detail="非法文件名")
    suf = Path(name).suffix.lower()
    if suf not in ALLOWED_UPLOAD_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"仅支持扩展名：{', '.join(sorted(ALLOWED_UPLOAD_EXT))}",
        )
    stem = Path(name).stem
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / name
    if not candidate.exists():
        return candidate
    for i in range(1, 10000):
        alt = root / f"{stem}_{i}{suf}"
        if not alt.exists():
            return alt
    return root / f"{stem}_{uuid.uuid4().hex[:10]}{suf}"


def _safe_translation_upload_path(original_name: str) -> Path:
    return _safe_unique_path_in_dir(TRANSLATION_UPLOAD_DIR, original_name)


@app.post("/api/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    library_id: str = Form("default"),
) -> Dict[str, Any]:
    """
    将拖入的文件保存到「目标库」对应的文档目录，随后对该库运行 ingest 即可建立索引。

    - default 库：保持旧行为（保存到 data/uploads/；ingest 扫描 data/ 全量）
    - 非 default 库：保存到 data/library_docs/<profile>__<library_id>/；ingest 仅扫描该目录
    """
    if not files:
        raise HTTPException(status_code=400, detail="未选择文件")
    profile = rag_engine.get_profile()
    lid = normalize_library_id(library_id)
    dest_root = _library_docs_dir(profile=profile, library_id=lid)
    saved: List[str] = []
    for f in files:
        if not f.filename:
            continue
        if lid == "default":
            dest = _safe_unique_upload_path(f.filename)
        else:
            dest = _safe_unique_path_in_dir(dest_root, f.filename)
        body = await f.read()
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"文件过大（>{MAX_UPLOAD_BYTES // (1024 * 1024)}MB）：{f.filename}",
            )
        dest.write_bytes(body)
        saved.append(dest.name)
    return {
        "saved": saved,
        "count": len(saved),
        "dir": str(dest_root).replace("\\", "/"),
        "profile": profile,
        "library_id": lid,
    }


@app.post("/api/ingest", response_model=IngestResponse)
def run_ingest() -> Dict[str, int]:
    """
    在前端点击按钮时运行 ingest 流程（无流式进度，兼容旧客户端）。
    """
    profile = rag_engine.get_profile()
    return run_ingest_pipeline(
        profile,
        rag_engine.params["EMBEDDING_MODEL"],
        data_dir="data",
        reload_sparse_cb=rag_engine.reload_sparse,
    )


@app.post("/api/ingest/stream")
def run_ingest_stream(req: IngestRequest = IngestRequest()):
    """
    NDJSON 流：每行一个 JSON，含 type=progress|done|error。
    """
    profile = rag_engine.get_profile()
    embed_model = rag_engine.params["EMBEDDING_MODEL"]
    library_id = normalize_library_id(req.library_id)
    lock = _get_lib_lock(profile, library_id)
    # 非 default：仅 ingest 该库对应的文档目录（上传时也会存入该目录）
    data_dir = str(_library_docs_dir(profile=profile, library_id=library_id))

    def ndjson_gen():
        if not lock.acquire(blocking=False):
            err = {"type": "error", "message": f"library 正在被写入：{profile}__{library_id}"}
            yield (json.dumps(err, ensure_ascii=False) + "\n").encode("utf-8")
            return
        try:
            for ev in ingest_event_stream(
                profile,
                embed_model,
                library_id=library_id,
                data_dir=data_dir,
                reload_sparse_cb=rag_engine.reload_sparse,
            ):
                yield (json.dumps(ev, ensure_ascii=False) + "\n").encode("utf-8")
        except Exception as e:
            err = {"type": "error", "message": str(e)}
            yield (json.dumps(err, ensure_ascii=False) + "\n").encode("utf-8")
        finally:
            try:
                lock.release()
            except Exception:
                pass

    return StreamingResponse(
        ndjson_gen(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/translation/projects")
def api_translation_projects() -> Dict[str, Any]:
    return {"projects": list_translation_projects()}


@app.get("/api/translation/glossary")
def api_translation_global_glossary(target_language: str = "") -> Dict[str, Any]:
    return {"glossary": load_global_glossary(target_language=target_language)}


@app.put("/api/translation/glossary")
def api_translation_save_global_glossary(req: TranslationGlobalGlossaryRequest) -> Dict[str, Any]:
    try:
        return {"glossary": save_global_glossary(req.glossary)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/translation/projects/{project_id}")
def api_translation_project(project_id: str) -> Dict[str, Any]:
    try:
        state = load_project(project_id)
        glossary = load_glossary(project_id, confirmed=False)
        confirmed = load_glossary(project_id, confirmed=True)
        return {"state": state, "glossary_draft": glossary, "glossary": confirmed}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="翻译项目不存在")


@app.post("/api/translation/prepare")
async def api_translation_prepare(
    file: UploadFile = File(...),
    target_language: str = Form("zh-CN"),
    llm_provider: str = Form("gemini"),
    llm_model: str = Form(""),
    project_id: str = Form(""),
) -> Dict[str, Any]:
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="未选择文件")
    dest = _safe_translation_upload_path(file.filename)
    body = await file.read()
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大（>{MAX_UPLOAD_BYTES // (1024 * 1024)}MB）：{file.filename}",
        )
    dest.write_bytes(body)
    try:
        events: List[Dict[str, Any]] = []
        state = prepare_translation_project(
            dest,
            target_language=target_language,
            provider=llm_provider,
            model=llm_model,
            project_id=project_id,
            emit=lambda ev: events.append(ev),
        )
        return {
            "state": state,
            "glossary_draft": load_glossary(state["project_id"], confirmed=False),
            "events": events,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/translation/projects/{project_id}/glossary")
def api_translation_save_glossary(project_id: str, req: TranslationGlossaryRequest) -> Dict[str, Any]:
    try:
        glossary = save_glossary(project_id, req.glossary)
        return {"project_id": project_id, "glossary": glossary}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="翻译项目不存在")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/translation/projects/{project_id}/translate")
def api_translation_run(project_id: str, req: TranslationRunRequest = TranslationRunRequest()) -> Dict[str, Any]:
    try:
        events: List[Dict[str, Any]] = []
        state = translate_project(
            project_id,
            provider=req.llm_provider,
            model=req.llm_model,
            max_chunks=max(0, int(req.max_chunks or 0)),
            concurrency=min(20, max(1, int(req.concurrency or 1))),
            resume=bool(req.resume),
            emit=lambda ev: events.append(ev),
        )
        return {"state": state, "events": events}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="翻译项目不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/translation/projects/{project_id}/export")
def api_translation_export(project_id: str, req: TranslationExportRequest = TranslationExportRequest()) -> Dict[str, Any]:
    fmt = (req.format or "txt").strip().lower().lstrip(".")
    if fmt not in {"txt", "docx"}:
        raise HTTPException(status_code=400, detail="仅支持 txt 或 docx")
    try:
        out = export_project(project_id, output_format=fmt, output_path=req.output_path or "")
        return {
            "project_id": project_id,
            "path": str(out).replace("\\", "/"),
            "download_url": f"/api/translation/projects/{project_id}/download?format={fmt}",
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="翻译项目不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/translation/projects/{project_id}/download")
def api_translation_download(project_id: str, format: str = "txt") -> FileResponse:
    fmt = (format or "txt").strip().lower().lstrip(".")
    if fmt not in {"txt", "docx"}:
        raise HTTPException(status_code=400, detail="仅支持 txt 或 docx")
    try:
        state = load_project(project_id)
        last = str(state.get("last_export") or "").strip()
        path = Path(last) if last and last.lower().endswith(f".{fmt}") else export_project(project_id, output_format=fmt)
        if not path.exists():
            raise FileNotFoundError(str(path))
        media = "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if fmt == "docx" else "text/plain"
        return FileResponse(path, media_type=media, filename=path.name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="导出文件不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/profile", response_model=ProfileResponse)
def get_profile() -> Dict[str, Any]:
    return {
        "profile": rag_engine.get_profile(),
        "available_profiles": rag_engine.get_profile_options(),
        "needs_reingest": True,
    }


@app.post("/api/profile", response_model=ProfileResponse)
def set_profile(req: ProfileRequest) -> Dict[str, Any]:
    ok = rag_engine.switch_profile(req.profile)
    if not ok:
        return {
            "profile": rag_engine.get_profile(),
            "available_profiles": rag_engine.get_profile_options(),
            "needs_reingest": True,
        }
    return {
        "profile": rag_engine.get_profile(),
        "available_profiles": rag_engine.get_profile_options(),
        "needs_reingest": True,
    }


@app.post("/api/answer", response_model=AnswerResponse)
def answer_question(req: QuestionRequest) -> Dict[str, Any]:
    """
    等价于 chat.py 中调用 RAGEngine().answer。
    """
    # conversation id: if not provided, generate one and return it
    conv_id = (req.conversation_id or "").strip() or uuid.uuid4().hex
    trace_id = new_trace_id()
    log_stage(
        trace_id=trace_id,
        stage="http.answer",
        event="start",
        extra={
            "conv": conv_id[:8],
            "provider": req.llm_provider,
            "model": req.llm_model,
            "rerank": bool(req.use_rerank),
            "hybrid": bool(req.use_hybrid),
            "style": req.answer_style,
        },
    )
    answer, docs, meta = rag_engine.answer(
        req.question,
        user_id=(req.user_id or "default"),
        conversation_id=conv_id,
        memory=bool(req.memory),
        history_max_turns=max(0, int(req.history_max_turns or 0)),
        history_max_chars=max(0, int(req.history_max_chars or 0)),
        wiki_max_chars=max(0, int(req.wiki_max_chars or 0)),
        use_concept_graph=bool(req.use_concept_graph),
        use_concept_index=bool(req.use_concept_index),
        keyword_terms=req.keyword_terms,
        source_filters=req.source_filters,
        auto_extract_keywords=req.auto_extract_keywords,
        use_hybrid=req.use_hybrid,
        use_rerank=req.use_rerank,
        use_sep_reference=req.use_sep_reference,
        answer_style=req.answer_style,
        llm_provider=req.llm_provider,
        llm_model=req.llm_model,
        ultra_long_answer=req.ultra_long_answer,
        library_ids=req.library_ids,
        library_weights=req.library_weights,
        trace_id=trace_id,
    )
    log_stage(trace_id=trace_id, stage="http.answer", event="ok", extra={"docs": len(docs), "reranked": bool(meta.get("reranked"))})
    return {
        "answer": answer,
        "docs": docs,
        "conversation_id": conv_id,
        "keyword_hit_docs": meta.get("keyword_hit_docs", []),
        "keyword_source_stats": meta.get("keyword_source_stats", []),
        "profile": meta.get("profile", "quality"),
        "keywords_used": meta.get("keywords_used", []),
        "source_filters_used": meta.get("source_filters_used", []),
        "user_terms_used": meta.get("user_terms_used", []),
        "auto_terms_used": meta.get("auto_terms_used", []),
        "dropped_terms": meta.get("dropped_terms", []),
        "keyword_query": meta.get("keyword_query", ""),
        "term_source": meta.get("term_source", "question"),
        "hybrid": bool(meta.get("hybrid", False)),
        "reranked": bool(meta.get("reranked", False)),
        "sep_reference_enabled": bool(meta.get("sep_reference_enabled", False)),
        "sep_docs_kept": int(meta.get("sep_docs_kept", 0)),
        "sep_max_docs": int(meta.get("sep_max_docs", 0)),
        "sep_weight": float(meta.get("sep_weight", 0.0)),
        "answer_style": meta.get("answer_style", req.answer_style),
        "answer_model": meta.get("answer_model", ""),
        "answer_max_output_tokens": int(meta.get("answer_max_output_tokens", 0)),
        "debug": meta.get("debug", {}),
    }


@app.post("/api/answer/stream")
def answer_question_stream(req: QuestionRequest) -> StreamingResponse:
    """
    NDJSON 流：首行 meta，随后多行 delta（将完整回答分块输出，便于前端边收边显示），末行 final 含完整结果。
    另：后台 wiki 更新仍由 RAGEngine 内部异步线程处理，与单次 /api/answer 一致。
    """
    conv_id = (req.conversation_id or "").strip() or uuid.uuid4().hex
    trace_id = new_trace_id()

    def ndjson_gen():
        log_stage(
            trace_id=trace_id,
            stage="http.answer_stream",
            event="start",
            extra={"conv": conv_id[:8], "provider": req.llm_provider, "model": req.llm_model},
        )
        answer, docs, meta = rag_engine.answer(
            req.question,
            user_id=(req.user_id or "default"),
            conversation_id=conv_id,
            memory=bool(req.memory),
            history_max_turns=max(0, int(req.history_max_turns or 0)),
            history_max_chars=max(0, int(req.history_max_chars or 0)),
            wiki_max_chars=max(0, int(req.wiki_max_chars or 0)),
            use_concept_graph=bool(req.use_concept_graph),
            use_concept_index=bool(req.use_concept_index),
            keyword_terms=req.keyword_terms,
            source_filters=req.source_filters,
            auto_extract_keywords=req.auto_extract_keywords,
            use_hybrid=req.use_hybrid,
            use_rerank=req.use_rerank,
            use_sep_reference=req.use_sep_reference,
            answer_style=req.answer_style,
            llm_provider=req.llm_provider,
            llm_model=req.llm_model,
            ultra_long_answer=req.ultra_long_answer,
            library_ids=req.library_ids,
            library_weights=req.library_weights,
            trace_id=trace_id,
        )
        yield (
            json.dumps(
                {"type": "start", "conversation_id": conv_id, "profile": meta.get("profile", "quality")},
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        chunk_size = 384
        text = answer or ""
        for i in range(0, len(text), chunk_size):
            yield (
                json.dumps({"type": "delta", "text": text[i : i + chunk_size]}, ensure_ascii=False).encode(
                    "utf-8"
                )
                + b"\n"
            )
        payload = {
            "type": "final",
            "answer": answer,
            "docs": docs,
            "conversation_id": conv_id,
            "keyword_hit_docs": meta.get("keyword_hit_docs", []),
            "keyword_source_stats": meta.get("keyword_source_stats", []),
            "profile": meta.get("profile", "quality"),
            "keywords_used": meta.get("keywords_used", []),
            "source_filters_used": meta.get("source_filters_used", []),
            "user_terms_used": meta.get("user_terms_used", []),
            "auto_terms_used": meta.get("auto_terms_used", []),
            "dropped_terms": meta.get("dropped_terms", []),
            "keyword_query": meta.get("keyword_query", ""),
            "term_source": meta.get("term_source", "question"),
            "hybrid": bool(meta.get("hybrid", False)),
            "reranked": bool(meta.get("reranked", False)),
            "sep_reference_enabled": bool(meta.get("sep_reference_enabled", False)),
            "sep_docs_kept": int(meta.get("sep_docs_kept", 0)),
            "sep_max_docs": int(meta.get("sep_max_docs", 0)),
            "sep_weight": float(meta.get("sep_weight", 0.0)),
            "answer_style": meta.get("answer_style", req.answer_style),
            "answer_model": meta.get("answer_model", ""),
            "answer_max_output_tokens": int(meta.get("answer_max_output_tokens", 0)),
            "debug": meta.get("debug", {}),
        }
        yield json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
        log_stage(trace_id=trace_id, stage="http.answer_stream", event="ok", extra={"docs": len(docs), "reranked": bool(meta.get("reranked"))})

    return StreamingResponse(
        ndjson_gen(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class ConceptIngestRequest(BaseModel):
    user_id: str = "default"
    texts: List[str] = Field(default_factory=list)
    anchor_prefix: str = "concept"


@app.post("/api/memory/concepts/add")
def memory_concepts_add(req: ConceptIngestRequest) -> Dict[str, Any]:
    """向「概念层」独立向量库写入短文本（与语料 chunks 分域，避免混检碰撞）。"""
    from src.concept_vector_store import add_concept_documents

    n = add_concept_documents(
        user_id=req.user_id,
        embedder=rag_engine.embedder,
        texts=req.texts or [],
        anchor_prefix=(req.anchor_prefix or "concept").strip() or "concept",
    )
    return {"ok": True, "added": n, "user_id": req.user_id}


class GraphEdgeRequest(BaseModel):
    user_id: str = "default"
    src_label: str
    dst_label: str
    relation: str = "RELATES_TO"


@app.post("/api/memory/graph/edge")
def memory_graph_edge(req: GraphEdgeRequest) -> Dict[str, Any]:
    from src.concept_graph import add_relation

    add_relation(
        user_id=req.user_id,
        src_label=req.src_label,
        dst_label=req.dst_label,
        relation=req.relation,
    )
    return {"ok": True}


class EvalSingleRequest(BaseModel):
    """批量实验用的单次检索预览（约 top_n 条），不写入数据库。"""

    question: str
    suite: str = "ablation"  # ablation | terminology
    mode: str = "full_rrf"
    top_n: int = 10
    doc_preview_chars: int = 400  # 0=片段全文写入 text_preview
    # 与正式 /api/answer 一致：限定文献（文件名子串等）全模式生效；限定关键词仅在 full_rrf 时由 runner 传入检索层
    keyword_terms: Optional[List[str]] = None
    source_filters: Optional[List[str]] = None


@app.post("/api/eval/single")
def eval_single(req: EvalSingleRequest) -> Dict[str, Any]:
    from src.eval.runner import EvalItem, run_retrieval_only

    item = EvalItem(qid="api", question=req.question, gold_chunk_ids=[], experiment="api")
    if req.suite in ("ablation", "terminology"):
        top_n = 30 if req.suite == "terminology" else req.top_n
        return run_retrieval_only(
            rag_engine,
            item=item,
            mode=req.mode,
            top_n=top_n,
            doc_preview_chars=req.doc_preview_chars,
            keyword_terms=req.keyword_terms,
            source_filters=req.source_filters,
        )
    raise HTTPException(status_code=400, detail="suite must be ablation or terminology")


class Top16CompareRequest(BaseModel):
    """
    Post-rerank Top-K 对比：Full（混合+RRF+重排）vs B2（纯稀疏+重排）。
    固定 use_rerank=True、use_sep_reference=False、retrieval_final_k_override=final_k。
    """

    question: str
    final_k: int = 16
    doc_preview_chars: int = 1200  # 0 = 全文（可能很大）
    keyword_terms: Optional[List[str]] = None
    source_filters: Optional[List[str]] = None
    auto_extract_keywords: bool = True
    answer_style: str = "哲学论述"


def _trim_retrieve_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    dbg = meta.get("debug") or {}
    return {
        "profile": meta.get("profile"),
        "hybrid": meta.get("hybrid"),
        "reranked": meta.get("reranked"),
        "retrieval_ablation": meta.get("retrieval_ablation"),
        "keywords_used": meta.get("keywords_used"),
        "source_filters_used": meta.get("source_filters_used"),
        "keyword_query": meta.get("keyword_query"),
        "sep_reference_enabled": meta.get("sep_reference_enabled"),
        "debug": {
            "final_k": dbg.get("final_k"),
            "dense_top_ids": (dbg.get("dense_top_ids") or [])[:12],
            "sparse_top_ids": (dbg.get("sparse_top_ids") or [])[:12],
            "fused_top_ids": (dbg.get("fused_top_ids") or [])[:12],
            "retrieval_params_effective": dbg.get("retrieval_params_effective"),
            "source_filter_balance": dbg.get("source_filter_balance"),
        },
    }


def _docs_for_compare(docs: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    raw_len = 0
    for d in docs:
        full = d.get("text") or ""
        raw_len = len(full)
        if max_chars and max_chars > 0:
            text = full[: max_chars]
        else:
            text = full
        out.append(
            {
                "chunk_id": d.get("chunk_id"),
                "source": d.get("source"),
                "page": d.get("page"),
                "text": text,
                "text_was_truncated": bool(max_chars and max_chars > 0 and raw_len > max_chars),
                "text_full_length": raw_len,
            }
        )
    return out


@app.post("/api/eval/top16_compare")
def eval_top16_compare(req: Top16CompareRequest) -> Dict[str, Any]:
    """
    并排返回 Full 与 sparse_only（B2）在相同后处理下的 Top-final_k 片段，供人工标注与辩论协议使用。
    """
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is required")
    fk = max(1, min(int(req.final_k), 200))

    base_kw: Dict[str, Any] = dict(
        question=q,
        keyword_terms=req.keyword_terms,
        source_filters=req.source_filters,
        auto_extract_keywords=req.auto_extract_keywords,
        use_hybrid=True,
        use_rerank=True,
        use_sep_reference=False,
        answer_style=req.answer_style,
        retrieval_final_k_override=fk,
    )

    docs_full, meta_full = rag_engine.retrieve(retrieval_ablation=None, **base_kw)
    docs_b2, meta_b2 = rag_engine.retrieve(retrieval_ablation="sparse_only", **base_kw)

    prev_chars = int(req.doc_preview_chars)
    return {
        "config": {
            "final_k": fk,
            "use_rerank": True,
            "use_sep_reference": False,
            "use_hybrid": True,
            "answer_style": req.answer_style,
            "doc_preview_chars": prev_chars,
            "note": "Full=混合检索+RRF+重排；B2=sparse_only+同一套重排与截断。",
        },
        "full": {
            "mode": "full_rrf",
            "mode_label": "Full（hybrid + RRF + rerank）",
            "count": len(docs_full),
            "docs": _docs_for_compare(docs_full, prev_chars),
            "meta": _trim_retrieve_meta(meta_full),
        },
        "b2_sparse_only": {
            "mode": "sparse_only",
            "mode_label": "B2（sparse_only + rerank）",
            "count": len(docs_b2),
            "docs": _docs_for_compare(docs_b2, prev_chars),
            "meta": _trim_retrieve_meta(meta_b2),
        },
    }


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

