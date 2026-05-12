from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .config import (
    ANSWER_MAX_OUTPUT_TOKENS_DEFAULT,
    GEMINI_ANSWER_MODEL,
    GEMINI_ANSWER_TEMPERATURE,
)
from .document_loader import load_single_document
from .llm_router import generate_answer
from .translation_export import export_translation


TRANSLATION_ROOT = Path("data") / "translations"
GLOBAL_GLOSSARY_NAME = "global_glossary.json"
DEFAULT_CHUNK_CHARS = 3600
DEFAULT_SEGMENT_CHARS = 4200
DEFAULT_CHAPTER_CHARS = 32000
OVERVIEW_SAMPLE_CHARS = 18000
MAX_GLOSSARY_TERMS = 80
MAX_GLOBAL_GLOSSARY_TERMS = 5000

EmitFn = Optional[Callable[[Dict[str, Any]], None]]


@dataclass
class TranslationPaths:
    root: Path
    state: Path
    glossary_draft: Path
    glossary: Path
    chunks_dir: Path
    translations_dir: Path
    exports_dir: Path


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _safe_slug(text: str, limit: int = 48) -> str:
    raw = re.sub(r"\.[^.]+$", "", Path(text or "translation").name)
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", raw).strip("._-")
    return (slug or "translation")[:limit]


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_load(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _emit(emit: EmitFn, event: str, **extra: Any) -> None:
    if emit:
        emit({"type": event, **extra})


def project_paths(project_id: str, root: str | Path = TRANSLATION_ROOT) -> TranslationPaths:
    base = Path(root) / project_id
    return TranslationPaths(
        root=base,
        state=base / "state.json",
        glossary_draft=base / "glossary.draft.json",
        glossary=base / "glossary.json",
        chunks_dir=base / "chunks",
        translations_dir=base / "translations",
        exports_dir=base / "exports",
    )


def global_glossary_path(root: str | Path = TRANSLATION_ROOT) -> Path:
    return Path(root) / GLOBAL_GLOSSARY_NAME


def list_translation_projects(root: str | Path = TRANSLATION_ROOT) -> List[Dict[str, Any]]:
    base = Path(root)
    out: List[Dict[str, Any]] = []
    if not base.exists():
        return out
    for state_path in base.glob("*/state.json"):
        try:
            state = _json_load(state_path, {})
            out.append(
                {
                    "project_id": state.get("project_id") or state_path.parent.name,
                    "source_name": state.get("source_name") or "",
                    "target_language": state.get("target_language") or "",
                    "status": state.get("status") or "",
                    "updated_at": state.get("updated_at") or "",
                    "progress": state.get("progress") or {},
                }
            )
        except Exception:
            continue
    return sorted(out, key=lambda x: str(x.get("updated_at") or ""), reverse=True)


def load_project(project_id: str, root: str | Path = TRANSLATION_ROOT) -> Dict[str, Any]:
    paths = project_paths(project_id, root)
    state = _json_load(paths.state)
    if not state:
        raise FileNotFoundError(f"Translation project not found: {project_id}")
    return state


def load_glossary(project_id: str, root: str | Path = TRANSLATION_ROOT, *, confirmed: bool = True) -> Dict[str, Any]:
    paths = project_paths(project_id, root)
    path = paths.glossary if confirmed and paths.glossary.exists() else paths.glossary_draft
    return _json_load(path, {"terms": []})


def _term_key(source: str, target_language: str = "") -> str:
    return f"{(target_language or '').strip().lower()}::{(source or '').strip().lower()}"


def _normalize_global_glossary(obj: Dict[str, Any]) -> Dict[str, Any]:
    terms = obj.get("terms") if isinstance(obj, dict) else []
    if not isinstance(terms, list):
        terms = []
    clean_terms: List[Dict[str, Any]] = []
    seen = set()
    for item in terms:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or item.get("term") or "").strip()
        target_language = str(item.get("target_language") or "").strip()
        if not source:
            continue
        key = _term_key(source, target_language)
        if key in seen:
            continue
        seen.add(key)
        projects = item.get("projects") if isinstance(item.get("projects"), list) else []
        clean_terms.append(
            {
                "source": source,
                "target": str(item.get("target") or "").strip(),
                "target_language": target_language,
                "note": str(item.get("note") or "").strip(),
                "confidence": item.get("confidence", ""),
                "projects": [str(x) for x in projects if str(x).strip()][:50],
                "usage_count": int(item.get("usage_count") or 0),
                "created_at": str(item.get("created_at") or "").strip(),
                "updated_at": str(item.get("updated_at") or "").strip(),
            }
        )
    return {
        "version": int(obj.get("version") or 1) if isinstance(obj, dict) else 1,
        "updated_at": str(obj.get("updated_at") or "").strip() if isinstance(obj, dict) else "",
        "terms": clean_terms[:MAX_GLOBAL_GLOSSARY_TERMS],
        "notes": str(obj.get("notes") or "").strip() if isinstance(obj, dict) else "",
    }


def load_global_glossary(
    root: str | Path = TRANSLATION_ROOT,
    *,
    target_language: str = "",
) -> Dict[str, Any]:
    payload = _normalize_global_glossary(_json_load(global_glossary_path(root), {"version": 1, "terms": []}) or {})
    lang = (target_language or "").strip().lower()
    if lang:
        payload["terms"] = [
            t for t in payload.get("terms") or []
            if not str(t.get("target_language") or "").strip()
            or str(t.get("target_language") or "").strip().lower() == lang
        ]
    return payload


def save_global_glossary(glossary: Dict[str, Any], root: str | Path = TRANSLATION_ROOT) -> Dict[str, Any]:
    payload = _normalize_global_glossary(glossary or {})
    now = _now()
    for term in payload.get("terms") or []:
        term["created_at"] = term.get("created_at") or now
        term["updated_at"] = term.get("updated_at") or now
    payload["updated_at"] = now
    _json_dump(global_glossary_path(root), payload)
    return payload


def merge_into_global_glossary(
    *,
    project_id: str,
    project_glossary: Dict[str, Any],
    target_language: str,
    root: str | Path = TRANSLATION_ROOT,
) -> Dict[str, Any]:
    now = _now()
    global_payload = load_global_glossary(root)
    by_key: Dict[str, Dict[str, Any]] = {}
    for term in global_payload.get("terms") or []:
        by_key[_term_key(str(term.get("source") or ""), str(term.get("target_language") or ""))] = dict(term)

    for term in _normalize_glossary(project_glossary).get("terms") or []:
        source = str(term.get("source") or "").strip()
        if not source:
            continue
        key = _term_key(source, target_language)
        existing = dict(by_key.get(key) or {})
        projects = list(existing.get("projects") or [])
        if project_id and project_id not in projects:
            projects.append(project_id)
        by_key[key] = {
            "source": source,
            "target": str(term.get("target") or existing.get("target") or "").strip(),
            "target_language": target_language,
            "note": str(term.get("note") or existing.get("note") or "").strip(),
            "confidence": term.get("confidence") or existing.get("confidence") or "",
            "projects": projects[:50],
            "usage_count": int(existing.get("usage_count") or 0) + 1,
            "created_at": existing.get("created_at") or now,
            "updated_at": now,
        }

    merged = {
        "version": 1,
        "updated_at": now,
        "notes": str(global_payload.get("notes") or ""),
        "terms": sorted(by_key.values(), key=lambda x: (str(x.get("target_language") or ""), str(x.get("source") or "").lower())),
    }
    return save_global_glossary(merged, root)


def combine_glossaries(global_glossary: Dict[str, Any], project_glossary: Dict[str, Any]) -> Dict[str, Any]:
    combined: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for item in (global_glossary.get("terms") or []):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip()
        if not source:
            continue
        key = source.lower()
        combined[key] = dict(item)
        order.append(key)
    for item in (_normalize_glossary(project_glossary).get("terms") or []):
        source = str(item.get("source") or "").strip()
        if not source:
            continue
        key = source.lower()
        if key not in order:
            order.append(key)
        merged = dict(combined.get(key) or {})
        merged.update(item)
        merged["source"] = source
        combined[key] = merged
    return {"terms": [combined[k] for k in order if k in combined]}


def save_glossary(project_id: str, glossary: Dict[str, Any], root: str | Path = TRANSLATION_ROOT) -> Dict[str, Any]:
    paths = project_paths(project_id, root)
    state = load_project(project_id, root)
    payload = _normalize_glossary(glossary)
    payload["confirmed_at"] = _now()
    _json_dump(paths.glossary, payload)
    merge_into_global_glossary(
        project_id=project_id,
        project_glossary=payload,
        target_language=str(state.get("target_language") or ""),
        root=root,
    )
    state["glossary_status"] = "confirmed"
    state["global_glossary_path"] = str(global_glossary_path(root).as_posix())
    state["updated_at"] = _now()
    _json_dump(paths.state, state)
    return payload


def _source_project_id(source_path: Path, target_language: str) -> str:
    try:
        st = source_path.stat()
        seed = f"{source_path.resolve()}|{st.st_size}|{int(st.st_mtime)}|{target_language}"
    except Exception:
        seed = f"{source_path}|{time.time()}|{target_language}"
    digest = hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{_safe_slug(source_path.name)}_{digest}"


def _clean_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _looks_like_heading(line: str) -> bool:
    s = (line or "").strip()
    if not s or len(s) > 90:
        return False
    patterns = [
        r"^(chapter|part|section|book)\s+[\divxlcdm]+[\s:.-]",
        r"^(introduction|preface|conclusion|afterword|appendix|bibliography)$",
        r"^第[一二三四五六七八九十百零〇\d]+[章节卷部篇]",
        r"^[一二三四五六七八九十百零〇\d]+[、.．]\s*\S+",
    ]
    lower = s.lower()
    return any(re.search(p, lower, re.IGNORECASE) for p in patterns)


def _page_chapter_key(page_value: Any) -> Optional[str]:
    s = str(page_value or "")
    m = re.search(r"chapter-(\d+)", s, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _make_chapters(pages: List[Dict[str, Any]], max_chapter_chars: int = DEFAULT_CHAPTER_CHARS) -> List[Dict[str, Any]]:
    if not pages:
        return []

    epub_keys = [_page_chapter_key(p.get("page")) for p in pages]
    if any(epub_keys):
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        order: List[str] = []
        for page, key in zip(pages, epub_keys):
            k = key or "0"
            if k not in grouped:
                grouped[k] = []
                order.append(k)
            grouped[k].append(page)
        chapters: List[Dict[str, Any]] = []
        for idx, key in enumerate(order, start=1):
            text = _clean_text("\n\n".join(str(p.get("text") or "") for p in grouped[key]))
            if text:
                chapters.append({"index": idx, "title": f"Chapter {key}", "text": text})
        return chapters

    chapters: List[Dict[str, Any]] = []
    current_title = "Chapter 1"
    current_parts: List[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current_parts, current_len, current_title
        text = _clean_text("\n\n".join(current_parts))
        if not text:
            current_parts = []
            current_len = 0
            return
        chapters.append({"index": len(chapters) + 1, "title": current_title, "text": text})
        current_parts = []
        current_len = 0

    for page in pages:
        page_text = _clean_text(str(page.get("text") or ""))
        if not page_text:
            continue
        lines = page_text.splitlines()
        first = next((ln.strip() for ln in lines if ln.strip()), "")
        if _looks_like_heading(first) and current_len > 1200:
            flush()
            current_title = first[:90]
        elif current_len >= max_chapter_chars:
            flush()
            current_title = f"Chapter {len(chapters) + 1}"
        current_parts.append(page_text)
        current_len += len(page_text)

    flush()
    return chapters


def _split_text(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n\n".join(buf).strip())
            buf = []
            buf_len = 0

    for para in paras:
        if len(para) > max_chars:
            flush()
            pieces = re.split(r"(?<=[。！？.!?])\s+", para)
            part = ""
            for piece in pieces:
                if not piece:
                    continue
                if len(part) + len(piece) + 1 > max_chars and part:
                    chunks.append(part.strip())
                    part = piece
                else:
                    part = (part + " " + piece).strip()
            if part:
                chunks.append(part.strip())
            continue
        if buf_len + len(para) + 2 > max_chars and buf:
            flush()
        buf.append(para)
        buf_len += len(para) + 2
    flush()
    return chunks


def _chunk_path(paths: TranslationPaths, chapter_index: int, chunk_index: int) -> Path:
    return paths.chunks_dir / f"ch{chapter_index:04d}_chunk{chunk_index:04d}.txt"


def _translation_path(paths: TranslationPaths, chapter_index: int, chunk_index: int) -> Path:
    return paths.translations_dir / f"ch{chapter_index:04d}_chunk{chunk_index:04d}.json"


def _block_type(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return "paragraph"
    if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", s):
        return "list_item"
    if _looks_like_heading(s):
        return "heading"
    return "paragraph"


def _extract_leading_footnote(text: str) -> Tuple[Optional[str], str]:
    s = (text or "").strip()
    patterns = [
        r"^\[(\d{1,4})\]\s*(.+)$",
        r"^\((\d{1,4})\)\s*(.+)$",
    ]
    for pat in patterns:
        m = re.match(pat, s, flags=re.S)
        if m and len(m.group(2).strip()) >= 12:
            return m.group(1), m.group(2).strip()
    return None, s


def _normalize_footnote_refs(text: str) -> str:
    s = text or ""
    s = re.sub(r"(?<!\w)\^(\d{1,4})(?!\w)", r"[\1]", s)
    s = re.sub(r"[（(](\d{1,4})[）)]", r"[\1]", s)
    return s


def _paragraphs_for_blocks(text: str) -> List[str]:
    clean = _clean_text(text)
    if not clean:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", clean) if p.strip()]
    out: List[str] = []
    for para in paras:
        if len(para) <= DEFAULT_SEGMENT_CHARS:
            out.append(para)
            continue
        pieces = re.split(r"(?<=[。！？.!?])\s+", para)
        buf = ""
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if len(buf) + len(piece) + 1 > DEFAULT_SEGMENT_CHARS and buf:
                out.append(buf.strip())
                buf = piece
            else:
                buf = (buf + " " + piece).strip()
        if buf:
            out.append(buf.strip())
    return out


def _build_block_records(
    paths: TranslationPaths,
    chapters: List[Dict[str, Any]],
    *,
    segment_chars: int = DEFAULT_SEGMENT_CHARS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    blocks: List[Dict[str, Any]] = []
    segments: List[Dict[str, Any]] = []
    paths.chunks_dir.mkdir(parents=True, exist_ok=True)

    for chapter in chapters:
        chapter_index = int(chapter["index"])
        chapter_blocks: List[Dict[str, Any]] = []
        for block_index, para in enumerate(_paragraphs_for_blocks(str(chapter.get("text") or "")), start=1):
            footnote_no, body = _extract_leading_footnote(para)
            typ = "footnote" if footnote_no else _block_type(body)
            block_id = f"ch{chapter_index:04d}_b{block_index:05d}"
            refs = re.findall(r"\[(\d{1,4})\]", _normalize_footnote_refs(body))
            block = {
                "block_id": block_id,
                "chapter_index": chapter_index,
                "block_index": block_index,
                "type": typ,
                "text": _normalize_footnote_refs(body),
                "chars": len(body),
                "footnote_number": footnote_no or "",
                "footnote_refs": refs,
            }
            blocks.append(block)
            chapter_blocks.append(block)

        seg_buf: List[Dict[str, Any]] = []
        seg_len = 0
        seg_idx = 1

        def flush_segment() -> None:
            nonlocal seg_buf, seg_len, seg_idx
            if not seg_buf:
                return
            p = _chunk_path(paths, chapter_index, seg_idx)
            payload = {
                "chapter_index": chapter_index,
                "chunk_index": seg_idx,
                "blocks": seg_buf,
            }
            p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            segments.append(
                {
                    "chapter_index": chapter_index,
                    "chunk_index": seg_idx,
                    "path": str(p.as_posix()),
                    "chars": seg_len,
                    "block_ids": [b["block_id"] for b in seg_buf],
                    "block_count": len(seg_buf),
                    "status": "pending",
                    "granularity": "blocks",
                }
            )
            seg_buf = []
            seg_len = 0
            seg_idx += 1

        for block in chapter_blocks:
            b_len = int(block.get("chars") or 0)
            if seg_buf and seg_len + b_len + 2 > segment_chars:
                flush_segment()
            seg_buf.append(block)
            seg_len += b_len + 2
        flush_segment()

    return blocks, segments


def _build_chunk_records(paths: TranslationPaths, chapters: List[Dict[str, Any]], chunk_chars: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    paths.chunks_dir.mkdir(parents=True, exist_ok=True)
    for chapter in chapters:
        chapter_index = int(chapter["index"])
        for idx, text in enumerate(_split_text(str(chapter.get("text") or ""), max_chars=chunk_chars), start=1):
            p = _chunk_path(paths, chapter_index, idx)
            p.write_text(text, encoding="utf-8")
            records.append(
                {
                    "chapter_index": chapter_index,
                    "chunk_index": idx,
                    "path": str(p.as_posix()),
                    "chars": len(text),
                    "status": "pending",
                }
            )
    return records


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"\s*```$", "", raw).strip()
    candidates = [raw]
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start : end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
            return obj if isinstance(obj, dict) else None
        except Exception:
            continue
    return None


def _normalize_glossary(obj: Dict[str, Any]) -> Dict[str, Any]:
    terms = obj.get("terms") if isinstance(obj, dict) else []
    if not isinstance(terms, list):
        terms = []
    clean_terms: List[Dict[str, Any]] = []
    seen = set()
    for item in terms:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or item.get("term") or "").strip()
        if not source or source.lower() in seen:
            continue
        seen.add(source.lower())
        clean_terms.append(
            {
                "source": source,
                "target": str(item.get("target") or "").strip(),
                "note": str(item.get("note") or "").strip(),
                "confidence": item.get("confidence", ""),
            }
        )
    return {
        "terms": clean_terms[:MAX_GLOSSARY_TERMS],
        "notes": str(obj.get("notes") or "").strip() if isinstance(obj, dict) else "",
    }


def _fallback_overview(source_name: str, chapters: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "overview": f"{source_name} loaded with {len(chapters)} chapter(s).",
        "structure": [{"index": c.get("index"), "title": c.get("title"), "chars": len(str(c.get("text") or ""))} for c in chapters],
        "style_notes": "",
        "chapter_summaries": [],
    }


def _glossary_prompt_preview(glossary: Dict[str, Any], limit: int = 60) -> str:
    lines: List[str] = []
    for item in (glossary.get("terms") or [])[:limit]:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip()
        target = str(item.get("target") or "").strip()
        note = str(item.get("note") or "").strip()
        if source:
            lines.append(f"- {source} => {target or '[待定]'}" + (f" ({note})" if note else ""))
    return "\n".join(lines) if lines else "（暂无长期术语）"


def _overview_prompt(
    source_name: str,
    target_language: str,
    chapters: List[Dict[str, Any]],
    *,
    global_glossary: Optional[Dict[str, Any]] = None,
) -> str:
    toc = "\n".join(
        f"{c.get('index')}. {c.get('title')} ({len(str(c.get('text') or ''))} chars)"
        for c in chapters[:80]
    )
    excerpts: List[str] = []
    budget = OVERVIEW_SAMPLE_CHARS
    for c in chapters:
        if budget <= 0:
            break
        text = str(c.get("text") or "")
        sample = text[: min(1800, budget)]
        budget -= len(sample)
        excerpts.append(f"## {c.get('index')}. {c.get('title')}\n{sample}")
    excerpt_text = "\n\n".join(excerpts)
    global_terms = _glossary_prompt_preview(global_glossary or {})
    return f"""你是学术翻译项目的总编辑。请先概览这本书/文章，提炼结构、主题、文体和关键术语。

文件名：{source_name}
目标语言：{target_language}

长期术语库中已有译法（若与本文语境冲突，可在术语表草稿中给出修正建议）：
{global_terms}

目录/章节候选：
{toc}

原文摘录：
{excerpt_text}

请只输出 JSON，不要 Markdown。格式：
{{
  "overview": "全书/全文概览，200-500字",
  "structure": [{{"index": 1, "title": "章节标题", "summary": "章节摘要"}}],
  "style_notes": "翻译风格建议",
  "terms": [
    {{"source": "原文术语", "target": "建议译名", "note": "含义/译法理由", "confidence": "high|medium|low"}}
  ]
}}
"""


def _call_llm(
    prompt: str,
    *,
    provider: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
    stage: str,
) -> Tuple[str, str]:
    return generate_answer(
        prompt=prompt,
        provider=provider,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        stage=stage,
    )


def prepare_translation_project(
    source_path: str | Path,
    *,
    target_language: str = "zh-CN",
    provider: str = "gemini",
    model: str = "",
    project_id: str = "",
    root: str | Path = TRANSLATION_ROOT,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    call_llm: bool = True,
    emit: EmitFn = None,
) -> Dict[str, Any]:
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    target = (target_language or "zh-CN").strip()
    pid = (project_id or "").strip() or _source_project_id(src, target)
    paths = project_paths(pid, root)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.translations_dir.mkdir(parents=True, exist_ok=True)
    paths.exports_dir.mkdir(parents=True, exist_ok=True)

    _emit(emit, "load_start", project_id=pid, source_path=str(src))
    pages = load_single_document(str(src))
    chapters = _make_chapters(pages)
    for c in chapters:
        c["chars"] = len(str(c.get("text") or ""))
        c.pop("text", None)

    raw_chapters = _make_chapters(pages)
    blocks, chunks = _build_block_records(paths, raw_chapters, segment_chars=max(800, int(chunk_chars or DEFAULT_SEGMENT_CHARS)))
    _emit(emit, "chunks_ready", project_id=pid, chapters=len(chapters), chunks=len(chunks))

    overview = _fallback_overview(src.name, raw_chapters)
    glossary = {"terms": [], "notes": ""}
    model_used = ""
    global_glossary = load_global_glossary(root, target_language=target)
    if call_llm:
        prompt = _overview_prompt(src.name, target, raw_chapters, global_glossary=global_glossary)
        _emit(emit, "overview_start", project_id=pid)
        text, model_used = _call_llm(
            prompt,
            provider=provider,
            model=model or GEMINI_ANSWER_MODEL,
            temperature=0.2,
            max_output_tokens=8192,
            stage="translation.overview",
        )
        parsed = _extract_json(text)
        if parsed:
            overview = {
                "overview": str(parsed.get("overview") or "").strip(),
                "structure": parsed.get("structure") if isinstance(parsed.get("structure"), list) else [],
                "style_notes": str(parsed.get("style_notes") or "").strip(),
                "chapter_summaries": parsed.get("structure") if isinstance(parsed.get("structure"), list) else [],
                "raw_model_output": text,
            }
            glossary = _normalize_glossary(parsed)
        else:
            overview["raw_model_output"] = text
        _emit(emit, "overview_done", project_id=pid, model_used=model_used, terms=len(glossary.get("terms") or []))

    state: Dict[str, Any] = {
        "project_id": pid,
        "source_path": str(src),
        "source_name": src.name,
        "target_language": target,
        "provider": provider,
        "model": model or "",
        "model_used_for_overview": model_used,
        "created_at": _now(),
        "updated_at": _now(),
        "status": "prepared",
        "glossary_status": "draft",
        "vector_policy": "no_persistent_vector_by_default",
        "chapters": chapters,
        "blocks": blocks,
        "chunks": chunks,
        "translation_granularity": "blocks",
        "progress": {"total_chunks": len(chunks), "translated_chunks": 0},
        "overview": overview,
        "paths": {
            "root": str(paths.root.as_posix()),
            "state": str(paths.state.as_posix()),
            "glossary_draft": str(paths.glossary_draft.as_posix()),
            "glossary": str(paths.glossary.as_posix()),
            "global_glossary": str(global_glossary_path(root).as_posix()),
            "exports": str(paths.exports_dir.as_posix()),
        },
    }
    _json_dump(paths.state, state)
    _json_dump(paths.glossary_draft, glossary)
    _emit(emit, "prepared", project_id=pid)
    return state


def _glossary_prompt_block(glossary: Dict[str, Any], *, source_text: str = "") -> str:
    terms = glossary.get("terms") or []
    if source_text and len(terms) > MAX_GLOSSARY_TERMS:
        src_lower = source_text.lower()
        matched = []
        rest = []
        for item in terms:
            source = str(item.get("source") or "").strip() if isinstance(item, dict) else ""
            if source and source.lower() in src_lower:
                matched.append(item)
            else:
                rest.append(item)
        terms = matched + rest
    lines = []
    for item in terms[:MAX_GLOSSARY_TERMS]:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip()
        target = str(item.get("target") or "").strip()
        note = str(item.get("note") or "").strip()
        if source:
            lines.append(f"- {source} => {target or '[待定]'}" + (f" ({note})" if note else ""))
    return "\n".join(lines) if lines else "（无已确认术语；请保持关键术语前后一致。）"


def _chapter_summary(state: Dict[str, Any], chapter_index: int) -> str:
    overview = state.get("overview") or {}
    for item in overview.get("chapter_summaries") or overview.get("structure") or []:
        if not isinstance(item, dict):
            continue
        try:
            if int(item.get("index") or 0) == int(chapter_index):
                return str(item.get("summary") or "").strip()
        except Exception:
            continue
    return ""


def _translation_prompt(
    *,
    state: Dict[str, Any],
    glossary: Dict[str, Any],
    chapter: Dict[str, Any],
    chunk: Dict[str, Any],
    source_text: str,
    previous_tail: str,
) -> str:
    overview = (state.get("overview") or {}).get("overview") or ""
    style_notes = (state.get("overview") or {}).get("style_notes") or ""
    chapter_index = int(chapter.get("index") or chunk.get("chapter_index") or 0)
    return f"""你是严谨的长文翻译引擎。请把以下原文翻译为 {state.get('target_language') or 'zh-CN'}。

要求：
1. 只输出译文，不要解释过程。
2. 保留段落层次；如原文有小标题，请译出小标题。
3. 严格遵守术语表；未列入术语表的关键概念也要前后一致。
4. 不要省略、概括或改写原文论证。
5. 遇到明显 OCR 噪声，可轻度整理，但不得增添原文没有的信息。

全书概览：
{overview}

章节：{chapter_index}. {chapter.get('title') or ''}
章节摘要：{_chapter_summary(state, chapter_index)}
风格建议：{style_notes}

确认术语表：
{_glossary_prompt_block(glossary, source_text=source_text)}

上一分块译文末尾（用于衔接，可忽略重复内容）：
{previous_tail[-1200:] if previous_tail else '（无）'}

当前原文：
{source_text}
"""


def _segment_translation_prompt(
    *,
    state: Dict[str, Any],
    glossary: Dict[str, Any],
    chapter: Dict[str, Any],
    chunk: Dict[str, Any],
    blocks: List[Dict[str, Any]],
) -> str:
    overview = (state.get("overview") or {}).get("overview") or ""
    style_notes = (state.get("overview") or {}).get("style_notes") or ""
    chapter_index = int(chapter.get("index") or chunk.get("chapter_index") or 0)
    block_lines: List[str] = []
    for b in blocks:
        block_lines.append(
            json.dumps(
                {
                    "block_id": b.get("block_id"),
                    "type": b.get("type") or "paragraph",
                    "footnote_number": b.get("footnote_number") or "",
                    "text": b.get("text") or "",
                },
                ensure_ascii=False,
            )
        )
    source_text = "\n".join(str(b.get("text") or "") for b in blocks)
    return f"""你是严谨的学术长文翻译引擎。请把下列结构块翻译为 {state.get('target_language') or 'zh-CN'}。

输出格式（强制）：
- 只输出 JSON 数组，不要输出 Markdown 代码围栏或解释。
- 数组每项必须是 {{"block_id": "...", "translated_text": "..."}}。
- 必须为每个输入 block_id 返回一项，不能漏译、不能新增 block_id。

翻译规则：
1. 保留标题、段落、列表项、脚注标记等结构。正文中的 [1]、[2] 等脚注标记必须原样保留。
2. type=footnote 的块是脚注内容，也要翻译；但书名、期刊名、专有名词、英文论文题名通常保持原文不译，必要时只翻译解释性语句。
3. 严格遵守术语表；未列入术语表的关键概念也要前后一致。
4. 不要省略、概括或改写原文论证。遇到明显 OCR 噪声可轻度整理，但不得增添原文没有的信息。

全书概览：
{overview}

章节：{chapter_index}. {chapter.get('title') or ''}
章节摘要：{_chapter_summary(state, chapter_index)}
风格建议：{style_notes}

确认术语表：
{_glossary_prompt_block(glossary, source_text=source_text)}

输入结构块（每行一个 JSON）：
{chr(10).join(block_lines)}
"""


def _extract_json_array(text: str) -> Optional[List[Dict[str, Any]]]:
    raw = (text or "").strip()
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"\s*```$", "", raw).strip()
    candidates = [raw]
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        candidates.append(raw[start : end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
        except Exception:
            continue
    return None


def _fallback_translated_blocks(blocks: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    pieces = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    out: List[Dict[str, Any]] = []
    for idx, block in enumerate(blocks):
        out.append(
            {
                "block_id": block.get("block_id"),
                "type": block.get("type") or "paragraph",
                "footnote_number": block.get("footnote_number") or "",
                "translated_text": pieces[idx] if idx < len(pieces) else (text.strip() if len(blocks) == 1 else str(block.get("text") or "")),
            }
        )
    return out


def _normalize_translated_blocks(blocks: List[Dict[str, Any]], model_text: str) -> List[Dict[str, Any]]:
    parsed = _extract_json_array(model_text)
    expected = {str(b.get("block_id") or ""): b for b in blocks}
    if not parsed:
        return _fallback_translated_blocks(blocks, model_text)
    by_id: Dict[str, Dict[str, Any]] = {}
    for item in parsed:
        bid = str(item.get("block_id") or "").strip()
        if bid in expected:
            by_id[bid] = item
    out: List[Dict[str, Any]] = []
    for block in blocks:
        bid = str(block.get("block_id") or "")
        item = by_id.get(bid) or {}
        text = str(item.get("translated_text") or item.get("translation") or "").strip()
        out.append(
            {
                "block_id": bid,
                "type": block.get("type") or "paragraph",
                "footnote_number": block.get("footnote_number") or "",
                "translated_text": text or str(block.get("text") or ""),
            }
        )
    return out


def _render_translated_blocks(blocks: List[Dict[str, Any]]) -> str:
    return "\n\n".join(str(b.get("translated_text") or "").strip() for b in blocks if str(b.get("translated_text") or "").strip())


def _load_translated_chunks(paths: TranslationPaths) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not paths.translations_dir.exists():
        return out
    for p in paths.translations_dir.glob("*.json"):
        try:
            obj = _json_load(p, {})
            if obj:
                out.append(obj)
        except Exception:
            continue
    return out


def translate_project(
    project_id: str,
    *,
    root: str | Path = TRANSLATION_ROOT,
    provider: str = "",
    model: str = "",
    max_chunks: int = 0,
    concurrency: int = 1,
    resume: bool = True,
    emit: EmitFn = None,
) -> Dict[str, Any]:
    paths = project_paths(project_id, root)
    state = load_project(project_id, root)
    project_glossary = load_glossary(project_id, root, confirmed=True)
    glossary = combine_glossaries(
        load_global_glossary(root, target_language=str(state.get("target_language") or "")),
        project_glossary,
    )
    provider = (provider or state.get("provider") or "gemini").strip()
    model = (model or state.get("model") or GEMINI_ANSWER_MODEL).strip()
    workers = min(20, max(1, int(concurrency or 1)))

    translated = {(
        int(x.get("chapter_index") or 0),
        int(x.get("chunk_index") or 0),
    ): x for x in _load_translated_chunks(paths)}

    chapters_by_idx = {int(c.get("index") or 0): c for c in state.get("chapters") or []}
    chunks = list(state.get("chunks") or [])
    done_count = 0
    translated_this_run = 0
    lock = threading.Lock()

    state["status"] = "translating"
    state["updated_at"] = _now()
    _json_dump(paths.state, state)

    pending: List[Dict[str, Any]] = []
    for chunk in chunks:
        chapter_index = int(chunk.get("chapter_index") or 0)
        chunk_index = int(chunk.get("chunk_index") or 0)
        key = (chapter_index, chunk_index)
        if resume and key in translated:
            done_count += 1
            continue
        if max_chunks and translated_this_run >= max_chunks:
            break
        pending.append(chunk)
        translated_this_run += 1

    def _load_segment_blocks(chunk: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
        src_path = Path(str(chunk.get("path") or ""))
        if not src_path.is_absolute():
            src_path = Path.cwd() / src_path
        raw = src_path.read_text(encoding="utf-8")
        try:
            obj = json.loads(raw)
            bs = obj.get("blocks") if isinstance(obj, dict) else None
            if isinstance(bs, list):
                return raw, [b for b in bs if isinstance(b, dict)]
        except Exception:
            pass
        return raw, [
            {
                "block_id": f"ch{int(chunk.get('chapter_index') or 0):04d}_c{int(chunk.get('chunk_index') or 0):04d}_legacy",
                "chapter_index": int(chunk.get("chapter_index") or 0),
                "block_index": 1,
                "type": "paragraph",
                "text": raw,
                "footnote_number": "",
            }
        ]

    def _translate_one(chunk: Dict[str, Any]) -> Dict[str, Any]:
        chapter_index = int(chunk.get("chapter_index") or 0)
        chunk_index = int(chunk.get("chunk_index") or 0)
        _raw, blocks = _load_segment_blocks(chunk)
        chapter = chapters_by_idx.get(chapter_index, {"index": chapter_index, "title": f"Chapter {chapter_index}"})
        _emit(emit, "translate_chunk_start", project_id=project_id, chapter_index=chapter_index, chunk_index=chunk_index)
        prompt = _segment_translation_prompt(
            state=state,
            glossary=glossary,
            chapter=chapter,
            chunk=chunk,
            blocks=blocks,
        )
        text, model_used = _call_llm(
            prompt,
            provider=provider,
            model=model,
            temperature=float(GEMINI_ANSWER_TEMPERATURE),
            max_output_tokens=int(ANSWER_MAX_OUTPUT_TOKENS_DEFAULT),
            stage="translation.chunk",
        )
        translated_blocks = _normalize_translated_blocks(blocks, text)
        item = {
            "chapter_index": chapter_index,
            "chunk_index": chunk_index,
            "translation": _render_translated_blocks(translated_blocks),
            "translated_blocks": translated_blocks,
            "model_used": model_used,
            "updated_at": _now(),
            "granularity": "blocks",
        }
        _json_dump(_translation_path(paths, chapter_index, chunk_index), item)
        return item

    if pending:
        if workers <= 1:
            for chunk in pending:
                item = _translate_one(chunk)
                key = (int(item.get("chapter_index") or 0), int(item.get("chunk_index") or 0))
                translated[key] = item
                state["progress"] = {"total_chunks": len(chunks), "translated_chunks": len(translated)}
                state["updated_at"] = _now()
                _json_dump(paths.state, state)
                _emit(
                    emit,
                    "translate_chunk_done",
                    project_id=project_id,
                    chapter_index=key[0],
                    chunk_index=key[1],
                    translated_chunks=len(translated),
                    total_chunks=len(chunks),
                    model_used=item.get("model_used"),
                )
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_translate_one, chunk) for chunk in pending]
                for fut in as_completed(futures):
                    item = fut.result()
                    key = (int(item.get("chapter_index") or 0), int(item.get("chunk_index") or 0))
                    with lock:
                        translated[key] = item
                        state["progress"] = {"total_chunks": len(chunks), "translated_chunks": len(translated)}
                        state["updated_at"] = _now()
                        _json_dump(paths.state, state)
                    _emit(
                        emit,
                        "translate_chunk_done",
                        project_id=project_id,
                        chapter_index=key[0],
                        chunk_index=key[1],
                        translated_chunks=len(translated),
                        total_chunks=len(chunks),
                        model_used=item.get("model_used"),
                    )

    total = len(chunks)
    state["progress"] = {"total_chunks": total, "translated_chunks": len(translated)}
    state["status"] = "translated" if len(translated) >= total else "partial"
    state["updated_at"] = _now()
    _json_dump(paths.state, state)
    return state


def export_project(
    project_id: str,
    *,
    root: str | Path = TRANSLATION_ROOT,
    output_format: str = "txt",
    output_path: str | Path = "",
) -> Path:
    paths = project_paths(project_id, root)
    state = load_project(project_id, root)
    fmt = (output_format or "txt").strip().lower().lstrip(".")
    if not output_path:
        output_path = paths.exports_dir / f"{_safe_slug(state.get('source_name') or project_id)}.{fmt}"
    translated_chunks = _load_translated_chunks(paths)
    out = export_translation(state, translated_chunks, output_path, output_format=fmt)
    state["last_export"] = str(out.as_posix())
    state["updated_at"] = _now()
    _json_dump(paths.state, state)
    return out


def run_translation(
    source_path: str | Path,
    *,
    target_language: str = "zh-CN",
    provider: str = "gemini",
    model: str = "",
    output_format: str = "txt",
    project_id: str = "",
    root: str | Path = TRANSLATION_ROOT,
    concurrency: int = 1,
    emit: EmitFn = None,
) -> Dict[str, Any]:
    state = prepare_translation_project(
        source_path,
        target_language=target_language,
        provider=provider,
        model=model,
        project_id=project_id,
        root=root,
        emit=emit,
    )
    pid = str(state["project_id"])
    if not project_paths(pid, root).glossary.exists():
        save_glossary(pid, load_glossary(pid, root, confirmed=False), root=root)
    state = translate_project(pid, root=root, provider=provider, model=model, concurrency=concurrency, emit=emit)
    out = export_project(pid, root=root, output_format=output_format)
    state["last_export"] = str(out.as_posix())
    return state
