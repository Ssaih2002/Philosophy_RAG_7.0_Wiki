from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
import unicodedata
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from . import config
from .llm_gemini import describe_llm_error
from .llm_router import generate_answer

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()

# 主四段 + 可选归档（程序在遗忘时追加）
WIKI_MAIN_SECTIONS: Tuple[str, ...] = (
    "## Confirmed",
    "## Preferences",
    "## Projects",
    "## Hypotheses",
)
ARCHIVE_SECTION = "## Archive（自动归档）"


def _lock_for(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lk = _LOCKS.get(key)
        if lk is None:
            lk = threading.Lock()
            _LOCKS[key] = lk
        return lk


def _safe_id(x: str, *, default: str) -> str:
    s = (x or "").strip()
    if not s:
        return default
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:80] or default


def _now_date() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _one_line(s: str, limit: int = 500) -> str:
    out = (s or "").strip().replace("\r", " ").replace("\n", " ")
    if len(out) > limit:
        out = out[:limit] + "..."
    return out


def _append_wiki_log(p: "WikiPaths", title: str, lines: List[str]) -> None:
    try:
        body = p.log.read_text(encoding="utf-8") if p.log.exists() else "# Log\n\n"
        if body and not body.endswith("\n"):
            body += "\n"
        body += f"## [{_now_date()}] {title} | {_now_ts()}\n\n"
        for line in lines:
            body += f"- {line}\n"
        body += "\n"
        p.log.write_text(body, encoding="utf-8")
    except Exception:
        pass


def _wiki_llm_error_summary(exc: Exception) -> str:
    diagnosis = describe_llm_error(exc)
    return f"{diagnosis}: {_one_line(str(exc), 700)}"


def _call_wiki_llm(
    *,
    p: "WikiPaths",
    stage: str,
    prompt: str,
    llm_provider: str,
    llm_model: str,
    temperature: float,
    max_output_tokens: int,
) -> Tuple[str, str]:
    provider = (llm_provider or "gemini").strip()
    primary = (llm_model or "").strip()
    candidates = [primary]
    if provider.lower() == "gemini":
        for m in getattr(config, "GEMINI_FALLBACK_MODELS", []) or []:
            m = str(m).strip()
            if m and m not in candidates:
                candidates.append(m)

    last_error: Optional[Exception] = None
    for idx, model_name in enumerate(candidates):
        try:
            text, used = generate_answer(
                prompt=prompt,
                provider=provider,
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                stage=f"wiki.{stage}",
            )
            if idx > 0:
                _append_wiki_log(
                    p,
                    "wiki_llm_fallback_ok",
                    [
                        f"stage: {stage}",
                        f"requested_model: {provider}:{primary}",
                        f"fallback_model: {provider}:{model_name}",
                        f"used_model: {used}",
                    ],
                )
            return text, used
        except Exception as e:
            last_error = e
            _append_wiki_log(
                p,
                "wiki_llm_failed",
                [
                    f"stage: {stage}",
                    f"requested_model: {provider}:{primary}",
                    f"attempted_model: {provider}:{model_name}",
                    f"error: {_wiki_llm_error_summary(e)}",
                ],
            )
    raise last_error or RuntimeError("Wiki LLM call failed without explicit error.")


@dataclass(frozen=True)
class WikiPaths:
    root: Path
    schema: Path
    index: Path
    log: Path
    user: Path
    meta: Path


def wiki_paths(*, user_id: str, data_dir: str = "data") -> WikiPaths:
    uid = _safe_id(user_id, default="default")
    root = Path(data_dir) / "memory" / "wiki" / uid
    return WikiPaths(
        root=root,
        schema=root / "schema.md",
        index=root / "index.md",
        log=root / "log.md",
        user=root / "user.md",
        meta=root / "user.meta.json",
    )


SCHEMA_MD = """# User Wiki Schema (A版)

本目录是“用户画像/偏好/长期项目”的持久化 wiki（一期 A 版）。

## 目标
- 让系统跨会话记住用户是谁、主要研究什么、偏好何种输出风格与约束。
- **不是**概念层知识库；不从新文献 ingest 自动生成概念页/实体页（后续任务再做）。

## 文件
- `user.md`: 用户画像主文档（核心）
- `user.meta.json`: 每条要点的访问计数、衰减权重、强化/遗忘元数据（程序维护）
- `index.md`: 目录（指向 user.md 与 log.md）
- `log.md`: 时间线（追加式）

## 规则（强制）
1. 只记录**稳定、可复用**的信息：身份、研究方向、术语偏好、语言偏好、项目、工作流偏好。
2. 不记录敏感信息（真实姓名/地址/电话/账号密钥等）。
3. 区分：Confirmed / Preferences / Projects / Hypotheses。
4. Hypotheses 不能当作事实；若被用户否认，立即移除或标注为否定。
5. Wiki 只用于“语境与偏好”，不得作为论文脚注证据来源。
6. 后端会**合并**你输出的内容与旧稿：旧稿中已有、但你未再写出的要点**默认保留**；长期未强化才会归档或删除。
7. 若需某条永不被遗忘，可在 `user.meta.json` 的对应 `items[hash].pinned` 设为 `true`（高级用法）。
"""


INDEX_MD = """# Index

- [user.md](user.md) — 用户画像（稳定信息）
- [user.meta.json](user.meta.json) — 要点权重 / 强化与遗忘（程序维护）
- [log.md](log.md) — 更新日志（追加式）
"""


USER_MD_TEMPLATE = """# User

## Confirmed
- （暂无）

## Preferences
- （暂无）

## Projects
- （暂无）

## Hypotheses
- （暂无）
"""

DECAY_LAMBDA_PER_DAY = 0.018


def _bullet_hash(text: str) -> str:
    t = (text or "").strip()
    return hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]


def _parse_bullet_lines(md: str) -> List[str]:
    out: List[str] = []
    for line in (md or "").splitlines():
        m = re.match(r"^\s*-\s+(.+?)\s*$", line)
        if m:
            out.append(m.group(1).strip())
    return out


def _parse_sections(md: str) -> Dict[str, List[str]]:
    """按 ## 标题切分；只认行首 `## `。"""
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in (md or "").splitlines():
        st = line.strip()
        if st.startswith("## "):
            current = st
            sections.setdefault(current, [])
            continue
        if current is None:
            continue
        m = re.match(r"^\s*-\s+(.+?)\s*$", line)
        if m:
            sections[current].append(m.group(1).strip())
    return sections


def _all_known_headers() -> Set[str]:
    return set(WIKI_MAIN_SECTIONS) | {ARCHIVE_SECTION}


def _validate_candidate_structure(md: str) -> bool:
    s = md or ""
    for h in WIKI_MAIN_SECTIONS:
        if h not in s:
            return False
    return True


def _global_candidate_hashes(cand_sections: Dict[str, List[str]]) -> Set[str]:
    out: Set[str] = set()
    for sec, bs in cand_sections.items():
        if sec == ARCHIVE_SECTION:
            continue
        for b in bs:
            out.add(_bullet_hash(b))
    return out


def _merge_old_and_candidate(
    old_sections: Dict[str, List[str]],
    cand_sections: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """候选优先；旧稿中未出现在候选任意位置的要点按原小节追加保留。"""
    # 只从候选的主四段收集 hash（归档段不参与「全局覆盖」）
    cand_global = _global_candidate_hashes(cand_sections)
    out: Dict[str, List[str]] = {}

    for sec in WIKI_MAIN_SECTIONS:
        old_bs = list(old_sections.get(sec) or [])
        cand_bs = list(cand_sections.get(sec) or [])
        merged: List[str] = []
        seen: Set[str] = set()

        for b in cand_bs:
            h = _bullet_hash(b)
            if h in seen:
                continue
            seen.add(h)
            merged.append(b)

        for b in old_bs:
            h = _bullet_hash(b)
            if h in seen:
                continue
            if h in cand_global:
                continue
            merged.append(b)
            seen.add(h)

        out[sec] = merged

    # 归档段：候选优先，再保留旧归档中未出现在候选归档中的条目
    oa = list(old_sections.get(ARCHIVE_SECTION) or [])
    ca = list(cand_sections.get(ARCHIVE_SECTION) or [])
    arch_seen: Set[str] = set()
    arch_out: List[str] = []
    for b in ca:
        h = _bullet_hash(b)
        if h in arch_seen:
            continue
        arch_seen.add(h)
        arch_out.append(b)
    arch_cand_hashes = {_bullet_hash(b) for b in ca}
    for b in oa:
        h = _bullet_hash(b)
        if h in arch_seen:
            continue
        if h in arch_cand_hashes:
            continue
        arch_out.append(b)
        arch_seen.add(h)
    if arch_out:
        out[ARCHIVE_SECTION] = arch_out

    return out


def _render_user_md(sections: Dict[str, List[str]]) -> str:
    lines: List[str] = ["# User", ""]
    for sec in WIKI_MAIN_SECTIONS:
        bs = sections.get(sec) or []
        lines.append(sec)
        lines.append("")
        for b in bs:
            lines.append(f"- {b}")
        lines.append("")
    arch = sections.get(ARCHIVE_SECTION) or []
    if arch:
        lines.append(ARCHIVE_SECTION)
        lines.append("")
        for b in arch:
            lines.append(f"- {b}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _norm_bullet_for_dedup(b: str) -> str:
    # strict “完全重复”：只做空白归一化，不做同义改写
    s = (b or "").strip()
    # canonicalize width/compat forms (e.g., fullwidth punctuation)
    s = unicodedata.normalize("NFKC", s)
    # normalize all whitespace (including non-breaking spaces) into single spaces
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _dedup_sections_exact(sections: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], int]:
    """
    Remove exact duplicate bullets across the whole wiki (after whitespace normalization).
    Keep the first occurrence in section order, preserving original order.
    """
    seen: Set[str] = set()
    removed = 0
    out: Dict[str, List[str]] = {}
    order = list(WIKI_MAIN_SECTIONS) + ([ARCHIVE_SECTION] if sections.get(ARCHIVE_SECTION) else [])
    for sec in order:
        bs = list(sections.get(sec) or [])
        kept: List[str] = []
        for b in bs:
            k = _norm_bullet_for_dedup(b)
            if not k:
                continue
            if k in seen:
                removed += 1
                continue
            seen.add(k)
            kept.append(b)
        out[sec] = kept
    # keep any other sections (shouldn't exist, but be defensive)
    for sec, bs in sections.items():
        if sec in out:
            continue
        out[sec] = bs
    return out, removed


def _build_compact_prompt(*, schema: str, current_user_md: str) -> str:
    req = "\n".join(f"- 必须包含小节标题行：{h}" for h in WIKI_MAIN_SECTIONS)
    return f"""You maintain a small Markdown wiki about the USER (A版).

You MUST follow this schema:
{schema}

Hard requirements:
{req}
- Start with a single top heading `# User` then the four sections in that order.
- Use `- ` bullet lines under each section.

Task (VERY IMPORTANT):
- DO NOT add any new facts, ideas, or bullet points that are not already present.
- You MAY rewrite/merge bullets ONLY by combining/rephrasing information that is already present.
- You MAY:
  (a) delete bullets that are redundant / overlapping with other bullets already present,
  (b) merge multiple overlapping bullets into one bullet (no new info),
  (c) reorder bullets to group related items.
- When you merge: keep the meaning and details; do not introduce new names, claims, or interpretations.
- Strong preference:
  - Aggressively reduce redundancy: if two bullets overlap substantially, MERGE them.
  - Try to reduce the total number of bullets while preserving all information already present.
  - Do not leave obviously repetitive bullets.

Current user.md:
{current_user_md}

Output:
- Output ONLY the full updated contents of user.md (Markdown). No extra commentary.
"""


def _build_subset_compact_prompt(*, schema: str, current_user_md: str) -> str:
    req = "\n".join(f"- 必须包含小节标题行：{h}" for h in WIKI_MAIN_SECTIONS)
    return f"""You maintain a small Markdown wiki about the USER (A版).

You MUST follow this schema:
{schema}

Hard requirements:
{req}
- Start with a single top heading `# User` then the four sections in that order.
- Use `- ` bullet lines under each section.

Task:
- Output a cleaned version of current user.md.
- You may DELETE duplicate or redundant bullets.
- You may REORDER bullets under the existing sections.
- You MUST NOT rewrite, paraphrase, merge, or add any bullet.
- Every output bullet must be copied verbatim from Current user.md.

Current user.md:
{current_user_md}

Output:
- Output ONLY the full updated contents of user.md (Markdown). No extra commentary.
"""


def _call_compact_llm_once(
    *,
    p: "WikiPaths",
    stage: str,
    prompt: str,
    llm_provider: str,
    llm_model: str,
    max_output_tokens: int,
) -> Tuple[str, str]:
    text, used = _call_wiki_llm(
        p=p,
        stage=stage,
        prompt=prompt,
        llm_provider=llm_provider,
        llm_model=llm_model,
        temperature=0.0,
        max_output_tokens=max_output_tokens,
    )
    cand = (text or "").strip()
    if cand:
        return cand, used
    text2, used2 = _call_wiki_llm(
        p=p,
        stage=f"{stage}_retry_empty",
        prompt=prompt,
        llm_provider=llm_provider,
        llm_model=llm_model,
        temperature=0.0,
        max_output_tokens=max_output_tokens,
    )
    return (text2 or "").strip(), used2


def _render_compact_candidate(cand: str) -> str:
    s2 = _parse_sections(cand)
    for h in WIKI_MAIN_SECTIONS:
        s2.setdefault(h, [])
    s2d, _rm2 = _dedup_sections_exact(s2)
    return _render_user_md(s2d)


def _compact_user_md_with_llm(
    *,
    p: "WikiPaths",
    schema: str,
    base_md: str,
    llm_provider: str,
    llm_model: str,
    stage: str,
) -> Tuple[Optional[str], str]:
    max_out = int(getattr(config, "WIKI_COMPACT_MAX_OUTPUT_TOKENS", 4096))
    details: List[str] = []

    try:
        prompt = _build_compact_prompt(schema=schema, current_user_md=base_md)
        cand, used = _call_compact_llm_once(
            p=p,
            stage=stage,
            prompt=prompt,
            llm_provider=llm_provider,
            llm_model=llm_model,
            max_output_tokens=max_out,
        )
        if cand and _validate_candidate_structure(cand):
            return _render_compact_candidate(cand), ""
        details.append(
            f"rewrite compact rejected: invalid structure or empty; used_model={used}; preview={_one_line(cand, 220)}"
        )
    except Exception as e:
        details.append(f"rewrite compact failed: {_wiki_llm_error_summary(e)}")

    # Fallback keeps the stricter prompt, but acceptance remains structural.
    # The behavioral constraints live in the prompt, not in brittle post-hoc
    # semantic/token validation.
    try:
        prompt = _build_subset_compact_prompt(schema=schema, current_user_md=base_md)
        cand, used = _call_compact_llm_once(
            p=p,
            stage=f"{stage}_strict_subset",
            prompt=prompt,
            llm_provider=llm_provider,
            llm_model=llm_model,
            max_output_tokens=max_out,
        )
        if cand and _validate_candidate_structure(cand):
            return _render_compact_candidate(cand), ""
        details.append(
            f"strict subset compact rejected: invalid structure or empty; used_model={used}; preview={_one_line(cand, 220)}"
        )
    except Exception as e:
        details.append(f"strict subset compact failed: {_wiki_llm_error_summary(e)}")

    return None, "; ".join(x for x in details if x)


def compact_user_wiki_sync(
    *,
    user_id: str,
    llm_provider: str,
    llm_model: str,
    data_dir: str = "data",
    reason: str = "manual",
) -> Dict[str, Any]:
    """
    Force a wiki compaction immediately (regardless of write_count), then reset write_count to 0.
    MUST NOT add new bullets; any violation will be rejected (no overwrite).
    """
    p = ensure_wiki_initialized(user_id=user_id, data_dir=data_dir)
    lk = _lock_for(str(p.root))
    now = time.time()
    ok = False
    detail = ""
    before_md = ""
    after_md = ""
    removed_exact = 0
    before_bullets = 0
    base_bullets = 0
    after_bullets = 0
    changed = False
    with lk:
        meta0 = _load_wiki_meta(p.meta)
        meta0 = _migrate_meta(meta0, now=now)
        try:
            schema = p.schema.read_text(encoding="utf-8")
        except Exception:
            schema = SCHEMA_MD
        try:
            before_md = p.user.read_text(encoding="utf-8")
        except Exception:
            before_md = USER_MD_TEMPLATE

    # baseline exact-dedup first (safe)
    s0 = _parse_sections(before_md)
    for h in WIKI_MAIN_SECTIONS:
        s0.setdefault(h, [])
    s0d, removed_exact = _dedup_sections_exact(s0)
    base_md = _render_user_md(s0d)
    before_bullets = len(_parse_bullet_lines(before_md))
    base_bullets = len(_parse_bullet_lines(base_md))

    compacted, detail = _compact_user_md_with_llm(
        p=p,
        schema=schema,
        base_md=base_md,
        llm_provider=llm_provider,
        llm_model=llm_model,
        stage="manual_compact",
    )
    if compacted:
        after_md = compacted
        after_bullets = len(_parse_bullet_lines(after_md))
        changed = (after_md != before_md)
        ok = True
    elif removed_exact:
        after_md = base_md
        after_bullets = base_bullets
        changed = (after_md != before_md)
        ok = True
        detail = detail or "llm compact unavailable; applied deterministic exact-dedup only"

    with lk:
        meta1 = _load_wiki_meta(p.meta)
        meta1 = _migrate_meta(meta1, now=time.time())
        if ok:
            meta1["write_count"] = 0
        meta1["last_app_use_ts"] = time.time()
        if ok and after_md:
            try:
                p.user.write_text(after_md, encoding="utf-8")
                # housekeeping: do not treat as reinforcement
                meta_out = _sync_meta_after_merge(
                    meta1,
                    after_md,
                    cand_hashes=set(),
                    next_round=int(meta1.get("qa_round", 0)),
                    now=time.time(),
                    forgiveness=False,
                )
                meta_out["write_count"] = 0
                _save_wiki_meta(p.meta, meta_out)
            except Exception as e:
                ok = False
                detail = f"write failed: {e}"
        else:
            _save_wiki_meta(p.meta, meta1)

        try:
            log = (p.log.read_text(encoding="utf-8") if p.log.exists() else "# Log\n\n") + (
                f"## [{_now_date()}] wiki_compact | {_now_ts()}\n\n"
                f"- ok: {'true' if ok else 'false'}\n"
                f"- reason: {reason}\n"
                + (f"- detail: {detail}\n" if (detail and not ok) else "")
                + (f"- baseline_dedup_exact_removed: {removed_exact}\n" if removed_exact else "")
                + "\n"
            )
            p.log.write_text(log, encoding="utf-8")
        except Exception:
            pass

    return {
        "ok": ok,
        "changed": bool(changed),
        "reason": reason,
        "detail": detail,
        "baseline_dedup_exact_removed": removed_exact,
        "before_bullets": before_bullets,
        "base_bullets": base_bullets,
        "after_bullets": after_bullets,
        "before_chars": len(before_md or ""),
        "after_chars": len(after_md or ""),
    }


def _forget_thresholds_for_section(sec: str) -> Tuple[float, int]:
    if sec.strip() == "## Hypotheses":
        return (
            float(getattr(config, "WIKI_HYPOTHESIS_FORGET_MIN_DAYS", 90.0)),
            int(getattr(config, "WIKI_HYPOTHESIS_FORGET_MIN_ROUNDS", 6)),
        )
    return (
        float(getattr(config, "WIKI_FORGET_MIN_DAYS", 120.0)),
        int(getattr(config, "WIKI_FORGET_MIN_ROUNDS", 8)),
    )


def _apply_forget_and_archive(
    sections: Dict[str, List[str]],
    meta: Dict[str, Any],
    *,
    qa_round: int,
    next_round: int,
    now: float,
    forgiveness: bool,
) -> Tuple[Dict[str, List[str]], List[str]]:
    """从主四段移除遗忘项，追加到 Archive。forgiveness 时不删除。"""
    if forgiveness:
        return sections, []

    items: Dict[str, Any] = dict(meta.get("items") or {})
    log_lines: List[str] = []
    arch: List[str] = list(sections.get(ARCHIVE_SECTION) or [])
    out_main: Dict[str, List[str]] = {k: list(v) for k, v in sections.items() if k != ARCHIVE_SECTION}

    for sec in WIKI_MAIN_SECTIONS:
        min_days, min_rounds = _forget_thresholds_for_section(sec)
        kept: List[str] = []
        for b in out_main.get(sec) or []:
            h = _bullet_hash(b)
            ent = dict(items.get(h) or {})
            if ent.get("pinned"):
                kept.append(b)
                continue
            # 无 last_reinforced_* 的旧数据：用 last_ts 近似「上次相关时间」；轮次用上一轮 qa_round
            lr = float(
                ent.get("last_reinforced_ts")
                or ent.get("last_ts")
                or now
            )
            lrr = int(ent.get("last_reinforced_round", qa_round))
            days = (now - lr) / 86400.0
            rounds_gap = next_round - lrr
            if days >= min_days and rounds_gap >= min_rounds:
                tag = f"（{_now_date()} 自 {sec.replace('## ', '')} 遗忘） {b}"
                arch.append(tag)
                log_lines.append(f"forgotten: {sec} / {h[:8]}…")
            else:
                kept.append(b)
        out_main[sec] = kept

    out: Dict[str, List[str]] = dict(out_main)
    if arch:
        out[ARCHIVE_SECTION] = arch
    return out, log_lines


def _sync_meta_after_merge(
    prev_meta: Dict[str, Any],
    merged_md: str,
    cand_hashes: Set[str],
    *,
    next_round: int,
    now: float,
    forgiveness: bool,
) -> Dict[str, Any]:
    """为最终文中的每条 bullet 维护 items；强化信息随合并结果更新。"""
    decay = float((prev_meta or {}).get("decay_lambda") or DECAY_LAMBDA_PER_DAY)
    old_items: Dict[str, Any] = dict((prev_meta or {}).get("items") or {})
    bullets = _parse_bullet_lines(merged_md)
    new_items: Dict[str, Any] = {}

    for b in bullets:
        h = _bullet_hash(b)
        ent = dict(old_items.get(h) or {})
        ent.setdefault("access", 0)
        ent.setdefault("w", 1.0)
        ent.setdefault("last_ts", now)
        ent.setdefault("pinned", False)

        if forgiveness:
            ent["last_reinforced_ts"] = now
            ent["last_reinforced_round"] = next_round
        elif h in cand_hashes:
            ent["last_reinforced_ts"] = now
            ent["last_reinforced_round"] = next_round
        else:
            # 候选未再写出、由合并保留的旧要点：保留上一轮强化时间，不刷新
            o = old_items.get(h) or {}
            if o.get("last_reinforced_ts") is not None:
                ent["last_reinforced_ts"] = float(o["last_reinforced_ts"])
            else:
                ent.setdefault("last_reinforced_ts", now)
            if o.get("last_reinforced_round") is not None:
                ent["last_reinforced_round"] = int(o["last_reinforced_round"])
            else:
                ent.setdefault("last_reinforced_round", max(0, next_round - 1))

        new_items[h] = ent

    return {
        "items": new_items,
        "decay_lambda": decay,
        "qa_round": next_round,
        "last_app_use_ts": now,
    }


def _migrate_meta(raw: Dict[str, Any], *, now: float) -> Dict[str, Any]:
    raw = dict(raw or {})
    raw.setdefault("items", {})
    raw.setdefault("decay_lambda", DECAY_LAMBDA_PER_DAY)
    raw.setdefault("qa_round", 0)
    raw.setdefault("last_app_use_ts", now)
    raw.setdefault("write_count", 0)
    return raw


def _load_wiki_meta(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return _migrate_meta({}, now=time.time())
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return _migrate_meta(obj, now=time.time())
    except Exception:
        pass
    return _migrate_meta({}, now=time.time())


def _save_wiki_meta(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _effective_weight(ent: Dict[str, Any], now_ts: float, lam: float) -> float:
    w = float(ent.get("w", 1.0))
    access = int(ent.get("access", 0))
    last_ts = float(ent.get("last_ts", now_ts))
    days = max(0.0, (now_ts - last_ts) / 86400.0)
    boost = 1.0 + 0.12 * math.log(1.0 + max(0, access))
    return max(0.05, w * math.exp(-lam * days) * boost)


def _question_touches_bullet(question: str, bullet: str) -> bool:
    q = (question or "").strip().lower()
    b = (bullet or "").strip().lower()
    if not q or not b:
        return False
    if len(b) >= 6 and b in q:
        return True
    toks = re.split(r"[^\w\u4e00-\u9fff]+", q)
    toks = [t for t in toks if len(t) >= 2]
    return any(t and t in b for t in toks[:24])


def _touch_meta_for_question(
    meta: Dict[str, Any],
    bullets: List[str],
    question: str,
) -> Dict[str, Any]:
    now = time.time()
    items: Dict[str, Any] = dict(meta.get("items") or {})
    changed = False
    for b in bullets:
        if not _question_touches_bullet(question, b):
            continue
        h = _bullet_hash(b)
        ent = dict(items.get(h) or {})
        ent["access"] = int(ent.get("access", 0)) + 1
        ent["last_ts"] = now
        ent["w"] = min(1.6, float(ent.get("w", 1.0)) * 1.03)
        items[h] = ent
        changed = True
    if changed:
        meta = dict(meta)
        meta["items"] = items
    return meta


def _format_weighted_wiki_for_prompt(
    raw_md: str,
    meta: Dict[str, Any],
    *,
    max_chars: int,
) -> str:
    if max_chars <= 0:
        return ""
    bullets = _parse_bullet_lines(raw_md)
    if not bullets:
        t = (raw_md or "").strip()
        return t[:max_chars] if max_chars > 0 else t
    now = time.time()
    lam = float(meta.get("decay_lambda") or DECAY_LAMBDA_PER_DAY)
    items: Dict[str, Any] = dict(meta.get("items") or {})
    scored: List[Tuple[float, str]] = []
    for b in bullets:
        h = _bullet_hash(b)
        ent = items.get(h) or {"access": 0, "last_ts": now, "w": 1.0}
        eff = _effective_weight(ent, now, lam)
        scored.append((eff, b))
    scored.sort(key=lambda x: -x[0])
    lines = [f"- {b}" for _, b in scored]
    blob = "User wiki bullets (sorted by relevance; decayed by time, boosted by access):\n" + "\n".join(
        lines
    )
    if max_chars > 0 and len(blob) > max_chars:
        return blob[: max_chars - 1].rstrip() + "…"
    return blob


def ensure_wiki_initialized(*, user_id: str, data_dir: str = "data") -> WikiPaths:
    p = wiki_paths(user_id=user_id, data_dir=data_dir)
    lk = _lock_for(str(p.root))
    with lk:
        p.root.mkdir(parents=True, exist_ok=True)
        if not p.schema.exists():
            p.schema.write_text(SCHEMA_MD, encoding="utf-8")
        if not p.index.exists():
            p.index.write_text(INDEX_MD, encoding="utf-8")
        if not p.user.exists():
            p.user.write_text(USER_MD_TEMPLATE, encoding="utf-8")
        if not p.log.exists():
            p.log.write_text("# Log\n\n", encoding="utf-8")
        if not p.meta.exists():
            _save_wiki_meta(
                p.meta,
                _migrate_meta({"items": {}}, now=time.time()),
            )
    return p


def read_user_wiki(
    *,
    user_id: str,
    data_dir: str = "data",
    max_chars: int = 4000,
    question: str = "",
) -> str:
    p = ensure_wiki_initialized(user_id=user_id, data_dir=data_dir)
    lk = _lock_for(str(p.root))
    with lk:
        try:
            text = p.user.read_text(encoding="utf-8")
        except Exception:
            text = ""
        meta = _load_wiki_meta(p.meta)
        bullets = _parse_bullet_lines(text)
        if question and bullets:
            meta = _touch_meta_for_question(meta, bullets, question)
            _save_wiki_meta(p.meta, meta)
        return _format_weighted_wiki_for_prompt(text, meta, max_chars=max_chars)


def _build_update_prompt(*, schema: str, current_user_md: str, q: str, a: str) -> str:
    req = "\n".join(f"- 必须包含小节标题行：{h}" for h in WIKI_MAIN_SECTIONS)
    return f"""You maintain a small Markdown wiki about the USER (A版).

You MUST follow this schema:
{schema}

Hard requirements:
{req}
- Never include sensitive data.
- Only record stable, reusable user facts/preferences/projects (per schema).

CRITICAL OUTPUT FORMAT (delta-only; do NOT rewrite the full user.md):
- Output ONLY a single JSON object with exactly these keys:
  - "add_confirmed": array of strings (bullets to add to ## Confirmed)
  - "add_preferences": array of strings (bullets to add to ## Preferences)
  - "add_projects": array of strings (bullets to add to ## Projects)
  - "add_hypotheses": array of strings (bullets to add to ## Hypotheses; for uncertain info)
  - "remove_bullets": array of strings (bullets to remove by exact text match; optional clean-up)
- Each string MUST be the bullet text WITHOUT leading "- ".
- Return empty arrays when nothing should change.
- Do not include any other keys, comments, markdown, or surrounding text.

Current user.md:
{current_user_md}

New conversation snippet (latest Q/A):
User question:
{q}

Assistant answer:
{a}

Task:
- Extract any NEW stable information worth remembering and propose it as additions in the JSON.
- If the assistant answer merely restates existing information, output all empty arrays.

Output:
- Output ONLY the JSON object. No extra commentary.
"""


def _parse_wiki_delta_json(text: str) -> Optional[Dict[str, List[str]]]:
    """
    Parse the LLM delta JSON. Return a dict with the exact keys, each mapped to a list of strings.
    Unknown keys are ignored and missing keys are treated as empty lists; values remain strict lists of strings.
    """
    if not text:
        return None
    t = (text or "").strip()
    if not t:
        return None
    # Defensive: allow fenced blocks from some models.
    if t.startswith("```"):
        t2 = re.sub(r"^\s*```(?:json)?\s*", "", t, flags=re.I)
        t2 = re.sub(r"\s*```\s*$", "", t2)
        t = t2.strip() or t

    # If the model added any extra text, try to extract the first JSON object.
    # This keeps safety (still strict keys) while being robust to "Here is the JSON:" wrappers.
    if not (t.startswith("{") and t.endswith("}")):
        i = t.find("{")
        j = t.rfind("}")
        if 0 <= i < j:
            cand = t[i : j + 1].strip()
            if cand.startswith("{") and cand.endswith("}"):
                t = cand

    try:
        obj = json.loads(t)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    allowed = {
        "add_confirmed",
        "add_preferences",
        "add_projects",
        "add_hypotheses",
        "remove_bullets",
    }
    out: Dict[str, List[str]] = {}
    for k in allowed:
        v = obj.get(k, [])
        if not isinstance(v, list):
            return None
        cleaned: List[str] = []
        for it in v:
            if not isinstance(it, str):
                return None
            s = it.strip()
            if not s:
                continue
            # Ensure caller doesn't include "- " prefix.
            if s.startswith("- "):
                s = s[2:].strip()
            if s:
                cleaned.append(s)
        out[k] = cleaned
    return out


def _wiki_delta_key_note(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^\s*```(?:json)?\s*", "", t, flags=re.I)
        t = re.sub(r"\s*```\s*$", "", t).strip() or t
    if not (t.startswith("{") and t.endswith("}")):
        i = t.find("{")
        j = t.rfind("}")
        if 0 <= i < j:
            t = t[i : j + 1].strip()
    try:
        obj = json.loads(t)
    except Exception:
        return "json_parse_failed"
    if not isinstance(obj, dict):
        return "json_not_object"
    allowed = {
        "add_confirmed",
        "add_preferences",
        "add_projects",
        "add_hypotheses",
        "remove_bullets",
    }
    keys = set(obj.keys())
    ignored = sorted(keys - allowed)
    missing = sorted(allowed - keys)
    notes = []
    if ignored:
        notes.append("ignored_keys=" + ",".join(ignored[:10]))
    if missing:
        notes.append("missing_keys_filled_empty=" + ",".join(missing[:10]))
    return "; ".join(notes) if notes else "keys_ok"


def _apply_wiki_delta(
    old_sections: Dict[str, List[str]],
    delta: Dict[str, List[str]],
) -> Tuple[Dict[str, List[str]], Set[str], int, int]:
    """
    Apply delta to existing sections.
    - Adds are appended within their target section (no reordering).
    - Removes are by exact normalized-match across ALL main sections + archive.
    Returns: (new_sections, added_hashes, added_count, removed_count)
    """
    sections: Dict[str, List[str]] = {k: list(v) for k, v in (old_sections or {}).items()}
    for h in WIKI_MAIN_SECTIONS:
        sections.setdefault(h, [])
    # Preserve archive if exists
    if ARCHIVE_SECTION in old_sections:
        sections.setdefault(ARCHIVE_SECTION, list(old_sections.get(ARCHIVE_SECTION) or []))

    remove_norm = {_norm_bullet_for_dedup(b) for b in (delta.get("remove_bullets") or [])}
    remove_norm.discard("")
    removed = 0
    if remove_norm:
        for sec, bs in list(sections.items()):
            kept: List[str] = []
            for b in bs or []:
                if _norm_bullet_for_dedup(b) in remove_norm:
                    removed += 1
                    continue
                kept.append(b)
            sections[sec] = kept

    # Build global set of existing bullets (normalized) to avoid duplicates across sections.
    existing_norm: Set[str] = set()
    for sec, bs in sections.items():
        for b in bs or []:
            existing_norm.add(_norm_bullet_for_dedup(b))

    add_map = {
        "add_confirmed": "## Confirmed",
        "add_preferences": "## Preferences",
        "add_projects": "## Projects",
        "add_hypotheses": "## Hypotheses",
    }
    added_hashes: Set[str] = set()
    added = 0
    for dk, sec in add_map.items():
        for b in (delta.get(dk) or []):
            nb = _norm_bullet_for_dedup(b)
            if not nb or nb in existing_norm:
                continue
            sections[sec].append(b.strip())
            existing_norm.add(nb)
            added += 1
            added_hashes.add(_bullet_hash(b.strip()))

    return sections, added_hashes, added, removed


def update_user_wiki_sync(
    *,
    user_id: str,
    question: str,
    answer: str,
    llm_provider: str,
    llm_model: str,
    data_dir: str = "data",
) -> Optional[str]:
    """
    候选全文 + 与旧稿合并 + 强化/遗忘 + 长期未使用保护。
    失败时返回 None，不覆盖磁盘。
    """
    p = ensure_wiki_initialized(user_id=user_id, data_dir=data_dir)
    now = time.time()
    grace_sec = float(getattr(config, "WIKI_ABSENCE_GRACE_DAYS", 45.0)) * 86400.0
    max_out = int(getattr(config, "WIKI_UPDATE_MAX_OUTPUT_TOKENS", 8192))

    lk = _lock_for(str(p.root))
    with lk:
        meta0 = _load_wiki_meta(p.meta)
        meta0 = _migrate_meta(meta0, now=now)
        prev_use = float(meta0.get("last_app_use_ts", now))
        forgiveness = (now - prev_use) > grace_sec
        qa_round = int(meta0.get("qa_round", 0))
        next_round = qa_round + 1
        write_count0 = int(meta0.get("write_count", 0))

        try:
            schema = p.schema.read_text(encoding="utf-8")
        except Exception:
            schema = SCHEMA_MD
        try:
            cur_user = p.user.read_text(encoding="utf-8")
        except Exception:
            cur_user = USER_MD_TEMPLATE

    prompt = _build_update_prompt(
        schema=schema,
        current_user_md=cur_user,
        q=question,
        a=answer,
    )
    # Gemini sometimes returns empty/ill-formed JSON transiently. Retry once with lower temperature.
    last_text = ""
    last_used = ""
    last_key_note = ""
    for attempt in range(1, 3):
        try:
            text, used = _call_wiki_llm(
                p=p,
                stage=f"update_delta_attempt_{attempt}",
                prompt=prompt,
                llm_provider=llm_provider,
                llm_model=llm_model,
                temperature=0.2 if attempt == 1 else 0.0,
                max_output_tokens=max_out,
            )
            last_used = used
        except Exception as e:
            with lk:
                _append_wiki_log(
                    p,
                    "wiki_update_skipped",
                    [
                        "reason: llm_call_failed",
                        f"attempt: {attempt}/2",
                        f"requested_model: {llm_provider}:{llm_model}",
                        f"error: {_wiki_llm_error_summary(e)}",
                    ],
                )
            return None

        candidate_raw = (text or "").strip()
        last_text = candidate_raw
        last_key_note = _wiki_delta_key_note(candidate_raw)
        delta = _parse_wiki_delta_json(candidate_raw)
        if delta:
            break
    else:
        delta = None

    if not delta:
        # Best-effort: log parse failure for debugging.
        with lk:
            try:
                log = (p.log.read_text(encoding="utf-8") if p.log.exists() else "# Log\n\n") + (
                    f"## [{_now_date()}] wiki_update_skipped | {_now_ts()}\n\n"
                    f"- reason: delta_json_parse_failed_or_empty\n"
                    f"- requested_model: {llm_provider}:{llm_model}\n"
                    f"- used_model: {last_used}\n"
                    f"- key_note: {last_key_note}\n"
                    f"- preview: {last_text[:220].replace(chr(10),' ').replace(chr(13),' ')}\n\n"
                )
                p.log.write_text(log, encoding="utf-8")
            except Exception:
                pass
        return None

    old_sec = _parse_sections(cur_user)
    merged, cand_hashes, added_cnt, removed_cnt = _apply_wiki_delta(old_sec, delta)

    merged_after_forget, forget_log = _apply_forget_and_archive(
        merged,
        meta0,
        qa_round=qa_round,
        next_round=next_round,
        now=now,
        forgiveness=forgiveness,
    )

    # Always do a deterministic exact-dedup pass to suppress obvious accumulation.
    merged_after_dedup, dedup_removed = _dedup_sections_exact(merged_after_forget)
    final_md = _render_user_md(merged_after_dedup)
    if not _validate_candidate_structure(final_md):
        return None

    meta1 = _sync_meta_after_merge(
        meta0,
        final_md,
        cand_hashes,
        next_round=next_round,
        now=now,
        forgiveness=forgiveness,
    )
    # bump write counter
    meta1["write_count"] = write_count0 + 1

    with lk:
        try:
            p.user.write_text(final_md, encoding="utf-8")
            _save_wiki_meta(p.meta, meta1)
            log_body = (
                (p.log.read_text(encoding="utf-8") if p.log.exists() else "# Log\n\n")
                + f"## [{_now_date()}] qa | {_now_ts()}\n\n"
                + f"- updated: user.md (delta+forget; round={next_round}; forgiveness={forgiveness}; write_count={meta1.get('write_count')})\n"
                + f"- requested_model: {llm_provider}:{llm_model}\n"
                + f"- used_model: {last_used}\n"
                + f"- delta_key_note: {last_key_note}\n"
                + f"- delta_added: {added_cnt}\n"
                + f"- delta_removed: {removed_cnt}\n"
            )
            if dedup_removed:
                log_body += f"- dedup_exact_removed: {dedup_removed}\n"
            if forget_log:
                log_body += "".join(f"- {x}\n" for x in forget_log[:20])
                if len(forget_log) > 20:
                    log_body += f"- … and {len(forget_log) - 20} more\n"
            log_body += "\n"
            p.log.write_text(log_body, encoding="utf-8")
        except Exception:
            return None

    # Periodic compaction after N writes: try rewrite/merge first, then a strict
    # verbatim-delete/reorder fallback. Failures preserve write_count for retry.
    try:
        n = int(getattr(config, "WIKI_COMPACT_EVERY_N_WRITES", 5))
    except Exception:
        n = 5
    if n > 0 and int(meta1.get("write_count") or 0) >= n:
        compacted = None
        compact_reason = ""
        compacted, compact_reason = _compact_user_md_with_llm(
            p=p,
            schema=schema,
            base_md=final_md,
            llm_provider=llm_provider,
            llm_model=llm_model,
            stage="auto_compact",
        )

        if compacted:
            with lk:
                try:
                    meta2 = _load_wiki_meta(p.meta)
                    meta2 = _migrate_meta(meta2, now=time.time())
                    # housekeeping: do not treat as reinforcement; just keep timestamps/rounds consistent
                    meta_out = _sync_meta_after_merge(
                        meta2,
                        compacted,
                        cand_hashes=set(),
                        next_round=int(meta2.get("qa_round", next_round)),
                        now=time.time(),
                        forgiveness=False,
                    )
                    meta_out["write_count"] = 0
                    p.user.write_text(compacted, encoding="utf-8")
                    _save_wiki_meta(p.meta, meta_out)
                    log = (p.log.read_text(encoding="utf-8") if p.log.exists() else "# Log\n\n") + (
                        f"## [{_now_date()}] wiki_compact | {_now_ts()}\n\n"
                        f"- ok: true (every_n={n})\n\n"
                    )
                    p.log.write_text(log, encoding="utf-8")
                except Exception:
                    pass
            return compacted
        else:
            with lk:
                try:
                    meta2 = _load_wiki_meta(p.meta)
                    meta2 = _migrate_meta(meta2, now=time.time())
                    _save_wiki_meta(p.meta, meta2)
                    log = (p.log.read_text(encoding="utf-8") if p.log.exists() else "# Log\n\n") + (
                        f"## [{_now_date()}] wiki_compact | {_now_ts()}\n\n"
                        f"- ok: false ({compact_reason or 'unknown'})\n"
                        f"- action: write_count preserved for retry (dedup already applied)\n\n"
                    )
                    p.log.write_text(log, encoding="utf-8")
                except Exception:
                    pass

    return final_md


def update_user_wiki_async(
    *,
    user_id: str,
    question: str,
    answer: str,
    llm_provider: str,
    llm_model: str,
    data_dir: str = "data",
) -> None:
    def _run():
        update_user_wiki_sync(
            user_id=user_id,
            question=question,
            answer=answer,
            llm_provider=llm_provider,
            llm_model=llm_model,
            data_dir=data_dir,
        )

    t = threading.Thread(target=_run, daemon=True)
    t.start()
