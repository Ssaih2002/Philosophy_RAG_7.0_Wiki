from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_LOCKS: Dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


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
    # keep it filesystem-friendly; allow a-zA-Z0-9_- only
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:80] or default


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


@dataclass(frozen=True)
class ConversationPaths:
    root: Path
    jsonl_path: Path


def conversation_paths(
    *,
    user_id: str,
    conversation_id: str,
    data_dir: str = "data",
) -> ConversationPaths:
    uid = _safe_id(user_id, default="default")
    cid = _safe_id(conversation_id, default="conv")
    root = Path(data_dir) / "memory" / "conversations" / uid
    return ConversationPaths(root=root, jsonl_path=root / f"{cid}.jsonl")


def append_event(
    *,
    user_id: str,
    conversation_id: str,
    role: str,
    content: str,
    meta: Optional[Dict[str, Any]] = None,
    data_dir: str = "data",
) -> None:
    paths = conversation_paths(user_id=user_id, conversation_id=conversation_id, data_dir=data_dir)
    lk = _lock_for(str(paths.jsonl_path))
    with lk:
        paths.root.mkdir(parents=True, exist_ok=True)
        rec: Dict[str, Any] = {
            "ts": _now_iso(),
            "role": (role or "").strip() or "user",
            "content": content or "",
        }
        if meta:
            rec["meta"] = meta
        line = json.dumps(rec, ensure_ascii=False)
        with paths.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _read_last_jsonl_lines(path: Path, *, max_lines: int) -> List[Dict[str, Any]]:
    if max_lines <= 0 or not path.exists():
        return []
    # Read from end in a simple (but safe) way; JSONL files are expected small/moderate.
    # If it grows huge, we can optimize later.
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []
    lines = [ln for ln in text.splitlines() if ln.strip()]
    tail = lines[-max_lines:] if max_lines > 0 else lines
    out: List[Dict[str, Any]] = []
    for ln in tail:
        try:
            obj = json.loads(ln)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out


def load_recent_turns(
    *,
    user_id: str,
    conversation_id: str,
    max_turns: int,
    max_chars: int,
    data_dir: str = "data",
) -> List[Tuple[str, str]]:
    """
    Return a list of (role, content) in chronological order, clipped by:
    - max_turns (user+assistant pairs approx; we treat as messages count/2)
    - max_chars (total content chars)
    """
    paths = conversation_paths(user_id=user_id, conversation_id=conversation_id, data_dir=data_dir)
    lk = _lock_for(str(paths.jsonl_path))
    with lk:
        # Read more than needed, then clip. Each turn is 2 messages typically.
        want_msgs = max(0, int(max_turns)) * 2
        want_msgs = max(want_msgs, 0)
        # if max_turns==0 => no history
        if want_msgs <= 0 or max_chars <= 0:
            return []
        raw = _read_last_jsonl_lines(paths.jsonl_path, max_lines=want_msgs)
    msgs: List[Tuple[str, str]] = []
    for it in raw:
        role = str(it.get("role") or "").strip() or "user"
        content = str(it.get("content") or "")
        if not content:
            continue
        if role not in ("user", "assistant"):
            # ignore other internal roles
            continue
        msgs.append((role, content))

    # clip by chars from end (keep most recent)
    if max_chars > 0:
        total = 0
        kept_rev: List[Tuple[str, str]] = []
        for role, content in reversed(msgs):
            c = len(content)
            if total + c > max_chars and kept_rev:
                break
            if total + c > max_chars and not kept_rev:
                # keep at least one message (truncate)
                kept_rev.append((role, content[-max_chars:]))
                total = max_chars
                break
            kept_rev.append((role, content))
            total += c
        msgs = list(reversed(kept_rev))
    return msgs


def _similar_enough(a: str, b: str, *, ratio: float = 0.92) -> bool:
    """Cheap similarity: prefix overlap for long repeated assistant errors."""
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return False
    if a == b:
        return True
    n = min(len(a), len(b), 800)
    if n < 24:
        return False
    ca, cb = a[:n], b[:n]
    same = sum(1 for i in range(n) if ca[i] == cb[i])
    return (same / float(n)) >= ratio


def dedupe_adjacent_messages(
    messages: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for role, content in messages:
        if out and out[-1][0] == role and _similar_enough(out[-1][1], content):
            continue
        out.append((role, content))
    return out


def clip_block(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if max_chars <= 0 or not t:
        return ""
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "…"


def format_history_block(messages: List[Tuple[str, str]]) -> str:
    if not messages:
        return ""
    parts: List[str] = []
    for role, content in messages:
        if role == "user":
            parts.append(f"User: {content}")
        else:
            parts.append(f"Assistant: {content}")
    return "\n\n".join(parts).strip()


def prepare_history_block_for_prompt(
    messages: List[Tuple[str, str]],
    *,
    history_max_chars: int,
) -> str:
    msgs = dedupe_adjacent_messages(messages)
    hb = format_history_block(msgs)
    return clip_block(hb, history_max_chars)

