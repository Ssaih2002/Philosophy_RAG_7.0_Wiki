from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import CHROMA_PATH, SPARSE_DB_PATH


LIB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_library_id(library_id: str) -> str:
    """
    library_id 仅允许 [A-Za-z0-9_-]，避免路径穿越/特殊字符导致的跨平台问题。
    """
    lid = (library_id or "").strip()
    if not lid:
        raise ValueError("library_id is required")
    if not LIB_ID_RE.match(lid):
        raise ValueError("library_id must match ^[A-Za-z0-9_-]{1,64}$")
    return lid


def normalize_library_id(library_id: Optional[str]) -> str:
    """
    Normalize user input into a safe library_id.

    - Trim whitespace
    - Convert any whitespace to '_'
    - Replace any non [A-Za-z0-9_-] with '_'
    - Collapse repeated '_' and strip leading/trailing '_'
    - Enforce max length 64
    - Fallback to 'default' when empty
    """
    raw = (library_id or "").strip()
    if not raw:
        return "default"
    # normalize whitespace to underscore
    s = re.sub(r"\s+", "_", raw)
    # replace other illegal chars
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
    # collapse underscores
    s = re.sub(r"_+", "_", s).strip("_")
    s = s[:64].strip("_")
    return s or "default"


@dataclass(frozen=True)
class LibraryKey:
    profile: str
    library_id: str  # normalized, never empty

    @property
    def key(self) -> str:
        return f"{self.profile}__{self.library_id}"


def chroma_dir_for(profile: str, library_id: Optional[str]) -> str:
    """
    - legacy: data/chroma_db_<profile>/  (library_id == default)
    - named:  data/chroma_db_<profile>__<library_id>/
    """
    lid = normalize_library_id(library_id)
    base = Path(CHROMA_PATH)
    if lid == "default":
        return str(base.parent / f"{base.name}_{profile}")
    validate_library_id(lid)
    return str(base.parent / f"{base.name}_{profile}__{lid}")


def chroma_collection_for(profile: str, library_id: Optional[str]) -> str:
    lid = normalize_library_id(library_id)
    if lid == "default":
        return f"philosophy_{profile}"
    validate_library_id(lid)
    return f"philosophy_{profile}__{lid}"


def sparse_db_for(profile: str, library_id: Optional[str]) -> str:
    """
    - legacy: data/sparse_fts_<profile>.db (library_id == default)
    - named:  data/sparse_fts_<profile>__<library_id>.db
    """
    lid = normalize_library_id(library_id)
    base = Path(SPARSE_DB_PATH)
    if lid == "default":
        return str(base.parent / f"{base.stem}_{profile}{base.suffix}")
    validate_library_id(lid)
    return str(base.parent / f"{base.stem}_{profile}__{lid}{base.suffix}")


def list_libraries(data_dir: str = "data") -> List[LibraryKey]:
    """
    通过扫描 data/ 下的 chroma_db_* 目录推断可用 libraries。
    """
    root = Path(data_dir)
    if not root.exists():
        return []

    out: List[LibraryKey] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if not name.startswith("chroma_db_"):
            continue
        tail = name[len("chroma_db_") :]
        # quality / quality__MEGA2
        if "__" in tail:
            prof, lid = tail.split("__", 1)
            lid = normalize_library_id(lid)
        else:
            prof, lid = tail, "default"
        if prof and lid:
            out.append(LibraryKey(profile=prof, library_id=lid))

    # Dedup + stable order
    seen = set()
    uniq: List[LibraryKey] = []
    for k in sorted(out, key=lambda x: (x.profile, x.library_id)):
        if k.key in seen:
            continue
        seen.add(k.key)
        uniq.append(k)
    return uniq


def libraries_manifest_path(data_dir: str = "data") -> Path:
    return Path(data_dir) / "libraries.json"


def read_libraries_manifest(data_dir: str = "data") -> Dict[str, Dict]:
    """
    可选：保存 display_name/created_at 等 UI 友好信息。
    缺失时返回 {}，系统仍可仅靠目录扫描工作。
    """
    p = libraries_manifest_path(data_dir)
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}
    return {}


def write_libraries_manifest(data: Dict[str, Dict], data_dir: str = "data") -> None:
    p = libraries_manifest_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

