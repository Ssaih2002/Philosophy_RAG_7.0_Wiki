from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sqlite3

from .library_manager import normalize_library_id, sparse_db_for


@dataclass(frozen=True)
class LanguageProfile:
    """
    Minimal language profile for a library.

    We intentionally avoid heavy deps (fastText/langid) to keep installation simple.
    This heuristic is designed for the project's main use case:
    - distinguish ZH vs Latin-script, and within Latin-script, roughly DE vs EN.
    """

    profile: str
    library_id: str
    dist: Dict[str, float]  # e.g. {"de":0.78,"en":0.18,"zh":0.04}
    sampled_chunks: int
    updated_at_ts: float


def _meta_dir(data_dir: str = "data") -> Path:
    p = Path(data_dir) / "library_meta"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _meta_path(profile: str, library_id: str, *, data_dir: str = "data") -> Path:
    lid = normalize_library_id(library_id)
    key = f"{profile}__{lid}"
    return _meta_dir(data_dir) / f"{key}.lang.json"


def load_cached_language_profile(
    profile: str,
    library_id: str,
    *,
    data_dir: str = "data",
) -> Optional[LanguageProfile]:
    p = _meta_path(profile, library_id, data_dir=data_dir)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return None
        dist = obj.get("dist") or {}
        if not isinstance(dist, dict):
            return None
        sampled = int(obj.get("sampled_chunks") or 0)
        ts = float(obj.get("updated_at_ts") or 0.0)
        lid = normalize_library_id(library_id)
        # normalize to floats
        out: Dict[str, float] = {}
        for k, v in dist.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        if not out:
            return None
        return LanguageProfile(
            profile=str(profile),
            library_id=lid,
            dist=out,
            sampled_chunks=sampled,
            updated_at_ts=ts,
        )
    except Exception:
        return None


def save_cached_language_profile(lp: LanguageProfile, *, data_dir: str = "data") -> None:
    p = _meta_path(lp.profile, lp.library_id, data_dir=data_dir)
    payload = {
        "profile": lp.profile,
        "library_id": lp.library_id,
        "dist": lp.dist,
        "sampled_chunks": int(lp.sampled_chunks),
        "updated_at_ts": float(lp.updated_at_ts),
        "updated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(lp.updated_at_ts)),
        "method": "heuristic_v1",
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


_RE_ZH = re.compile(r"[\u4e00-\u9fff]")


def _is_zh(text: str) -> bool:
    return bool(_RE_ZH.search(text or ""))


def _score_latin_de_en(text: str) -> Tuple[float, float]:
    """
    Return (de_score, en_score) for latin-script text.
    """
    t = (text or "").lower()
    if not t:
        return (0.0, 0.0)
    # strong DE marks
    de = 0.0
    if any(x in t for x in (" ä", "ö", "ü", "ß", "ä", "ö", "ü")):
        de += 2.2
    # stopwords (very lightweight)
    de_sw = (" der ", " die ", " das ", " und ", " nicht ", " ist ", " eine ", " ein ", " im ", " zu ", " von ")
    en_sw = (" the ", " and ", " of ", " to ", " is ", " are ", " in ", " for ", " that ", " with ", " as ")
    for w in de_sw:
        if w in f" {t} ":
            de += 0.35
    en = 0.0
    for w in en_sw:
        if w in f" {t} ":
            en += 0.35
    return (de, en)


def _iter_sparse_sample_texts(
    *,
    profile: str,
    library_id: str,
    data_dir: str,
    sample_limit: int,
) -> List[str]:
    db_path = sparse_db_for(profile, library_id)
    p = Path(db_path)
    if not p.exists():
        return []
    con = sqlite3.connect(str(p))
    try:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        # random-ish sample without ORDER BY random() (slow on huge tables):
        # take every Nth row using rowid modulo.
        # If the table is small, this yields most rows.
        want = max(20, int(sample_limit))
        stride = 17
        rows = cur.execute(
            "SELECT text FROM chunks WHERE (abs(rowid) % ?) = 0 LIMIT ?",
            (stride, want),
        ).fetchall()
        out: List[str] = []
        for r in rows:
            txt = str(r["text"] or "")
            if txt:
                out.append(txt[:1200])
        return out
    except Exception:
        return []
    finally:
        con.close()


def compute_and_cache_language_profile(
    profile: str,
    library_id: str,
    *,
    data_dir: str = "data",
    sample_limit: int = 420,
    force_recompute: bool = False,
) -> Optional[LanguageProfile]:
    lid = normalize_library_id(library_id)
    if not force_recompute:
        cached = load_cached_language_profile(profile, lid, data_dir=data_dir)
        if cached:
            return cached

    texts = _iter_sparse_sample_texts(
        profile=profile, library_id=lid, data_dir=data_dir, sample_limit=sample_limit
    )
    if not texts:
        return None

    zh = 0
    de_score = 0.0
    en_score = 0.0
    n = 0
    for t in texts:
        if not t:
            continue
        n += 1
        if _is_zh(t):
            zh += 1
            continue
        d, e = _score_latin_de_en(t)
        de_score += d
        en_score += e

    if n <= 0:
        return None

    dist: Dict[str, float] = {}
    dist["zh"] = float(zh) / float(n)
    latin_n = max(1, n - zh)
    # Normalize DE/EN scores within latin samples
    if de_score <= 0.0 and en_score <= 0.0:
        # unknown latin: split evenly
        dist["de"] = 0.5 * (1.0 - dist["zh"])
        dist["en"] = 0.5 * (1.0 - dist["zh"])
    else:
        tot = max(1e-9, de_score + en_score)
        dist["de"] = (1.0 - dist["zh"]) * (de_score / tot)
        dist["en"] = (1.0 - dist["zh"]) * (en_score / tot)

    now = time.time()
    lp = LanguageProfile(
        profile=str(profile),
        library_id=lid,
        dist=dist,
        sampled_chunks=int(n),
        updated_at_ts=float(now),
    )
    save_cached_language_profile(lp, data_dir=data_dir)
    return lp


def top_languages(dist: Dict[str, float], *, min_share: float = 0.2, max_langs: int = 3) -> List[str]:
    items = [(k, float(v)) for k, v in (dist or {}).items()]
    items.sort(key=lambda x: x[1], reverse=True)
    out: List[str] = []
    for k, v in items:
        if v >= float(min_share):
            out.append(k)
        if len(out) >= int(max_langs):
            break
    return out

