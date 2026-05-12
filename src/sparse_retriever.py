import sqlite3
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .library_manager import sparse_db_for, normalize_library_id


def _db_path(profile: str, library_id: str) -> Path:
    path = Path(sparse_db_for(profile, library_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _escape_fts_term(s: str) -> str:
    return s.replace('"', '""').strip()


def _prefix_stems(term: str) -> List[str]:
    """
    Build robust prefix stems for morphological/compound variants.
    Example: Zusammenarbeit -> zusammenarbeit*, zusammen*
    """
    t = (term or "").strip().lower()
    if not t:
        return []
    # Keep letters/numbers, normalize separators.
    norm = re.sub(r"[\s\-_]+", " ", t)
    norm = re.sub(r"[^\w\s]", "", norm, flags=re.UNICODE).strip()
    if not norm:
        return []

    stems: List[str] = []
    # Full collapsed form supports common compound matches.
    collapsed = norm.replace(" ", "")
    if len(collapsed) >= 4:
        stems.append(collapsed)

    # Token-level prefixes support split compounds / line-break artifacts.
    for tok in norm.split():
        if len(tok) >= 4:
            stems.append(tok)

    # Heuristic for very long compounds: add first 8 chars.
    if len(collapsed) >= 10:
        stems.append(collapsed[:8])

    out: List[str] = []
    seen = set()
    for s in stems:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def build_sparse_query(terms: List[str], fallback_question: str) -> str:
    cleaned = [t.strip() for t in terms if t and t.strip()]
    if not cleaned:
        cleaned = [fallback_question.strip()]
    clauses: List[str] = []
    for t in cleaned:
        esc = _escape_fts_term(t)
        if esc:
            # keep exact phrase
            clauses.append(f'"{esc}"')
        # add prefix variants to improve German inflection/compound recall
        for stem in _prefix_stems(t):
            clauses.append(f"{stem}*")

    # Fallback safety
    if not clauses:
        fq = _escape_fts_term(fallback_question.strip())
        clauses = [f'"{fq}"'] if fq else []
    return " OR ".join(clauses)


class SparseRetriever:
    def __init__(self, profile: str = "quality", library_id: str = "default"):
        self.profile = profile
        self.library_id = normalize_library_id(library_id)
        self.path = str(_db_path(self.profile, self.library_id))
        self._ready = False
        self.reload()

    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    page TEXT,
                    source TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(
                    text,
                    chunk_id UNINDEXED,
                    tokenize='unicode61'
                )
                """
            )
            conn.commit()

    def rebuild(self, chunks: List[Dict[str, Any]], show_progress: bool = True) -> None:
        self._ensure_schema()
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM chunks_fts")
            rows = []
            fts_rows = []
            chunk_iter = enumerate(chunks)
            if show_progress and chunks:
                chunk_iter = enumerate(
                    tqdm(chunks, desc="FTS 索引", unit="chunk")
                )
            for i, c in chunk_iter:
                chunk_id = f"chunk_{i}"
                text = c.get("text", "")
                page = str(c.get("page", ""))
                source = c.get("source", "Unknown")
                rows.append((chunk_id, text, page, source))
                fts_rows.append((text, chunk_id))
            conn.executemany(
                "INSERT INTO chunks(chunk_id, text, page, source) VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.executemany(
                "INSERT INTO chunks_fts(text, chunk_id) VALUES (?, ?)",
                fts_rows,
            )
            conn.commit()
        self._ready = True

    def next_chunk_id_start(self) -> int:
        """下一个可用的 chunk 编号（与 chunk_0, chunk_1 … 约定一致）。"""
        self._ensure_schema()
        best = -1
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM chunks WHERE chunk_id GLOB 'chunk_*'"
            ).fetchall()
        for row in rows:
            cid = row["chunk_id"] if hasattr(row, "keys") else row[0]
            if not cid or not str(cid).startswith("chunk_"):
                continue
            suf = str(cid)[6:]
            try:
                best = max(best, int(suf))
            except ValueError:
                continue
        return best + 1

    def append_chunks(self, chunks: List[Dict[str, Any]], id_start: int, show_progress: bool = False) -> None:
        """在不清空表的前提下追加 chunks（与 VectorStore.add 的 id 规则一致）。"""
        if not chunks:
            return
        self._ensure_schema()
        rows = []
        fts_rows = []
        chunk_iter = enumerate(chunks)
        if show_progress and chunks:
            chunk_iter = enumerate(tqdm(chunks, desc="FTS 追加", unit="chunk"))
        for j, c in chunk_iter:
            i = id_start + j
            chunk_id = f"chunk_{i}"
            text = c.get("text", "")
            page = str(c.get("page", ""))
            source = c.get("source", "Unknown")
            rows.append((chunk_id, text, page, source))
            fts_rows.append((text, chunk_id))
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO chunks(chunk_id, text, page, source) VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.executemany(
                "INSERT INTO chunks_fts(text, chunk_id) VALUES (?, ?)",
                fts_rows,
            )
            conn.commit()
        self._ready = True

    def reload(self) -> None:
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(1) AS c FROM chunks").fetchone()
            self._ready = bool(row and row["c"] > 0)

    def is_ready(self) -> bool:
        return self._ready

    def search(
        self, query: str, k: Optional[int], source_filters: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        if not self._ready or not query.strip():
            return []
        sql = """
            SELECT c.chunk_id, c.text, c.page, c.source
            FROM chunks_fts f
            JOIN chunks c ON c.chunk_id = f.chunk_id
            WHERE chunks_fts MATCH ?
        """
        params: List[Any] = [query]
        if source_filters:
            vals = [s for s in source_filters if s and str(s).strip()]
            if vals:
                placeholders = ",".join("?" for _ in vals)
                sql += f" AND c.source IN ({placeholders})"
                params.extend(vals)
        sql += " ORDER BY bm25(chunks_fts)"
        if k is not None and int(k) > 0:
            sql += " LIMIT ?"
            params.append(int(k))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "page": r["page"],
                "source": r["source"] or "Unknown",
            }
            for r in rows
        ]

    def get_doc(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT chunk_id, text, page, source FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "chunk_id": row["chunk_id"],
            "text": row["text"],
            "page": row["page"],
            "source": row["source"] or "Unknown",
        }
