"""One-off: print Chroma sqlite collections + embedding counts.

用途：排查 `data/chroma_db_<profile>/` 是否存在、集合名是否正确、是否真的写入了向量。
注意：这不是“合并”脚本；合并请用 `merge_profile.py`。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ["chroma_db_quality", "chroma_db_fast", "chroma_db_sep"]:
        p = root / "data" / name / "chroma.sqlite3"
        print("===", p, "exists=", p.exists(), "===")
        if not p.exists():
            continue
        con = sqlite3.connect(str(p))
        cur = con.cursor()
        tables = [
            r[0]
            for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        print("tables:", tables)
        if "collections" in tables:
            rows = list(cur.execute("SELECT id, name, dimension FROM collections"))
            print("collections rows:", rows)
        if "embeddings" in tables:
            n = cur.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            print("embeddings count:", n)
        con.close()
        print()


if __name__ == "__main__":
    main()

