"""
Rename a Chroma collection inside a persistence directory WITHOUT re-ingesting.

Why this exists
---------------
Users sometimes copy/rename a Chroma persistence directory (e.g. into
`data/chroma_db_quality__MEGA2/`) but the internal collection name stays
`philosophy_quality`. The app expects `philosophy_quality__MEGA2`, so it ends up
creating an empty target collection and querying that.

This tool:
- deletes an empty target collection (if it already exists), then
- renames the source collection to the target name by updating `chroma.sqlite3`.

It does NOT touch embeddings/segments; it only updates the collection name.

Usage (PowerShell)
-----------------
python tools/rename_chroma_collection.py --path data/chroma_db_quality__MEGA2 --from philosophy_quality --to philosophy_quality__MEGA2
python tools/rename_chroma_collection.py --path data/chroma_db_quality__5758SECOND --from philosophy_quality --to philosophy_quality__5758SECOND
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import chromadb


def _count(client: chromadb.PersistentClient, name: str) -> int | None:
    try:
        col = client.get_collection(name)
    except Exception:
        return None
    try:
        return int(col.count())
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Chroma persistence dir (contains chroma.sqlite3)")
    ap.add_argument("--from", dest="src_name", required=True, help="source collection name")
    ap.add_argument("--to", dest="dst_name", required=True, help="target collection name")
    args = ap.parse_args()

    db_dir = Path(args.path)
    db_path = db_dir / "chroma.sqlite3"
    if not db_path.exists():
        raise SystemExit(f"chroma.sqlite3 not found: {db_path}")

    client = chromadb.PersistentClient(path=str(db_dir))

    src_n = _count(client, args.src_name)
    dst_n = _count(client, args.dst_name)

    print(f"[rename] path={db_dir}")
    print(f"[rename] from={args.src_name} count={src_n}")
    print(f"[rename] to  ={args.dst_name} count={dst_n}")

    if src_n is None:
        raise SystemExit(f"source collection not found: {args.src_name}")
    if src_n <= 0:
        raise SystemExit("source collection is empty; refusing to rename")

    # If target exists and is empty, delete it first so we can rename.
    if dst_n is not None:
        if dst_n > 0:
            raise SystemExit("target collection is non-empty; refusing to overwrite")
        print("[rename] deleting empty target collection first…")
        client.delete_collection(args.dst_name)

    # Rename in sqlite.
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute("UPDATE collections SET name = ? WHERE name = ?", (args.dst_name, args.src_name))
        if cur.rowcount != 1:
            raise SystemExit(f"unexpected UPDATE rowcount={cur.rowcount} (expected 1)")
        con.commit()
    finally:
        con.close()

    # Verify.
    client2 = chromadb.PersistentClient(path=str(db_dir))
    new_n = _count(client2, args.dst_name)
    old_n = _count(client2, args.src_name)
    print(f"[rename] verify to   count={new_n}")
    print(f"[rename] verify from count={old_n}")
    if new_n is None or new_n <= 0:
        raise SystemExit("rename verification failed: target missing/empty")
    if old_n is not None:
        raise SystemExit("rename verification failed: source still exists")

    print("[rename] OK")


if __name__ == "__main__":
    main()

