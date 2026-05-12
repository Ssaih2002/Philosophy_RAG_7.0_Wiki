import chromadb
from pathlib import Path
import shutil
from tqdm import tqdm

from .library_manager import chroma_collection_for, chroma_dir_for, normalize_library_id


class VectorStore:
    def __init__(self, profile: str = "quality", library_id: str = "default"):
        self.profile = profile
        self.library_id = normalize_library_id(library_id)
        self.db_path = chroma_dir_for(self.profile, self.library_id)
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection_name = chroma_collection_for(self.profile, self.library_id)
        try:
            # Normal path: open the profile-specific collection.
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            # Generic compatibility:
            # Some users rename/copy a Chroma persistence directory without renaming the
            # internal collection name. Then `get_or_create_collection(target)` creates an
            # empty collection even though another non-empty one exists in the same DB dir.
            # If the target collection is empty, prefer the most-populated existing collection.
            def _safe_count(col) -> int:
                try:
                    return int(col.count())
                except Exception:
                    return -1

            try:
                tgt_n = _safe_count(self.collection)
            except Exception:
                tgt_n = -1

            if tgt_n == 0:
                try:
                    cols = self.client.list_collections() or []
                except Exception:
                    cols = []

                best = None
                best_n = 0
                for c in cols:
                    n = _safe_count(c)
                    if n > best_n:
                        best = c
                        best_n = n

                if best is not None and best_n > 0:
                    self.collection = best
                    try:
                        self.collection_name = getattr(best, "name", self.collection_name)
                    except Exception:
                        pass

            # Compatibility: some users build a SEP vector DB by ingesting under "quality"
            # and then moving the persistence directory into chroma_db_sep/.
            # In that case the only populated collection is often "philosophy_quality".
            # Avoid silently querying an empty freshly-created "philosophy_sep".
            if self.profile == "sep":
                def _try_get(name: str):
                    try:
                        return self.client.get_collection(name)
                    except Exception:
                        return None

                # Strategy (path-based, not "suffix-based"):
                # - If philosophy_sep exists and has data, use it.
                # - Else if philosophy_quality exists and has data, use it.
                # - Else keep the default get_or_create_collection result.
                # Backward compatibility:
                # - Some users may have only philosophy_sep / philosophy_quality inside the sep DB dir.
                sep_col = _try_get("philosophy_sep")
                qual_col = _try_get("philosophy_quality")
                sep_n = _safe_count(sep_col) if sep_col else -1
                qual_n = _safe_count(qual_col) if qual_col else -1

                if sep_col and sep_n > 0:
                    self.collection = sep_col
                    self.collection_name = "philosophy_sep"
                elif qual_col and qual_n > 0:
                    self.collection = qual_col
                    self.collection_name = "philosophy_quality"
        except Exception as e:
            # 常见原因：用户手动删了部分 Chroma 文件（如 segment/bin），导致 sqlite 元数据与磁盘不一致。
            # 此时会报 NotFoundError: Collection [uuid] does not exist.
            msg = str(e)
            if "Collection" in msg and "does not exist" in msg:
                # 尝试清空并重建当前 profile 的 collection（保守做法：仅重建 collection，不动目录）
                try:
                    self.client.delete_collection(self.collection_name)
                except Exception:
                    pass
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            else:
                raise

    def reset_collection(self):
        """
        彻底重建当前 profile 的 Chroma 存储。

        仅 delete_collection 在某些“磁盘段文件已丢失/被删除”的情况下不足以恢复，
        会导致 query 时出现 NotFoundError: Collection [uuid] does not exist。
        因此这里直接清空整个持久化目录并重新创建 collection。
        """
        try:
            # 关闭/重建 client：不同版本 Chroma 对资源释放行为不同，这里直接重建最稳。
            self.client = None  # type: ignore[assignment]
        except Exception:
            pass
        try:
            shutil.rmtree(self.db_path, ignore_errors=True)
        except Exception:
            # 忽略：若文件被占用，后续会在 query/ingest 时抛更明确错误
            pass
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks, embeddings, batch_size=5000, show_progress=True, id_offset: int = 0):
        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch")

        n = len(chunks)
        if n == 0:
            return

        batch_iter = range(0, n, batch_size)
        if show_progress:
            batch_iter = tqdm(
                batch_iter,
                total=(n + batch_size - 1) // batch_size,
                desc="写入 Chroma",
                unit="batch",
            )

        for start in batch_iter:
            end = min(start + batch_size, n)
            batch_chunks = chunks[start:end]
            batch_embeddings = embeddings[start:end]

            ids = [f"chunk_{id_offset + start + i}" for i in range(len(batch_chunks))]
            docs = [c["text"] for c in batch_chunks]
            metas = [
                {
                    "page": c["page"],
                    "source": c["source"],
                    "chunk_id": f"chunk_{id_offset + start + i}",
                }
                for i, c in enumerate(batch_chunks)
            ]

            self.collection.add(
                ids=ids,
                documents=docs,
                embeddings=batch_embeddings,
                metadatas=metas,
            )

    @staticmethod
    def _source_matches_filters(meta: dict, vals: list) -> bool:
        """metadata.source 子串匹配；vals 非空时至少命中其一即保留（与 README 一致）。"""
        src = str((meta or {}).get("source") or "")
        return any(v in src for v in vals)

    def search(self, embedding, k, source_filters=None):
        vals: list = []
        if source_filters:
            vals = [s.strip() for s in source_filters if s and str(s).strip()]

        # 无文献限定时直接向量检索
        if not vals:
            try:
                return self.collection.query(
                    query_embeddings=[embedding],
                    n_results=k,
                )
            except Exception as e:
                msg = str(e)
                if "Collection" in msg and "does not exist" in msg:
                    raise RuntimeError(
                        "Chroma 向量库已损坏或磁盘文件不一致（collection uuid 不存在）。"
                        "请先运行一次 Ingest 以重建向量库；必要时删除对应目录 data/chroma_db_<profile>/ 后再 Ingest。"
                    ) from e
                raise

        # 有文献限定：不在 Chroma where 里用 $contains/$or（版本差异大，且 $in 回退无法匹配子串）。
        # 先多取候选再在 Python 里按 source 子串过滤，保证多关键词并集都生效。
        n_fetch = min(2000, max(int(k) * 30, 200 * len(vals), int(k) + 20))
        try:
            raw = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_fetch,
            )
        except Exception as e:
            msg = str(e)
            if "Collection" in msg and "does not exist" in msg:
                raise RuntimeError(
                    "Chroma 向量库已损坏或磁盘文件不一致（collection uuid 不存在）。"
                    "请先运行一次 Ingest 以重建向量库；必要时删除对应目录 data/chroma_db_<profile>[__<library_id>]/ 后再 Ingest。"
                ) from e
            raise

        docs0 = raw["documents"][0]
        metas0 = raw["metadatas"][0]
        ids0 = raw.get("ids", [[]])[0]
        dists0 = raw.get("distances")
        dist_list = dists0[0] if dists0 else None

        out_docs: list = []
        out_metas: list = []
        out_ids: list = []
        out_dists: list = []
        for i, m in enumerate(metas0):
            if self._source_matches_filters(m, vals):
                out_docs.append(docs0[i])
                out_metas.append(m)
                if i < len(ids0):
                    out_ids.append(ids0[i])
                if dist_list is not None:
                    out_dists.append(dist_list[i])
                if len(out_docs) >= int(k):
                    break

        # 若仍不足，再放大拉取一次（只扫新增 tail，避免与首轮候选重复）
        if len(out_docs) < int(k) and n_fetch < 2000:
            n2 = min(2000, n_fetch * 2)
            try:
                raw2 = self.collection.query(
                    query_embeddings=[embedding],
                    n_results=n2,
                )
            except Exception:
                raw2 = None
            if raw2 is not None:
                seen_cid = {m.get("chunk_id") for m in out_metas if m}
                docs1 = raw2["documents"][0]
                metas1 = raw2["metadatas"][0]
                ids1 = raw2.get("ids", [[]])[0]
                d1 = raw2.get("distances")
                dl1 = d1[0] if d1 else None
                for i in range(n_fetch, len(metas1)):
                    m = metas1[i]
                    cid = m.get("chunk_id") if m else None
                    if cid and cid in seen_cid:
                        continue
                    if not self._source_matches_filters(m, vals):
                        continue
                    out_docs.append(docs1[i])
                    out_metas.append(m)
                    if i < len(ids1):
                        out_ids.append(ids1[i])
                    if dl1 is not None:
                        out_dists.append(dl1[i])
                    if cid:
                        seen_cid.add(cid)
                    if len(out_docs) >= int(k):
                        break

        result: dict = {
            "documents": [out_docs],
            "metadatas": [out_metas],
        }
        if out_ids:
            result["ids"] = [out_ids]
        if out_dists:
            result["distances"] = [out_dists]
        return result
