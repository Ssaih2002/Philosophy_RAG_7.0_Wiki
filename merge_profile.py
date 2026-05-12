import sys
import sqlite3
from typing import List

from tqdm import tqdm

from src.vector_store import VectorStore
from src.sparse_retriever import SparseRetriever


def merge_profiles(source_profile: str, target_profile: str) -> None:
    """
    将「source_profile」对应的 Chroma 向量库中的所有文献片段，
    追加合并到「target_profile」的向量库 + 稀疏 FTS5 索引中。

    设计要点：
    - 只追加，不会删除 target 中已有内容；
    - 为避免 chunk_id 冲突，新片段使用前缀型 ID：`{source_profile}_extra_{i}`；
    - 稠密向量与稀疏索引共享同一组新 chunk_id，保持 RAGEngine 内部一致。
    """
    if source_profile == target_profile:
        raise ValueError("source_profile 与 target_profile 不能相同。")

    print(f"[merge_profile] source_profile = {source_profile}")
    print(f"[merge_profile] target_profile = {target_profile}")

    # 1. 打开源/目标向量库（Chroma）
    src_vs = VectorStore(source_profile)
    tgt_vs = VectorStore(target_profile)

    print("[merge_profile] 读取源 Chroma 集合…")
    # 注意：当前 chromadb 版本不接受 where={}；不传 where 即全量读取。
    src = src_vs.collection.get(
        include=["documents", "embeddings", "metadatas"],  # 取出文本 + 向量 + 元数据
    )
    docs: List[str] = src.get("documents") or []
    metas = src.get("metadatas") or []
    embs = src.get("embeddings")
    # chromadb 可能返回 numpy 数组，这里统一转为普通列表以便后续处理
    if embs is None:
        embs_list = []
    else:
        try:
            # numpy.ndarray or similar
            embs_list = list(embs)
        except TypeError:
            embs_list = embs  # 已经是列表

    n = len(docs)
    if n == 0:
        print("[merge_profile] 源集合为空，无需合并。")
        return

    if not (len(embs_list) == n and len(metas) == n):
        raise RuntimeError(
            f"源集合数据长度不一致：docs={n}, embeddings={len(embs_list)}, metas={len(metas)}"
        )

    print(f"[merge_profile] 源 profile 共 {n} 个片段，将追加到 {target_profile}。")

    # 2. 确保目标稀疏索引存在
    sparse = SparseRetriever(target_profile)  # 构造函数会确保 schema 就绪
    conn = sqlite3.connect(sparse.path)
    conn.row_factory = sqlite3.Row

    # 3. 逐条追加：Chroma + SQLite FTS5
    pbar = tqdm(range(n), desc="合并片段", unit="chunk")
    for i in pbar:
        doc = docs[i] or ""
        meta = metas[i] or {}

        page = meta.get("page", "")
        source = meta.get("source", "Unknown")

        new_id = f"{source_profile}_extra_{i}"

        # 3.1 追加到目标 Chroma
        tgt_vs.collection.add(
            ids=[new_id],
            documents=[doc],
            embeddings=[embs[i]],
            metadatas=[
                {
                    "page": page,
                    "source": source,
                    "chunk_id": new_id,
                }
            ],
        )

        # 3.2 追加到目标 SQLite FTS5（chunks + chunks_fts）
        conn.execute(
            "INSERT OR REPLACE INTO chunks(chunk_id, text, page, source) VALUES (?, ?, ?, ?)",
            (new_id, doc, str(page), source or "Unknown"),
        )
        conn.execute(
            "INSERT INTO chunks_fts(text, chunk_id) VALUES (?, ?)",
            (doc, new_id),
        )

    conn.commit()
    conn.close()

    print(
        f"[merge_profile] 合并完成：从 {source_profile} 追加 {n} 个片段到 {target_profile}。"
    )


def main():
    if len(sys.argv) != 3:
        print(
            "用法：\n"
            "  python merge_profile.py <source_profile> <target_profile>\n\n"
            "示例：\n"
            "  # 将单独 ingest 的 tmp profile 合并进 quality 主库\n"
            "  python merge_profile.py tmp quality\n"
        )
        sys.exit(1)

    source_profile = sys.argv[1].strip()
    target_profile = sys.argv[2].strip()
    merge_profiles(source_profile, target_profile)


if __name__ == "__main__":
    main()

