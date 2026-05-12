import argparse

from src.ingest_pipeline import run_ingest_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="为单文献/小批文献创建临时 profile 索引，便于后续合并到主库。"
    )
    parser.add_argument(
        "--profile",
        default="tmp",
        help="临时索引 profile 名称（默认: tmp）",
    )
    parser.add_argument(
        "--data-dir",
        default="data_single",
        help="单文献数据目录（默认: data_single）",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-m3",
        help="embedding 模型（默认: BAAI/bge-m3）",
    )
    args = parser.parse_args()

    print("[ingest_single_tmp] start")
    print(f"[ingest_single_tmp] profile = {args.profile}")
    print(f"[ingest_single_tmp] data_dir = {args.data_dir}")
    print(f"[ingest_single_tmp] embedding_model = {args.embedding_model}")

    result = run_ingest_pipeline(
        profile=args.profile,
        embedding_model=args.embedding_model,
        data_dir=args.data_dir,
    )
    print(
        f"[ingest_single_tmp] done: total_pages={result['total_pages']}, "
        f"total_chunks={result['total_chunks']}"
    )


if __name__ == "__main__":
    main()

