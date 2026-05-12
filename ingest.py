from src.config import CURRENT_PROFILE, PROFILE_SETTINGS
from src.ingest_pipeline import run_ingest_pipeline


def main():
    profile = CURRENT_PROFILE
    params = PROFILE_SETTINGS[profile]
    print(f"Using retrieval profile: {profile}")
    library_id = "default"
    print(f"Using library_id: {library_id}")

    def emit(ev):
        if ev["type"] == "progress":
            print(f"  [{ev['percent']:5.1f}%] {ev.get('message', '')}")
        elif ev["type"] == "done":
            print(
                f"Ingestion complete. total_pages={ev['total_pages']} total_chunks={ev['total_chunks']}"
            )

    run_ingest_pipeline(
        profile,
        params["EMBEDDING_MODEL"],
        library_id=library_id,
        data_dir="data",
        emit=emit,
        reload_sparse_cb=None,
    )


if __name__ == "__main__":
    main()
