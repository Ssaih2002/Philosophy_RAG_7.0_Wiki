import os
import sys
from pathlib import Path


def main() -> int:
    """
    Prefetch embedding + reranker models into the local Hugging Face cache.

    Usage (Windows, in project root):
      .venv\\Scripts\\python.exe tools\\prefetch_models.py

    Optional:
      - set HF_HOME to control cache directory (start_app.bat already sets it to data/hf_cache)
      - set HF_ENDPOINT to a mirror if you are behind a slow connection
    """
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(f"[prefetch] huggingface_hub not available: {e}")
        print("[prefetch] Please ensure transformers is installed (it usually brings huggingface_hub).")
        return 1

    # Make sure we can import `src` even when invoked from other directories.
    # Project root is one level above /tools.
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from src import config
    except Exception as e:
        print(f"[prefetch] cannot import src.config: {e}")
        print(f"[prefetch] tip: run from project root, e.g. {root}")
        return 1

    models = [
        getattr(config, "EMBEDDING_MODEL", ""),
        getattr(config, "RERANKER_MODEL", ""),
    ]
    models = [m for m in models if m and str(m).strip()]
    if not models:
        print("[prefetch] No models found in config.")
        return 0

    cache_dir = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or ""
    print(f"[prefetch] cache_dir={cache_dir or '(default)'}")

    for m in models:
        print(f"[prefetch] downloading: {m}")
        snapshot_download(
            repo_id=m,
            cache_dir=cache_dir or None,
            local_files_only=False,
            resume_download=True,
        )

    print("[prefetch] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

