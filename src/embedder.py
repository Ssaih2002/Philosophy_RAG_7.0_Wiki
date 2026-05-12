from typing import Any, Dict, List, Optional, Sequence, Union
import threading

from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL

_MODEL_CACHE: Dict[str, SentenceTransformer] = {}
_CACHE_LOCK = threading.Lock()

class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        # Cache models by name to avoid repeated heavy loads (e.g. when SEP reference
        # uses the same embedding model as the main profile).
        with _CACHE_LOCK:
            m = _MODEL_CACHE.get(self.model_name)
            if m is None:
                print("Loading embedding model...")
                m = SentenceTransformer(self.model_name)
                _MODEL_CACHE[self.model_name] = m
        self.model = m

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        show_progress_bar: Optional[bool] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Encode texts into embeddings.

        Notes:
        - Default SentenceTransformer behavior may use CUDA if available.
        - In real-world desktop usage, CUDA OOM can happen (shared GPU memory, other apps, etc).
          We catch OOM and transparently fall back to CPU for robustness.
        """
        if isinstance(texts, str):
            texts = [texts]
        else:
            texts = list(texts)
        if show_progress_bar is None:
            show_progress_bar = len(texts) > 1
        try:
            emb = self.model.encode(
                texts,
                show_progress_bar=show_progress_bar,
                **kwargs,
            )
            return emb.tolist()
        except Exception as e:
            msg = str(e).lower()
            is_oom = ("out of memory" in msg) or ("cuda" in msg and "memory" in msg)
            if not is_oom:
                raise
            # Robust fallback: retry on CPU with a small batch size.
            try:
                self.model.to("cpu")
            except Exception:
                pass
            kwargs2 = dict(kwargs)
            kwargs2.setdefault("device", "cpu")
            kwargs2.setdefault("batch_size", 8)
            emb2 = self.model.encode(
                texts,
                show_progress_bar=show_progress_bar,
                **kwargs2,
            )
            return emb2.tolist()
