from typing import Dict, List, Optional

import os

from sentence_transformers import CrossEncoder

from .config import RERANKER_MODEL, RERANKER_DEVICE


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self._model: Optional[CrossEncoder] = None

    def _pick_device(self) -> str:
        # Env override wins (useful for debugging / forcing CPU)
        env = (os.getenv("RERANKER_DEVICE") or "").strip().lower()
        want = (env or str(RERANKER_DEVICE or "auto")).strip().lower()
        if want not in ("auto", "cuda", "cpu"):
            want = "auto"
        if want == "cpu":
            return "cpu"
        # Try CUDA when requested/auto
        try:
            import torch  # type: ignore

            if getattr(torch, "cuda", None) and torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            device = self._pick_device()
            print(f"Loading reranker model... ({self.model_name}, device={device})")
            # CrossEncoder will download to HF cache on first run (unless already cached).
            self._model = CrossEncoder(self.model_name, device=device)
        return self._model

    def rerank(self, question: str, docs: List[Dict], top_k: int) -> List[Dict]:
        if not docs:
            return docs
        model = self._get_model()
        pairs = [[question, d.get("text", "")] for d in docs]
        scores = model.predict(pairs)
        ranked = sorted(
            zip(docs, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        out = []
        for doc, score in ranked[:top_k]:
            item = dict(doc)
            item["rerank_score"] = float(score)
            out.append(item)
        return out
