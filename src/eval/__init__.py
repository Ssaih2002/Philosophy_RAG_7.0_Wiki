"""Batch evaluation helpers (ablation, metrics, CLI)."""

from .metrics import hit_at_k, mean_reciprocal_rank, ndcg_at_k

__all__ = ["hit_at_k", "mean_reciprocal_rank", "ndcg_at_k"]
