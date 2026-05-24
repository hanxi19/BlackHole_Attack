"""
Cross-encoder reranking for LongRAG second stage.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class CrossEncoderReranker:
    """Score (query, passage) pairs with a HuggingFace cross-encoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._device = device

    def _ensure_model(self):
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.model_name, device=self._device)

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """Return relevance scores for (query, passage) pairs."""
        if not pairs:
            return np.array([], dtype=np.float32)
        self._ensure_model()
        scores = self._model.predict(
            list(pairs),
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return np.asarray(scores, dtype=np.float32)

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[int, str]],
        top_k: int,
    ) -> list[tuple[int, float]]:
        """Rerank candidate passages; return top_k as (passage_idx, score)."""
        if not candidates:
            return []
        pairs = [(query, text) for _, text in candidates]
        scores = self.score_pairs(pairs)
        ranked = sorted(
            zip((idx for idx, _ in candidates), scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]
