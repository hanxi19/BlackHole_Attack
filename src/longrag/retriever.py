"""
LongRAG retriever: coarse search over long units + optional cross-encoder rerank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import faiss
import numpy as np

from process_data.data_manager import DataManager
from .units import LongUnitIndex, build_long_units
from .rerank import CrossEncoderReranker


@dataclass
class LongRAGSearchResult:
    scores: np.ndarray       # (n_queries, k)
    indices: np.ndarray      # (n_queries, k) passage indices in corpus
    unit_hits: np.ndarray    # (n_queries, unit_k) first-stage unit indices


class LongRAGRetriever:
    """Two-stage LongRAG retrieval on a DataManager corpus (clean or poisoned)."""

    def __init__(
        self,
        dm: DataManager,
        *,
        max_chars: int = 16_000,
        group_by_title: bool = True,
        index_type: Literal["FlatIP", "IVF", "HNSW"] = "FlatIP",
        reranker: Optional[CrossEncoderReranker] = None,
        **index_kwargs,
    ):
        if dm.corpus_vecs is None or dm.corpus_texts is None:
            raise RuntimeError("DataManager must have corpus loaded")
        if dm.query_vecs is None or dm.query_texts is None:
            raise RuntimeError("DataManager must have queries loaded")

        self.dm = dm
        self.max_chars = max_chars
        self.group_by_title = group_by_title
        self.index_type = index_type
        self.index_kwargs = index_kwargs
        self.reranker = reranker

        self.units: Optional[LongUnitIndex] = None
        self._unit_index: Optional[faiss.Index] = None

    def build_units(self) -> LongUnitIndex:
        """Group corpus into long units and build the unit-level FAISS index."""
        self.units = build_long_units(
            self.dm.corpus_texts,
            self.dm.corpus_vecs,
            max_chars=self.max_chars,
            group_by_title=self.group_by_title,
        )
        self._build_unit_index()
        n_passages = len(self.dm.corpus_texts)
        n_units = len(self.units.unit_ids)
        ratio = n_passages / max(n_units, 1)
        print(f"  LongRAG units: {n_units} (from {n_passages} passages, {ratio:.1f}x compression)")
        return self.units

    def _build_unit_index(self) -> None:
        if self.units is None:
            raise RuntimeError("Call build_units() first")

        vecs = self.units.unit_vecs.copy()
        faiss.normalize_L2(vecs)
        dim = vecs.shape[1]
        n = vecs.shape[0]

        if self.index_type == "FlatIP":
            index = faiss.IndexFlatIP(dim)
        elif self.index_type == "IVF":
            nlist = self.index_kwargs.get("nlist")
            if nlist is None:
                nlist = min(4096, max(64, int(4 * np.sqrt(n))))
            nprobe = self.index_kwargs.get("nprobe", min(128, max(16, nlist // 4)))
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            index.train(vecs)
            faiss.extract_index_ivf(index).nprobe = nprobe
        elif self.index_type == "HNSW":
            m = self.index_kwargs.get("hnsw_M", 64)
            index = faiss.IndexHNSWFlat(dim, m)
            index.hnsw.efSearch = self.index_kwargs.get("ef_search", 256)
        else:
            raise ValueError(f"Unsupported unit index type: {self.index_type}")

        index.add(vecs)
        self._unit_index = index

    def _search_units(
        self,
        query_vecs: np.ndarray,
        unit_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._unit_index is None or self.units is None:
            raise RuntimeError("Unit index not built. Call build_units() first.")

        q = query_vecs.copy().astype(np.float32)
        faiss.normalize_L2(q)
        scores, unit_ids = self._unit_index.search(q, unit_k)
        return scores, unit_ids

    def _expand_candidates(self, unit_ids_row: np.ndarray) -> list[int]:
        """Union of passage indices from retrieved long units."""
        assert self.units is not None
        seen: set[int] = set()
        ordered: list[int] = []
        for uid in unit_ids_row:
            if uid < 0:
                continue
            for pi in self.units.unit_passage_indices[int(uid)]:
                if pi not in seen:
                    seen.add(pi)
                    ordered.append(pi)
        return ordered

    def _vector_rank(
        self,
        query_vec: np.ndarray,
        candidate_indices: list[int],
        top_k: int,
    ) -> tuple[list[int], list[float]]:
        if not candidate_indices:
            return [], []
        q = query_vec.astype(np.float32)
        q /= max(np.linalg.norm(q), 1e-9)
        vecs = self.dm.corpus_vecs[candidate_indices]
        vecs = vecs.copy()
        faiss.normalize_L2(vecs)
        sims = vecs @ q
        order = np.argsort(-sims)[:top_k]
        idxs = [candidate_indices[i] for i in order]
        scores = [float(sims[i]) for i in order]
        return idxs, scores

    def search(
        self,
        *,
        k: int = 10,
        unit_k: int = 20,
        use_rerank: bool = False,
        query_vecs: Optional[np.ndarray] = None,
        query_texts: Optional[Sequence[str]] = None,
        sample: Optional[int] = None,
    ) -> LongRAGSearchResult:
        """Retrieve top-k passages via long-unit search, optionally with rerank."""
        if self.units is None:
            self.build_units()

        qvecs = query_vecs if query_vecs is not None else self.dm.query_vecs
        qtexts_df = self.dm.query_texts
        if qtexts_df is None:
            raise RuntimeError("Query texts required for rerank")

        if sample is not None and sample < qvecs.shape[0]:
            rng = np.random.default_rng(42)
            sel = rng.choice(qvecs.shape[0], size=sample, replace=False)
            qvecs = qvecs[sel]
            qtexts = qtexts_df.iloc[sel]["text"].astype(str).tolist()
        else:
            qtexts = qtexts_df["text"].astype(str).tolist()

        unit_scores, unit_hits = self._search_units(qvecs, unit_k)
        n_queries = qvecs.shape[0]
        out_scores = np.zeros((n_queries, k), dtype=np.float32)
        out_indices = np.full((n_queries, k), -1, dtype=np.int64)

        corpus_texts = self.dm.corpus_texts

        for qi in range(n_queries):
            candidates = self._expand_candidates(unit_hits[qi])
            if not candidates:
                continue

            if use_rerank:
                if self.reranker is None:
                    self.reranker = CrossEncoderReranker()
                query = qtexts[qi]
                cand_pairs = [
                    (
                        pi,
                        str(corpus_texts.iloc[pi].get("text", "") or "").strip()
                        or str(corpus_texts.iloc[pi]["_id"]),
                    )
                    for pi in candidates
                ]
                ranked = self.reranker.rerank(query, cand_pairs, k)
                for j, (pi, sc) in enumerate(ranked):
                    out_indices[qi, j] = pi
                    out_scores[qi, j] = sc
            else:
                idxs, scores = self._vector_rank(qvecs[qi], candidates, k)
                for j, (pi, sc) in enumerate(zip(idxs, scores)):
                    out_indices[qi, j] = pi
                    out_scores[qi, j] = sc

        return LongRAGSearchResult(
            scores=out_scores,
            indices=out_indices,
            unit_hits=unit_hits,
        )
