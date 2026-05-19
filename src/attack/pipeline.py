"""
Black-hole attack pipeline.

Orchestrates: preprocess → cluster → centroid perturbation → insertion → save.

The attack creates adversarial "black hole" vectors from cluster centroids that
attract nearest-neighbor queries, diverting them from genuine relevant documents.
"""

from __future__ import annotations

import sys
import os
from typing import Optional, Literal

import numpy as np
import faiss

# Allow running this module directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from process_data.data_manager import DataManager
from attack.preprocess import apply_preprocess
from attack.cluster import apply_clustering
from attack.centroid import perturb_centroids


class BlackHolePipeline:
    """Orchestrate the full black-hole attack on a retrieval index."""

    def __init__(
        self,
        source: DataManager,
        *,
        preprocess_mode: str = "default",
        cluster_method: str = "kmeans",
        n_clusters: int = 100,
        num_copies: int = 10,
        epsilon: float = 0.01,
        seed: int = 42,
    ):
        self.source = source
        self.preprocess_mode = preprocess_mode
        self.cluster_method = cluster_method
        self.n_clusters = n_clusters
        self.num_copies = num_copies
        self.epsilon = epsilon
        self.seed = seed

        # Results populated after run()
        self.preprocessed_vecs: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.cluster_centers: Optional[np.ndarray] = None
        self.adversarial_vecs: Optional[np.ndarray] = None
        self.result: Optional[DataManager] = None

    def run(self) -> DataManager:
        """Execute the full attack pipeline and return the poisoned DataManager."""
        if self.source.corpus_vecs is None or self.source.corpus_texts is None:
            raise RuntimeError("Source DataManager has no corpus loaded")

        print("=" * 60)
        print("  Black-Hole Attack Pipeline")
        print(f"  model={self.source.model}  dataset={self.source.dataset}")
        print(f"  preprocess={self.preprocess_mode}  cluster={self.cluster_method}")
        print(f"  n_clusters={self.n_clusters}  num_copies={self.num_copies}  epsilon={self.epsilon}")
        print("=" * 60)
        print()

        # Step 1: Preprocess
        print("[1/4] Preprocess ...")
        self.preprocessed_vecs = apply_preprocess(
            self.source.corpus_vecs, mode=self.preprocess_mode
        )
        print(f"  vectors: {self.preprocessed_vecs.shape}")
        print()

        # Step 2: Cluster
        print("[2/4] Cluster ...")
        self.labels, self.cluster_centers = apply_clustering(
            self.preprocessed_vecs,
            method=self.cluster_method,
            n_clusters=self.n_clusters,
        )
        print(f"  labels: {self.labels.shape}")
        print(f"  centers: {self.cluster_centers.shape}")
        print()

        # Step 3: Perturb centroids → adversarial vectors
        print("[3/4] Generate adversarial vectors ...")
        self.adversarial_vecs = perturb_centroids(
            self.cluster_centers,
            num_copies=self.num_copies,
            epsilon=self.epsilon,
            seed=self.seed,
        )
        print(f"  adversarial vectors: {self.adversarial_vecs.shape}")
        print()

        # Step 4: Build poisoned index
        print("[4/4] Build poisoned index ...")
        self.result = self._build_poisoned()
        print()
        print(self.result.summarize())
        print("=" * 60)
        return self.result

    def _build_poisoned(self) -> DataManager:
        """Insert adversarial vectors into a copy of the source and rebuild index."""
        src = self.source
        adv = self.adversarial_vecs

        # Generate fake doc ids for adversarial vectors
        n_adv = adv.shape[0]
        fake_ids = [f"bh_{i:08d}" for i in range(n_adv)]
        fake_texts = [""] * n_adv

        # Build poisoned corpus
        poisoned_vecs = np.vstack([src.corpus_vecs, adv]).astype(np.float32)

        import pandas as pd
        adv_rows = pd.DataFrame({"_id": fake_ids, "text": fake_texts, "title": [""] * n_adv})
        poisoned_texts = pd.concat(
            [src.corpus_texts.copy(), adv_rows], ignore_index=True
        )

        # Compute internal dimension for FAISS
        dim = poisoned_vecs.shape[1]

        # Build FAISS index (FlatIP for simplicity; rebuildable later)
        v = poisoned_vecs.copy()
        faiss.normalize_L2(v)
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(v)

        # Construct result DataManager (no config needed, copy from source)
        result = DataManager.__new__(DataManager)
        result.model = src.model
        result.dataset = src.dataset
        result.vector_dir = src.vector_dir
        result.dataset_dir = src.dataset_dir
        result._config = src._config

        result.corpus_vecs = poisoned_vecs
        result.corpus_texts = poisoned_texts
        result.query_vecs = src.query_vecs.copy() if src.query_vecs is not None else None
        result.query_texts = src.query_texts.copy() if src.query_texts is not None else None
        result.qrels = src.qrels.copy() if src.qrels is not None else None

        result.ann_index = faiss_index
        result._index_type = "FlatIP"
        result._corpus_dirty = False

        print(f"  original docs: {src.corpus_vecs.shape[0]}")
        print(f"  adversarial:   {n_adv}")
        print(f"  total:          {poisoned_vecs.shape[0]}")
        return result

    def save(self, output_dir: str) -> DataManager:
        """Save the poisoned result to a new directory."""
        if self.result is None:
            raise RuntimeError("Pipeline has not been run. Call run() first.")
        self.result.save(output_dir)
        return self.result
