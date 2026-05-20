"""
Clustering strategies for the black-hole attack pipeline.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


def cluster_kmeans(vecs: np.ndarray, n_clusters: int, *,
                   random_state: int = 42, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """K-means clustering on vectors.

    Returns:
        labels: (N,) int array, cluster index for each vector
        centers: (n_clusters, dim) float32 array of cluster centroids
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, **kwargs)
    labels = km.fit_predict(vecs)
    return labels, km.cluster_centers_.astype(np.float32)


def cluster_minibatch_kmeans(vecs: np.ndarray, n_clusters: int, *,
                              batch_size: int = 4096,
                              random_state: int = 42, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """MiniBatch K-means — much faster for large vector sets (>1M).

    Returns:
        labels: (N,) int array, cluster index for each vector
        centers: (n_clusters, dim) float32 array of cluster centroids
    """
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        # batch_size=batch_size,
        batch_size=n_clusters * 100,
        random_state=random_state,
        n_init="auto",
        **kwargs,
    )
    labels = km.fit_predict(vecs)
    return labels, km.cluster_centers_.astype(np.float32)


CLUSTERERS = {
    "kmeans": cluster_kmeans,
    "minibatch_kmeans": cluster_minibatch_kmeans,
}


def apply_clustering(vecs: np.ndarray, method: str = "kmeans",
                     n_clusters: int = 100, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    if method not in CLUSTERERS:
        raise ValueError(f"Unknown clustering method: {method}. Available: {list(CLUSTERERS)}")
    return CLUSTERERS[method](vecs, n_clusters=n_clusters, **kwargs)
