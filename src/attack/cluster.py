"""
Clustering strategies for the black-hole attack pipeline.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

import faiss


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
        batch_size=batch_size,
        # batch_size=n_clusters * 50,
        random_state=random_state,
        n_init="auto",
        **kwargs,
    )
    labels = km.fit_predict(vecs)
    return labels, km.cluster_centers_.astype(np.float32)


def cluster_faiss_gpu(vecs: np.ndarray, n_clusters: int, *,
                      random_state: int = 42,
                      gpu_id: int = 1,
                      niter: int = 25,
                      use_float16: bool = False,
                      **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """GPU-accelerated k-means via FAISS.

    Requires faiss-gpu. Typical speedup over CPU k-means: 10–50×.

    Returns:
        labels: (N,) int array, cluster index for each vector
        centers: (n_clusters, dim) float32 array of cluster centroids
    """
    n_gpus = faiss.get_num_gpus()
    if n_gpus == 0:
        raise RuntimeError("No GPU available for FAISS GPU clustering")

    # Normalize for inner product (cosine similarity in k-means)
    data = vecs.astype(np.float32).copy()
    faiss.normalize_L2(data)

    d = data.shape[1]
    res = faiss.StandardGpuResources()

    cfg = faiss.GpuIndexFlatConfig()
    cfg.device = gpu_id
    cfg.useFloat16 = use_float16

    # GpuIndexFlatIP: inner-product search on GPU
    index = faiss.GpuIndexFlatIP(res, d, cfg)

    clus = faiss.Clustering(d, n_clusters)
    clus.seed = random_state
    clus.niter = niter
    clus.verbose = True
    # Use all data (disable the default 256*k subsampling)
    clus.max_points_per_centroid = max(1, data.shape[0] // n_clusters)

    print(f"  FAISS GPU k-means: n={data.shape[0]}, d={d}, k={n_clusters}, "
          f"GPU={gpu_id}, niter={niter}")
    clus.train(data, index)

    # Extract centroids
    centroids = faiss.vector_float_to_array(clus.centroids).reshape(n_clusters, d)

    # Assign labels: load centroids into GPU index, search all points
    index.reset()
    index.add(centroids)
    D, I = index.search(data, 1)
    labels = I.flatten().astype(np.int64)

    return labels, centroids


from utils.teb_mean import cluster_teb  # noqa: E402
from utils.adaptive_mean import adaptive_clustering  # noqa: E402

CLUSTERERS = {
    "kmeans": cluster_kmeans,
    "minibatch_kmeans": cluster_minibatch_kmeans,
    "faiss_gpu": cluster_faiss_gpu,
    "teb": cluster_teb,
    "adaptive": adaptive_clustering,
}


def apply_clustering(vecs: np.ndarray, method: str = "kmeans",
                     n_clusters: int = 100, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    if method not in CLUSTERERS:
        raise ValueError(f"Unknown clustering method: {method}. Available: {list(CLUSTERERS)}")
    return CLUSTERERS[method](vecs, n_clusters=n_clusters, **kwargs)
