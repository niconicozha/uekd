"""Domain partition utilities for cross-domain teacher validation (Sec. IV-B3).

The paper partitions the training set into k non-overlapping domains:
    * Weibo21   -> 9 official domain labels
    * Twitter   -> 17 official event labels
    * GossipCop -> 10 domains clustered by Sentence-BERT embeddings
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def knn_cluster_domains(text_feats: torch.Tensor, k: int = 10, seed: int = 42) -> np.ndarray:
    """Cluster samples into k domains from pooled text embeddings.

    Mirrors the GossipCop protocol of the paper: samples are clustered into
    10 domains based on sentence embeddings. Here the mean-pooled frozen text
    features play the role of the sentence embedding; sklearn KMeans is used
    for the clustering step.

    Args:
        text_feats: (N, L, D) or (N, D) text features.
        k: number of domains.

    Returns:
        (N,) integer domain ids in [0, k).
    """
    from sklearn.cluster import KMeans

    x = text_feats
    if isinstance(x, torch.Tensor):
        x = x.float()
        if x.dim() == 3:
            x = x.mean(dim=1)  # mean pooling over the sequence
        x = x.numpy()
    x = np.asarray(x, dtype=np.float32)
    # L2-normalise so that cosine-like geometry drives the clustering
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    x = x / norm

    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return km.fit_predict(x).astype(np.int64)


def load_or_compute_domains(
    domains: Optional[torch.Tensor],
    text_feats: torch.Tensor,
    num_domains: int,
    domain_source: str = "official",
    cache_path: Optional[str] = None,
    seed: int = 42,
) -> torch.Tensor:
    """Return domain ids, clustering them on the fly when no labels exist.

    Args:
        domains: official domain labels, or None when unavailable.
        text_feats: pooled source for clustering when ``domains`` is None.
        num_domains: target number of domains k.
        domain_source: 'official' or 'knn-cluster' (dataset preset field).
        cache_path: optional file to persist computed clusters.
    """
    if domains is not None and domain_source == "official":
        return domains.long()

    if cache_path is not None:
        try:
            return torch.load(cache_path, map_location="cpu", weights_only=True).long()
        except FileNotFoundError:
            pass

    ids = knn_cluster_domains(text_feats, k=num_domains, seed=seed)
    result = torch.from_numpy(ids)
    if cache_path is not None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save(result, cache_path)
    return result
