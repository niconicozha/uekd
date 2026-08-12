"""Feature-level multimodal datasets.

The paper extracts frozen features before training (Sec. V-A4):
    * text  : BERT last hidden states           -> (seq_len, 768)
    * image : CLIP-ResNet50 pooled embedding    -> (1024,)

This module consumes those pre-extracted tensors. Expected on-disk layout
under ``<data_root>/<dataset>/``::

    text_feats.pt    Tensor (N, seq_len, text_dim)   float16/32
    image_feats.pt   Tensor (N, image_dim)           float16/32
    labels.pt        Tensor (N,)                     long {0=real, 1=fake}
    domains.pt       Tensor (N,) long, optional      official domain/event ids
    split.json       {"train": [idx...], "test": [idx...]}

Use ``uekd/extract/extract_text.py`` and ``uekd/extract/extract_image.py``
to produce the feature files from raw data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from uekd.utils import default_num_workers, load_json, seed_worker


@dataclass
class UEKDBatch:
    """A collated training/testing batch."""

    text: torch.Tensor        # (B, L, text_dim)
    image: torch.Tensor       # (B, image_dim)
    labels: torch.Tensor      # (B,) long
    index: torch.Tensor       # (B,) position in the dataset (for teacher-label lookup)
    domains: Optional[torch.Tensor] = None  # (B,) long


class FeatureDataset(Dataset):
    """Wraps pre-extracted feature tensors with an optional index subset."""

    def __init__(
        self,
        text_feats: torch.Tensor,
        image_feats: torch.Tensor,
        labels: torch.Tensor,
        domains: Optional[torch.Tensor] = None,
        indices: Optional[List[int]] = None,
    ):
        self.text_feats = text_feats
        self.image_feats = image_feats
        self.labels = labels
        self.domains = domains
        self.indices = (
            torch.as_tensor(indices, dtype=torch.long)
            if indices is not None
            else torch.arange(len(labels), dtype=torch.long)
        )

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        idx = int(self.indices[i])
        item = {
            "text": self.text_feats[idx].float(),
            "image": self.image_feats[idx].float(),
            "label": self.labels[idx].long(),
            "index": idx,
        }
        if self.domains is not None:
            item["domain"] = self.domains[idx].long()
        return item


def collate_uekd(batch: List[Dict[str, torch.Tensor]]) -> UEKDBatch:
    return UEKDBatch(
        text=torch.stack([b["text"] for b in batch]),
        image=torch.stack([b["image"] for b in batch]),
        labels=torch.stack([b["label"] for b in batch]),
        index=torch.tensor([b["index"] for b in batch], dtype=torch.long),
        domains=torch.stack([b["domain"] for b in batch]) if "domain" in batch[0] else None,
    )


def load_preextracted(data_dir: str) -> Dict[str, torch.Tensor]:
    """Load the standard pre-extracted feature directory (see module docstring)."""
    def _load(name: str) -> torch.Tensor:
        pt = os.path.join(data_dir, name + ".pt")
        npy = os.path.join(data_dir, name + ".npy")
        if os.path.exists(pt):
            return torch.load(pt, map_location="cpu", weights_only=True)
        if os.path.exists(npy):
            return torch.from_numpy(np.load(npy))
        raise FileNotFoundError(f"Missing feature file: {pt} (or .npy)")

    data = {
        "text": _load("text_feats"),
        "image": _load("image_feats"),
        "labels": _load("labels").long(),
    }
    if os.path.exists(os.path.join(data_dir, "domains.pt")) or os.path.exists(
        os.path.join(data_dir, "domains.npy")
    ):
        data["domains"] = _load("domains").long()
    split_path = os.path.join(data_dir, "split.json")
    if os.path.exists(split_path):
        data["split"] = load_json(split_path)
    return data


def build_dataloaders(
    data: Dict[str, torch.Tensor],
    batch_size: int = 32,
    train_indices: Optional[List[int]] = None,
    test_indices: Optional[List[int]] = None,
    num_workers: int = -1,
    device: str = "cpu",
) -> Dict[str, DataLoader]:
    """Build train/test DataLoaders (platform / CUDA aware).

    If ``split.json`` is present in ``data`` it is honoured; otherwise an
    event-unaware 8:2 split is performed (only for datasets without an
    official split -- the paper splits Weibo21 by event at 8:2).

    ``num_workers=-1`` auto-selects a sensible count per platform (see
    :func:`uekd.utils.default_num_workers`). ``pin_memory`` is enabled when
    ``device`` is a CUDA device so host->GPU copies are faster on Linux
    training nodes.
    """
    from uekd.runtime import is_cuda

    domains = data.get("domains")
    if train_indices is None or test_indices is None:
        split = data.get("split")
        if split is not None:
            train_indices, test_indices = split["train"], split["test"]
        else:
            n = len(data["labels"])
            perm = torch.randperm(n).tolist()
            cut = int(n * 0.8)
            train_indices, test_indices = perm[:cut], perm[cut:]

    common = dict(
        text_feats=data["text"],
        image_feats=data["image"],
        labels=data["labels"],
        domains=domains,
    )
    train_ds = FeatureDataset(indices=train_indices, **common)
    test_ds = FeatureDataset(indices=test_indices, **common)

    workers = default_num_workers(num_workers)
    pin_memory = is_cuda(device)
    gen = torch.Generator()
    gen.manual_seed(0)

    shared = dict(
        batch_size=batch_size,
        collate_fn=collate_uekd,
        num_workers=workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        persistent_workers=(workers > 0),
    )

    return {
        "train": DataLoader(
            train_ds, shuffle=True, drop_last=False, generator=gen, **shared
        ),
        "test": DataLoader(test_ds, shuffle=False, **shared),
    }
