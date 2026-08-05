"""Shared utilities: seeding, metrics, early stopping, tensor I/O."""

from __future__ import annotations

import json
import os
import random
from typing import Dict, Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility (numpy / torch / cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Platform-aware DataLoader helpers (Linux / Windows / macOS)
# ---------------------------------------------------------------------------
def default_num_workers(requested: int = -1) -> int:
    """Pick a sensible ``num_workers`` for the current platform.

    ``requested >= 0`` is honoured as-is. With ``-1`` (auto) we use several
    workers on Linux/macOS (fork start method, cheap) but keep ``0`` on
    Windows, where each worker needs a full process spawn and small models do
    not benefit. Multi-worker loading is one of the main Linux training wins.
    """
    if requested is not None and requested >= 0:
        return requested
    import platform

    if platform.system() == "Windows":
        return 0
    try:
        return min(4, len(os.sched_getaffinity(0)))  # Linux: usable cores
    except AttributeError:
        return min(4, os.cpu_count() or 1)


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` keeping multi-worker sampling seeded.

    Without this, ``num_workers > 0`` (the Linux default) re-introduces
    non-determinism through per-worker numpy/random state.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def binary_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Accuracy / precision / recall / F1 for binary predictions (Table III)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    preds = (probs >= 0.5).astype(np.int64)

    tp = float(np.sum((preds == 1) & (labels == 1)))
    fp = float(np.sum((preds == 1) & (labels == 0)))
    fn = float(np.sum((preds == 0) & (labels == 1)))
    tn = float(np.sum((preds == 0) & (labels == 0)))

    acc = (tp + tn) / max(tp + fp + fn + tn, 1.0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


class EarlyStopper:
    """Early stopping on a monitored accuracy (Sec. V-A4 of the paper)."""

    def __init__(self, patience: int = 30, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best: Optional[float] = None
        self.counter = 0
        self.best_state: Optional[dict] = None

    def _is_better(self, value: float) -> bool:
        if self.best is None:
            return True
        return value > self.best if self.mode == "max" else value < self.best

    def step(self, value: float, model: Optional[torch.nn.Module] = None) -> bool:
        """Returns True when training should stop."""
        if self._is_better(value):
            self.best = value
            self.counter = 0
            if model is not None:
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Tensor / json I/O helpers
# ---------------------------------------------------------------------------
def save_tensor_dict(path: str, data: Dict[str, torch.Tensor]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(data, path)


def load_tensor_dict(path: str) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_metrics(metrics: Dict[str, float]) -> str:
    return " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
