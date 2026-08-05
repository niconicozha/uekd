"""Uni-modal teacher training via cross-domain validation (Sec. IV-B, Alg. 1).

For every domain d_i the procedure:
    1. initialises a fresh teacher with the SAME architecture as the student;
    2. trains it on D - d_i with only modality m visible (other modalities
       Gaussian-masked, Eq. 9) using the BCE objective;
    3. collects the out-of-domain predictions on d_i -- these are the
       event-agnostic labels;
    4. refines them toward the neutral 0.5 so that a modality without fake
       evidence does not conflict with the ground truth (Alg. 1, line 8).

Indexing convention: all ``Subset`` selections use DATASET-GLOBAL sample ids
(0..N-1). Predictions are finally re-aligned to the training-split order.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from uekd.data.dataset import FeatureDataset, collate_uekd
from uekd.models.backbone import UEKDWrapper
from uekd.runtime import AMP, mv
from uekd.utils import EarlyStopper


def refine_teacher_predictions(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Refinement step of Algorithm 1 (line 8).

    t_hat = max(t_hat, 0.5) if y == 1 else min(t_hat, 0.5)

    Keeping the prediction on the same side as the label but allowing it to
    stay neutral lets the other, truly-tampered modality dominate the
    classification of that sample.
    """
    refined = preds.clone()
    fake = labels == 1
    refined[fake] = torch.maximum(refined[fake], torch.full_like(refined[fake], 0.5))
    refined[~fake] = torch.minimum(refined[~fake], torch.full_like(refined[~fake], 0.5))
    return refined


def _train_one_teacher(
    model_factory: Callable[[], UEKDWrapper],
    dataset: FeatureDataset,
    train_global: List[int],
    modality: str,
    device: str,
    epochs: int = 30,
    patience: int = 8,
    batch_size: int = 32,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.1,
    seed: int = 42,
    use_amp: bool = True,
) -> UEKDWrapper:
    """Train a single uni-modal teacher T^m_{d_i} on D - d_i until converged.

    ``train_global`` are the DATASET-GLOBAL ids of the samples in D - d_i.
    """
    torch.manual_seed(seed)

    # carve a small internal validation split from D - d_i for early stopping
    rng = np.random.default_rng(seed)
    arr = np.asarray(train_global)
    perm = rng.permutation(len(arr))
    n_val = max(1, int(len(arr) * val_fraction))
    val_ids = arr[perm[:n_val]].tolist()
    tr_ids = arr[perm[n_val:]].tolist()

    train_loader = DataLoader(
        Subset(dataset, tr_ids), batch_size=batch_size, shuffle=True, collate_fn=collate_uekd
    )
    val_loader = DataLoader(
        Subset(dataset, val_ids), batch_size=batch_size, shuffle=False, collate_fn=collate_uekd
    )

    teacher = model_factory().to(device)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=lr, weight_decay=weight_decay)
    stopper = EarlyStopper(patience=patience)
    amp = AMP(device, enabled=use_amp)

    for _epoch in range(epochs):
        teacher.train()
        for batch in train_loader:
            text = mv(batch.text, device)
            image = mv(batch.image, device)
            labels = mv(batch.labels, device)
            # Eq. 9: teacher sees only its own modality, the rest is masked
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                logits = teacher(text, image, view=modality)
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            amp.backward(loss)
            amp.step(optimizer)
            amp.update()

        # early stopping on the internal validation subset
        teacher.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                text, image = mv(batch.text, device), mv(batch.image, device)
                probs = torch.sigmoid(teacher(text, image, view=modality))
                correct += int(((probs >= 0.5).long() == mv(batch.labels, device)).sum().item())
                total += int(batch.labels.numel())
        val_acc = correct / max(total, 1)
        if stopper.step(val_acc, teacher):
            break

    if stopper.best_state is not None:
        teacher.load_state_dict(stopper.best_state)
    return teacher


@torch.no_grad()
def _predict_teacher(teacher: UEKDWrapper, dataset: FeatureDataset, global_ids: List[int],
                     modality: str, device: str, batch_size: int = 64) -> torch.Tensor:
    """Return sigmoid probabilities for ``global_ids`` in the given order."""
    teacher.eval()
    loader = DataLoader(Subset(dataset, global_ids), batch_size=batch_size,
                        shuffle=False, collate_fn=collate_uekd)
    from uekd.runtime import is_cuda

    use_cuda = is_cuda(device)
    preds = []
    for batch in loader:
        text, image = mv(batch.text, device), mv(batch.image, device)
        with torch.amp.autocast("cuda", enabled=use_cuda):
            probs = torch.sigmoid(teacher(text, image, view=modality))
        preds.append(probs.float().cpu())
    return torch.cat(preds, dim=0)


def train_event_agnostic_teachers(
    model_factory: Callable[[], UEKDWrapper],
    text_feats: torch.Tensor,
    image_feats: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    train_indices: List[int],
    device: str,
    epochs: int = 30,
    patience: int = 8,
    batch_size: int = 32,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    save_dir: Optional[str] = None,
    use_amp: bool = True,
) -> Dict[str, torch.Tensor]:
    """Algorithm 1: event-agnostic teacher predictions for BOTH modalities.

    Args:
        model_factory: zero-arg callable returning a fresh UEKDWrapper with
            the student's architecture (teachers share the student structure,
            Sec. IV-B1).
        text_feats/image_feats/labels: full-dataset feature tensors.
        domains: (N,) domain/event ids (official labels or KNN clusters).
        train_indices: global indices belonging to the training split.
        save_dir: when given, predictions are stored as
            ``<save_dir>/teacher_preds_<modality>.pt``.

    Returns:
        {"t": Tensor aligned with train_indices, "v": ...}
    """
    from uekd.runtime import is_cuda

    dataset = FeatureDataset(text_feats, image_feats, labels, domains=domains)
    idx_arr = np.asarray(train_indices, dtype=np.int64)
    # global sample id -> position within the training split
    pos_of_global = {int(g): pos for pos, g in enumerate(train_indices)}

    train_domains = domains[torch.from_numpy(idx_arr)].cpu().numpy()
    unique_domains = np.unique(train_domains)

    results: Dict[str, torch.Tensor] = {
        "t": torch.zeros(len(train_indices)),
        "v": torch.zeros(len(train_indices)),
    }

    for modality in ("t", "v"):
        preds = torch.zeros(len(train_indices))
        for d in unique_domains:
            in_domain_global = idx_arr[train_domains == d].tolist()
            out_domain_global = idx_arr[train_domains != d].tolist()

            teacher = _train_one_teacher(
                model_factory, dataset, out_domain_global, modality, device,
                epochs=epochs, patience=patience, batch_size=batch_size,
                lr=lr, weight_decay=weight_decay, seed=seed, use_amp=use_amp,
            )

            raw = _predict_teacher(teacher, dataset, in_domain_global, modality, device)
            y_d = labels[torch.as_tensor(in_domain_global, dtype=torch.long)]
            refined = refine_teacher_predictions(raw, y_d)

            # store back at the training-split positions of these samples
            positions = [pos_of_global[int(g)] for g in in_domain_global]
            preds[torch.as_tensor(positions, dtype=torch.long)] = refined

            del teacher
            if is_cuda(device):
                torch.cuda.empty_cache()

        results[modality] = preds
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"teacher_preds_{modality}.pt")
            torch.save({"global_indices": torch.from_numpy(idx_arr), "preds": preds}, path)

    return results


def load_teacher_predictions(save_dir: str, train_indices: List[int]) -> Dict[str, torch.Tensor]:
    """Load saved teacher predictions and align them with ``train_indices``."""
    out = {}
    for modality in ("t", "v"):
        payload = torch.load(os.path.join(save_dir, f"teacher_preds_{modality}.pt"),
                             map_location="cpu", weights_only=True)
        pos_of_global = {int(g): i for i, g in enumerate(payload["global_indices"].tolist())}
        aligned = torch.stack([payload["preds"][pos_of_global[int(g)]] for g in train_indices])
        out[modality] = aligned
    return out
