"""Student (multimodal) model training with the UEKD objective (Eq. 18)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from uekd.config import TrainConfig
from uekd.framework.distill import UEKDLoss, shapley_modality_weights
from uekd.framework.evaluate import evaluate_multimodal
from uekd.models.masking import VIEW_IMAGE, VIEW_MULTIMODAL, VIEW_TEXT
from uekd.runtime import AMP, mv
from uekd.utils import EarlyStopper


@dataclass
class TrainerState:
    """Result of a student training run."""

    best_acc: float
    best_state: Dict[str, torch.Tensor]
    history: List[dict] = field(default_factory=list)


def scatter_teacher_labels(train_indices: List[int], preds: torch.Tensor, n_total: int,
                           default: float = 0.5) -> torch.Tensor:
    """Map teacher predictions (ordered by ``train_indices``) onto a
    full-sized tensor indexable by the dataset-global sample id."""
    out = torch.full((n_total,), default, dtype=torch.float32)
    idx = torch.as_tensor(train_indices, dtype=torch.long)
    out[idx] = preds.float()
    return out


def train_student(
    model: torch.nn.Module,
    train_loader: DataLoader,
    monitor_loader: DataLoader,
    teacher_t: torch.Tensor,
    teacher_v: torch.Tensor,
    cfg: TrainConfig,
    device: str,
    val_loader: Optional[DataLoader] = None,
) -> TrainerState:
    """Train the multimodal student under UEKD distillation.

    Args:
        model: UEKD-wrapped student backbone.
        train_loader: training batches (collated with UEKDBatch, carrying the
            dataset-global ``index`` used to fetch teacher labels).
        monitor_loader: loader used for early stopping -- the test loader by
            default because the paper stops when *test* accuracy plateaus
            (Sec. V-A4); pass ``val_loader`` for a stricter protocol.
        teacher_t/teacher_v: full-sized tensors (N,) of event-agnostic
            teacher predictions, indexable by the global sample id.
        cfg: training hyper-parameters (alpha, beta, lr, patience...).
    """
    model = model.to(device)
    criterion = UEKDLoss(alpha=cfg.alpha, beta=cfg.beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    stopper = EarlyStopper(patience=cfg.patience)
    amp = AMP(device, enabled=cfg.use_amp)

    teacher_t = mv(teacher_t, device)
    teacher_v = mv(teacher_v, device)

    state = TrainerState(best_acc=0.0, best_state={})
    if cfg.monitor == "val" and val_loader is None:
        raise ValueError("cfg.monitor='val' requires a validation loader")
    eval_loader = val_loader if cfg.monitor == "val" else monitor_loader

    for epoch in range(cfg.epochs):
        model.train()
        epoch_stats = {"loss": 0.0, "lam_t": 0.0, "phi_t": 0.0, "n_batches": 0}

        for batch in train_loader:
            text = mv(batch.text, device)
            image = mv(batch.image, device)
            labels = mv(batch.labels, device)
            idx = mv(batch.index, device)
            t_t = teacher_t[idx]
            t_v = teacher_v[idx]

            # dynamic per-batch loss weights from Shapley values (Eq. 13-17)
            lam, stats = shapley_modality_weights(
                model, text, image, labels, t_t, t_v, eps=cfg.shapley_eps
            )

            model.train()
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                logits_mm = model(text, image, view=VIEW_MULTIMODAL)
                logits_t = model(text, image, view=VIEW_TEXT)
                logits_v = model(text, image, view=VIEW_IMAGE)
                losses = criterion(logits_mm, logits_t, logits_v, labels, t_t, t_v, lam)

            amp.backward(losses["loss"])
            amp.step(optimizer)
            amp.update()

            epoch_stats["loss"] += float(losses["loss"].item())
            epoch_stats["lam_t"] += float(lam["t"])
            epoch_stats["phi_t"] += float(stats.phi["t"])
            epoch_stats["n_batches"] += 1

        # ---- end-of-epoch evaluation & early stopping -------------------
        metrics = evaluate_multimodal(model, eval_loader, device)
        acc = metrics["accuracy"]
        n = max(epoch_stats["n_batches"], 1)
        record = {
            "epoch": epoch,
            "train_loss": epoch_stats["loss"] / n,
            "avg_lambda_t": epoch_stats["lam_t"] / n,
            "avg_phi_t": epoch_stats["phi_t"] / n,
            f"{cfg.monitor}_accuracy": acc,
        }
        state.history.append(record)

        if epoch % max(cfg.log_every // 5, 1) == 0 or epoch == cfg.epochs - 1:
            print(
                f"[epoch {epoch:3d}] loss={record['train_loss']:.4f} "
                f"lambda_t={record['avg_lambda_t']:.3f} "
                f"{cfg.monitor}_acc={acc:.4f} (best={stopper.best if stopper.best else acc:.4f})"
            )

        if stopper.step(acc, model):
            print(f"Early stopping at epoch {epoch} (no improvement for {cfg.patience} epochs).")
            break

    state.best_acc = float(stopper.best or 0.0)
    state.best_state = stopper.best_state or {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return state
