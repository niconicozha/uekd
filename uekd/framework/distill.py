"""Distillation losses and Shapley-value adaptive modality weights.

Implements:
    * Eq. 10  L_KD^m  -- MSE between event-agnostic teacher prediction and
                         the student's masked uni-modal prediction
    * Eq. 11  L_GT^m  -- BCE of the uni-modal channel against ground truth
    * Eq. 12  L^m     -- L_KD^m + beta * L_GT^m
    * Eq. 13-16       -- Shapley value phi^m and under-optimization gamma^m
    * Eq. 17          -- normalised loss weight lambda^m
    * Eq. 18          -- overall objective L = alpha*L_bce + sum lambda^m L^m
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from uekd.models.masking import VIEW_IMAGE, VIEW_MULTIMODAL, VIEW_NONE, VIEW_TEXT


# ---------------------------------------------------------------------------
# Shapley-value based adaptive learning speeds (Sec. IV-D)
# ---------------------------------------------------------------------------
@dataclass
class ShapleyStats:
    """Diagnostics for logging/inspection."""

    phi: Dict[str, float]             # Shapley values phi^m (Eq. 15)
    value_fn: Dict[str, float]        # V(S) accuracies (Eq. 13)
    teacher_acc: Dict[str, float]     # information upper bound per modality (Eq. 16)
    gamma: Dict[str, float]           # under-optimization degree gamma^m (Eq. 16)
    lam: Dict[str, float]             # normalised weights lambda^m (Eq. 17)


def _batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits.detach()) >= 0.5).long()
    return (preds == labels).float().mean().item()


@torch.no_grad()
def shapley_modality_weights(
    model: torch.nn.Module,
    text: torch.Tensor,
    image: torch.Tensor,
    labels: torch.Tensor,
    teacher_text: torch.Tensor,
    teacher_image: torch.Tensor,
    eps: float = 1e-3,
) -> Tuple[Dict[str, float], ShapleyStats]:
    """Compute the per-modality loss weights lambda^m for one batch.

    Args:
        model: UEKD-wrapped student.
        text/image/labels: current training batch.
        teacher_text/teacher_image: event-agnostic teacher predictions t_hat
            for the samples of this batch (already refined, Eq. in Alg. 1).
        eps: numerical guard for phi^m.

    Returns:
        ({"t": lambda_t, "v": lambda_v}, ShapleyStats)
    """
    was_training = model.training
    model.eval()

    # value function V(S): batch accuracy with the input set S active (Eq. 13)
    v_full = _batch_accuracy(model(text, image, view=VIEW_MULTIMODAL), labels)
    v_text = _batch_accuracy(model(text, image, view=VIEW_TEXT), labels)
    v_image = _batch_accuracy(model(text, image, view=VIEW_IMAGE), labels)
    v_none = _batch_accuracy(model(text, image, view=VIEW_NONE), labels)

    if was_training:
        model.train()

    # two-modality Shapley values (Eq. 15)
    phi_t = 0.5 * (v_full - v_image) + 0.5 * (v_text - v_none)
    phi_v = 0.5 * (v_full - v_text) + 0.5 * (v_image - v_none)

    # information upper bound: accuracy of the event-agnostic teacher labels
    # themselves, i.e. (1/|B|) * sum I(t_hat_i^m) == y_i  (Eq. 16)
    acc_t = ((teacher_text >= 0.5).long() == labels).float().mean().item()
    acc_v = ((teacher_image >= 0.5).long() == labels).float().mean().item()

    # under-optimization degree gamma^m = (1 / phi^m) * teacher_acc^m (Eq. 16)
    phi_t_c = max(phi_t, eps)
    phi_v_c = max(phi_v, eps)
    gamma_t = acc_t / phi_t_c
    gamma_v = acc_v / phi_v_c

    # normalised weights lambda^m (Eq. 17)
    z = gamma_t + gamma_v
    lam_t = gamma_t / z if z > 0 else 0.5
    lam_v = gamma_v / z if z > 0 else 0.5

    stats = ShapleyStats(
        phi={"t": phi_t, "v": phi_v},
        value_fn={"mm": v_full, "t": v_text, "v": v_image, "none": v_none},
        teacher_acc={"t": acc_t, "v": acc_v},
        gamma={"t": gamma_t, "v": gamma_v},
        lam={"t": lam_t, "v": lam_v},
    )
    return {"t": lam_t, "v": lam_v}, stats


# ---------------------------------------------------------------------------
# Losses (Eq. 10-12, Eq. 18)
# ---------------------------------------------------------------------------
class UEKDLoss(torch.nn.Module):
    """Overall UEKD objective.

    L = alpha * L_bce(multimodal)
        + lambda_t * (L_KD^t + beta * L_GT^t)
        + lambda_v * (L_KD^v + beta * L_GT^v)
    """

    def __init__(self, alpha: float = 0.25, beta: float = 0.25):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        logits_mm: torch.Tensor,          # multimodal logits
        logits_t: torch.Tensor,           # masked text-only student logits s_hat^t
        logits_v: torch.Tensor,           # masked image-only student logits s_hat^v
        labels: torch.Tensor,             # ground truth y in {0,1}
        teacher_t: torch.Tensor,          # event-agnostic t_hat^t (probabilities)
        teacher_v: torch.Tensor,          # event-agnostic t_hat^v
        lam: Dict[str, float],            # Shapley-normalised weights
    ) -> Dict[str, torch.Tensor]:
        labels = labels.float()

        # multimodal classification loss (the backbone's original objective)
        l_bce = F.binary_cross_entropy_with_logits(logits_mm, labels)

        # uni-modal channel losses, per modality m in {t, v}
        p_t, p_v = torch.sigmoid(logits_t), torch.sigmoid(logits_v)

        l_kd_t = F.mse_loss(p_t, teacher_t)                      # Eq. 10
        l_kd_v = F.mse_loss(p_v, teacher_v)                      # Eq. 10
        l_gt_t = F.binary_cross_entropy_with_logits(logits_t, labels)  # Eq. 11
        l_gt_v = F.binary_cross_entropy_with_logits(logits_v, labels)  # Eq. 11

        l_t = l_kd_t + self.beta * l_gt_t                        # Eq. 12
        l_v = l_kd_v + self.beta * l_gt_v                        # Eq. 12

        total = self.alpha * l_bce + lam["t"] * l_t + lam["v"] * l_v  # Eq. 18

        return {
            "loss": total,
            "l_bce": l_bce.detach(),
            "l_kd_t": l_kd_t.detach(),
            "l_kd_v": l_kd_v.detach(),
            "l_gt_t": l_gt_t.detach(),
            "l_gt_v": l_gt_v.detach(),
        }
