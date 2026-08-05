"""Evaluation utilities.

* :func:`evaluate_multimodal`       -- standard test metrics (Table III):
                                       accuracy, precision, recall, F1.
* :func:`evaluate_unimodal_channels` -- uni-modal performance of the channels
                                       inside the multimodal model, obtained
                                       with the masking protocol of Eq. 8
                                       (Table IV / Fig. 2 of the paper).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from uekd.models.masking import VIEW_IMAGE, VIEW_MULTIMODAL, VIEW_TEXT
from uekd.runtime import is_cuda, mv
from uekd.utils import binary_metrics


@torch.no_grad()
def _collect(model: torch.nn.Module, loader: DataLoader, device: str, view: str):
    model.eval()
    use_cuda = is_cuda(device)
    probs_all, labels_all = [], []
    for batch in loader:
        text, image = mv(batch.text, device), mv(batch.image, device)
        with torch.amp.autocast("cuda", enabled=use_cuda):
            probs = torch.sigmoid(model(text, image, view=view))
        probs_all.append(probs.float().cpu().numpy())
        labels_all.append(batch.labels.numpy())
    return np.concatenate(probs_all), np.concatenate(labels_all)


def evaluate_multimodal(model: torch.nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    """Standard multimodal evaluation: accuracy / precision / recall / F1."""
    model.eval()
    probs, labels = _collect(model, loader, device, VIEW_MULTIMODAL)
    return binary_metrics(probs, labels)


def evaluate_unimodal_channels(model: torch.nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    """Accuracy of each masked uni-modal channel inside the multimodal model.

    Reproduces the diagnosis protocol of Sec. III-B / Table IV: the same
    jointly-trained model is tested with only one modality visible while the
    other modalities (and psi) are Gaussian-masked.
    """
    model.eval()
    out = {}
    for name, view in (("text_in_multimodal", VIEW_TEXT), ("image_in_multimodal", VIEW_IMAGE)):
        probs, labels = _collect(model, loader, device, view)
        out[name + "_accuracy"] = binary_metrics(probs, labels)["accuracy"]
    return out


def full_evaluation_report(model: torch.nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    """Combined report: multimodal metrics + uni-modal channel accuracies."""
    report = evaluate_multimodal(model, loader, device)
    report.update(evaluate_unimodal_channels(model, loader, device))
    return report
