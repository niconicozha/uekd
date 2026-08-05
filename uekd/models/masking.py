"""Gaussian modality masking (Eq. 8 / Eq. 9 of the paper).

To separate uni-modal predictions inside a multimodal model, the inputs of
the non-target modalities are replaced -- *before the fusion stage* -- by
uninformative Gaussian noise whose per-dimension mean/variance match those of
the current batch:

    g~(x) ~ N(mean(x), var(x)),  statistics computed over i in the batch B.

The paper stresses this implementation is non-trivial: it does not shift the
input distribution, while zero-padding would break the numerical stability of
BatchNorm-style layers. The cross-modal consistency feature psi is ALWAYS
masked when computing uni-modal outputs (it is meaningless in that setting).
"""

from __future__ import annotations

import torch

#: views supported by :class:`uekd.models.backbone.UEKDWrapper`
VIEW_MULTIMODAL = "mm"     # both modalities active
VIEW_TEXT = "t"            # text only, image + psi masked
VIEW_IMAGE = "v"           # image only, text + psi masked
VIEW_NONE = "none"         # both modalities masked (used for V(empty))
ALL_VIEWS = (VIEW_MULTIMODAL, VIEW_TEXT, VIEW_IMAGE, VIEW_NONE)


def gaussian_noise_like(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Sample Gaussian noise approximating the batch distribution of ``x``.

    Args:
        x: tensor whose first dimension is the batch dimension,
           e.g. (B, d) or (B, L, d).
        eps: numerical guard added to the standard deviation.

    Returns:
        noise tensor with the same shape as ``x``; gradients are detached
        since the noise only plays the role of an uninformative placeholder.
    """
    with torch.no_grad():
        mean = x.detach().mean(dim=0, keepdim=True)
        var = x.detach().var(dim=0, unbiased=False, keepdim=True)
    return mean + torch.randn_like(x) * (var.sqrt() + eps)


def apply_view_mask(reprs: dict, view: str) -> dict:
    """Mask modality representations in-place according to ``view``.

    Args:
        reprs: dict with keys 'text', 'image' and optionally 'psi'
               (outputs of the uni-modal encoders, before fusion).
        view: one of :data:`ALL_VIEWS`.

    Returns:
        a new dict where the masked entries are replaced by Gaussian noise.
    """
    if view not in ALL_VIEWS:
        raise ValueError(f"Unknown view '{view}', expected one of {ALL_VIEWS}")

    masked = dict(reprs)
    mask_text = view in (VIEW_IMAGE, VIEW_NONE)
    mask_image = view in (VIEW_TEXT, VIEW_NONE)

    if mask_text:
        masked["text"] = gaussian_noise_like(reprs["text"])
    if mask_image:
        masked["image"] = gaussian_noise_like(reprs["image"])
    # psi is always masked when separating uni-modal predictions
    if view != VIEW_MULTIMODAL and masked.get("psi") is not None:
        masked["psi"] = gaussian_noise_like(reprs["psi"])
    return masked
