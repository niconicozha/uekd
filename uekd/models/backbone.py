"""Backbone abstraction + UEKD plug-and-play wrapper.

The paper unifies late / early / hybrid fusion backbones as

    y_hat = sigma( f( phi_t(x_t), phi_v(x_v), psi(x) ) )          (Eq. 1)

where phi_m are uni-modal encoders, psi the (optional) cross-modal
consistency module and f the fusion stage. UEKD only touches the inputs and
outputs of the backbone, which is why the wrapper below works for any class
implementing this interface (plug-and-play, Sec. IV).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from uekd.models.masking import VIEW_MULTIMODAL, apply_view_mask


class MultimodalBackbone(nn.Module):
    """Interface every backbone must implement.

    ``encode`` returns the per-modality representations BEFORE fusion:
        {'text': Tensor, 'image': Tensor, 'psi': Tensor | None}
    ``fuse`` consumes those representations and outputs the class logit.
    """

    def encode(self, text: torch.Tensor, image: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        raise NotImplementedError

    def fuse(self, reprs: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, text: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return self.fuse(self.encode(text, image))


class UEKDWrapper(nn.Module):
    """Adds modality-masked uni-modal prediction channels to any backbone.

    Views (see uekd.models.masking):
        'mm'   -> standard multimodal prediction  y_hat      (Eq. 1)
        't'    -> text-only prediction            s_hat^t    (Eq. 8)
        'v'    -> image-only prediction            s_hat^v    (Eq. 8)
        'none' -> both modalities masked (only used for V(0), Eq. 13)

    The same wrapper is used for the student and the uni-modal teachers
    (Eq. 9): a teacher for modality m is simply trained with view=m.
    """

    def __init__(self, backbone: MultimodalBackbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, text: torch.Tensor, image: torch.Tensor, view: str = VIEW_MULTIMODAL) -> torch.Tensor:
        reprs = self.backbone.encode(text, image)
        if view != VIEW_MULTIMODAL:
            reprs = apply_view_mask(reprs, view)
        return self.backbone.fuse(reprs)  # (B,) logits

    # convenience --------------------------------------------------------
    def predict_proba(self, text: torch.Tensor, image: torch.Tensor, view: str = VIEW_MULTIMODAL) -> torch.Tensor:
        return torch.sigmoid(self.forward(text, image, view=view))
