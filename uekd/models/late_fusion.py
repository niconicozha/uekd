"""Late-fusion backbone (simplified SpotFake+ [42]-like structure).

Each modality is encoded independently, the representations are concatenated
and passed through an MLP classifier. This backbone has NO cross-modal
consistency module psi (pure late fusion, first row of Table I).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from uekd.models.backbone import MultimodalBackbone


class LateFusionBackbone(MultimodalBackbone):
    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 1024,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        **_,
    ):
        super().__init__()
        # phi_t: mean-pool the BERT token sequence, then project
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # phi_v: project the pooled CLIP image embedding
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # f: concatenation fusion + MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, text: torch.Tensor, image: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # text: (B, L, D_text) -> mean pool -> (B, D_text)
        text_pooled = text.mean(dim=1)
        return {
            "text": self.text_encoder(text_pooled),   # (B, d)
            "image": self.image_encoder(image),       # (B, d)
            "psi": None,                              # no cross-modal module
        }

    def fuse(self, reprs: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        fused = torch.cat([reprs["text"], reprs["image"]], dim=-1)
        return self.classifier(fused).squeeze(-1)     # (B,) logits
