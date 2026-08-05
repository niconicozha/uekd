"""Early-fusion backbone (simplified MCAN [23] / HMCAN [22]-like structure).

Architecture split to respect the UEKD masking protocol (Eq. 8):

    encode() -> phi_t / phi_v outputs ONLY (token projections, no
                cross-modal interaction yet) + the consistency feature psi
    fuse()   -> the fusion stage f: intra-modal self-attention,
                cross-modal co-attention, pooling and the MLP classifier

Because UEKDWrapper inserts Gaussian masking between ``encode`` and ``fuse``,
this split guarantees that "the mask operation is performed before the
modality fusion stage f(.)" exactly as the paper requires -- even for
early-fusion models where uni-modal effects are entangled inside attention.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from uekd.models.backbone import MultimodalBackbone


class CoAttentionBackbone(MultimodalBackbone):
    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 1024,
        seq_len: int = 120,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        n_img_tokens: int = 8,
        dropout: float = 0.3,
        use_psi: bool = True,
        **_,
    ):
        super().__init__()
        self.n_img_tokens = n_img_tokens
        self.use_psi = use_psi

        # phi_t: project BERT hidden states into the shared attention space
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        # phi_v: expand the pooled CLIP embedding into pseudo visual tokens
        self.image_proj = nn.Linear(image_dim, n_img_tokens * hidden_dim)
        self.img_pos = nn.Parameter(torch.randn(1, n_img_tokens, hidden_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.text_self_attn = nn.TransformerEncoder(layer, num_layers=n_layers)
        # co-attention: text <-> image (part of the fusion stage f)
        self.cross_t2i = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_i2t = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(hidden_dim)
        self.norm_i = nn.LayerNorm(hidden_dim)

        # psi: cross-modal consistency feature (hybrid-fusion family, CAFE-like)
        if use_psi:
            self.psi_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        in_dim = hidden_dim * 2 + (hidden_dim if use_psi else 0)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    def encode(self, text: torch.Tensor, image: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """Uni-modal encoder outputs BEFORE any cross-modal interaction."""
        bsz = text.size(0)
        t = self.text_proj(text)                                            # (B, L, d)
        v = self.image_proj(image).view(bsz, self.n_img_tokens, -1) + self.img_pos  # (B, k, d)

        psi = None
        if self.use_psi:
            psi = self.psi_mlp(torch.cat([t.mean(dim=1), v.mean(dim=1)], dim=-1))  # (B, d)

        return {"text": t, "image": v, "psi": psi}

    # ------------------------------------------------------------------
    def fuse(self, reprs: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        """Fusion stage f: attention interactions + MLP classifier."""
        t, v = reprs["text"], reprs["image"]

        t = self.text_self_attn(t)                    # intra-modal (text)
        t2i, _ = self.cross_t2i(t, v, v)              # text attends to image
        i2t, _ = self.cross_i2t(v, t, t)              # image attends to text
        t = self.norm_t(t + t2i)
        v = self.norm_i(v + i2t)

        parts = [t.mean(dim=1), v.mean(dim=1)]
        if reprs.get("psi") is not None:
            parts.append(reprs["psi"])
        fused = torch.cat(parts, dim=-1)
        return self.classifier(fused).squeeze(-1)     # (B,) logits
