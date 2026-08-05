"""Factory building a UEKD-wrapped backbone from config."""

from __future__ import annotations

from uekd.config import ModelConfig, DATASET_PRESETS
from uekd.models.attn_fusion import CoAttentionBackbone
from uekd.models.backbone import UEKDWrapper
from uekd.models.late_fusion import LateFusionBackbone


def build_backbone(model_cfg: ModelConfig, dataset: str = "weibo21") -> UEKDWrapper:
    """Instantiate and wrap a backbone according to ``model_cfg.backbone``."""
    preset = DATASET_PRESETS[dataset]
    common = dict(
        text_dim=preset["text_dim"],
        image_dim=preset["image_dim"],
        hidden_dim=model_cfg.hidden_dim,
        dropout=model_cfg.dropout,
    )
    if model_cfg.backbone == "late_fusion":
        backbone = LateFusionBackbone(**common)
    elif model_cfg.backbone == "co_attention":
        backbone = CoAttentionBackbone(
            seq_len=preset["seq_len"],
            n_heads=model_cfg.n_heads,
            n_layers=model_cfg.n_layers,
            n_img_tokens=model_cfg.n_img_tokens,
            use_psi=True if model_cfg.use_psi is None else model_cfg.use_psi,
            **common,
        )
    else:
        raise ValueError(f"Unknown backbone '{model_cfg.backbone}'")
    return UEKDWrapper(backbone)
