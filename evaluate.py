"""Standalone evaluation of a trained UEKD student checkpoint.

Reports the multimodal metrics (Table III style) and the uni-modal channel
accuracies inside the multimodal model (Table IV / Fig. 2 style, via the
masking protocol of Eq. 8).

Example:
    python evaluate.py --dataset weibo21 --backbone co_attention \
        --checkpoint ./checkpoints/weibo21/student_co_attention.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uekd.config import DATASET_PRESETS, ModelConfig, resolve_device
from uekd.data.dataset import build_dataloaders, load_preextracted
from uekd.data.synthetic import make_synthetic_dataset
from uekd.framework.evaluate import full_evaluation_report
from uekd.models.factory import build_backbone
from uekd.runtime import setup_backend
from uekd.utils import format_metrics, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="weibo21", choices=list(DATASET_PRESETS))
    p.add_argument("--data-root", default="./data")
    p.add_argument("--backbone", default="late_fusion", choices=["late_fusion", "co_attention"])
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    setup_backend(device)
    preset = DATASET_PRESETS[args.dataset]

    if args.dataset == "synthetic":
        data = make_synthetic_dataset(
            n_events=preset["num_domains"] + 2,
            seq_len=preset["seq_len"],
            text_dim=preset["text_dim"],
            image_dim=preset["image_dim"],
            seed=args.seed,
        )
    else:
        data = load_preextracted(os.path.join(args.data_root, args.dataset))

    loaders = build_dataloaders(data, batch_size=args.batch_size, device=device)

    model = build_backbone(
        ModelConfig(backbone=args.backbone, hidden_dim=args.hidden_dim),
        dataset=args.dataset,
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)

    report = full_evaluation_report(model, loaders["test"], device)
    print(f"===== evaluation | {args.dataset} | {args.backbone} =====")
    print(format_metrics(report))


if __name__ == "__main__":
    main()
