"""Stage 1: train uni-modal teachers via cross-domain validation (Alg. 1).

Produces the event-agnostic teacher predictions stored under
``--output-dir`` and consumed by train_student.py.

Examples:
    # Weibo21 with official domain labels, co-attention backbone
    python train_teacher.py --dataset weibo21 --data-root ./data \
        --backbone co_attention --output-dir ./checkpoints/weibo21

    # Synthetic demo (no real data needed)
    python train_teacher.py --dataset synthetic --backbone late_fusion \
        --output-dir ./checkpoints/synthetic --teacher-epochs 6
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uekd.config import DATASET_PRESETS, ModelConfig, TrainConfig, resolve_device
from uekd.data.dataset import load_preextracted
from uekd.data.domains import load_or_compute_domains
from uekd.data.synthetic import make_synthetic_dataset
from uekd.framework.teacher import train_event_agnostic_teachers
from uekd.models.factory import build_backbone
from uekd.runtime import setup_backend
from uekd.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="weibo21", choices=list(DATASET_PRESETS))
    p.add_argument("--data-root", default="./data")
    p.add_argument("--backbone", default="late_fusion", choices=["late_fusion", "co_attention"])
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--output-dir", default="./checkpoints")
    p.add_argument("--teacher-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    # runtime / platform flags
    p.add_argument("--no-amp", action="store_true",
                   help="disable CUDA mixed precision (auto-enabled on GPU)")
    p.add_argument("--deterministic", action="store_true",
                   help="use deterministic CUDA kernels (slower, reproducible)")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    setup_backend(device, deterministic=args.deterministic)
    preset = DATASET_PRESETS[args.dataset]
    print(f"[train_teacher] dataset={args.dataset} backbone={args.backbone} device={device}")

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    if args.dataset == "synthetic":
        data = make_synthetic_dataset(
            n_events=preset["num_domains"] + 2,
            seq_len=preset["seq_len"],
            text_dim=preset["text_dim"],
            image_dim=preset["image_dim"],
            seed=args.seed,
        )
        train_indices = data["split"]["train"]
    else:
        data_dir = os.path.join(args.data_root, args.dataset)
        data = load_preextracted(data_dir)
        split = data.get("split")
        if split is None:
            raise FileNotFoundError(
                f"split.json missing under {data_dir}; provide official train/test indices"
            )
        train_indices = split["train"]

    domains = load_or_compute_domains(
        data.get("domains"),
        data["text"],
        num_domains=preset["num_domains"],
        domain_source=preset["domain_source"],
        cache_path=None if args.dataset == "synthetic"
        else os.path.join(args.data_root, args.dataset, "domains_cached.pt"),
        seed=args.seed,
    )
    print(f"[train_teacher] k={len(torch.unique(domains[train_indices]))} domains on train split")

    # ------------------------------------------------------------------
    # teachers (same architecture as the future student, Sec. IV-B1)
    # ------------------------------------------------------------------
    model_cfg = ModelConfig(backbone=args.backbone, hidden_dim=args.hidden_dim)
    train_cfg = TrainConfig(batch_size=args.batch_size, lr=args.lr)

    def factory():
        return build_backbone(model_cfg, dataset=args.dataset)

    results = train_event_agnostic_teachers(
        model_factory=factory,
        text_feats=data["text"],
        image_feats=data["image"],
        labels=data["labels"],
        domains=domains,
        train_indices=train_indices,
        device=device,
        epochs=args.teacher_epochs,
        patience=train_cfg.teacher_patience,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        save_dir=args.output_dir,
        use_amp=not args.no_amp,
    )

    for m, preds in results.items():
        n_fake = float((preds > 0.5).sum())
        print(
            f"[train_teacher] modality '{m}': {len(preds)} predictions, "
            f"{n_fake:.0f} judged fake-ish, mean={preds.mean():.3f} "
            f"-> saved to {os.path.join(args.output_dir, f'teacher_preds_{m}.pt')}"
        )


if __name__ == "__main__":
    main()
