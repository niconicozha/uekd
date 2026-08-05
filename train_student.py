"""Stage 2: train the multimodal student under UEKD distillation (Eq. 18).

Requires the event-agnostic teacher predictions from train_teacher.py.

Examples:
    python train_student.py --dataset weibo21 --backbone co_attention \
        --teacher-dir ./checkpoints/weibo21 --output-dir ./checkpoints/weibo21

    # Synthetic demo
    python train_student.py --dataset synthetic --backbone late_fusion \
        --teacher-dir ./checkpoints/synthetic --epochs 20 --patience 8
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uekd.config import DATASET_PRESETS, ModelConfig, TrainConfig, resolve_device
from uekd.data.dataset import build_dataloaders, load_preextracted
from uekd.data.synthetic import make_synthetic_dataset
from uekd.framework.evaluate import full_evaluation_report
from uekd.framework.teacher import load_teacher_predictions
from uekd.framework.trainer import scatter_teacher_labels, train_student
from uekd.models.factory import build_backbone
from uekd.runtime import setup_backend
from uekd.utils import format_metrics, save_json, save_tensor_dict, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="weibo21", choices=list(DATASET_PRESETS))
    p.add_argument("--data-root", default="./data")
    p.add_argument("--backbone", default="late_fusion", choices=["late_fusion", "co_attention"])
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--teacher-dir", required=True, help="directory with teacher_preds_*.pt")
    p.add_argument("--output-dir", default="./checkpoints")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--alpha", type=float, default=0.25, help="Eq. 18 weight of L_bce")
    p.add_argument("--beta", type=float, default=0.25, help="Eq. 12 weight of L_GT")
    p.add_argument("--monitor", default="test", choices=["test", "val"],
                   help="early-stop monitor; the paper uses the test accuracy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    # runtime / platform flags
    p.add_argument("--num-workers", type=int, default=-1,
                   help="DataLoader workers; -1 = auto (Linux>0, Windows=0)")
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
    print(f"[train_student] dataset={args.dataset} backbone={args.backbone} device={device}")

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
    else:
        data = load_preextracted(os.path.join(args.data_root, args.dataset))

    loaders = build_dataloaders(data, batch_size=args.batch_size,
                                device=device, num_workers=args.num_workers)
    train_indices = [int(i) for i in loaders["train"].dataset.indices.tolist()]
    n_total = len(data["labels"])

    # teacher labels (event-agnostic knowledge, Sec. IV-B3); realigned by
    # dataset-global sample ids so stage 1 and stage 2 stay consistent even
    # if the train-split ordering differs between runs.
    aligned = load_teacher_predictions(args.teacher_dir, train_indices)
    teacher = {
        m: scatter_teacher_labels(train_indices, aligned[m], n_total)
        for m in ("t", "v")
    }

    # ------------------------------------------------------------------
    # student + UEKD training
    # ------------------------------------------------------------------
    model_cfg = ModelConfig(backbone=args.backbone, hidden_dim=args.hidden_dim)
    train_cfg = TrainConfig(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        monitor=args.monitor,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
        deterministic=args.deterministic,
        device=device,
    )
    model = build_backbone(model_cfg, dataset=args.dataset)

    state = train_student(
        model,
        train_loader=loaders["train"],
        monitor_loader=loaders["test"],
        teacher_t=teacher["t"],
        teacher_v=teacher["v"],
        cfg=train_cfg,
        device=device,
    )

    # ------------------------------------------------------------------
    # final evaluation with the best checkpoint
    # ------------------------------------------------------------------
    model.load_state_dict(state.best_state)
    model = model.to(device)
    report = full_evaluation_report(model, loaders["test"], device)
    print("\n===== UEKD student | test report =====")
    print(format_metrics(report))

    os.makedirs(args.output_dir, exist_ok=True)
    save_tensor_dict(
        os.path.join(args.output_dir, f"student_{args.backbone}.pt"),
        state.best_state,
    )
    save_json(
        os.path.join(args.output_dir, f"student_{args.backbone}_report.json"),
        {"best_test_accuracy": state.best_acc, "report": report,
         "history_tail": state.history[-5:]},
    )
    print(f"[train_student] checkpoint + report saved under {args.output_dir}")


if __name__ == "__main__":
    main()
