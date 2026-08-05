"""End-to-end smoke test on synthetic data (CPU friendly, no downloads).

Runs the complete UEKD pipeline with tiny budgets to verify correctness:
    1. synthetic event-disjoint dataset generation
    2. backbone forward passes + Gaussian masking shapes (Eq. 8)
    3. cross-domain teacher validation (Alg. 1) for both modalities
    4. student distillation with Shapley-adaptive weights (Eq. 13-18)
    5. evaluation: multimodal metrics + uni-modal channels

Usage:
    python smoke_test.py [--backbone late_fusion|co_attention|both]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uekd.config import ModelConfig, TrainConfig, resolve_device
from uekd.data.dataset import build_dataloaders
from uekd.data.synthetic import make_synthetic_dataset
from uekd.framework.distill import UEKDLoss, shapley_modality_weights
from uekd.framework.evaluate import full_evaluation_report
from uekd.framework.teacher import (
    refine_teacher_predictions,
    train_event_agnostic_teachers,
)
from uekd.framework.trainer import scatter_teacher_labels, train_student
from uekd.models.factory import build_backbone
from uekd.models.masking import VIEW_IMAGE, VIEW_MULTIMODAL, VIEW_NONE, VIEW_TEXT
from uekd.utils import format_metrics, set_seed


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        raise AssertionError(f"Smoke test failed: {name} {detail}")


def test_masking_shapes(device: str) -> None:
    print("\n== Step 1: backbone forward + modality masking (Eq. 8) ==")
    cfg = ModelConfig(backbone="co_attention", hidden_dim=32, n_heads=2)
    B, L, Dt, Dv = 4, 16, 32, 32
    text = torch.randn(B, L, Dt, device=device)
    image = torch.randn(B, Dv, device=device)

    model = build_backbone(cfg, dataset="synthetic").to(device)
    reprs = model.backbone.encode(text, image)
    for view in (VIEW_MULTIMODAL, VIEW_TEXT, VIEW_IMAGE, VIEW_NONE):
        logits = model(text, image, view=view)
        check(f"view='{view}' output shape", tuple(logits.shape) == (B,))
        check(f"view='{view}' finite", torch.isfinite(logits).all().item())

    # masked representations must differ from originals but keep shape/stats
    from uekd.models.masking import apply_view_mask

    masked = apply_view_mask(reprs, VIEW_TEXT)
    check("text kept under view='t'", torch.equal(masked["text"], reprs["text"]))
    check("image replaced under view='t'", not torch.equal(masked["image"], reprs["image"]))
    check("masked image shape", masked["image"].shape == reprs["image"].shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="both",
                        choices=["late_fusion", "co_attention", "both"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    set_seed(args.seed)
    device = resolve_device("cpu")  # smoke test always on CPU
    print(f"UEKD smoke test on device={device}")

    test_masking_shapes(device)

    # ------------------------------------------------------------------
    print("\n== Step 2: synthetic dataset ==")
    data = make_synthetic_dataset(n_events=6, n_per_event=90, seq_len=16,
                                  text_dim=32, image_dim=32, seed=args.seed)
    n = len(data["labels"])
    check("dataset size", n > 400, f"N={n}")
    check("event-disjoint split",
          len(set(data["split"]["train"]) & set(data["split"]["test"])) == 0)
    loaders = build_dataloaders(data, batch_size=32)
    train_indices = [int(i) for i in loaders["train"].dataset.indices.tolist()]

    # ------------------------------------------------------------------
    print("\n== Step 3: refinement logic sanity check ==")
    preds = torch.tensor([0.9, 0.3, 0.6, 0.2])
    labels = torch.tensor([1, 1, 0, 0])
    refined = refine_teacher_predictions(preds, labels)
    check("fake preds >= 0.5", bool(((refined[labels == 1] >= 0.5).all())))
    check("real preds <= 0.5", bool(((refined[labels == 0] <= 0.5).all())))

    backbones = ["late_fusion", "co_attention"] if args.backbone == "both" else [args.backbone]

    for bb in backbones:
        print(f"\n############ backbone = {bb} ############")
        model_cfg = ModelConfig(backbone=bb, hidden_dim=32, n_heads=2, n_img_tokens=4)

        # --------------------------------------------------------------
        print("\n== Step 4: cross-domain teachers (Alg. 1) ==")
        factory = lambda: build_backbone(model_cfg, dataset="synthetic")
        teacher_preds = train_event_agnostic_teachers(
            model_factory=factory,
            text_feats=data["text"],
            image_feats=data["image"],
            labels=data["labels"],
            domains=data["domains"],
            train_indices=train_indices,
            device=device,
            epochs=4,
            patience=3,
            batch_size=32,
            lr=2e-3,
            seed=args.seed,
        )
        for m in ("t", "v"):
            p = teacher_preds[m]
            check(f"teacher '{m}' preds finite", torch.isfinite(p).all().item())
            check(f"teacher '{m}' preds in [0,1]", bool(((p >= 0) & (p <= 1)).all()))
            y_tr = data["labels"][torch.tensor(train_indices)]
            check(f"teacher '{m}' refinement side-consistent",
                  bool(((p[y_tr == 1] >= 0.5).all() and (p[y_tr == 0] <= 0.5).all())))
            acc = float(((p >= 0.5).long() == y_tr).float().mean())
            print(f"      teacher '{m}' train-label agreement = {acc:.3f}")

        # --------------------------------------------------------------
        print("\n== Step 5: student distillation (Eq. 13-18) ==")
        model = build_backbone(model_cfg, dataset="synthetic")
        teacher_t = scatter_teacher_labels(train_indices, teacher_preds["t"], n)
        teacher_v = scatter_teacher_labels(train_indices, teacher_preds["v"], n)

        # one Shapley-weight sanity call
        batch = next(iter(loaders["train"]))
        lam, stats = shapley_modality_weights(
            model.to(device), batch.text, batch.image, batch.labels,
            teacher_t[batch.index], teacher_v[batch.index],
        )
        check("lambda sums to 1", abs(lam["t"] + lam["v"] - 1.0) < 1e-6,
              f"lam_t={lam['t']:.3f}")
        check("shapley stats keys", set(stats.phi) == {"t", "v"})

        train_cfg = TrainConfig(epochs=6, patience=5, batch_size=32, lr=2e-3,
                                alpha=0.25, beta=0.25, log_every=100)
        state = train_student(
            model,
            train_loader=loaders["train"],
            monitor_loader=loaders["test"],
            teacher_t=teacher_t,
            teacher_v=teacher_v,
            cfg=train_cfg,
            device=device,
        )
        check("training history recorded", len(state.history) > 0)
        check("best accuracy > chance", state.best_acc > 0.55,
              f"best_acc={state.best_acc:.3f}")

        # --------------------------------------------------------------
        print("\n== Step 6: evaluation report ==")
        model.load_state_dict(state.best_state)
        model = model.to(device)
        report = full_evaluation_report(model, loaders["test"], device)
        print("      " + format_metrics(report))
        check("test accuracy finite & sane", 0.0 <= report["accuracy"] <= 1.0)

    print(f"\nALL SMOKE TESTS PASSED in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
