"""Frozen CLIP-ResNet50 visual feature extraction.

The paper uses ResNet50 pre-trained by CLIP [51] for the visual modality and
keeps it frozen (Sec. V-A4). The pooled image embedding (1024-d for RN50) is
pre-extracted here so training runs purely on feature tensors.

Usage:
    python -m uekd.extract.extract_image \
        --dataset weibo21 \
        --images data/weibo21/images.list \
        --output data/weibo21/image_feats.pt

``--images`` is a plain text file with one image path per line (absolute or
relative to ``--base-dir``). Pass ``--base-dir`` to prefix all entries.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import torch

from uekd.config import DATASET_PRESETS, resolve_device


def read_image_paths(path: str, base_dir: str = "") -> List[str]:
    paths = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            paths.append(os.path.join(base_dir, line) if base_dir else line)
    return paths


@torch.no_grad()
def extract(
    image_paths: List[str],
    clip_model_name: str,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(clip_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model.eval()  # frozen encoder

    feats = []
    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        images = []
        for p in chunk:
            with Image.open(p) as img:
                images.append(img.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt").to(device)
        emb = model.get_image_features(**inputs)  # (B, 1024) for RN50
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        feats.append(emb.cpu())
        if (start // batch_size) % 20 == 0:
            print(f"  extracted {start + len(chunk)}/{len(image_paths)}")

    return torch.cat(feats, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="weibo21", choices=list(DATASET_PRESETS))
    parser.add_argument("--images", required=True, help="text file listing image paths")
    parser.add_argument("--base-dir", default="", help="prefix for relative image paths")
    parser.add_argument("--output", required=True, help="output .pt path")
    parser.add_argument("--model", default=None, help="override the CLIP model name")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    preset = DATASET_PRESETS[args.dataset]
    clip_name = args.model or preset.get("image_model")
    device = resolve_device(args.device)
    if clip_name is None:
        raise ValueError("No default CLIP model for this dataset; pass --model")

    print(f"[extract_image] clip={clip_name} device={device}")
    paths = read_image_paths(args.images, args.base_dir)
    print(f"[extract_image] loaded {len(paths)} image paths")

    feats = extract(paths, clip_name, args.batch_size, device)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(feats, args.output)
    print(f"[extract_image] saved {tuple(feats.shape)} -> {args.output}")


if __name__ == "__main__":
    main()
