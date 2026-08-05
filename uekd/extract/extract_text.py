"""Frozen BERT textual feature extraction.

The paper keeps the pre-trained encoders FROZEN during all training stages
(Sec. V-A4):
    Weibo21   -> bert-base-chinese,  max length 120
    GossipCop -> bert-base-uncased,  max length 200
    Twitter   -> twhin-bert-base,    max length 170

This script pre-extracts the last hidden states so that the training pipeline
can work purely on feature tensors (much faster, matches the paper setup).

Usage:
    python -m uekd.extract.extract_text \
        --dataset weibo21 \
        --texts data/weibo21/raw_texts.jsonl \
        --output data/weibo21/text_feats.pt

The input ``--texts`` file is JSON Lines: one record per line with a ``text``
field (and optionally ``id``). A plain .txt file (one post per line) is also
accepted.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import torch

from uekd.config import DATASET_PRESETS, resolve_device


def read_texts(path: str) -> List[str]:
    texts: List[str] = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                texts.append(obj.get("text", ""))
    else:
        with open(path, "r", encoding="utf-8") as f:
            texts = [ln.rstrip("\n") for ln in f if ln.strip()]
    return texts


@torch.no_grad()
def extract(
    texts: List[str],
    model_name: str,
    max_len: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()  # frozen encoder

    feats = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = tokenizer(
            chunk,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        out = model(**enc)
        feats.append(out.last_hidden_state.cpu())  # (B, L, 768)
        if (start // batch_size) % 20 == 0:
            print(f"  extracted {start + len(chunk)}/{len(texts)}")

    return torch.cat(feats, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="weibo21", choices=list(DATASET_PRESETS))
    parser.add_argument("--texts", required=True, help="path to .jsonl or .txt raw texts")
    parser.add_argument("--output", required=True, help="output .pt path")
    parser.add_argument("--model", default=None, help="override the text encoder name")
    parser.add_argument("--max-len", type=int, default=None, help="override max text length")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    preset = DATASET_PRESETS[args.dataset]
    model_name = args.model or preset.get("text_model")
    max_len = args.max_len or preset["seq_len"]
    device = resolve_device(args.device)
    if model_name is None:
        raise ValueError(f"Dataset '{args.dataset}' has no default text encoder; pass --model")

    print(f"[extract_text] encoder={model_name} max_len={max_len} device={device}")
    texts = read_texts(args.texts)
    print(f"[extract_text] loaded {len(texts)} texts")

    feats = extract(texts, model_name, max_len, args.batch_size, device)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(feats, args.output)
    print(f"[extract_text] saved {tuple(feats.shape)} -> {args.output}")


if __name__ == "__main__":
    main()
