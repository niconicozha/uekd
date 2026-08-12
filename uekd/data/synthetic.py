"""Synthetic multimodal fake-news dataset for smoke testing.

Mimics the phenomena the paper studies (Fig. 5 & Fig. 7):

* **event-specific noise** -- each event carries a spurious direction that a
  teacher trained on the whole set can memorise but that does not transfer;
* **uni-modal tampering** -- for many samples only ONE modality contains the
  fake signal, while the other modality looks normal;
* **generalisable signal** -- a shared direction per modality that truly
  separates fake from real across all events.

The train/test split is event-disjoint (as the Weibo21 protocol), so relying
on event-specific noise hurts generalisation -- exactly the setting where
cross-domain (event-agnostic) teachers help.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def make_synthetic_dataset(
    n_events: int = 6,
    n_per_event: int = 120,
    seq_len: int = 16,
    text_dim: int = 32,
    image_dim: int = 32,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Generate the synthetic dataset.

    Returns a dict compatible with :func:`uekd.data.dataset.build_dataloaders`:
        text (N, L, text_dim), image (N, image_dim), labels (N,),
        domains (N,), split {"train": [...], "test": [...]}.
    """
    rng = np.random.default_rng(seed)

    # generalisable fake-signal directions (shared across events)
    w_text = rng.normal(size=(text_dim,)).astype(np.float32)
    w_image = rng.normal(size=(image_dim,)).astype(np.float32)
    w_text /= np.linalg.norm(w_text)
    w_image /= np.linalg.norm(w_image)

    texts, images, labels, domains = [], [], [], []

    for e in range(n_events):
        # event-specific spurious direction (memorisable noise)
        a_text = rng.normal(size=(text_dim,)).astype(np.float32)
        a_image = rng.normal(size=(image_dim,)).astype(np.float32)
        a_text /= np.linalg.norm(a_text)
        a_image /= np.linalg.norm(a_image)

        n = n_per_event + int(rng.integers(-10, 10))
        y = rng.integers(0, 2, size=n)

        # which modality is tampered for each sample:
        # ~45% text-only, ~45% image-only, ~10% both  (uni-modal tampering)
        mode = rng.choice([0, 1, 2], size=n, p=[0.45, 0.45, 0.10])

        strength = y.astype(np.float32)[:, None]  # fake -> +1 shift, real -> 0

        # ---- text features (n, L, D) -------------------------------------
        t_noise = rng.normal(scale=0.8, size=(n, seq_len, text_dim)).astype(np.float32)
        t_signal = strength[:, :, None] * (mode[:, None, None] != 1) * w_text[None, None, :]
        t_event = rng.normal(scale=1.2, size=(n, 1, 1)).astype(np.float32) * a_text[None, None, :]
        t = t_noise + 1.6 * t_signal + t_event
        texts.append(t)

        # ---- image features (n, D) --------------------------------------
        v_noise = rng.normal(scale=0.8, size=(n, image_dim)).astype(np.float32)
        v_signal = strength * (mode[:, None] != 0) * w_image[None, :]
        v_event = rng.normal(scale=1.2, size=(n, 1)).astype(np.float32) * a_image[None, :]
        v = v_noise + 1.6 * v_signal + v_event
        images.append(v)

        labels.append(y)
        domains.append(np.full(n, e, dtype=np.int64))

    text = np.concatenate(texts, axis=0)
    image = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    domains = np.concatenate(domains, axis=0)

    # event-disjoint 80/20 split (hold out whole events for testing)
    events = np.arange(n_events)
    rng.shuffle(events)
    n_test_events = max(1, int(round(n_events * 0.2)))
    test_events = set(events[:n_test_events].tolist())

    train_idx = [i for i, e in enumerate(domains) if e not in test_events]
    test_idx = [i for i, e in enumerate(domains) if e in test_events]

    return {
        "text": torch.from_numpy(text),
        "image": torch.from_numpy(image),
        "labels": torch.from_numpy(labels.astype(np.int64)),
        "domains": torch.from_numpy(domains.astype(np.int64)),
        "split": {"train": train_idx, "test": test_idx},
    }
