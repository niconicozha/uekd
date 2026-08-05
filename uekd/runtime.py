"""Runtime device & platform layer.

Centralises all CUDA / MPS / CPU and Linux / Windows handling so that the
rest of the codebase stays device-agnostic:

* :func:`resolve_device`   -- pick a torch device ('auto' | cpu | cuda | cuda:N | mps)
* :func:`setup_backend`    -- CUDA performance flags (cuDNN benchmark, TF32)
* :func:`mv`               -- non-blocking tensor transfer to a device
* :class:`AMP`             -- optional mixed-precision helper (CUDA only)
* :func:`is_cuda`          -- device predicate

Everything degrades gracefully: on a CPU-only / Windows machine the helpers
become no-ops, so the same code runs unchanged on Linux + GPU clusters.
"""

from __future__ import annotations

from typing import Union

import torch

Tensor = torch.Tensor
DeviceLike = Union[str, torch.device]


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------
def resolve_device(requested: str = "auto") -> str:
    """Resolve a device string, validating CUDA availability.

    Args:
        requested: one of ``auto`` | ``cpu`` | ``cuda`` | ``cuda:N`` | ``mps``.

    ``auto`` prefers CUDA, then Apple MPS, then CPU. Requesting ``cuda*``
    without an available GPU raises a clear error instead of silently falling
    back, which would otherwise produce hard-to-debug OOM/speed surprises.
    """
    requested = (requested or "auto").strip().lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but torch.cuda.is_available() is False. "
                "Check your driver / CUDA-enabled torch build, or pass --device cpu."
            )
        return requested

    return requested


def is_cuda(device: DeviceLike) -> bool:
    return str(device).startswith("cuda")


def device_type(device: DeviceLike) -> str:
    """Return the torch device-type string ('cuda' | 'cpu' | 'mps')."""
    return str(device).split(":")[0]


# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------
def setup_backend(device: DeviceLike, deterministic: bool = False) -> None:
    """Apply device-specific performance settings.

    On CUDA: enable cuDNN autotuning and TF32 acceleration (safe for fp32
    training of this kind of model). ``deterministic=True`` trades speed for
    reproducible kernels. On CPU / MPS this is a no-op.
    """
    if not is_cuda(device):
        return

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    # TF32 speeds up matmul/conv on Ampere+ GPUs with negligible accuracy loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Tensor transfer
# ---------------------------------------------------------------------------
def mv(x: Tensor, device: DeviceLike) -> Tensor:
    """Move a tensor to ``device``.

    Uses ``non_blocking=True`` for CUDA transfers so host->device copies can
    overlap with compute (a standard Linux/GPU training optimisation). No-op
    when already on the target device.
    """
    if x.device == torch.device(device):
        return x
    return x.to(device, non_blocking=is_cuda(device))


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------
class AMP:
    """Automatic mixed precision helper, active only on CUDA.

    Wraps :class:`torch.amp.GradScaler` and :func:`torch.amp.autocast` behind
    a tiny interface. When AMP is disabled (CPU / MPS / ``enabled=False``) the
    context manager is a no-op and ``scale``/``step``/``update`` fall back to
    plain optimiser behaviour, so training loops read identically either way.

    Usage::

        amp = AMP(device, enabled=cfg.use_amp)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast():
            loss = compute_loss(...)
        amp.backward(loss)
        amp.step(optimizer)
        amp.update()
    """

    def __init__(self, device: DeviceLike, enabled: bool = True):
        self.enabled = bool(enabled) and is_cuda(device)
        self._device_type = "cuda"
        self.scaler = torch.amp.GradScaler(self._device_type, enabled=self.enabled)

    def autocast(self):
        return torch.amp.autocast(self._device_type, enabled=self.enabled)

    def backward(self, loss: Tensor) -> None:
        self.scaler.scale(loss).backward()

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.scaler.step(optimizer)

    def update(self) -> None:
        self.scaler.update()
