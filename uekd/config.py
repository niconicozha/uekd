"""Configuration and hyper-parameters for UEKD.

Values follow Section V-A4 "Implementation Details" of the paper unless noted:
    * batch size 32, Adam with lr 2e-4
    * 150 epochs, early stopping when the monitored accuracy does not
      improve for 30 epochs
    * alpha = beta = 0.25 (Fig. 9 of the paper)
"""

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Dataset-specific presets (Table: Datasets in Sec. V-A1 / Sec. IV-B3)
# ---------------------------------------------------------------------------
DATASET_PRESETS = {
    # Weibo21: official 9 domain labels, bert-base-chinese, max text len 120
    "weibo21": dict(
        text_model="bert-base-chinese",
        image_model="openai/clip-resnet50",
        seq_len=120,
        text_dim=768,
        image_dim=1024,
        num_domains=9,
        domain_source="official",
        train_size=(4640, 4487),   # (real, fake)
        lang="zh",
    ),
    # Twitter: official 17 event labels used as domains, twhin-bert-base, len 170
    "twitter": dict(
        text_model="twhin-bert-base",
        image_model="openai/clip-resnet50",
        seq_len=170,
        text_dim=768,
        image_dim=1024,
        num_domains=17,
        domain_source="official",
        train_size=(3923, 6276),
        test_size=(408, 1072),
        lang="en",
    ),
    # GossipCop (FakeNewsNet): balanced down-sampling, 10 KNN-clustered domains
    "gossipcop": dict(
        text_model="bert-base-uncased",
        image_model="openai/clip-resnet50",
        seq_len=200,
        text_dim=768,
        image_dim=1024,
        num_domains=10,
        domain_source="knn-cluster",
        train_size=(2036, 2036),
        test_size=(545, 545),
        lang="en",
    ),
    # Synthetic demo dataset used by smoke_test.py
    "synthetic": dict(
        text_model=None,
        image_model=None,
        seq_len=16,
        text_dim=32,
        image_dim=32,
        num_domains=4,
        domain_source="official",
        lang="n/a",
    ),
}


@dataclass
class ModelConfig:
    """Backbone configuration (simplified versions of the paper's backbones)."""

    backbone: str = "late_fusion"      # 'late_fusion' (SpotFake+-like) or 'co_attention' (MCAN/HMCAN-like)
    hidden_dim: int = 128              # projection dimension d for each modality branch
    dropout: float = 0.3
    use_psi: Optional[bool] = None     # cross-modal consistency module; default: True for co_attention
    # co-attention specific
    n_heads: int = 4
    n_layers: int = 1
    n_img_tokens: int = 8              # number of pseudo visual tokens expanded from the pooled image feat


@dataclass
class TrainConfig:
    """Student / teacher training hyper-parameters."""

    batch_size: int = 32               # Sec. V-A4
    lr: float = 2e-4                   # Sec. V-A4
    epochs: int = 150                  # Sec. V-A4
    patience: int = 30                 # early-stop patience, Sec. V-A4
    alpha: float = 0.25                # Eq. 18 weight of L_bce (Fig. 9)
    beta: float = 0.25                 # Eq. 12 weight of L_GT (Fig. 9)
    teacher_epochs: int = 30           # teachers converge much faster (Sec. VI-F)
    teacher_patience: int = 8
    shapley_eps: float = 1e-3          # numerical guard for phi^m (Eq. 16)
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"               # 'auto' | 'cpu' | 'cuda' | 'cuda:N' | 'mps'
    monitor: str = "test"              # early-stop monitor: 'test' (paper) or 'val'
    log_every: int = 20
    # runtime / platform handling
    use_amp: bool = True               # mixed precision, auto-enabled only on CUDA
    num_workers: int = -1              # DataLoader workers; -1 = platform auto (Linux>0, Windows=0)
    deterministic: bool = False        # reproducible CUDA kernels (slower)


@dataclass
class UEKDConfig:
    """Top-level config bundling everything."""

    dataset: str = "weibo21"
    data_root: str = "./data"
    output_dir: str = "./checkpoints"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def preset(self) -> dict:
        return DATASET_PRESETS[self.dataset]

    @property
    def num_domains(self) -> int:
        return self.preset["num_domains"]


def resolve_device(requested: str = "auto") -> str:
    """Pick the device string, honouring the 'auto' keyword.

    Delegates to :func:`uekd.runtime.resolve_device`, which validates CUDA
    availability and also supports 'cuda:N' and 'mps'. Kept here as the
    public import path used by the entry scripts.
    """
    from uekd.runtime import resolve_device as _resolve

    return _resolve(requested)
