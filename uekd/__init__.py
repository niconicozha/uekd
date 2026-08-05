"""UEKD: Uni-modal Event-agnostic Knowledge Distillation for Multimodal Fake News Detection.

Reference:
    Liu et al., "Uni-Modal Event-Agnostic Knowledge Distillation for
    Multimodal Fake News Detection", IEEE TKDE, vol. 36, no. 12, 2024.

Package layout:
    uekd.config     -- hyper-parameters (Sec. V-A4 of the paper)
    uekd.data       -- feature datasets, domain partition, synthetic data
    uekd.models     -- backbone models + Gaussian modality masking (Eq. 8/9)
    uekd.framework  -- cross-domain teacher (Alg. 1), distillation losses
                       (Eq. 10-12), Shapley-value adaptive weights (Eq. 13-18),
                       student trainer and evaluation
    uekd.extract    -- frozen BERT / CLIP-ResNet50 feature extraction scripts
"""

__version__ = "0.1.0"
