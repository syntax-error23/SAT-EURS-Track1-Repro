"""
SAT-EURS Track 1 — V18 Models Package
Condition-Aware Zero-Residual Restoration Refiners and DINOv3 Dual-Encoder Segmenters
"""

from .restoration_refiner import ConditionedRestorationRefinerV18, build_v18_restoration_refiner
from .segmenter import EURSAPEXV18Segmenter, build_v18_segmenter
from .encoder_dinov3 import DINOv3ViTBackbone, DINOv3ConvNeXtBackbone
from .neck_upernet import V18UPerNetNeck, V18DetailStem

__all__ = [
    "ConditionedRestorationRefinerV18",
    "build_v18_restoration_refiner",
    "EURSAPEXV18Segmenter",
    "build_v18_segmenter",
    "DINOv3ViTBackbone",
    "DINOv3ConvNeXtBackbone",
    "V18UPerNetNeck",
    "V18DetailStem",
]
