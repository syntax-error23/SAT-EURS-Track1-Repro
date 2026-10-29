"""
SAT-EURS Track 1 — V18 Dual-Encoder Semantic Segmenter (EURSAPEXV18Segmenter)
Architecture:
1. Main Semantic Path: Meta DINOv3 ViT-L/16 on Refined Restored Image (blocks 5, 11, 17, 23)
2. Raw-Image Complementary Path: Official DINOv3 ConvNeXt-Base on Raw Degraded Image (strides 4, 8, 16, 32)
3. Detail Contour Stem: Two-stage convolutional stem on Raw Degraded Image (strides 2, 4)
4. Fusion & Neck: UPerNet/FPN with dynamic learned spatial gating and bounded FiLM condition gates
5. Heads: Five-class categorical logits + auxiliary boundary logits + binary objectness logits
"""

from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_dinov3 import DINOv3ViTBackbone, DINOv3ConvNeXtBackbone
from .neck_upernet import V18DetailStem, V18UPerNetNeck


class ConditionEncoder(nn.Module):
    """Encodes water type and depth into a compact condition vector for neck FiLM modulation."""
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.wt_embed = nn.Embedding(11, embed_dim)
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU()
        )

    def forward(self, wt: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        wt_e = self.wt_embed(wt.clamp(1, 10))
        depth_norm = (depth.float() / 100.0).unsqueeze(-1)
        dp_e = self.depth_mlp(depth_norm)
        return self.fusion(torch.cat([wt_e, dp_e], dim=-1))


class EURSAPEXV18Segmenter(nn.Module):
    """
    Unified V18 Dual-Encoder Segmenter for Track 1 SOTA.
    Combines transformer global semantics with CNN high-frequency details.
    """
    def __init__(
        self,
        num_classes: int = 5,
        vit_pretrained: bool = False,
        conv_pretrained: bool = False,
        use_mock: bool = False,
        neck_dim: int = 256
    ):
        super().__init__()
        self.num_classes = num_classes

        # 1. Main Path: DINOv3 ViT-L/16
        self.vit_backbone = DINOv3ViTBackbone(
            pretrained=vit_pretrained,
            use_mock=use_mock,
            freeze_blocks_until=18
        )

        # 2. Complementary Path: DINOv3 ConvNeXt-Base
        self.conv_backbone = DINOv3ConvNeXtBackbone(
            pretrained=conv_pretrained,
            use_mock=use_mock,
            freeze_stages_until=2
        )

        # 3. Two-Stage Raw Detail Stem
        self.detail_stem = V18DetailStem(in_channels=3, out_channels=128)

        # 4. Condition Vector Encoder
        self.cond_encoder = ConditionEncoder(embed_dim=64)

        # 5. UPerNet / FPN Neck
        self.neck = V18UPerNetNeck(
            vit_dim=1024,
            conv_dims=[128, 256, 512, 1024],
            neck_dim=neck_dim,
            num_classes=num_classes,
            cond_embed_dim=64
        )

        # ImageNet normalization constants
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Applies ImageNet mean/std normalization."""
        return (x - self.mean) / self.std

    def forward(
        self,
        restored_rgb: torch.Tensor,
        raw_deg_rgb: torch.Tensor,
        wt: torch.Tensor,
        depth: torch.Tensor,
        unpadded_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            restored_rgb: [B, 3, H_pad, W_pad] in [0, 1] from restoration refiner
            raw_deg_rgb:  [B, 3, H_pad, W_pad] in [0, 1] raw degraded input
            wt:           [B] integer water type in [1..10]
            depth:        [B] integer depth in [5..100]
            unpadded_size: optional target (H_unpad, W_unpad) before multiple-of-32 padding.
                          If None, outputs at (H_pad, W_pad).
        Returns:
            dict containing:
              - 'logits': [B, 5, H, W] main five-class semantic logits
              - 'aux_boundary': [B, 1, H, W] auxiliary boundary logits
              - 'aux_objectness': [B, 1, H, W] auxiliary marine/objectness logits
              - 'gate_s16': learned gate map at stride 16
              - 'gate_s4': learned gate map at stride 4
        """
        b, _, h_pad, w_pad = restored_rgb.shape
        target_size = unpadded_size if unpadded_size is not None else (h_pad, w_pad)

        # 1. Condition Embedding
        cond_embed = self.cond_encoder(wt, depth)  # [B, 64]

        # 2. Main Semantic Path (DINOv3 ViT-L on restored image)
        norm_restored = self.normalize(restored_rgb)
        vit_feats = self.vit_backbone(norm_restored)  # 4 x [B, 1024, H/16, W/16]

        # 3. Complementary Path (DINOv3 ConvNeXt-B on raw degraded image)
        norm_raw = self.normalize(raw_deg_rgb)
        conv_feats = self.conv_backbone(norm_raw)     # 4 x [strides 4, 8, 16, 32]

        # 4. Detail Stem on raw image
        _, detail_s4 = self.detail_stem(raw_deg_rgb)  # [B, 128, H/4, W/4]

        # 5. UPerNet / FPN Neck Fusion & Decoding
        neck_out = self.neck(
            vit_features=vit_feats,
            conv_features=conv_feats,
            detail_s4=detail_s4,
            cond_embed=cond_embed,
            target_size=target_size
        )

        return neck_out


def build_v18_segmenter(
    num_classes: int = 5,
    vit_pretrained: bool = False,
    conv_pretrained: bool = False,
    use_mock: bool = False,
    neck_dim: int = 256
) -> EURSAPEXV18Segmenter:
    return EURSAPEXV18Segmenter(
        num_classes=num_classes,
        vit_pretrained=vit_pretrained,
        conv_pretrained=conv_pretrained,
        use_mock=use_mock,
        neck_dim=neck_dim
    )
