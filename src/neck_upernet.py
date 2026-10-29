"""
SAT-EURS Track 1 — V18 Neck & Multi-Scale Decoder (UPerNet / FPN Style)
Features:
1. Two-stage raw RGB detail stem at strides 2 and 4 for thin contours & rare species details
2. Feature Pyramid Neck combining ViT-L dense block tokens with ConvNeXt multi-stride pyramids
3. Learned, regularized spatial gating (dynamic semantic vs. convolutional feature fusion, no fixed alpha)
4. Bounded FiLM water-type and depth condition gates in the decoder
5. Multi-head output: Five-class main logits + auxiliary boundary logits + binary objectness logits
"""

from typing import List, Tuple, Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class V18DetailStem(nn.Module):
    """
    Two-stage raw RGB detail stem at strides 2 and 4 for small-object contours.
    Preserves thin appendages (e.g. seahorse snout, starfish spines, ray margins).
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 128):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns stride 2 [B, 64, H/2, W/2] and stride 4 [B, 128, H/4, W/4]."""
        s2 = self.stage1(x)
        s4 = self.stage2(s2)
        return s2, s4


class LearnedSpatialGate(nn.Module):
    """
    Learned spatial gate between semantic (ViT / restored) and spatial (ConvNeXt / raw) features.
    Computes a pixel-wise gating map in [0, 1].
    Regularized toward balanced fusion at initialization.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=1)
        )
        # Initialize gate bias to 0.0 (sigmoid(0) = 0.5 balanced fusion start)
        nn.init.zeros_(self.gate_conv[-1].weight)
        nn.init.zeros_(self.gate_conv[-1].bias)

    def forward(self, f_sem: torch.Tensor, f_conv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cat_feat = torch.cat([f_sem, f_conv], dim=1)
        gate = torch.sigmoid(self.gate_conv(cat_feat))
        f_fused = gate * f_sem + (1.0 - gate) * f_conv
        return f_fused, gate


class DecoderFiLMGate(nn.Module):
    """Bounded FiLM modulation in decoder: (1 + 0.2 * tanh(gamma)) * f + 0.2 * tanh(beta)."""
    def __init__(self, embed_dim: int, channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, channels * 2)
        )
        nn.init.zeros_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)

    def forward(self, f: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        film = self.mlp(cond_embed)  # [B, 2*C]
        gamma, beta = film.chunk(2, dim=-1)
        gamma = 0.20 * torch.tanh(gamma).view(-1, gamma.shape[-1], 1, 1)
        beta = 0.20 * torch.tanh(beta).view(-1, beta.shape[-1], 1, 1)
        return (1.0 + gamma) * f + beta


class V18UPerNetNeck(nn.Module):
    """
    UPerNet / FPN-style top-down neck combining ViT multi-block features at stride 16
    with ConvNeXt multi-stride features (strides 4, 8, 16, 32) and raw detail stem.
    """
    def __init__(
        self,
        vit_dim: int = 1024,
        conv_dims: list = [128, 256, 512, 1024],
        neck_dim: int = 256,
        num_classes: int = 5,
        cond_embed_dim: int = 64
    ):
        super().__init__()
        self.neck_dim = neck_dim
        self.num_classes = num_classes

        # Projections for 4 ViT tapped blocks (all stride 16)
        self.vit_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(vit_dim, neck_dim, kernel_size=1),
                nn.BatchNorm2d(neck_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])
        # ViT block fusion at stride 16
        self.vit_fuse = nn.Sequential(
            nn.Conv2d(neck_dim * 4, neck_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(neck_dim),
            nn.ReLU(inplace=True)
        )

        # Projections for ConvNeXt multi-scale features
        self.conv_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(conv_dims[0], neck_dim, kernel_size=1),  # stride 4
                nn.BatchNorm2d(neck_dim),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(conv_dims[1], neck_dim, kernel_size=1),  # stride 8
                nn.BatchNorm2d(neck_dim),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(conv_dims[2], neck_dim, kernel_size=1),  # stride 16
                nn.BatchNorm2d(neck_dim),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(conv_dims[3], neck_dim, kernel_size=1),  # stride 32
                nn.BatchNorm2d(neck_dim),
                nn.ReLU(inplace=True)
            ),
        ])

        # Learned Gating at Stride 16 (fusing ViT fused block feature with ConvNeXt stride 16)
        self.gate_stride16 = LearnedSpatialGate(neck_dim)

        # Learned Gating at Stride 4 (fusing detail stem with FPN stride 4)
        self.detail_proj = nn.Sequential(
            nn.Conv2d(128, neck_dim, kernel_size=1),
            nn.BatchNorm2d(neck_dim),
            nn.ReLU(inplace=True)
        )
        self.gate_stride4 = LearnedSpatialGate(neck_dim)

        # Top-down FPN smooth convolutions
        self.smooth_s32 = nn.Conv2d(neck_dim, neck_dim, kernel_size=3, padding=1)
        self.smooth_s16 = nn.Conv2d(neck_dim, neck_dim, kernel_size=3, padding=1)
        self.smooth_s8 = nn.Conv2d(neck_dim, neck_dim, kernel_size=3, padding=1)
        self.smooth_s4 = nn.Conv2d(neck_dim, neck_dim, kernel_size=3, padding=1)

        # Decoder FiLM condition gates
        self.film_s4 = DecoderFiLMGate(cond_embed_dim, neck_dim)

        # Multi-scale feature aggregation head
        self.fpn_agg = nn.Sequential(
            nn.Conv2d(neck_dim * 4, neck_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(neck_dim),
            nn.ReLU(inplace=True)
        )

        # 1. Main Head: Five-Class Semantic Logits
        self.cls_head = nn.Sequential(
            nn.Conv2d(neck_dim, neck_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(neck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.10),
            nn.Conv2d(neck_dim, num_classes, kernel_size=1)
        )

        # 2. Auxiliary Head: 1-channel Boundary Logits
        self.boundary_head = nn.Sequential(
            nn.Conv2d(neck_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        # 3. Auxiliary Head: 1-channel Binary Marine / Objectness Logits
        self.objectness_head = nn.Sequential(
            nn.Conv2d(neck_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(
        self,
        vit_features: List[torch.Tensor],
        conv_features: List[torch.Tensor],
        detail_s4: torch.Tensor,
        cond_embed: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            vit_features: 4 tensors at stride 16 from ViT blocks 5, 11, 17, 23
            conv_features: 4 tensors at strides 4, 8, 16, 32 from ConvNeXt
            detail_s4: [B, 128, H/4, W/4] from detail stem
            cond_embed: [B, cond_embed_dim] from condition encoder
            target_size: (H, W) exact unpadded image height & width
        """
        # 1. Project and fuse ViT features at stride 16
        vit_proj_list = [p(f) for p, f in zip(self.vit_projs, vit_features)]
        vit_fused_s16 = self.vit_fuse(torch.cat(vit_proj_list, dim=1))

        # 2. Project ConvNeXt features
        c_s4 = self.conv_projs[0](conv_features[0])
        c_s8 = self.conv_projs[1](conv_features[1])
        c_s16 = self.conv_projs[2](conv_features[2])
        c_s32 = self.conv_projs[3](conv_features[3])

        # 3. Dynamic Learned Spatial Gating at Stride 16
        p16, gate_s16 = self.gate_stride16(vit_fused_s16, c_s16)

        # 4. Top-down FPN pathway
        p32 = self.smooth_s32(c_s32)

        # p32 upsampled into p16
        p16 = self.smooth_s16(p16 + F.interpolate(p32, size=p16.shape[-2:], mode="bilinear", align_corners=False))

        # p16 upsampled into s8
        p8 = self.smooth_s8(c_s8 + F.interpolate(p16, size=c_s8.shape[-2:], mode="bilinear", align_corners=False))

        # p8 upsampled into s4
        p4 = c_s4 + F.interpolate(p8, size=c_s4.shape[-2:], mode="bilinear", align_corners=False)

        # 5. Fuse detail stem at Stride 4 via learned gate
        detail_proj = self.detail_proj(detail_s4)
        p4, gate_s4 = self.gate_stride4(p4, detail_proj)
        p4 = self.smooth_s4(p4)

        # 6. Apply condition FiLM modulation at Stride 4
        p4 = self.film_s4(p4, cond_embed)

        # 7. Multi-scale aggregation to Stride 4
        p8_up = F.interpolate(p8, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        p16_up = F.interpolate(p16, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        p32_up = F.interpolate(p32, size=p4.shape[-2:], mode="bilinear", align_corners=False)

        cat_all = torch.cat([p4, p8_up, p16_up, p32_up], dim=1)
        f_agg = self.fpn_agg(cat_all)

        # 8. Decode to target unpadded size (H, W)
        f_full = F.interpolate(f_agg, size=target_size, mode="bilinear", align_corners=False)

        # 9. Multi-Head Predictions
        logits_5class = self.cls_head(f_full)
        aux_boundary = self.boundary_head(f_full)
        aux_objectness = self.objectness_head(f_full)

        return {
            "logits": logits_5class,
            "aux_boundary": aux_boundary,
            "aux_objectness": aux_objectness,
            "gate_s16": gate_s16,
            "gate_s4": gate_s4
        }
