"""
Condition-Aware, Zero-Residual Restoration Refiner (V18)
Architecture:
- Base physical prior: Scale-Consistent 19-term RSUIGM matrix operator + verified 1.34 unsharp filter
- Refiner input: Concatenated [degraded RGB, base-restored RGB] (6 channels)
- FiLM conditioning: Water-type and depth embeddings injected at each multi-scale stage
- Multi-scale NAFNet encoder-decoder (channels: 32, 64, 128, 192)
- Dual output heads:
  (a) Low-resolution local affine delta field (3x3 transform + bias), identity-initialized
  (b) Full-resolution RGB residual, strictly zero-initialized
Mathematical Guarantee:
  At step 0, affine = Identity and residual = 0 -> J_refined = J_base exactly within float32 tolerance.
"""

import os
import os.path as osp
from typing import Dict, Tuple, Optional, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rsuigm_poly_features_gpu(img_bchw: torch.Tensor) -> torch.Tensor:
    b, _, h, w = img_bchw.shape
    s1 = F.interpolate(
        F.avg_pool2d(img_bchw, kernel_size=16, stride=16),
        size=(h, w),
        mode='bilinear',
        align_corners=False,
    )
    s2 = F.interpolate(
        F.avg_pool2d(img_bchw, kernel_size=64, stride=64),
        size=(h, w),
        mode='bilinear',
        align_corners=False,
    )
    x = img_bchw.permute(0, 2, 3, 1).reshape(b, h * w, 3)
    b1 = s1.permute(0, 2, 3, 1).reshape(b, h * w, 3)
    b2 = s2.permute(0, 2, 3, 1).reshape(b, h * w, 3)
    r = x[:, :, 0:1]
    g = x[:, :, 1:2]
    bl = x[:, :, 2:3]
    ones = torch.ones_like(r)
    return torch.cat(
        [ones, x, x * x, r * g, r * bl, g * bl, x * x * x, b1, b2],
        dim=-1,
    )



class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlockFiLM(nn.Module):
    """
    Nonlinear Activation Free Block with FiLM condition modulation.
    Uses Simplified Channel Attention (SCA) and SimpleGate.
    """
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, groups=dw_channel, bias=True)
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, bias=True)
        )

        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, bias=True)

        self.norm1 = nn.GroupNorm(1, c)
        self.norm2 = nn.GroupNorm(1, c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor, film_scale: Optional[torch.Tensor] = None, film_shift: Optional[torch.Tensor] = None) -> torch.Tensor:
        inp = x
        if film_scale is not None and film_shift is not None:
            # FiLM modulation: (1 + gamma) * x + beta
            inp = inp * (1.0 + film_scale) + film_shift

        y = self.norm1(inp)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg1(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        y = inp + y * self.beta

        z = self.norm2(y)
        z = self.conv4(z)
        z = self.sg2(z)
        z = self.conv5(z)
        return y + z * self.gamma


class ConditionEmbedding(nn.Module):
    """
    Embeds discrete water type (1..10) and continuous depth (5..100m).
    Generates multi-scale FiLM scale and shift parameters for each stage.
    """
    def __init__(self, channels_list: list, embed_dim: int = 64):
        super().__init__()
        self.wt_embed = nn.Embedding(11, embed_dim)  # index 1..10
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        self.film_heads = nn.ModuleDict()
        for i, c in enumerate(channels_list):
            head = nn.Linear(embed_dim, c * 2)
            # Initialize FiLM scale and shift strictly to zero
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.film_heads[f"stage_{i}"] = head

    def forward(self, wt: torch.Tensor, depth: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        wt_e = self.wt_embed(wt.clamp(1, 10))
        depth_norm = (depth.float() / 100.0).unsqueeze(-1)
        dp_e = self.depth_mlp(depth_norm)
        cond = self.fusion(torch.cat([wt_e, dp_e], dim=-1))  # [B, embed_dim]

        film_params = {}
        for k, head in self.film_heads.items():
            out = head(cond)  # [B, 2 * C]
            scale, shift = out.chunk(2, dim=-1)
            film_params[k] = (scale.view(-1, scale.shape[-1], 1, 1), shift.view(-1, shift.shape[-1], 1, 1))

        return film_params


class ConditionedRestorationRefinerV18(nn.Module):
    """
    Condition-Aware Zero-Residual Restoration Refiner.
    Multi-scale NAFNet architecture (32, 64, 128, 192) with FiLM conditioning,
    identity-initialized low-res local affine field, and zero-initialized residual head.
    """
    def __init__(
        self,
        channels: list = [32, 64, 128, 192],
        rsuigm_table: Optional[Dict[Tuple[int, int], torch.Tensor]] = None,
        use_neural_refiner: bool = True
    ):
        super().__init__()
        self.channels = channels
        self.use_neural_refiner = use_neural_refiner
        self.rsuigm_table = rsuigm_table

        # Intro: Degraded RGB (3ch) + Base Restored RGB (3ch) = 6ch
        self.intro = nn.Conv2d(6, channels[0], kernel_size=3, padding=1)

        # Condition FiLM Embedder
        self.cond_embed = ConditionEmbedding(channels, embed_dim=64)

        # Multi-scale Encoder
        self.enc0 = NAFBlockFiLM(channels[0])
        self.down0 = nn.Conv2d(channels[0], channels[1], kernel_size=2, stride=2)

        self.enc1 = NAFBlockFiLM(channels[1])
        self.down1 = nn.Conv2d(channels[1], channels[2], kernel_size=2, stride=2)

        self.enc2 = NAFBlockFiLM(channels[2])
        self.down2 = nn.Conv2d(channels[2], channels[3], kernel_size=2, stride=2)

        # Bottleneck
        self.middle = nn.Sequential(
            NAFBlockFiLM(channels[3]),
            NAFBlockFiLM(channels[3])
        )

        # Multi-scale Decoder with Skip Connections
        self.up2 = nn.Sequential(
            nn.Conv2d(channels[3], channels[2], kernel_size=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        )
        self.dec2 = NAFBlockFiLM(channels[2])

        self.up1 = nn.Sequential(
            nn.Conv2d(channels[2], channels[1], kernel_size=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        )
        self.dec1 = NAFBlockFiLM(channels[1])

        self.up0 = nn.Sequential(
            nn.Conv2d(channels[1], channels[0], kernel_size=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        )
        self.dec0 = NAFBlockFiLM(channels[0])

        # Head A: Low-Resolution Local Affine Field (1/8 scale)
        # Predicts 3x3 matrix (9 params) + 3-dim bias (3 params) = 12 params per cell
        self.affine_head = nn.Sequential(
            nn.Conv2d(channels[3], 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 12, kernel_size=1)
        )
        # Identity initialization for Affine Head:
        # Diagonal elements of 3x3 matrix initialized to 1.0, off-diagonal and bias to 0.0
        self._init_affine_identity()

        # Head B: Full-Resolution RGB Residual Head
        self.residual_head = nn.Conv2d(channels[0], 3, kernel_size=3, padding=1)
        # Zero-initialization: Exact 0.0 residual start
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _init_affine_identity(self):
        """Initializes affine head output to exact Identity transform: A = I_3, b = 0."""
        last_conv = self.affine_head[-1]
        nn.init.zeros_(last_conv.weight)
        # Channel layout: [A00, A01, A02, A10, A11, A12, A20, A21, A22, b0, b1, b2]
        bias_init = torch.zeros(12)
        bias_init[0] = 1.0  # A00
        bias_init[4] = 1.0  # A11
        bias_init[8] = 1.0  # A22
        last_conv.bias.data.copy_(bias_init)

    def set_rsuigm_table(self, table: Dict[Tuple[int, int], torch.Tensor]):
        """Sets the underlying closed-form RSUIGM table."""
        self.rsuigm_table = table

    def compute_base_restoration(
        self,
        deg_rgb: torch.Tensor,
        wt: torch.Tensor,
        depth: torch.Tensor,
        unsharp: bool = True
    ) -> torch.Tensor:
        """
        Computes closed-form physical RSUIGM inversion + 1.34 unsharp post-processing.
        Follows the verified production pipeline exactly.
        """
        if self.rsuigm_table is None:
            return deg_rgb

        b, c, h, w = deg_rgb.shape
        restored_list = []

        for i in range(b):
            w_val = int(wt[i].item())
            d_val = int(depth[i].item())
            key = (w_val, d_val)

            if key not in self.rsuigm_table:
                restored_list.append(deg_rgb[i:i+1])
                continue

            W = self.rsuigm_table[key].to(deg_rgb.device)  # [19, 3]
            feats = build_rsuigm_poly_features_gpu(deg_rgb[i:i+1])  # [1, H*W, 19]
            res_flat = torch.matmul(feats.reshape(-1, 19), W).clamp_(0.0, 1.0)
            res_im = res_flat.reshape(1, h, w, 3).permute(0, 3, 1, 2)

            if unsharp:
                # GPU Gaussian unsharp (sigma=2.0, alpha=1.34, beta=-0.34)
                # Matches cv2.GaussianBlur(res, (0, 0), 2.0)
                res_np = (res_im.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
                blur = cv2.GaussianBlur(res_np, (0, 0), 2.0)
                sharp = np.clip(cv2.addWeighted(res_np, 1.34, blur, -0.34, 0), 0, 255)
                sharp_t = torch.from_numpy(sharp.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(deg_rgb.device)
                restored_list.append(sharp_t)
            else:
                restored_list.append(res_im)

        return torch.cat(restored_list, dim=0)

    def forward(
        self,
        deg_rgb: torch.Tensor,
        wt: torch.Tensor,
        depth: torch.Tensor,
        base_restored_rgb: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            deg_rgb: [B, 3, H, W] degraded image in [0, 1]
            wt: [B] integer water type in [1..10]
            depth: [B] integer depth in [5..100]
            base_restored_rgb: optional precomputed base output. If None, computes via table.
        Returns:
            dict containing:
              - 'refined_image': [B, 3, H, W] final output in [0, 1]
              - 'base_image': [B, 3, H, W] base prior output
              - 'affine_transformed': [B, 3, H, W] local affine intermediate
              - 'residual': [B, 3, H, W] full-res delta
              - 'affine_params': [B, 12, H/8, W/8] local affine parameter map
        """
        b, _, h, w = deg_rgb.shape

        if base_restored_rgb is None:
            base_restored_rgb = self.compute_base_restoration(deg_rgb, wt, depth, unsharp=True)

        if not self.use_neural_refiner:
            # Clean fallback to base physical restoration
            return {
                "refined_image": base_restored_rgb,
                "base_image": base_restored_rgb,
                "affine_transformed": base_restored_rgb,
                "residual": torch.zeros_like(base_restored_rgb),
                "affine_params": torch.zeros((b, 12, h // 8, w // 8), device=deg_rgb.device)
            }

        # 1. Conditioning FiLM modulation parameters
        films = self.cond_embed(wt, depth)

        # 2. Encoder
        x_in = torch.cat([deg_rgb, base_restored_rgb], dim=1)  # [B, 6, H, W]
        f0 = self.intro(x_in)
        s0, b0 = films["stage_0"]
        f0 = self.enc0(f0, s0, b0)

        f1 = self.down0(f0)
        s1, b1 = films["stage_1"]
        f1 = self.enc1(f1, s1, b1)

        f2 = self.down1(f1)
        s2, b2 = films["stage_2"]
        f2 = self.enc2(f2, s2, b2)

        f3 = self.down2(f2)
        s3, b3 = films["stage_3"]
        f3 = self.middle[0](f3, s3, b3)
        f3 = self.middle[1](f3, s3, b3)

        # 3. Head A: Low-Resolution Local Affine Field at 1/8 scale
        affine_low = self.affine_head(f3)  # [B, 12, H/8, W/8]
        affine_up = F.interpolate(affine_low, size=(h, w), mode="bilinear", align_corners=False)  # [B, 12, H, W]

        # Apply local affine matrix A(x) * I_base + b(x)
        # affine_up: [B, 12, H, W] -> matrix [B, 3, 3, H, W], bias [B, 3, H, W]
        A = affine_up[:, :9].view(b, 3, 3, h, w)
        b_vec = affine_up[:, 9:12]  # [B, 3, H, W]

        # Batch-wise RGB matrix multiplication: A[b, :, :, y, x] @ I_base[b, :, y, x]
        # Reshape for einsum: 'b c d h w, b d h w -> b c h w'
        I_affine = torch.einsum("bcdhw,bdhw->bchw", A, base_restored_rgb) + b_vec

        # 4. Decoder
        d2 = self.up2(f3) + f2
        d2 = self.dec2(d2, s2, b2)

        d1 = self.up1(d2) + f1
        d1 = self.dec1(d1, s1, b1)

        d0 = self.up0(d1) + f0
        d0 = self.dec0(d0, s0, b0)

        # 5. Head B: Full-Resolution RGB Residual
        delta_res = self.residual_head(d0)  # [B, 3, H, W]

        # Final Refined Output (Exact Identity at step 0)
        refined = torch.clamp(I_affine + delta_res, 0.0, 1.0)

        return {
            "refined_image": refined,
            "base_image": base_restored_rgb,
            "affine_transformed": I_affine,
            "residual": delta_res,
            "affine_params": affine_low
        }


def build_v18_restoration_refiner(
    rsuigm_table: Optional[Dict[Tuple[int, int], torch.Tensor]] = None,
    use_neural_refiner: bool = True
) -> ConditionedRestorationRefinerV18:
    return ConditionedRestorationRefinerV18(
        channels=[32, 64, 128, 192],
        rsuigm_table=rsuigm_table,
        use_neural_refiner=use_neural_refiner
    )
