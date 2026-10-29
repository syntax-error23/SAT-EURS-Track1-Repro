"""
SAT-EURS Track 1 — V18 DINOv3 Encoder Backbones
1. DINOv3ViTBackbone: Meta DINOv3 ViT-L/16 (timm 'vit_large_patch16_dinov3')
   - Extracts dense token features from blocks 5, 11, 17, 23
   - Drops 5 prefix tokens (1 CLS + 4 register tokens)
   - Reshapes to [B, 1024, H/16, W/16]
   - Freezes patch embedding and early blocks; adapts blocks 18..23
2. DINOv3ConvNeXtBackbone: Official DINOv3 ConvNeXt-Base (timm 'convnext_base.dinov3_lvd1689m')
   - Extracts multi-scale features at strides 4, 8, 16, 32
   - Channels: 128, 256, 512, 1024
Includes mock/offline CPU mode for safe syntax and synthetic tests without network access.
"""

from typing import List, Tuple, Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MockViTBackbone(nn.Module):
    """Synthetic ViT-L/16 backbone for CPU testing without requiring downloaded weights."""
    def __init__(self, embed_dim: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)

    def forward_intermediate(self, x: torch.Tensor) -> List[torch.Tensor]:
        b, c, h, w = x.shape
        feat = self.conv(x)  # [B, embed_dim, H/16, W/16]
        return [feat, feat, feat, feat]


class MockConvNeXtBackbone(nn.Module):
    """Synthetic ConvNeXt-Base backbone for CPU testing without requiring downloaded weights."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 128, kernel_size=4, stride=4)
        self.s2 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.s3 = nn.Conv2d(256, 512, kernel_size=2, stride=2)
        self.s4 = nn.Conv2d(512, 1024, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        f1 = self.stem(x)   # stride 4
        f2 = self.s2(f1)    # stride 8
        f3 = self.s3(f2)    # stride 16
        f4 = self.s4(f3)    # stride 32
        return [f1, f2, f3, f4]


class DINOv3ViTBackbone(nn.Module):
    """
    Meta DINOv3 ViT-L/16 Backbone via timm.
    Features tapped at blocks 5, 11, 17, 23.
    """
    def __init__(
        self,
        model_name: str = "vit_large_patch16_dinov3",
        pretrained: bool = False,
        use_mock: bool = False,
        freeze_blocks_until: int = 18
    ):
        super().__init__()
        self.use_mock = use_mock
        self.embed_dim = 1024
        self.tapped_blocks = (5, 11, 17, 23)
        self.freeze_blocks_until = freeze_blocks_until

        if use_mock:
            self.model = MockViTBackbone(embed_dim=self.embed_dim)
        else:
            m_name = model_name
            if pretrained and not m_name.endswith(".lvd1689m"):
                m_name = f"{model_name}.lvd1689m"
            try:
                self.model = timm.create_model(
                    m_name,
                    pretrained=pretrained,
                    num_classes=0
                )
            except Exception as e:
                raise RuntimeError(
                    f"CRITICAL: Failed to initialize DINOv3 ViT backbone '{m_name}' (pretrained={pretrained}): {e}. "
                    f"Silent mock fallback is strictly forbidden in real runs. Set use_mock=True explicitly only in isolated unit tests."
                ) from e

            # Validate architecture contracts
            if not hasattr(self.model, "blocks") or len(self.model.blocks) < 24:
                raise RuntimeError(
                    f"CRITICAL: Model '{m_name}' has invalid block count: {len(getattr(self.model, 'blocks', []))}. Expected >= 24 blocks."
                )
            self.num_prefix_tokens = getattr(self.model, "num_prefix_tokens", 5)
            if self.num_prefix_tokens != 5:
                raise RuntimeError(
                    f"CRITICAL: Model '{m_name}' has unexpected prefix token count: {self.num_prefix_tokens}. Expected 5 (1 CLS + 4 register tokens)."
                )

            # Validate weight non-triviality
            param_norm = sum(p.norm().item() for p in self.model.parameters() if p.requires_grad or not p.requires_grad)
            if param_norm < 1.0:
                raise RuntimeError(f"CRITICAL: Model '{m_name}' has trivial parameter norm ({param_norm}). Pretrained weights may be empty.")

            if hasattr(self.model, "set_grad_checkpointing"):
                self.model.set_grad_checkpointing(True)

            self._apply_freezing()

    def _apply_freezing(self):
        """Freezes patch embed and blocks 0..freeze_blocks_until."""
        if hasattr(self.model, "patch_embed"):
            for p in self.model.patch_embed.parameters():
                p.requires_grad = False
        if hasattr(self.model, "cls_token") and self.model.cls_token is not None:
            self.model.cls_token.requires_grad = False
        if hasattr(self.model, "reg_token") and self.model.reg_token is not None:
            self.model.reg_token.requires_grad = False
        if hasattr(self.model, "pos_embed") and self.model.pos_embed is not None:
            self.model.pos_embed.requires_grad = False

        if hasattr(self.model, "blocks"):
            for idx, blk in enumerate(self.model.blocks):
                if idx < self.freeze_blocks_until:
                    for p in blk.parameters():
                        p.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] normalized image tensor (H, W multiples of 16)
        Returns:
            List of 4 feature maps at stride 16: [B, 1024, H/16, W/16]
        """
        if self.use_mock:
            return self.model.forward_intermediate(x)

        return self.model.forward_intermediates(
            x,
            indices=list(self.tapped_blocks),
            output_fmt="NCHW",
            intermediates_only=True
        )


class DINOv3ConvNeXtBackbone(nn.Module):
    """
    Official DINOv3 ConvNeXt-Base Backbone via timm.
    Outputs multi-scale features at strides 4, 8, 16, 32.
    """
    def __init__(
        self,
        model_name: str = "convnext_base",
        pretrained_tag: str = "dinov3_lvd1689m",
        pretrained: bool = False,
        use_mock: bool = False,
        freeze_stages_until: int = 2
    ):
        super().__init__()
        self.use_mock = use_mock
        self.out_channels = [128, 256, 512, 1024]

        if use_mock:
            self.model = MockConvNeXtBackbone()
        else:
            m_name = f"{model_name}.{pretrained_tag}" if pretrained and pretrained_tag else model_name
            try:
                self.model = timm.create_model(
                    m_name,
                    pretrained=pretrained,
                    features_only=True
                )
            except Exception as e:
                raise RuntimeError(
                    f"CRITICAL: Failed to initialize DINOv3 ConvNeXt backbone '{m_name}' (pretrained={pretrained}): {e}. "
                    f"Silent mock fallback is strictly forbidden in real runs. Set use_mock=True explicitly only in isolated unit tests."
                ) from e

            # Validate parameter non-triviality
            param_norm = sum(p.norm().item() for p in self.model.parameters() if p.requires_grad or not p.requires_grad)
            if param_norm < 1.0:
                raise RuntimeError(f"CRITICAL: Model '{m_name}' has trivial parameter norm ({param_norm}). Pretrained weights may be empty.")

            self._apply_freezing(freeze_stages_until)

    def _apply_freezing(self, freeze_stages_until: int):
        """Freezes early stages of ConvNeXt (stem and stages 0..freeze_stages_until-1)."""
        for name, module in self.model.named_children():
            if "stem" in name:
                for p in module.parameters():
                    p.requires_grad = False
            for stage_idx in range(freeze_stages_until):
                if f"stages_{stage_idx}" in name or f"stage_{stage_idx}" in name:
                    for p in module.parameters():
                        p.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] raw degraded RGB image
        Returns:
            List of 4 feature maps:
              - stride 4:  [B, 128, H/4, W/4]
              - stride 8:  [B, 256, H/8, W/8]
              - stride 16: [B, 512, H/16, W/16]
              - stride 32: [B, 1024, H/32, W/32]
        """
        return self.model(x)
