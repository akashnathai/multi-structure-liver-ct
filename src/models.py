"""
Lean multi-head 3D U-Net for liver, tumour, and vessel segmentation. 

- Outputs logits (sigmoid handled later for stability).
- Gradient checkpointing enabled to squeeze into 8GB VRAM.
- Intentionally skipped the transformer bottleneck to keep the architecture simple, letting us cleanly ablate the loss functions (clDice & constraints).
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import STRUCTURES


class ResBlock3D(nn.Module):
    """(Conv3d -> InstanceNorm3d -> LeakyReLU) x2 with a 1x1 residual skip."""

    def __init__(self, in_ch: int, out_ch: int, slope: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(slope, inplace=True)
        self.skip = (nn.Conv3d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class MultiHeadUNet3D(nn.Module):
    """Shared-trunk 3D U-Net with one sigmoid head per structure.

    Parameters
    ----------
    base_channels : encoder level-0 width; channels = [base, 2b, 4b] + 8b bottleneck.
    structures    : output head names (default liver/tumour/vessel).
    use_checkpointing : gradient-checkpoint each residual block during training.
    """

    def __init__(self, base_channels: int = 32, in_channels: int = 1,
                 slope: float = 0.1, use_checkpointing: bool = True,
                 structures: List[str] = STRUCTURES) -> None:
        super().__init__()
        b = base_channels
        self.use_checkpointing = use_checkpointing
        self.structures = list(structures)

        # ---- Encoder (3 down levels) ----
        self.enc0 = ResBlock3D(in_channels, b, slope)
        self.enc1 = ResBlock3D(b, 2 * b, slope)
        self.enc2 = ResBlock3D(2 * b, 4 * b, slope)
        self.pool = nn.MaxPool3d(2)

        # ---- Bottleneck ----
        self.bottleneck = ResBlock3D(4 * b, 8 * b, slope)

        # ---- Decoder (3 up levels) ----
        self.up2 = nn.ConvTranspose3d(8 * b, 4 * b, 2, stride=2)
        self.dec2 = ResBlock3D(4 * b + 4 * b, 4 * b, slope)
        self.up1 = nn.ConvTranspose3d(4 * b, 2 * b, 2, stride=2)
        self.dec1 = ResBlock3D(2 * b + 2 * b, 2 * b, slope)
        self.up0 = nn.ConvTranspose3d(2 * b, b, 2, stride=2)
        self.dec0 = ResBlock3D(b + b, b, slope)

        # ---- Heads (linear; sigmoid applied downstream) ----
        self.heads = nn.ModuleDict(
            {s: nn.Conv3d(b, 1, 1) for s in self.structures}
        )

        self._init_weights()

    # -- init +------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for head in self.heads.values():
            nn.init.zeros_(head.bias)
            nn.init.normal_(head.weight, std=1e-3)

    # -- checkpoint helper -------------------------------------------------
    def _run(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    @staticmethod
    def _match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Trilinear-resize x to ref's spatial size if they differ (odd dims)."""
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode="trilinear",
                              align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        s0 = self._run(self.enc0, x)
        s1 = self._run(self.enc1, self.pool(s0))
        s2 = self._run(self.enc2, self.pool(s1))
        bott = self._run(self.bottleneck, self.pool(s2))

        d2 = self.up2(bott)
        d2 = self._run(self.dec2, torch.cat([self._match(d2, s2), s2], dim=1))
        d1 = self.up1(d2)
        d1 = self._run(self.dec1, torch.cat([self._match(d1, s1), s1], dim=1))
        d0 = self.up0(d1)
        d0 = self._run(self.dec0, torch.cat([self._match(d0, s0), s0], dim=1))

        return {s: head(d0) for s, head in self.heads.items()}

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {s: torch.sigmoid(v) for s, v in self.forward(x).items()}


def build_model(cfg, base_channels: int = None) -> MultiHeadUNet3D:
    """Construct the model from a Config (optionally overriding base_channels)."""
    return MultiHeadUNet3D(
        base_channels=cfg.base_channels if base_channels is None else base_channels,
        slope=cfg.leaky_slope,
        use_checkpointing=cfg.use_checkpointing,
    )


def load_model_from_ckpt(cfg, ckpt_state: dict, device) -> MultiHeadUNet3D:
    """Rebuild a model honouring the architecture stored in the checkpoint.

    Robust to OOM-ladder downgrades where the trained ``base_channels`` differs
    from the default config.
    """
    arch = ckpt_state.get("arch", {})
    model = build_model(cfg, base_channels=arch.get("base_channels")).to(device)
    model.load_state_dict(ckpt_state["model"])
    return model
