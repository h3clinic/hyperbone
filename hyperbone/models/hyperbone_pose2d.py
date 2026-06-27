"""
HyperBonePose2D: explicit animal joint heatmap predictor.

Architecture:
  Encoder: 4 conv blocks with downsampling (stride 2)
  Decoder: 4 transposed conv blocks with upsampling
  Output heads: heatmaps [J, H, W] + visibility logits [J]

This is a U-Net-lite with skip connections for spatial precision.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample + concat skip + ConvBlock"""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class HyperBonePose2D(nn.Module):
    """
    U-Net-lite for animal joint heatmap prediction.

    Input: RGB crop [B, 3, H, W]
    Output:
      heatmaps: [B, J, H, W] - Gaussian peaks at joint locations
      visibility_logits: [B, J] - joint presence confidence
    """

    def __init__(
        self,
        num_joints: int = 19,
        base_channels: int = 32,
    ):
        super().__init__()
        self.num_joints = num_joints
        c = base_channels

        # Encoder
        self.enc1 = ConvBlock(3, c)        # -> [c, H, W]
        self.enc2 = ConvBlock(c, c * 2)    # -> [2c, H/2, W/2]
        self.enc3 = ConvBlock(c * 2, c * 4)  # -> [4c, H/4, W/4]
        self.enc4 = ConvBlock(c * 4, c * 8)  # -> [8c, H/8, W/8]

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(c * 8, c * 16)  # -> [16c, H/16, W/16]

        # Decoder
        self.dec4 = UpBlock(c * 16, c * 8, c * 8)   # -> [8c, H/8, W/8]
        self.dec3 = UpBlock(c * 8, c * 4, c * 4)    # -> [4c, H/4, W/4]
        self.dec2 = UpBlock(c * 4, c * 2, c * 2)    # -> [2c, H/2, W/2]
        self.dec1 = UpBlock(c * 2, c, c)             # -> [c, H, W]

        # Heatmap head: full resolution
        self.heatmap_head = nn.Conv2d(c, num_joints, 1)

        # Visibility head: global pool from bottleneck features
        self.vis_pool = nn.AdaptiveAvgPool2d(1)
        self.vis_head = nn.Sequential(
            nn.Linear(c * 16, c * 4),
            nn.ReLU(inplace=True),
            nn.Linear(c * 4, num_joints),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encoder
        e1 = self.enc1(x)           # [B, c, H, W]
        e2 = self.enc2(self.pool(e1))  # [B, 2c, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # [B, 4c, H/4, W/4]
        e4 = self.enc4(self.pool(e3))  # [B, 8c, H/8, W/8]

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # [B, 16c, H/16, W/16]

        # Decoder with skip connections
        d4 = self.dec4(b, e4)       # [B, 8c, H/8, W/8]
        d3 = self.dec3(d4, e3)      # [B, 4c, H/4, W/4]
        d2 = self.dec2(d3, e2)      # [B, 2c, H/2, W/2]
        d1 = self.dec1(d2, e1)      # [B, c, H, W]

        # Heatmap output
        heatmaps = torch.sigmoid(self.heatmap_head(d1))  # [B, J, H, W]

        # Visibility from bottleneck
        vis_feat = self.vis_pool(b).flatten(1)  # [B, 16c]
        visibility_logits = self.vis_head(vis_feat)  # [B, J]

        return {
            "heatmaps": heatmaps,
            "visibility_logits": visibility_logits,
        }


def soft_argmax_2d(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Differentiable 2D argmax from heatmaps.

    Input: [B, J, H, W]
    Output: [B, J, 2] - (x, y) normalized to [0, 1]
    """
    B, J, H, W = heatmaps.shape
    # Normalize heatmaps to probability distributions
    flat = heatmaps.view(B, J, -1)
    probs = F.softmax(flat * 10.0, dim=-1)  # temperature-scaled softmax
    probs = probs.view(B, J, H, W)

    # Create coordinate grids
    device = heatmaps.device
    yy = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).expand(B, J, H, W)
    xx = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).expand(B, J, H, W)

    # Weighted sum
    x_coords = (probs * xx).sum(dim=(2, 3))  # [B, J]
    y_coords = (probs * yy).sum(dim=(2, 3))  # [B, J]

    return torch.stack([x_coords, y_coords], dim=-1)  # [B, J, 2]


class HyperBonePose2DLoss(nn.Module):
    """
    Combined loss for pose2d:
    - Heatmap MSE (only for visible joints)
    - Visibility BCE
    - Optional coordinate regression loss via soft-argmax
    """

    def __init__(self, coord_weight: float = 1.0, vis_weight: float = 0.1):
        super().__init__()
        self.coord_weight = coord_weight
        self.vis_weight = vis_weight

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt_heatmaps: torch.Tensor,
        gt_visibility: torch.Tensor,
        gt_joints_xy: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        heatmaps = pred["heatmaps"]       # [B, J, H, W]
        vis_logits = pred["visibility_logits"]  # [B, J]

        B, J, H, W = heatmaps.shape

        # Heatmap loss: MSE only for visible joints
        vis_mask = gt_visibility.unsqueeze(-1).unsqueeze(-1)  # [B, J, 1, 1]
        heatmap_loss = F.mse_loss(heatmaps * vis_mask, gt_heatmaps * vis_mask)

        # Visibility loss
        vis_loss = F.binary_cross_entropy_with_logits(vis_logits, gt_visibility)

        # Coordinate loss via soft-argmax
        pred_coords = soft_argmax_2d(heatmaps)  # [B, J, 2]
        coord_mask = gt_visibility.unsqueeze(-1)  # [B, J, 1]
        coord_loss = F.smooth_l1_loss(
            pred_coords * coord_mask,
            gt_joints_xy * coord_mask,
        )

        total = heatmap_loss + self.vis_weight * vis_loss + self.coord_weight * coord_loss

        return {
            "total": total,
            "heatmap_loss": heatmap_loss,
            "vis_loss": vis_loss,
            "coord_loss": coord_loss,
        }
