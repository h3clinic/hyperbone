"""
HyperBone3D v0 — Supervised internal 3D pose predictor.

Small ConvNet encoder + MLP heads for predicting canonical 3D joint positions
from RGB + mask + depth input.

Input: [B, 5, H, W] (RGB=3 + mask=1 + depth=1)
Output:
  joint_xyz_canonical: [B, J, 3]
  joint_visibility_logits: [B, J]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from ..pose3d.joint_map import NUM_JOINTS


class ConvBlock(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU + optional downsample."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class HyperBone3D(nn.Module):
    """
    HyperBone3D v0 model.

    Architecture:
      5-channel input → 4 ConvBlocks → global avg pool → MLP → joint predictions

    Designed to be small and debuggable (< 2M parameters).
    """

    def __init__(
        self,
        num_joints: int = NUM_JOINTS,
        input_channels: int = 5,
        base_channels: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_joints = num_joints

        # Encoder: 5ch → 64 → 128 → 256 → 512
        self.encoder = nn.Sequential(
            ConvBlock(input_channels, base_channels, stride=2),      # /2
            ConvBlock(base_channels, base_channels * 2, stride=2),   # /4
            ConvBlock(base_channels * 2, base_channels * 4, stride=2),  # /8
            ConvBlock(base_channels * 4, base_channels * 8, stride=2),  # /16
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        feat_dim = base_channels * 8  # 512

        # Joint position head: predicts [J, 3] canonical xyz
        self.xyz_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_joints * 3),
        )

        # Visibility head: predicts [J] logits
        self.vis_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_joints),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [B, 5, H, W] input (RGB + mask + depth)

        Returns:
            Dict with:
              joint_xyz_canonical: [B, J, 3]
              joint_visibility_logits: [B, J]
        """
        feat = self.encoder(x)                      # [B, 512, H/16, W/16]
        feat = self.global_pool(feat)               # [B, 512, 1, 1]
        feat = feat.flatten(1)                      # [B, 512]

        xyz = self.xyz_head(feat)                   # [B, J*3]
        xyz = xyz.view(-1, self.num_joints, 3)      # [B, J, 3]

        vis = self.vis_head(feat)                   # [B, J]

        return {
            "joint_xyz_canonical": xyz,
            "joint_visibility_logits": vis,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HyperBone3DLoss(nn.Module):
    """
    Combined loss for HyperBone3D training.

    Components:
      - Joint position loss (Huber/L1) on visible joints only
      - Visibility BCE loss
      - Bone length consistency loss
    """

    def __init__(
        self,
        pos_weight: float = 1.0,
        vis_weight: float = 0.5,
        bone_weight: float = 0.2,
        bone_edges=None,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.vis_weight = vis_weight
        self.bone_weight = bone_weight
        self.bone_edges = bone_edges  # [E, 2] tensor

        self.huber = nn.SmoothL1Loss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(
        self,
        pred_xyz: torch.Tensor,        # [B, J, 3]
        pred_vis: torch.Tensor,         # [B, J]
        gt_xyz: torch.Tensor,           # [B, J, 3]
        gt_visible: torch.Tensor,       # [B, J]
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss."""
        B, J, _ = pred_xyz.shape

        # Position loss on visible joints only
        vis_mask = gt_visible.unsqueeze(-1).expand_as(pred_xyz)  # [B, J, 3]
        pos_loss = self.huber(pred_xyz, gt_xyz)  # [B, J, 3]
        pos_loss = (pos_loss * vis_mask).sum() / (vis_mask.sum() + 1e-8)

        # Visibility loss
        vis_loss = self.bce(pred_vis, gt_visible)

        # Bone length consistency loss
        bone_loss = torch.tensor(0.0, device=pred_xyz.device)
        if self.bone_edges is not None and len(self.bone_edges) > 0:
            edges = self.bone_edges.to(pred_xyz.device)
            parent_idx = edges[:, 0]  # [E]
            child_idx = edges[:, 1]   # [E]

            # Predicted bone lengths
            pred_parent = pred_xyz[:, parent_idx]  # [B, E, 3]
            pred_child = pred_xyz[:, child_idx]    # [B, E, 3]
            pred_lengths = torch.norm(pred_child - pred_parent, dim=-1)  # [B, E]

            # GT bone lengths
            gt_parent = gt_xyz[:, parent_idx]
            gt_child = gt_xyz[:, child_idx]
            gt_lengths = torch.norm(gt_child - gt_parent, dim=-1)  # [B, E]

            # Only compute for bones where both joints are visible
            parent_vis = gt_visible[:, parent_idx]  # [B, E]
            child_vis = gt_visible[:, child_idx]    # [B, E]
            bone_vis = parent_vis * child_vis       # [B, E]

            bone_err = torch.abs(pred_lengths - gt_lengths)  # [B, E]
            bone_loss = (bone_err * bone_vis).sum() / (bone_vis.sum() + 1e-8)

        total = (self.pos_weight * pos_loss +
                 self.vis_weight * vis_loss +
                 self.bone_weight * bone_loss)

        return {
            "total": total,
            "pos_loss": pos_loss,
            "vis_loss": vis_loss,
            "bone_loss": bone_loss,
        }
