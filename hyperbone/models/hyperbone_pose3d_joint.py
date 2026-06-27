"""
HyperBonePose3D-Joint: heatmap-conditioned 3D canonical joint predictor.

Architecture:
  1. Encoder-decoder backbone produces feature maps
  2. Heatmap head predicts per-joint spatial probability [J, H, W]
  3. Soft-argmax extracts 2D joint locations from heatmaps
  4. Feature sampling at each predicted 2D location (per-joint)
  5. Per-joint MLP lifts sampled features -> 3D xyz
  6. Visibility + confidence heads

This ensures the model must LOCALIZE joints in the image before predicting 3D.
It cannot just regress a global pose vector from pooled features.

Input: cropped animal image [B, C, H, W]
Output:
  joint_xyz_canonical: [B, J, 3]       (THE PRODUCT)
  joint_visibility_logits: [B, J]
  joint_confidence: [B, J]
  joint_heatmaps_2d: [B, J, Hm, Wm]   (auxiliary, proves image grounding)
  projected_joint_xy: [B, J, 2]        (soft-argmax of heatmaps, normalized)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from ..pose2d.quadruped_schema import NUM_JOINTS_2D, QUADRUPED_BONES_2D


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN -> ReLU with stride."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsample + skip concat + Conv for decoder."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


def spatial_soft_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Differentiable soft-argmax over spatial heatmaps.

    Input: [B, J, H, W] (unnormalized logits or probabilities)
    Output: [B, J, 2] (x, y) normalized to [0, 1]
    """
    B, J, H, W = heatmaps.shape
    # Convert to probabilities
    flat = heatmaps.view(B, J, -1)
    probs = F.softmax(flat, dim=-1)
    probs = probs.view(B, J, H, W)

    device = heatmaps.device
    # Coordinate grids [0, 1]
    yy = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).expand(B, J, H, W)
    xx = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).expand(B, J, H, W)

    x_coords = (probs * xx).sum(dim=(2, 3))  # [B, J]
    y_coords = (probs * yy).sum(dim=(2, 3))  # [B, J]

    return torch.stack([x_coords, y_coords], dim=-1)  # [B, J, 2]


def sample_features_at_joints(
    feature_map: torch.Tensor,   # [B, C, H, W]
    joint_xy: torch.Tensor,      # [B, J, 2] normalized [0,1]
) -> torch.Tensor:
    """
    Bilinear sample features at predicted joint locations.

    Returns: [B, J, C]
    """
    B, C, H, W = feature_map.shape
    J = joint_xy.shape[1]

    # Convert from [0,1] to grid_sample coords [-1, 1]
    grid = joint_xy * 2 - 1  # [B, J, 2]
    grid = grid.unsqueeze(2)  # [B, J, 1, 2] for grid_sample

    # grid_sample expects [B, C, H_out, W_out] with grid [B, H_out, W_out, 2]
    # Reshape: treat J as H_out, 1 as W_out
    sampled = F.grid_sample(
        feature_map, grid, mode='bilinear', align_corners=True
    )  # [B, C, J, 1]

    return sampled.squeeze(-1).permute(0, 2, 1)  # [B, J, C]


class HyperBonePose3DJoint(nn.Module):
    """
    Heatmap-conditioned 3D joint predictor.

    Pipeline:
      1. Encode image -> multi-scale features
      2. Decode to heatmap resolution -> [J, Hm, Wm] heatmaps
      3. Soft-argmax -> 2D joint locations [J, 2]
      4. Sample features at each joint location -> per-joint features
      5. Per-joint MLP -> 3D xyz lift
      6. Visibility + confidence from per-joint features

    This forces the model to LOCALIZE before LIFTING.
    """

    def __init__(
        self,
        num_joints: int = NUM_JOINTS_2D,
        input_channels: int = 3,
        base_channels: int = 48,
        hidden_dim: int = 256,
        use_aux_heatmaps: bool = True,
        heatmap_resolution: int = 48,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.use_aux_heatmaps = use_aux_heatmaps
        self.heatmap_resolution = heatmap_resolution
        c = base_channels

        # Encoder
        self.enc1 = ConvBlock(input_channels, c, stride=2)        # /2
        self.enc2 = ConvBlock(c, c * 2, stride=2)                 # /4
        self.enc3 = ConvBlock(c * 2, c * 4, stride=2)             # /8
        self.enc4 = ConvBlock(c * 4, c * 8, stride=2)             # /16

        # Decoder (produces feature map for heatmaps + feature sampling)
        self.dec3 = UpBlock(c * 8, c * 4, c * 4)   # /8
        self.dec2 = UpBlock(c * 4, c * 2, c * 2)   # /4
        self.dec1 = UpBlock(c * 2, c, c)            # /2

        # Heatmap head: produces [J, H/2, W/2] logits
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, num_joints, 1),
        )

        # Feature map for sampling (decoder output at /2 resolution)
        feat_ch = c  # feature channels at sampling resolution

        # Per-joint 3D lift MLP (input: sampled features at joint location)
        self.xyz_lift = nn.Sequential(
            nn.Linear(feat_ch, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),  # xyz per joint
        )

        # Per-joint visibility (from sampled features)
        self.vis_head = nn.Sequential(
            nn.Linear(feat_ch, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Per-joint confidence (from sampled features + heatmap peak value)
        self.conf_head = nn.Sequential(
            nn.Linear(feat_ch + 1, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.shape[0]

        # Encoder
        e1 = self.enc1(x)     # [B, c, H/2, W/2]
        e2 = self.enc2(e1)    # [B, 2c, H/4, W/4]
        e3 = self.enc3(e2)    # [B, 4c, H/8, W/8]
        e4 = self.enc4(e3)    # [B, 8c, H/16, W/16]

        # Decoder
        d3 = self.dec3(e4, e3)  # [B, 4c, H/8, W/8]
        d2 = self.dec2(d3, e2)  # [B, 2c, H/4, W/4]
        d1 = self.dec1(d2, e1)  # [B, c, H/2, W/2]

        # Heatmap logits
        heatmap_logits = self.heatmap_head(d1)  # [B, J, H/2, W/2]

        # Resize heatmaps to standard resolution
        if heatmap_logits.shape[2] != self.heatmap_resolution:
            heatmap_logits = F.interpolate(
                heatmap_logits,
                size=(self.heatmap_resolution, self.heatmap_resolution),
                mode='bilinear', align_corners=False
            )

        # Soft-argmax: extract 2D joint locations from heatmap
        joint_xy = spatial_soft_argmax(heatmap_logits)  # [B, J, 2]

        # Resize feature map for sampling to same as heatmap
        feat_for_sampling = F.interpolate(
            d1, size=(self.heatmap_resolution, self.heatmap_resolution),
            mode='bilinear', align_corners=False
        )  # [B, c, Hm, Wm]

        # Sample features at each predicted joint location
        joint_features = sample_features_at_joints(feat_for_sampling, joint_xy)  # [B, J, c]

        # Per-joint 3D lift
        xyz = self.xyz_lift(joint_features)  # [B, J, 3]

        # Per-joint visibility
        vis = self.vis_head(joint_features).squeeze(-1)  # [B, J]

        # Per-joint confidence (include heatmap peak value as input)
        heatmap_probs = torch.sigmoid(heatmap_logits)
        # Get peak value at each joint's predicted location
        peak_vals = sample_features_at_joints(
            heatmap_probs, joint_xy
        )  # [B, J, J] - we want diagonal
        # Actually, sample from the per-joint heatmap at that joint's location
        # Simpler: take max of each joint's heatmap channel
        peak_max = heatmap_probs.flatten(2).max(dim=-1).values  # [B, J]

        conf_input = torch.cat([joint_features, peak_max.unsqueeze(-1)], dim=-1)  # [B, J, c+1]
        conf = torch.sigmoid(self.conf_head(conf_input).squeeze(-1))  # [B, J]

        result = {
            "joint_xyz_canonical": xyz,            # [B, J, 3] - THE PRODUCT
            "joint_visibility_logits": vis,        # [B, J]
            "joint_confidence": conf,              # [B, J]
            "projected_joint_xy": joint_xy,        # [B, J, 2] - from soft-argmax
            "joint_heatmaps_2d": heatmap_probs,    # [B, J, Hm, Wm] - auxiliary
        }

        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HyperBonePose3DJointLoss(nn.Module):
    """
    Combined loss forcing image-grounded 3D joint prediction.

    The model cannot get low total loss without both:
    - Correct 3D joint positions
    - Correct 2D joint localization (heatmap peaks on the animal)
    """

    def __init__(
        self,
        pos_weight: float = 1.0,
        heatmap_weight: float = 1.0,
        coord2d_weight: float = 0.5,
        vis_weight: float = 0.2,
        bone_weight: float = 0.2,
        bone_edges=None,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.heatmap_weight = heatmap_weight
        self.coord2d_weight = coord2d_weight
        self.vis_weight = vis_weight
        self.bone_weight = bone_weight

        if bone_edges is None:
            bone_edges = QUADRUPED_BONES_2D
        self.register_buffer(
            "bone_edges",
            torch.tensor(bone_edges, dtype=torch.long)
        )

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt_xyz: torch.Tensor,              # [B, J, 3]
        gt_visible: torch.Tensor,          # [B, J]
        gt_joints_2d: Optional[torch.Tensor] = None,   # [B, J, 2] normalized [0,1]
        gt_heatmaps: Optional[torch.Tensor] = None,    # [B, J, H, W]
    ) -> Dict[str, torch.Tensor]:

        pred_xyz = pred["joint_xyz_canonical"]      # [B, J, 3]
        pred_vis = pred["joint_visibility_logits"]  # [B, J]
        pred_2d = pred["projected_joint_xy"]        # [B, J, 2]
        pred_hm = pred["joint_heatmaps_2d"]         # [B, J, Hm, Wm]
        B, J, _ = pred_xyz.shape

        # === PRIMARY: 3D position loss (visible joints only) ===
        vis_mask_3d = gt_visible.unsqueeze(-1).expand_as(pred_xyz)
        pos_loss = F.smooth_l1_loss(pred_xyz * vis_mask_3d, gt_xyz * vis_mask_3d, reduction='sum')
        pos_loss = pos_loss / (vis_mask_3d.sum() + 1e-8)

        # === PRIMARY: Heatmap loss (forces image grounding) ===
        heatmap_loss = torch.tensor(0.0, device=pred_xyz.device)
        if gt_heatmaps is not None:
            if pred_hm.shape[2:] != gt_heatmaps.shape[2:]:
                gt_heatmaps = F.interpolate(
                    gt_heatmaps, size=pred_hm.shape[2:],
                    mode='bilinear', align_corners=False
                )
            hm_mask = gt_visible.unsqueeze(-1).unsqueeze(-1)  # [B, J, 1, 1]
            heatmap_loss = F.mse_loss(pred_hm * hm_mask, gt_heatmaps * hm_mask)

        # === PRIMARY: 2D coordinate loss (soft-argmax must match GT projection) ===
        coord2d_loss = torch.tensor(0.0, device=pred_xyz.device)
        if gt_joints_2d is not None:
            vis_mask_2d = gt_visible.unsqueeze(-1)  # [B, J, 1]
            coord2d_loss = F.smooth_l1_loss(
                pred_2d * vis_mask_2d, gt_joints_2d * vis_mask_2d, reduction='sum'
            )
            coord2d_loss = coord2d_loss / (vis_mask_2d.sum() + 1e-8)

        # === Visibility BCE ===
        vis_loss = F.binary_cross_entropy_with_logits(pred_vis, gt_visible)

        # === Bone length consistency ===
        bone_loss = torch.tensor(0.0, device=pred_xyz.device)
        edges = self.bone_edges.to(pred_xyz.device)
        if len(edges) > 0:
            pi, ci = edges[:, 0], edges[:, 1]
            pred_bl = torch.norm(pred_xyz[:, ci] - pred_xyz[:, pi], dim=-1)
            gt_bl = torch.norm(gt_xyz[:, ci] - gt_xyz[:, pi], dim=-1)
            bone_vis = gt_visible[:, pi] * gt_visible[:, ci]
            bone_err = torch.abs(pred_bl - gt_bl)
            bone_loss = (bone_err * bone_vis).sum() / (bone_vis.sum() + 1e-8)

        total = (
            self.pos_weight * pos_loss +
            self.heatmap_weight * heatmap_loss +
            self.coord2d_weight * coord2d_loss +
            self.vis_weight * vis_loss +
            self.bone_weight * bone_loss
        )

        return {
            "total": total,
            "pos_loss": pos_loss,
            "heatmap_loss": heatmap_loss,
            "coord2d_loss": coord2d_loss,
            "vis_loss": vis_loss,
            "bone_loss": bone_loss,
        }


def project_canonical_to_crop(
    xyz_canonical: torch.Tensor,
    crop_size: int = 192,
) -> torch.Tensor:
    """
    Project canonical 3D joints to 2D crop coordinates (approximate orthographic).
    """
    xy = xyz_canonical[..., :2].clone()
    xy = (xy * 0.5 + 0.5) * crop_size
    xy[..., 1] = crop_size - xy[..., 1]
    return xy


def project_crop_to_fullframe(
    crop_xy: torch.Tensor,
    bbox_xywh: torch.Tensor,
    crop_size: int = 192,
) -> torch.Tensor:
    """
    Map crop-space 2D coordinates back to full-frame image coordinates.
    """
    if bbox_xywh.dim() == 1:
        bx, by, bw, bh = bbox_xywh
        full_x = crop_xy[..., 0] / crop_size * bw + bx
        full_y = crop_xy[..., 1] / crop_size * bh + by
        return torch.stack([full_x, full_y], dim=-1)
    else:
        bx = bbox_xywh[..., 0:1].unsqueeze(-1)
        by = bbox_xywh[..., 1:2].unsqueeze(-1)
        bw = bbox_xywh[..., 2:3].unsqueeze(-1)
        bh = bbox_xywh[..., 3:4].unsqueeze(-1)
        full_x = crop_xy[..., 0:1] / crop_size * bw + bx
        full_y = crop_xy[..., 1:2] / crop_size * bh + by
        return torch.cat([full_x, full_y], dim=-1)
