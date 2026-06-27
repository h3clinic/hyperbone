"""
HyperBone Static Rig Graph Model — predicts skeleton from point cloud.

Architecture:
- Backbone (PointNet or DGCNN) -> global + per-point features
- Joint decoder with query-to-point attention
- Edge predictor from joint tokens
- Skinning head (optional)
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbone.models.rig_backbones import PointNetBackbone, DGCNNBackbone


class JointDecoder(nn.Module):
    """Decode joints with query-to-point attention over local geometry."""

    def __init__(self, feat_dim: int = 512, max_joints: int = 128, query_dim: int = 128):
        super().__init__()
        self.max_joints = max_joints
        self.query_dim = query_dim

        self.node_queries = nn.Parameter(torch.randn(max_joints, query_dim) * 0.02)
        self.global_proj = nn.Linear(feat_dim, query_dim)
        self.point_proj = nn.Linear(feat_dim, query_dim)

        token_dim = query_dim * 3
        self.token_mlp = nn.Sequential(
            nn.Linear(token_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.pos_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

        self.active_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, global_feat: torch.Tensor, point_feat: torch.Tensor) -> tuple:
        """
        Args:
            global_feat: [B, feat_dim]
            point_feat: [B, P, feat_dim]

        Returns:
            joint_pos: [B, J, 3]
            active_logits: [B, J]
            node_tokens: [B, J, 256]
        """
        B, P, D = point_feat.shape
        J = self.max_joints

        q = self.node_queries.unsqueeze(0).expand(B, J, -1)  # [B, J, Q]
        g = self.global_proj(global_feat).unsqueeze(1).expand(B, J, -1)  # [B, J, Q]

        p = self.point_proj(point_feat)  # [B, P, Q]
        attn = torch.matmul(q, p.transpose(1, 2)) / (self.query_dim ** 0.5)  # [B, J, P]
        attn = torch.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, p)  # [B, J, Q]

        token = self.token_mlp(torch.cat([q, g, ctx], dim=-1))
        joint_pos = self.pos_head(token)
        active_logits = self.active_head(token).squeeze(-1)
        return joint_pos, active_logits, token


class EdgePredictor(nn.Module):
    """Predict adjacency from joint features."""

    def __init__(self, joint_feat_dim: int = 64, node_token_dim: int = 256, max_joints: int = 128):
        super().__init__()
        self.max_joints = max_joints

        # Project joint positions to features
        self.joint_proj = nn.Sequential(
            nn.Linear(3 + node_token_dim, 64),
            nn.ReLU(),
            nn.Linear(64, joint_feat_dim),
        )

        # Pairwise edge predictor
        self.edge_mlp = nn.Sequential(
            nn.Linear(joint_feat_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, joint_pos: torch.Tensor, node_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joint_pos: [B, J, 3]
            node_tokens: [B, J, T]

        Returns:
            adj_logits: [B, J, J]
        """
        B, J, _ = joint_pos.shape

        joint_in = torch.cat([joint_pos, node_tokens], dim=-1)
        joint_feat = self.joint_proj(joint_in)  # [B, J, D]

        # Pairwise concatenation (efficient: broadcast)
        fi = joint_feat.unsqueeze(2).expand(B, J, J, -1)  # [B, J, J, D]
        fj = joint_feat.unsqueeze(1).expand(B, J, J, -1)  # [B, J, J, D]
        pair_feat = torch.cat([fi, fj], dim=-1)  # [B, J, J, 2D]

        # Predict edges
        adj_logits = self.edge_mlp(pair_feat).squeeze(-1)  # [B, J, J]

        # Symmetrize
        adj_logits = (adj_logits + adj_logits.transpose(1, 2)) / 2

        return adj_logits


class SkinningHead(nn.Module):
    """Predict per-point skinning weights over joints."""

    def __init__(self, point_feat_dim: int = 512, joint_feat_dim: int = 3, max_joints: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(point_feat_dim + joint_feat_dim * max_joints, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, max_joints),
        )
        self.max_joints = max_joints

    def forward(self, point_feat: torch.Tensor, joint_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_feat: [B, P, 256]
            joint_pos: [B, J, 3]

        Returns:
            skin_logits: [B, P, J]
        """
        B, P, _ = point_feat.shape

        # Broadcast joint positions to each point
        joint_flat = joint_pos.reshape(B, -1)  # [B, J*3]
        joint_broadcast = joint_flat.unsqueeze(1).expand(B, P, -1)  # [B, P, J*3]

        combined = torch.cat([point_feat, joint_broadcast], dim=-1)  # [B, P, 256 + J*3]
        skin_logits = self.mlp(combined)  # [B, P, J]

        return skin_logits


class HyperBoneStaticRigModel(nn.Module):
    """
    Full static rig prediction model.

    Input: point cloud [B, P, 3]
    Output: joint positions, active mask, adjacency, skinning weights
    """

    def __init__(
        self,
        in_channels: int = 3,
        feat_dim: int = 512,
        max_joints: int = 128,
        predict_skinning: bool = True,
        backbone: str = "pointnet",
        knn_k: int = 20,
    ):
        super().__init__()
        self.max_joints = max_joints
        self.predict_skinning = predict_skinning
        self.backbone_name = backbone

        if backbone == "pointnet":
            self.encoder = PointNetBackbone(in_channels=in_channels, feat_dim=feat_dim)
        elif backbone == "dgcnn":
            self.encoder = DGCNNBackbone(in_channels=in_channels, feat_dim=feat_dim, k=knn_k)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.joint_decoder = JointDecoder(feat_dim=feat_dim, max_joints=max_joints)
        self.edge_predictor = EdgePredictor(max_joints=max_joints)

        if predict_skinning:
            self.skinning_head = SkinningHead(
                point_feat_dim=feat_dim,
                joint_feat_dim=3,
                max_joints=max_joints,
            )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pc = batch["pc"]  # [B, P, 3+]

        # Only use xyz for encoder
        pc_xyz = pc[:, :, :3]

        # Encode
        global_feat, point_feat = self.encoder(pc_xyz)

        # Decode joints with local point attention
        joint_pos, active_logits, node_tokens = self.joint_decoder(global_feat, point_feat)

        # Predict edges
        adj_logits = self.edge_predictor(joint_pos, node_tokens)

        result = {
            "joint_pos": joint_pos,
            "active_logits": active_logits,
            "adj_logits": adj_logits,
        }

        # Predict skinning
        if self.predict_skinning:
            skin_logits = self.skinning_head(point_feat, joint_pos)
            result["skin_logits"] = skin_logits

        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
