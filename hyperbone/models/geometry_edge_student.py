"""Track B: Geometry-only edge scorer (student model).

Learns to predict topology edges from local mesh patches around joints,
WITHOUT skinning weights at inference. Trained using GT adjacency and
optionally v4.1 teacher scores as targets.

Architecture:
  Per-joint: PointNet-lite encodes local mesh patch -> joint_mesh_token
  Per-edge:  concat(token_i, token_j, edge_geom_feats) -> MLP -> score
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetLite(nn.Module):
    """Lightweight PointNet for encoding local mesh patches.

    Input: [B, N_points, C_in] (xyz + optional normals/curvature)
    Output: [B, out_dim] global feature via max-pool
    """

    def __init__(self, in_channels: int = 6, out_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: [B, N, C]
        feat = self.mlp(points)     # [B, N, out_dim]
        pooled = feat.max(dim=1)[0]  # [B, out_dim]
        return pooled


class GeometryEdgeStudent(nn.Module):
    """Geometry-only edge scorer for Track B topology.

    For each candidate edge (i, j):
      1. Encode local mesh patches around joint i and j with PointNetLite
      2. Encode mesh corridor between i and j
      3. Concatenate with geometric edge features
      4. MLP to scalar edge score

    No skinning features used at any point.
    """

    def __init__(
        self,
        patch_in_channels: int = 6,
        patch_out_dim: int = 64,
        geom_feat_dim: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_encoder = PointNetLite(
            in_channels=patch_in_channels, out_dim=patch_out_dim,
        )
        self.corridor_encoder = PointNetLite(
            in_channels=patch_in_channels, out_dim=patch_out_dim,
        )

        edge_in = patch_out_dim * 3 + geom_feat_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        patch_i: torch.Tensor,
        patch_j: torch.Tensor,
        corridor: torch.Tensor,
        geom_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            patch_i: [B, N_pts, C] local mesh patch around joint i
            patch_j: [B, N_pts, C] local mesh patch around joint j
            corridor: [B, N_pts, C] mesh vertices along edge corridor
            geom_feats: [B, geom_feat_dim] edge geometry features

        Returns:
            score: [B] edge logit
        """
        tok_i = self.patch_encoder(patch_i)         # [B, D]
        tok_j = self.patch_encoder(patch_j)         # [B, D]
        tok_c = self.corridor_encoder(corridor)     # [B, D]
        x = torch.cat([tok_i, tok_j, tok_c, geom_feats], dim=-1)
        return self.edge_mlp(x).squeeze(-1)
