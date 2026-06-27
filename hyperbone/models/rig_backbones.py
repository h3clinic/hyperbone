"""
Backbone modules for static rig prediction.

Provides:
- PointNetBackbone (baseline)
- DGCNNBackbone (EdgeConv with kNN graph)

Both backbones return:
- global_feature: [B, D]
- point_features: [B, P, D]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """kNN on point features.

    Args:
        x: [B, C, P]

    Returns:
        idx: [B, P, k]
    """
    # Pairwise distance in feature space
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # [B, P, P]
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # [B, 1, P]
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # [B, P, P]
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x: torch.Tensor, k: int = 20) -> torch.Tensor:
    """Build edge features for EdgeConv.

    Args:
        x: [B, C, P]

    Returns:
        edge_feat: [B, 2C, P, k] with concat(x_i, x_j - x_i)
    """
    B, C, P = x.shape
    idx = knn(x, k=k)  # [B, P, k]

    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * P
    idx = idx + idx_base
    idx = idx.view(-1)

    x_t = x.transpose(2, 1).contiguous()  # [B, P, C]
    feature = x_t.view(B * P, C)[idx, :].view(B, P, k, C)  # neighbors
    x_i = x_t.view(B, P, 1, C).repeat(1, 1, k, 1)

    edge_feat = torch.cat((x_i, feature - x_i), dim=3).permute(0, 3, 1, 2).contiguous()
    return edge_feat


class PointNetBackbone(nn.Module):
    """PointNet baseline backbone."""

    def __init__(self, in_channels: int = 3, feat_dim: int = 512):
        super().__init__()
        self.feat_dim = feat_dim

        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, P, C = x.shape
        x_flat = x.reshape(B * P, C)

        point_feat = self.mlp1(x_flat).reshape(B, P, 256)
        high_feat = self.mlp2(point_feat.reshape(B * P, 256)).reshape(B, P, self.feat_dim)
        global_feat = high_feat.max(dim=1)[0]

        return global_feat, high_feat


class DGCNNBackbone(nn.Module):
    """DGCNN EdgeConv backbone for local geometry encoding."""

    def __init__(self, in_channels: int = 3, feat_dim: int = 512, k: int = 20):
        super().__init__()
        self.k = k
        self.feat_dim = feat_dim

        self.ec1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.ec2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.ec3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.ec4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.point_proj = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.global_proj = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, P, C] -> [B, C, P]
        x = x.transpose(2, 1).contiguous()

        x1 = self.ec1(get_graph_feature(x, k=self.k)).max(dim=-1)[0]  # [B,64,P]
        x2 = self.ec2(get_graph_feature(x1, k=self.k)).max(dim=-1)[0]  # [B,64,P]
        x3 = self.ec3(get_graph_feature(x2, k=self.k)).max(dim=-1)[0]  # [B,128,P]
        x4 = self.ec4(get_graph_feature(x3, k=self.k)).max(dim=-1)[0]  # [B,256,P]

        cat = torch.cat((x1, x2, x3, x4), dim=1)
        point_feat = self.point_proj(cat)  # [B, D, P]

        g_max = F.adaptive_max_pool1d(point_feat, 1).squeeze(-1)
        g_avg = F.adaptive_avg_pool1d(point_feat, 1).squeeze(-1)
        global_feat = self.global_proj(torch.cat([g_max, g_avg], dim=1))

        return global_feat, point_feat.transpose(2, 1).contiguous()
