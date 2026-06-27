"""
HyperNodeNet v0: small encoder-decoder CNN for general HyperNode prediction.

Architecture:
- Encoder: 4 conv blocks with downsampling
- Decoder: 4 conv blocks with upsampling + skip connections
- Dense heads: node_type_heatmaps, radius_map
- Graph token head: active, xy, xyz, node_type, confidence, edges
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbone.hypernodes.dataset import NUM_NODE_TYPES, NUM_EDGE_TYPES


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class HyperNodeNet(nn.Module):
    """Small U-Net with graph token head for HyperNode prediction."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        max_nodes: int = 64,
        num_node_types: int = NUM_NODE_TYPES,
        num_edge_types: int = NUM_EDGE_TYPES,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        C = base_channels

        # Encoder
        self.enc1 = ConvBlock(in_channels, C)
        self.enc2 = ConvBlock(C, C * 2)
        self.enc3 = ConvBlock(C * 2, C * 4)
        self.enc4 = ConvBlock(C * 4, C * 8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(C * 8, C * 8)

        # Decoder
        self.up4 = nn.ConvTranspose2d(C * 8, C * 8, 2, stride=2)
        self.dec4 = ConvBlock(C * 16, C * 8)
        self.up3 = nn.ConvTranspose2d(C * 8, C * 4, 2, stride=2)
        self.dec3 = ConvBlock(C * 8, C * 4)
        self.up2 = nn.ConvTranspose2d(C * 4, C * 2, 2, stride=2)
        self.dec2 = ConvBlock(C * 4, C * 2)
        self.up1 = nn.ConvTranspose2d(C * 2, C, 2, stride=2)
        self.dec1 = ConvBlock(C * 2, C)

        # Dense heads
        self.heatmap_head = nn.Conv2d(C, num_node_types, 1)
        self.radius_head = nn.Conv2d(C, 1, 1)

        # Graph token head (from bottleneck features)
        # Global pool bottleneck -> FC
        bottleneck_dim = C * 8
        token_hidden = 256
        self.token_pool = nn.AdaptiveAvgPool2d(1)
        self.token_fc = nn.Sequential(
            nn.Linear(bottleneck_dim, token_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(token_hidden, token_hidden),
            nn.ReLU(inplace=True),
        )
        # Per-node predictions
        self.active_head = nn.Linear(token_hidden, max_nodes)
        self.xy_head = nn.Linear(token_hidden, max_nodes * 2)
        self.xyz_head = nn.Linear(token_hidden, max_nodes * 3)
        self.node_type_head = nn.Linear(token_hidden, max_nodes * num_node_types)
        self.confidence_head = nn.Linear(token_hidden, max_nodes)

        # Edge predictions (from node embeddings)
        self.node_embed = nn.Linear(token_hidden, max_nodes * 32)
        self.edge_active_fc = nn.Linear(64, 1)
        self.edge_type_fc = nn.Linear(64, num_edge_types)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B = x.shape[0]
        N = self.max_nodes

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        bn = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Dense heads
        heatmaps = self.heatmap_head(d1)  # [B, T, H, W]
        radius_map = self.radius_head(d1)  # [B, 1, H, W]

        # Graph token head
        pooled = self.token_pool(bn).flatten(1)  # [B, C*8]
        token_feat = self.token_fc(pooled)  # [B, 256]

        active_logits = self.active_head(token_feat)  # [B, N]
        node_xy = torch.sigmoid(self.xy_head(token_feat).view(B, N, 2))  # [B, N, 2]
        node_xyz = self.xyz_head(token_feat).view(B, N, 3)  # [B, N, 3]
        node_type_logits = self.node_type_head(token_feat).view(
            B, N, self.num_node_types
        )  # [B, N, T]
        node_confidence = torch.sigmoid(
            self.confidence_head(token_feat)
        )  # [B, N]

        # Edge predictions
        node_emb = self.node_embed(token_feat).view(B, N, 32)  # [B, N, 32]
        # Pairwise: concat embeddings for all pairs
        emb_i = node_emb.unsqueeze(2).expand(B, N, N, 32)
        emb_j = node_emb.unsqueeze(1).expand(B, N, N, 32)
        pair_feat = torch.cat([emb_i, emb_j], dim=-1)  # [B, N, N, 64]

        edge_logits = self.edge_active_fc(pair_feat).squeeze(-1)  # [B, N, N]
        edge_type_logits = self.edge_type_fc(pair_feat)  # [B, N, N, E]

        return {
            "heatmaps": heatmaps,
            "radius_map": radius_map,
            "active_logits": active_logits,
            "node_xy": node_xy,
            "node_xyz": node_xyz,
            "node_type_logits": node_type_logits,
            "node_confidence": node_confidence,
            "edge_logits": edge_logits,
            "edge_type_logits": edge_type_logits,
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
