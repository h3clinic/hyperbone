"""Track B v2: Edge-graph message passing student for topology.

Scores candidate edges with graph context instead of independently.
Each candidate edge becomes a node in a line graph; edges sharing a
joint endpoint are connected. Message passing lets each edge see its
neighbors before scoring.

Architecture:
  1. PointNetLite encodes mesh patches -> initial edge embeddings
  2. Build line graph from candidate edge connectivity
  3. 2-4 rounds of message passing over line graph
  4. Output context-aware edge logits

No skinning features at any point.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetLite(nn.Module):
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
        feat = self.mlp(points)
        return feat.max(dim=1)[0]


class EdgeMessagePassingBlock(nn.Module):
    """One round of message passing on the line graph.

    Uses max aggregation to preserve discriminative signal across
    high-degree neighborhoods. Pre-norm residual connection.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.msg_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return x

        x_normed = self.norm(x)
        src, dst = edge_index[0], edge_index[1]
        src_feat = x_normed[src]
        dst_feat = x_normed[dst]

        msg = self.msg_mlp(torch.cat([src_feat, dst_feat], dim=-1))

        # Max aggregation instead of mean — preserves discriminative signal
        fill = torch.full_like(x, float('-inf'))
        idx_exp = dst.unsqueeze(1).expand_as(msg)
        agg = fill.scatter_reduce(0, idx_exp, msg, reduce='amax',
                                  include_self=False)
        no_msg = (agg == float('-inf')).all(dim=-1)
        agg = torch.where(no_msg.unsqueeze(1), x_normed, agg)

        combined = torch.cat([x_normed, agg], dim=-1)
        update = self.update_mlp(combined)
        # Gated residual: let the model learn how much context to mix in
        g = self.gate(combined)
        return x + g * update


class GeometryEdgeGraphStudent(nn.Module):
    """Edge-graph message passing student for Track B v2.

    Builds a line graph from candidate edges, runs message passing,
    then scores edges with graph context.
    """

    def __init__(
        self,
        patch_in_channels: int = 6,
        patch_out_dim: int = 64,
        geom_feat_dim: int = 16,
        edge_dim: int = 128,
        n_mp_rounds: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_encoder = PointNetLite(patch_in_channels, patch_out_dim)
        self.corridor_encoder = PointNetLite(patch_in_channels, patch_out_dim)

        raw_edge_dim = patch_out_dim * 3 + geom_feat_dim
        self.edge_proj = nn.Sequential(
            nn.Linear(raw_edge_dim, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.ReLU(inplace=True),
        )

        self.mp_blocks = nn.ModuleList([
            EdgeMessagePassingBlock(edge_dim, dropout)
            for _ in range(n_mp_rounds)
        ])

        self.score_head = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(edge_dim, 1),
        )

    def forward(
        self,
        patch_i: torch.Tensor,
        patch_j: torch.Tensor,
        corridor: torch.Tensor,
        geom_feats: torch.Tensor,
        line_graph_edges: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            patch_i: [E, N_pts, C]
            patch_j: [E, N_pts, C]
            corridor: [E, N_pts, C]
            geom_feats: [E, geom_feat_dim]
            line_graph_edges: [2, M] line graph connectivity

        Returns:
            logits: [E] per-edge scores with graph context
        """
        tok_i = self.patch_encoder(patch_i)
        tok_j = self.patch_encoder(patch_j)
        tok_c = self.corridor_encoder(corridor)

        raw = torch.cat([tok_i, tok_j, tok_c, geom_feats], dim=-1)
        x = self.edge_proj(raw)

        for block in self.mp_blocks:
            x = block(x, line_graph_edges)

        return self.score_head(x).squeeze(-1)


def build_line_graph(edge_pairs: torch.Tensor) -> torch.Tensor:
    """Build line graph edges from candidate edge pairs.

    Two edge-nodes are connected if they share a joint endpoint.

    Args:
        edge_pairs: [E, 2] candidate edges (joint indices)

    Returns:
        line_graph_edges: [2, M] undirected line graph connectivity
    """
    E = edge_pairs.shape[0]
    if E <= 1:
        return torch.zeros(2, 0, dtype=torch.long, device=edge_pairs.device)

    joint_to_edges = {}
    for e_idx in range(E):
        i, j = int(edge_pairs[e_idx, 0]), int(edge_pairs[e_idx, 1])
        joint_to_edges.setdefault(i, []).append(e_idx)
        joint_to_edges.setdefault(j, []).append(e_idx)

    src, dst = [], []
    for joint_id, e_list in joint_to_edges.items():
        for a_idx in range(len(e_list)):
            for b_idx in range(a_idx + 1, len(e_list)):
                ea, eb = e_list[a_idx], e_list[b_idx]
                src.extend([ea, eb])
                dst.extend([eb, ea])

    if not src:
        return torch.zeros(2, 0, dtype=torch.long, device=edge_pairs.device)

    return torch.tensor([src, dst], dtype=torch.long, device=edge_pairs.device)
