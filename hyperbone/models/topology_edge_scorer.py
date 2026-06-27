from __future__ import annotations

import torch
import torch.nn as nn


class EdgeMLPResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class TopologyEdgeScorer(nn.Module):
    """Dedicated topology edge scorer with residual MLP blocks."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        node_local_dim: int,
        global_context_dim: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.node_local_dim = node_local_dim
        self.global_context_dim = global_context_dim

        pair_in = (
            edge_feature_dim
            + (2 * node_feature_dim)
            + (2 * node_local_dim)
            + global_context_dim
            + 1
        )

        self.input_proj = nn.Sequential(
            nn.Linear(pair_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [EdgeMLPResidualBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        base_scores: torch.Tensor,
        edge_features: torch.Tensor,
        node_features: torch.Tensor,
        node_local_features: torch.Tensor,
        global_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, n_nodes, _ = base_scores.shape

        node_i = node_features.unsqueeze(2).expand(batch_size, n_nodes, n_nodes, -1)
        node_j = node_features.unsqueeze(1).expand(batch_size, n_nodes, n_nodes, -1)

        local_i = node_local_features.unsqueeze(2).expand(batch_size, n_nodes, n_nodes, -1)
        local_j = node_local_features.unsqueeze(1).expand(batch_size, n_nodes, n_nodes, -1)

        if global_context is None:
            global_context = torch.zeros(
                (batch_size, self.global_context_dim),
                dtype=base_scores.dtype,
                device=base_scores.device,
            )
        global_ij = global_context.unsqueeze(1).unsqueeze(1).expand(batch_size, n_nodes, n_nodes, -1)

        x = torch.cat(
            [
                edge_features,
                node_i,
                node_j,
                local_i,
                local_j,
                global_ij,
                base_scores.unsqueeze(-1),
            ],
            dim=-1,
        )

        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return self.out_head(x).squeeze(-1)
