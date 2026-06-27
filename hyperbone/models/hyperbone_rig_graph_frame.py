"""
Graph-token frame model for topology-aware rig prediction.

Instead of predicting fixed joint slots, this model predicts a SET of nodes
with positions, types, and edges. Training uses Hungarian matching to align
predicted nodes with GT joints regardless of topology.

Input: RGB/mask/depth [B, 5, H, W]
Output:
- node_active [B, N] — probability each slot is a real node
- node_xyz [B, N, 3] — 3D position per node
- node_type [B, N, T] — type logits per node
- edge_logits [B, N, N] — pairwise edge probability
- node_conf [B, N] — confidence
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphTokenFrameModel(nn.Module):
    """
    Predicts a variable-size rig graph from a single image.

    Architecture:
    1. CNN encoder → global + spatial features
    2. Learned node queries (like DETR object queries)
    3. Cross-attention from queries to image features
    4. Per-node heads: xyz, active, type
    5. Pairwise edge head
    """

    def __init__(self, in_channels: int = 5, max_nodes: int = 128,
                 n_node_types: int = 6, base_dim: int = 64,
                 n_query_layers: int = 3):
        super().__init__()
        self.max_nodes = max_nodes
        self.n_node_types = n_node_types
        self.feat_dim = base_dim * 4  # 256

        # Image encoder
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, base_dim, stride=2),    # /2
            self._conv_block(base_dim, base_dim * 2, stride=2),   # /4
            self._conv_block(base_dim * 2, base_dim * 4, stride=2),  # /8
            self._conv_block(base_dim * 4, self.feat_dim, stride=2),  # /16
        )

        # Spatial position encoding for image features
        self.feat_proj = nn.Conv2d(self.feat_dim, self.feat_dim, 1)

        # Learned node queries (like DETR)
        self.node_queries = nn.Embedding(max_nodes, self.feat_dim)

        # Cross-attention layers (query nodes attend to image features)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(self.feat_dim, num_heads=4, batch_first=True)
            for _ in range(n_query_layers)
        ])
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(self.feat_dim) for _ in range(n_query_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim * 2),
                nn.GELU(),
                nn.Linear(self.feat_dim * 2, self.feat_dim),
            ) for _ in range(n_query_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(self.feat_dim) for _ in range(n_query_layers)
        ])

        # Global context
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Per-node prediction heads
        self.xyz_head = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim // 2, 3),
        )
        self.active_head = nn.Linear(self.feat_dim, 1)
        self.type_head = nn.Linear(self.feat_dim, n_node_types)
        self.conf_head = nn.Linear(self.feat_dim, 1)

        # Edge prediction: pairwise bilinear
        self.edge_proj_a = nn.Linear(self.feat_dim, self.feat_dim // 2)
        self.edge_proj_b = nn.Linear(self.feat_dim, self.feat_dim // 2)
        self.edge_bias = nn.Parameter(torch.zeros(1))

    def _conv_block(self, in_c, out_c, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> dict:
        B = x.shape[0]
        device = x.device

        # Encode image
        feat_map = self.encoder(x)  # [B, D, H/16, W/16]
        feat_map = self.feat_proj(feat_map)

        # Flatten spatial dims for attention: [B, HW, D]
        feat_flat = feat_map.flatten(2).permute(0, 2, 1)  # [B, S, D]

        # Node queries: [B, N, D]
        queries = self.node_queries.weight.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention layers
        for attn, norm, ffn, ffn_norm in zip(
            self.cross_attn_layers, self.cross_norms,
            self.ffn_layers, self.ffn_norms
        ):
            # Cross attention: queries attend to image features
            q_out, _ = attn(queries, feat_flat, feat_flat)
            queries = norm(queries + q_out)
            # FFN
            queries = ffn_norm(queries + ffn(queries))

        # node_tokens: [B, N, D]
        node_tokens = queries

        # Per-node predictions
        node_xyz = self.xyz_head(node_tokens)  # [B, N, 3]
        node_active_logits = self.active_head(node_tokens).squeeze(-1)  # [B, N]
        node_type_logits = self.type_head(node_tokens)  # [B, N, T]
        node_conf = torch.sigmoid(self.conf_head(node_tokens).squeeze(-1))  # [B, N]

        # Edge prediction: bilinear
        ea = self.edge_proj_a(node_tokens)  # [B, N, D/2]
        eb = self.edge_proj_b(node_tokens)  # [B, N, D/2]
        edge_logits = torch.bmm(ea, eb.transpose(1, 2)) + self.edge_bias  # [B, N, N]

        return {
            "node_xyz": node_xyz,
            "node_active_logits": node_active_logits,
            "node_active": torch.sigmoid(node_active_logits),
            "node_type_logits": node_type_logits,
            "node_conf": node_conf,
            "edge_logits": edge_logits,
        }
