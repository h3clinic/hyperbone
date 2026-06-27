from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbone.models.rig_backbones import PointNetBackbone, DGCNNBackbone
from hyperbone.models.hyperbone_rig_graph_static import JointDecoder
from hyperbone.rigs.root_features import compute_root_structural_features, ROOT_STRUCTURAL_FEATURE_DIM


class ParentPointerHead(nn.Module):
    def __init__(self, feat_dim: int = 512, max_joints: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.max_joints = max_joints
        self.root_idx = max_joints

        self.node_proj = nn.Sequential(
            nn.Linear(256 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.candidate_proj = nn.Sequential(
            nn.Linear(256 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.root_head = nn.Linear(hidden_dim, 1)
        self.parent_pair = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.offset_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.edge_confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, joint_pos: torch.Tensor, node_tokens: torch.Tensor) -> dict:
        B, J, _ = joint_pos.shape
        node_feat = self.node_proj(torch.cat([joint_pos, node_tokens], dim=-1))
        cand_feat = self.candidate_proj(torch.cat([joint_pos, node_tokens], dim=-1))

        child = node_feat.unsqueeze(2).expand(B, J, J, -1)
        parent = cand_feat.unsqueeze(1).expand(B, J, J, -1)
        rel = joint_pos.unsqueeze(2) - joint_pos.unsqueeze(1)
        pair = torch.cat([child, parent, rel], dim=-1)
        pair_logits = self.parent_pair(pair).squeeze(-1)

        # Prevent self-parent.
        diag = torch.eye(J, device=joint_pos.device, dtype=torch.bool).unsqueeze(0)
        pair_logits = pair_logits.masked_fill(diag, -1e4)

        root_logits = self.root_head(node_feat).squeeze(-1)
        parent_logits = torch.cat([pair_logits, root_logits.unsqueeze(-1)], dim=-1)
        parent_offset = self.offset_head(node_feat)
        edge_confidence_logits = self.edge_confidence_head(node_feat).squeeze(-1)
        edge_confidence = torch.sigmoid(edge_confidence_logits)

        return {
            "parent_logits": parent_logits,
            "parent_offset": parent_offset,
            "edge_confidence_logits": edge_confidence_logits,
            "edge_confidence": edge_confidence,
        }


class PairwiseParentHead(nn.Module):
    def __init__(
        self,
        node_feat_dim: int = 256,
        hidden_dim: int = 256,
        root_bias_init: float = -3.47,
        root_feature_mode: str = "none",
        root_struct_k: int = 12,
    ):
        super().__init__()
        if root_feature_mode not in {"none", "structural"}:
            raise ValueError(f"Unknown root_feature_mode: {root_feature_mode}")
        self.root_feature_mode = root_feature_mode
        self.root_struct_k = root_struct_k

        pair_in = (node_feat_dim * 2) + 3 + 1 + 3 + 3 + node_feat_dim
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Dedicated root head.
        # - none: node feature + xyz + centroid distance + radius
        # - structural: node feature + structural root features
        root_in = node_feat_dim + (ROOT_STRUCTURAL_FEATURE_DIM if root_feature_mode == "structural" else (3 + 1 + 1))
        self.root_mlp = nn.Sequential(
            nn.Linear(root_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.root_logit = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.root_logit.bias, root_bias_init)
        self.offset_head = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.edge_confidence_head = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        joint_pos: torch.Tensor,
        node_features: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        candidate_indices: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        B, J, _ = joint_pos.shape
        child_feat = node_features.unsqueeze(2).expand(B, J, J, -1)
        parent_feat = node_features.unsqueeze(1).expand(B, J, J, -1)

        rel = joint_pos.unsqueeze(2) - joint_pos.unsqueeze(1)
        dist = torch.norm(rel, dim=-1, keepdim=True)
        child_xyz = joint_pos.unsqueeze(2).expand(B, J, J, -1)
        parent_xyz = joint_pos.unsqueeze(1).expand(B, J, J, -1)
        interaction = child_feat * parent_feat

        pair_input = torch.cat(
            [child_feat, parent_feat, rel, dist, child_xyz, parent_xyz, interaction],
            dim=-1,
        )
        pair_logits = self.pair_mlp(pair_input).squeeze(-1)

        diag = torch.eye(J, device=joint_pos.device, dtype=torch.bool).unsqueeze(0)
        pair_logits = pair_logits.masked_fill(diag, -1e4)

        if active_mask is not None:
            active_mask = active_mask.bool()
            inactive_parent = (~active_mask).unsqueeze(1).expand(B, J, J)
            pair_logits = pair_logits.masked_fill(inactive_parent, -1e4)

        # Root head input can be plain geometric or structural.
        if self.root_feature_mode == "structural":
            if active_mask is None:
                active_for_root = torch.ones(joint_pos.shape[0], joint_pos.shape[1], dtype=torch.bool, device=joint_pos.device)
            else:
                active_for_root = active_mask.bool()
            root_struct_feat = compute_root_structural_features(
                joint_pos,
                active_for_root,
                candidate_indices=candidate_indices,
                candidate_mask=candidate_mask,
                pair_logits=None,
                k=self.root_struct_k,
            )
            root_feat_in = torch.cat([node_features, root_struct_feat], dim=-1)
        else:
            if active_mask is not None:
                active_float = active_mask.float().unsqueeze(-1)
                centroid = (joint_pos * active_float).sum(dim=1, keepdim=True) / active_float.sum(dim=1, keepdim=True).clamp(min=1.0)
            else:
                centroid = joint_pos.mean(dim=1, keepdim=True)
            dist_to_centroid = torch.norm(joint_pos - centroid, dim=-1, keepdim=True)
            radius = dist_to_centroid.max(dim=1, keepdim=True)[0].clamp(min=1e-6)
            norm_dist = dist_to_centroid / radius
            root_feat_in = torch.cat([node_features, joint_pos, dist_to_centroid, norm_dist], dim=-1)

        root_hidden = self.root_mlp(root_feat_in)
        root_logits = self.root_logit(root_hidden).squeeze(-1)
        if active_mask is not None:
            root_logits = root_logits.masked_fill(~active_mask.bool(), -1e4)

        parent_logits = torch.cat([pair_logits, root_logits.unsqueeze(-1)], dim=-1)
        parent_offset = self.offset_head(node_features)
        edge_confidence_logits = self.edge_confidence_head(node_features).squeeze(-1)
        edge_confidence = torch.sigmoid(edge_confidence_logits)

        return {
            "parent_logits": parent_logits,
            "parent_offset": parent_offset,
            "edge_confidence_logits": edge_confidence_logits,
            "edge_confidence": edge_confidence,
            "root_logits": root_logits,
            "parent_pair_logits": pair_logits,
        }


class HyperBoneStaticParentModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        feat_dim: int = 512,
        max_joints: int = 128,
        predict_skinning: bool = True,
        backbone: str = "pointnet",
        knn_k: int = 20,
        parent_head: str = "slot",
        root_bias_init: float | None = None,
        root_feature_mode: str = "none",
    ):
        super().__init__()
        self.max_joints = max_joints
        self.predict_skinning = predict_skinning
        self.backbone_name = backbone
        self.parent_head_name = parent_head

        if backbone == "pointnet":
            self.encoder = PointNetBackbone(in_channels=in_channels, feat_dim=feat_dim)
        elif backbone == "dgcnn":
            self.encoder = DGCNNBackbone(in_channels=in_channels, feat_dim=feat_dim, k=knn_k)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.joint_decoder = JointDecoder(feat_dim=feat_dim, max_joints=max_joints)
        self.gt_pos_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )

        if parent_head == "pairwise":
            _root_bias = root_bias_init if root_bias_init is not None else -3.47
            self.parent_head = PairwiseParentHead(
                node_feat_dim=256,
                hidden_dim=256,
                root_bias_init=_root_bias,
                root_feature_mode=root_feature_mode,
                root_struct_k=12,
            )
        elif parent_head == "slot":
            self.parent_head = ParentPointerHead(feat_dim=feat_dim, max_joints=max_joints)
        else:
            raise ValueError(f"Unknown parent_head: {parent_head}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pc = batch["pc"]
        pc_xyz = pc[:, :, :3]

        global_feat, point_feat = self.encoder(pc_xyz)
        joint_pos, active_logits, node_tokens = self.joint_decoder(global_feat, point_feat)
        if self.parent_head_name == "pairwise":
            active_mask = torch.sigmoid(active_logits) > 0.5
            parent_out = self.parent_head(joint_pos, node_tokens, active_mask=active_mask)
        else:
            parent_out = self.parent_head(joint_pos, node_tokens)

        result = {
            "joint_pos": joint_pos,
            "active_logits": active_logits,
            "node_tokens": node_tokens,
            "parent_logits": parent_out["parent_logits"],
            "parent_pair_logits": parent_out.get("parent_pair_logits"),
            "root_logits": parent_out.get("root_logits"),
            "parent_offset": parent_out["parent_offset"],
            "edge_confidence_logits": parent_out["edge_confidence_logits"],
            "edge_confidence": parent_out["edge_confidence"],
        }
        return result

    def forward_parent_from_joints(
        self,
        joint_pos: torch.Tensor,
        node_tokens: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
        no_backbone_for_gt_nodes: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run only the parent head on externally supplied joint coordinates."""
        if self.parent_head_name == "pairwise":
            if no_backbone_for_gt_nodes or node_tokens is None:
                node_tokens = self.gt_pos_encoder(joint_pos)
            if active_mask is None:
                active_mask = torch.ones(joint_pos.shape[0], joint_pos.shape[1], dtype=torch.bool, device=joint_pos.device)
            parent_out = self.parent_head(joint_pos, node_tokens, active_mask=active_mask)
        else:
            if node_tokens is None:
                token_dim = self.parent_head.node_proj[0].in_features - 3
                node_tokens = torch.zeros(joint_pos.shape[0], joint_pos.shape[1], token_dim, device=joint_pos.device, dtype=joint_pos.dtype)
            parent_out = self.parent_head(joint_pos, node_tokens)

        return {
            "joint_pos": joint_pos,
            "node_tokens": node_tokens,
            "parent_logits": parent_out["parent_logits"],
            "parent_pair_logits": parent_out.get("parent_pair_logits"),
            "root_logits": parent_out.get("root_logits"),
            "parent_offset": parent_out["parent_offset"],
            "edge_confidence_logits": parent_out["edge_confidence_logits"],
            "edge_confidence": parent_out["edge_confidence"],
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
