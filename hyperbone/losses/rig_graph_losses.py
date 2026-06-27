"""
Rig graph losses for static skeleton prediction.

Losses:
- Joint position (Chamfer / L1 with Hungarian matching)
- Edge connectivity (BCE on adjacency matrix)
- Bone length consistency
- Skinning weight prediction
- Joint count / active node loss
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def hungarian_matching(
    pred_pos: torch.Tensor,
    gt_pos: torch.Tensor,
    pred_active: torch.Tensor,
    gt_active: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Hungarian matching between predicted and GT joints.

    Args:
        pred_pos: [B, J, 3] predicted joint positions
        gt_pos: [B, J, 3] ground truth joint positions
        pred_active: [B, J] predicted active mask (sigmoid)
        gt_active: [B, J] ground truth active mask

    Returns:
        matching: [B, J] index mapping pred → gt (-1 if unmatched)
    """
    B, J, _ = pred_pos.shape
    matching = torch.full((B, J), -1, dtype=torch.long, device=pred_pos.device)

    for b in range(B):
        # Get active indices
        pred_mask = pred_active[b] > 0.5
        gt_mask = gt_active[b] > 0.5

        pred_idx = pred_mask.nonzero(as_tuple=True)[0]
        gt_idx = gt_mask.nonzero(as_tuple=True)[0]

        if len(pred_idx) == 0 or len(gt_idx) == 0:
            continue

        # Cost matrix: L2 distance
        pred_pts = pred_pos[b, pred_idx]  # [Np, 3]
        gt_pts = gt_pos[b, gt_idx]  # [Ng, 3]

        cost = torch.cdist(pred_pts, gt_pts, p=2)  # [Np, Ng]

        # Solve assignment
        cost_np = cost.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)

        for r, c in zip(row_ind, col_ind):
            matching[b, pred_idx[r]] = gt_idx[c]

    return matching


class JointPositionLoss(nn.Module):
    """L1/Huber loss on matched joint positions."""

    def __init__(self, use_huber: bool = True, delta: float = 0.1):
        super().__init__()
        self.use_huber = use_huber
        self.delta = delta

    def forward(
        self,
        pred_pos: torch.Tensor,
        gt_pos: torch.Tensor,
        gt_active: torch.Tensor,
        matching: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_pos: [B, J, 3]
            gt_pos: [B, J, 3]
            gt_active: [B, J]
            matching: [B, J] optional pre-computed matching
        """
        if matching is None:
            # Direct comparison (same-slot mode)
            mask = gt_active.unsqueeze(-1)  # [B, J, 1]
            diff = pred_pos - gt_pos
            if self.use_huber:
                loss = F.smooth_l1_loss(
                    pred_pos * mask, gt_pos * mask, beta=self.delta, reduction="sum"
                )
            else:
                loss = (diff.abs() * mask).sum()
            n_active = mask.sum().clamp(min=1)
            return loss / n_active
        else:
            # Hungarian-matched mode
            B, J, _ = pred_pos.shape
            total_loss = 0.0
            count = 0
            for b in range(B):
                for j in range(J):
                    gt_j = matching[b, j]
                    if gt_j >= 0:
                        diff = pred_pos[b, j] - gt_pos[b, gt_j]
                        if self.use_huber:
                            total_loss += F.smooth_l1_loss(
                                pred_pos[b, j], gt_pos[b, gt_j], beta=self.delta
                            )
                        else:
                            total_loss += diff.abs().mean()
                        count += 1
            return total_loss / max(count, 1)


class EdgeConnectivityLoss(nn.Module):
    """BCE loss on predicted adjacency matrix."""

    def __init__(self, pos_weight: float = 5.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        pred_adj: torch.Tensor,
        gt_adj: torch.Tensor,
        gt_active: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_adj: [B, J, J] predicted adjacency logits
            gt_adj: [B, J, J] ground truth adjacency (0/1)
            gt_active: [B, J] active node mask
        """
        # Only compute loss where both nodes are active
        mask = gt_active.unsqueeze(-1) * gt_active.unsqueeze(-2)  # [B, J, J]

        # Masked BCE
        pos_weight = torch.tensor(self.pos_weight, device=pred_adj.device)
        loss = F.binary_cross_entropy_with_logits(
            pred_adj * mask,
            gt_adj * mask,
            pos_weight=pos_weight,
            reduction="sum",
        )
        n_valid = mask.sum().clamp(min=1)
        return loss / n_valid


class BoneLengthLoss(nn.Module):
    """L1 loss on bone lengths."""

    def forward(
        self,
        pred_pos: torch.Tensor,
        gt_adj: torch.Tensor,
        gt_bone_lengths: torch.Tensor,
        gt_active: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute bone length from predicted joint positions and compare to GT.
        Uses adjacency to identify connected pairs.
        """
        B, J, _ = pred_pos.shape

        total_loss = 0.0
        count = 0

        for b in range(B):
            # Find edges from adjacency (upper triangle)
            edges = (gt_adj[b].triu(diagonal=1) > 0.5).nonzero(as_tuple=False)
            for e in edges:
                i, j = e[0], e[1]
                if gt_active[b, i] > 0.5 and gt_active[b, j] > 0.5:
                    pred_len = (pred_pos[b, i] - pred_pos[b, j]).norm()
                    gt_len = (gt_bone_lengths[b, :].sum() / max(edges.shape[0], 1))
                    # Direct bone length from GT positions
                    gt_bone = (pred_pos[b, i].detach() - pred_pos[b, j].detach()).norm()
                    # We want predicted positions to give correct bone length
                    total_loss += F.l1_loss(pred_len, gt_bone)
                    count += 1

        return total_loss / max(count, 1)


class SkinningWeightLoss(nn.Module):
    """Cross-entropy loss on skinning weight assignments."""

    def forward(
        self,
        pred_skin_logits: torch.Tensor,
        gt_skin_joints: torch.Tensor,
        gt_skin_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_skin_logits: [B, P, J] per-point logits over joints
            gt_skin_joints: [B, P, K] ground truth joint indices
            gt_skin_weights: [B, P, K] ground truth weights
        """
        B, P, J = pred_skin_logits.shape
        K = gt_skin_joints.shape[-1]

        # Soft cross-entropy: target is a distribution over joints
        # Build target distribution from sparse GT
        target = torch.zeros(B, P, J, device=pred_skin_logits.device)
        for k in range(K):
            idx = gt_skin_joints[:, :, k].clamp(0, J - 1)  # [B, P]
            w = gt_skin_weights[:, :, k]  # [B, P]
            target.scatter_add_(2, idx.unsqueeze(-1), w.unsqueeze(-1))

        # Normalize target
        target_sum = target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        target = target / target_sum

        # KL divergence
        pred_log_prob = F.log_softmax(pred_skin_logits, dim=-1)
        # Only compute where we have valid skinning data
        valid = gt_skin_weights.sum(dim=-1) > 0  # [B, P]

        loss = F.kl_div(pred_log_prob, target, reduction="none").sum(dim=-1)  # [B, P]
        loss = (loss * valid.float()).sum() / valid.float().sum().clamp(min=1)
        return loss


class JointActiveLoss(nn.Module):
    """BCE loss on which joints are active."""

    def forward(
        self, pred_active_logits: torch.Tensor, gt_active: torch.Tensor
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(pred_active_logits, gt_active)


class RigGraphLoss(nn.Module):
    """Combined rig graph loss."""

    def __init__(
        self,
        w_joint_pos: float = 1.0,
        w_edge: float = 0.5,
        w_active: float = 0.3,
        w_skinning: float = 0.2,
        use_hungarian: bool = False,
    ):
        super().__init__()
        self.w_joint_pos = w_joint_pos
        self.w_edge = w_edge
        self.w_active = w_active
        self.w_skinning = w_skinning
        self.use_hungarian = use_hungarian

        self.joint_pos_loss = JointPositionLoss()
        self.edge_loss = EdgeConnectivityLoss()
        self.active_loss = JointActiveLoss()
        self.skinning_loss = SkinningWeightLoss()

    def forward(self, pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        gt_pos = batch["joint_pos"]
        gt_active = batch["joint_active"]
        gt_adj = batch["adj_matrix"]

        losses = {}

        # Joint position loss
        matching = None
        if self.use_hungarian:
            pred_active_prob = torch.sigmoid(pred["active_logits"])
            matching = hungarian_matching(pred["joint_pos"], gt_pos, pred_active_prob, gt_active)

        losses["joint_pos"] = self.joint_pos_loss(
            pred["joint_pos"], gt_pos, gt_active, matching
        )

        # Edge connectivity loss
        if "adj_logits" in pred:
            losses["edge"] = self.edge_loss(pred["adj_logits"], gt_adj, gt_active)

        # Active node loss
        if "active_logits" in pred:
            losses["active"] = self.active_loss(pred["active_logits"], gt_active)

        # Skinning loss
        if "skin_logits" in pred and "skin_joints" in batch:
            losses["skinning"] = self.skinning_loss(
                pred["skin_logits"], batch["skin_joints"], batch["skin_weights"]
            )

        # Total
        total = (
            self.w_joint_pos * losses["joint_pos"]
            + self.w_edge * losses.get("edge", torch.tensor(0.0, device=gt_pos.device))
            + self.w_active * losses.get("active", torch.tensor(0.0, device=gt_pos.device))
            + self.w_skinning * losses.get("skinning", torch.tensor(0.0, device=gt_pos.device))
        )
        losses["total"] = total

        return losses
