"""
Loss functions for HyperNodeNet training.

Combined multi-task loss with configurable weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    heatmap: float = 1.0
    node_active: float = 0.5
    node_xy: float = 1.0
    node_type: float = 0.5
    edge_active: float = 0.5
    edge_type: float = 0.2
    radius: float = 0.2
    chamfer: float = 0.5


def focal_mse_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Focal MSE: upweights hard (non-zero target) locations."""
    mse = (pred - target) ** 2
    weight = 1.0 + gamma * target  # higher weight where target > 0
    return (weight * mse).mean()


def chamfer_loss_2d(
    pred_xy: torch.Tensor,
    pred_active: torch.Tensor,
    target_xy: torch.Tensor,
    target_active: torch.Tensor,
) -> torch.Tensor:
    """Chamfer distance between predicted and target active node sets.
    
    Args:
        pred_xy: [B, N, 2] predicted coordinates
        pred_active: [B, N] active probabilities
        target_xy: [B, N, 2] target coordinates
        target_active: [B, N] binary active mask
    """
    B = pred_xy.shape[0]
    total_loss = torch.tensor(0.0, device=pred_xy.device)
    count = 0

    for b in range(B):
        # Get active predictions (threshold at 0.5)
        p_mask = pred_active[b] > 0.5
        t_mask = target_active[b] > 0.5

        p_pts = pred_xy[b][p_mask]  # [M, 2]
        t_pts = target_xy[b][t_mask]  # [K, 2]

        if p_pts.shape[0] == 0 or t_pts.shape[0] == 0:
            continue

        # Pairwise distance [M, K]
        dist = torch.cdist(p_pts, t_pts, p=2)

        # Pred -> target nearest
        d_p2t = dist.min(dim=1).values.mean()
        # Target -> pred nearest
        d_t2p = dist.min(dim=0).values.mean()

        total_loss = total_loss + (d_p2t + d_t2p) * 0.5
        count += 1

    return total_loss / max(count, 1)


class HyperNodeLoss(nn.Module):
    """Combined multi-task loss for HyperNodeNet."""

    def __init__(self, weights: LossWeights | None = None):
        super().__init__()
        self.w = weights or LossWeights()

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        losses = {}

        # Heatmap loss (focal MSE)
        losses["heatmap"] = focal_mse_loss(
            pred["heatmaps"], batch["heatmaps"]
        )

        # Radius map
        losses["radius"] = F.l1_loss(pred["radius_map"], batch["radius_map"])

        # Node active (BCE)
        losses["node_active"] = F.binary_cross_entropy_with_logits(
            pred["active_logits"], batch["node_active"]
        )

        # Node xy (L1 on active nodes only)
        active_mask = batch["node_active"].unsqueeze(-1)  # [B, N, 1]
        xy_diff = F.l1_loss(
            pred["node_xy"] * active_mask,
            batch["node_xy"] * active_mask,
            reduction="sum",
        )
        n_active = active_mask.sum().clamp(min=1.0)
        losses["node_xy"] = xy_diff / n_active

        # Node type (CE on active nodes)
        B, N, T = pred["node_type_logits"].shape
        active_flat = batch["node_active"].view(-1) > 0.5
        type_logits_flat = pred["node_type_logits"].view(-1, T)
        type_target_flat = batch["node_type"].view(-1)
        if active_flat.any():
            losses["node_type"] = F.cross_entropy(
                type_logits_flat[active_flat],
                type_target_flat[active_flat],
            )
        else:
            losses["node_type"] = torch.tensor(0.0, device=pred["heatmaps"].device)

        # Edge active (BCE)
        losses["edge_active"] = F.binary_cross_entropy_with_logits(
            pred["edge_logits"], batch["edge_active"]
        )

        # Edge type (CE on active edges)
        edge_mask = batch["edge_active"].view(-1) > 0.5
        E = pred["edge_type_logits"].shape[-1]
        etype_logits_flat = pred["edge_type_logits"].view(-1, E)
        etype_target_flat = batch["edge_type"].view(-1)
        if edge_mask.any():
            losses["edge_type"] = F.cross_entropy(
                etype_logits_flat[edge_mask],
                etype_target_flat[edge_mask],
            )
        else:
            losses["edge_type"] = torch.tensor(0.0, device=pred["heatmaps"].device)

        # Chamfer loss
        pred_active_prob = torch.sigmoid(pred["active_logits"])
        losses["chamfer"] = chamfer_loss_2d(
            pred["node_xy"], pred_active_prob,
            batch["node_xy"], batch["node_active"],
        )

        # Weighted total
        total = (
            self.w.heatmap * losses["heatmap"]
            + self.w.node_active * losses["node_active"]
            + self.w.node_xy * losses["node_xy"]
            + self.w.node_type * losses["node_type"]
            + self.w.edge_active * losses["edge_active"]
            + self.w.edge_type * losses["edge_type"]
            + self.w.radius * losses["radius"]
            + self.w.chamfer * losses["chamfer"]
        )

        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict["total"] = total.item()

        return total, loss_dict
