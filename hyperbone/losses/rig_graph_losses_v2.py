"""
Rig graph losses v2 — optimized for training speed.

Hungarian matching + Chamfer + Repulsion, but vectorized where possible.
The per-sample Hungarian matching (scipy) is unavoidable, but everything
else is batched tensor operations.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def hungarian_match_batch(
    pred_pos: torch.Tensor,
    gt_pos: torch.Tensor,
    pred_active: torch.Tensor,
    gt_active: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hungarian matching between predicted and GT active joints.

    Returns:
        pred_matched_idx: [B, M] matched pred slot indices (padded with -1)
        gt_matched_idx: [B, M] matched gt slot indices (padded with -1)
        valid_mask: [B, M] bool mask for valid matches
    """
    B = pred_pos.shape[0]
    device = pred_pos.device

    # Pre-compute on CPU for scipy
    pred_pos_cpu = pred_pos.detach().cpu().numpy()
    gt_pos_cpu = gt_pos.detach().cpu().numpy()
    pred_active_cpu = pred_active.detach().cpu().numpy()
    gt_active_cpu = gt_active.detach().cpu().numpy()

    all_pred_idx = []
    all_gt_idx = []
    max_matches = 0

    for b in range(B):
        pred_mask = pred_active_cpu[b] > 0.5
        gt_mask = gt_active_cpu[b] > 0.5

        pred_idx = np.where(pred_mask)[0]
        gt_idx = np.where(gt_mask)[0]

        if len(pred_idx) == 0 or len(gt_idx) == 0:
            all_pred_idx.append(np.array([], dtype=np.int64))
            all_gt_idx.append(np.array([], dtype=np.int64))
            continue

        # Cost matrix: L2 distance
        pred_pts = pred_pos_cpu[b, pred_idx]  # [Np, 3]
        gt_pts = gt_pos_cpu[b, gt_idx]  # [Ng, 3]
        diff = pred_pts[:, None, :] - gt_pts[None, :, :]  # [Np, Ng, 3]
        cost = np.sqrt((diff ** 2).sum(axis=-1))  # [Np, Ng]

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_pred = pred_idx[row_ind]
        matched_gt = gt_idx[col_ind]

        all_pred_idx.append(matched_pred)
        all_gt_idx.append(matched_gt)
        max_matches = max(max_matches, len(row_ind))

    if max_matches == 0:
        max_matches = 1

    # Pack into tensors
    pred_matched = torch.full((B, max_matches), -1, dtype=torch.long, device=device)
    gt_matched = torch.full((B, max_matches), -1, dtype=torch.long, device=device)
    valid_mask = torch.zeros(B, max_matches, dtype=torch.bool, device=device)

    for b in range(B):
        n = len(all_pred_idx[b])
        if n > 0:
            pred_matched[b, :n] = torch.from_numpy(all_pred_idx[b]).to(device)
            gt_matched[b, :n] = torch.from_numpy(all_gt_idx[b]).to(device)
            valid_mask[b, :n] = True

    return pred_matched, gt_matched, valid_mask


class RigGraphLossV2(nn.Module):
    """
    Combined v2 loss — vectorized for speed.

    Components:
    1. Hungarian position loss (Huber on matched pairs)
    2. Chamfer loss (bidirectional set distance)
    3. Repulsion loss (prevent collapse)
    4. Matched edge loss
    5. Active node BCE
    6. Unmatched penalty (vectorized)
    7. Bone length loss (matched)
    """

    def __init__(
        self,
        w_hungarian_pos: float = 2.0,
        w_chamfer: float = 1.0,
        w_repulsion: float = 0.5,
        w_edge: float = 0.5,
        w_bone_length: float = 0.3,
        w_active: float = 0.3,
        w_unmatched: float = 0.3,
        w_count: float = 0.5,
        w_skinning: float = 0.1,
        min_repulsion_dist: float = 0.05,
        huber_delta: float = 0.05,
        edge_pos_weight: float = 3.0,
        edge_fp_weight: float = 2.0,
        count_overpredict_scale: float = 2.0,
        count_underpredict_scale: float = 1.0,
        active_pos_weight: float = 1.0,
    ):
        super().__init__()
        self.w_hungarian_pos = w_hungarian_pos
        self.w_chamfer = w_chamfer
        self.w_repulsion = w_repulsion
        self.w_edge = w_edge
        self.w_bone_length = w_bone_length
        self.w_active = w_active
        self.w_unmatched = w_unmatched
        self.w_count = w_count
        self.w_skinning = w_skinning
        self.min_repulsion_dist = min_repulsion_dist
        self.huber_delta = huber_delta
        self.edge_pos_weight = edge_pos_weight
        self.edge_fp_weight = edge_fp_weight
        self.count_overpredict_scale = count_overpredict_scale
        self.count_underpredict_scale = count_underpredict_scale
        self.active_pos_weight = active_pos_weight

    def forward(
        self, pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
        schedule: Dict[str, float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            schedule: optional dict with ramp multipliers (0.0 to 1.0) for:
                'count', 'bone', 'edge_fp', 'unmatched'
                Allows gradual introduction of structural losses.
        """
        if schedule is None:
            schedule = {}

        gt_pos = batch["joint_pos"]  # [B, J, 3] float32
        gt_active = batch["joint_active"]  # [B, J] float32
        gt_adj = batch["adj_matrix"]  # [B, J, J] float32

        # Cast pred to float32 (AMP may produce fp16)
        pred_pos = pred["joint_pos"].float()  # [B, J, 3]
        pred_active_logits = pred["active_logits"].float()  # [B, J]
        pred_active_prob = torch.sigmoid(pred_active_logits)

        B, J, _ = pred_pos.shape
        device = pred_pos.device
        losses = {}

        # --- Hungarian matching (runs on CPU, no grad needed) ---
        pred_matched, gt_matched, match_valid = hungarian_match_batch(
            pred_pos, gt_pos, pred_active_prob, gt_active
        )

        # === 1. Hungarian Position Loss (vectorized gather) ===
        losses["hungarian_pos"] = self._hungarian_pos_loss(
            pred_pos, gt_pos, pred_matched, gt_matched, match_valid
        )

        # === 2. Chamfer Loss ===
        losses["chamfer"] = self._chamfer_loss(
            pred_pos, gt_pos, pred_active_prob, gt_active
        )

        # === 3. Repulsion Loss ===
        losses["repulsion"] = self._repulsion_loss(pred_pos, pred_active_prob)

        # === 4. Edge Loss (matched, FP penalty ramped by schedule) ===
        if "adj_logits" in pred:
            pred_adj_logits = pred["adj_logits"].float()
            s_edge_fp = schedule.get("edge_fp", 1.0)
            losses["edge"] = self._matched_edge_loss(
                pred_adj_logits, gt_adj, pred_matched, gt_matched, match_valid,
                edge_fp_scale=s_edge_fp
            )

        # === 5. Active node BCE (positive-weighted) ===
        active_pw = torch.tensor(self.active_pos_weight, device=device)
        losses["active"] = F.binary_cross_entropy_with_logits(
            pred_active_logits, gt_active, pos_weight=active_pw
        )

        # === 6. Unmatched penalty (vectorized) ===
        losses["unmatched"] = self._unmatched_penalty(
            pred_active_logits, pred_active_prob, gt_active,
            pred_matched, gt_matched, match_valid
        )

        # === 7. Bone length loss ===
        losses["bone_length"] = self._bone_length_loss(
            pred_pos, gt_pos, gt_adj, pred_matched, gt_matched, match_valid
        )

        # === 8. Active count loss ===
        losses["count"] = self._active_count_loss(
            pred_active_prob, gt_active
        )

        # === 9. Skinning (optional) ===
        if self.w_skinning > 0 and "skin_logits" in pred and "skin_joints" in batch:
            losses["skinning"] = self._skinning_loss(
                pred["skin_logits"].float(), batch["skin_joints"], batch["skin_weights"]
            )

        # --- Total (with schedule ramp) ---
        s_count = schedule.get("count", 1.0)
        s_bone = schedule.get("bone", 1.0)
        s_edge_fp = schedule.get("edge_fp", 1.0)
        s_unmatched = schedule.get("unmatched", 1.0)

        total = (
            self.w_hungarian_pos * losses["hungarian_pos"]
            + self.w_chamfer * losses["chamfer"]
            + self.w_repulsion * losses["repulsion"]
            + self.w_edge * losses.get("edge", torch.tensor(0.0, device=device))
            + self.w_active * losses["active"]
            + self.w_unmatched * s_unmatched * losses["unmatched"]
            + self.w_bone_length * s_bone * losses["bone_length"]
            + self.w_count * s_count * losses["count"]
            + self.w_skinning * losses.get("skinning", torch.tensor(0.0, device=device))
        )
        losses["total"] = total
        return losses

    def _hungarian_pos_loss(self, pred_pos, gt_pos, pred_matched, gt_matched, match_valid):
        """Huber loss on matched pairs — vectorized gather."""
        if not match_valid.any():
            return torch.tensor(0.0, device=pred_pos.device, requires_grad=True)

        B, M = pred_matched.shape
        # Clamp indices for safe gather (invalid entries masked out)
        p_idx = pred_matched.clamp(min=0).unsqueeze(-1).expand(-1, -1, 3)  # [B, M, 3]
        g_idx = gt_matched.clamp(min=0).unsqueeze(-1).expand(-1, -1, 3)  # [B, M, 3]

        pred_pts = torch.gather(pred_pos, 1, p_idx)  # [B, M, 3]
        gt_pts = torch.gather(gt_pos, 1, g_idx)  # [B, M, 3]

        # Masked Huber
        mask = match_valid.unsqueeze(-1).float()  # [B, M, 1]
        loss = F.smooth_l1_loss(
            pred_pts * mask, gt_pts * mask,
            beta=self.huber_delta, reduction="sum"
        )
        n_valid = match_valid.sum().clamp(min=1).float()
        return loss / (n_valid * 3)

    def _chamfer_loss(self, pred_pos, gt_pos, pred_active_prob, gt_active):
        """Bidirectional Chamfer — per-sample (variable sizes)."""
        B = pred_pos.shape[0]
        total = torch.tensor(0.0, device=pred_pos.device, requires_grad=True)
        count = 0

        for b in range(B):
            pred_mask = pred_active_prob[b] > 0.5
            gt_mask = gt_active[b] > 0.5

            if pred_mask.sum() == 0 or gt_mask.sum() == 0:
                continue

            pred_pts = pred_pos[b, pred_mask]  # [Np, 3]
            gt_pts = gt_pos[b, gt_mask]  # [Ng, 3]

            dist = torch.cdist(pred_pts, gt_pts, p=2)  # [Np, Ng]
            chamfer = dist.min(dim=1)[0].mean() + dist.min(dim=0)[0].mean()
            total = total + chamfer
            count += 1

        return total / max(count, 1)

    def _repulsion_loss(self, pred_pos, pred_active_prob):
        """Repulsion — penalize nodes closer than threshold."""
        B = pred_pos.shape[0]
        total = torch.tensor(0.0, device=pred_pos.device, requires_grad=True)
        count = 0

        for b in range(B):
            mask = pred_active_prob[b] > 0.5
            if mask.sum() < 2:
                continue

            pts = pred_pos[b, mask]  # [N, 3]
            N = pts.shape[0]
            dist = torch.cdist(pts, pts, p=2)  # [N, N]

            # Upper triangle
            triu_mask = torch.triu(torch.ones(N, N, device=pts.device), diagonal=1).bool()
            dists = dist[triu_mask]

            violations = F.relu(self.min_repulsion_dist - dists)
            if violations.sum() > 0:
                total = total + violations.mean()
                count += 1

        return total / max(count, 1)

    def _matched_edge_loss(self, pred_adj_logits, gt_adj, pred_matched, gt_matched, match_valid,
                           edge_fp_scale: float = 1.0):
        """Edge loss on matched subgraph."""
        B = pred_adj_logits.shape[0]
        device = pred_adj_logits.device
        total = torch.tensor(0.0, device=device, requires_grad=True)
        count = 0

        for b in range(B):
            valid = match_valid[b]
            if valid.sum() < 2:
                continue

            p_idx = pred_matched[b, valid]
            g_idx = gt_matched[b, valid]
            M = p_idx.shape[0]

            pred_sub = pred_adj_logits[b][p_idx][:, p_idx]  # [M, M]
            gt_sub = gt_adj[b][g_idx][:, g_idx]  # [M, M]

            triu = torch.triu(torch.ones(M, M, device=device), diagonal=1).bool()
            pred_flat = pred_sub[triu]
            gt_flat = gt_sub[triu]

            if pred_flat.numel() == 0:
                continue

            # Balanced BCE with configurable pos_weight
            bce_loss = F.binary_cross_entropy_with_logits(
                pred_flat, gt_flat,
                pos_weight=torch.tensor(self.edge_pos_weight, device=device),
            )

            # Extra false-positive penalty (scaled by schedule ramp)
            if edge_fp_scale > 0 and self.edge_fp_weight > 0:
                fp_mask = gt_flat < 0.5
                if fp_mask.any():
                    fp_loss = F.binary_cross_entropy_with_logits(
                        pred_flat[fp_mask], gt_flat[fp_mask]
                    )
                    loss = bce_loss + edge_fp_scale * self.edge_fp_weight * fp_loss
                else:
                    loss = bce_loss
            else:
                loss = bce_loss

            total = total + loss
            count += 1

        return total / max(count, 1)

    def _unmatched_penalty(self, pred_active_logits, pred_active_prob, gt_active,
                           pred_matched, gt_matched, match_valid):
        """Vectorized penalty for over/under-prediction."""
        B, J = pred_active_logits.shape
        device = pred_active_logits.device

        # Build mask of matched pred indices
        matched_pred_mask = torch.zeros(B, J, dtype=torch.bool, device=device)
        matched_gt_mask = torch.zeros(B, J, dtype=torch.bool, device=device)
        for b in range(B):
            valid_p = pred_matched[b, match_valid[b]]
            valid_g = gt_matched[b, match_valid[b]]
            if valid_p.numel() > 0:
                matched_pred_mask[b].scatter_(0, valid_p, True)
            if valid_g.numel() > 0:
                matched_gt_mask[b].scatter_(0, valid_g, True)

        # Overprediction: active but not matched → push inactive
        pred_active_mask = pred_active_prob > 0.5
        overpred_mask = pred_active_mask & ~matched_pred_mask

        if overpred_mask.any():
            overpred_logits = pred_active_logits[overpred_mask]
            overpred_loss = F.binary_cross_entropy_with_logits(
                overpred_logits, torch.zeros_like(overpred_logits)
            )
        else:
            overpred_loss = torch.tensor(0.0, device=device)

        # Recall penalty: fraction of GT not matched
        gt_active_mask = gt_active > 0.5
        unmatched_gt = gt_active_mask & ~matched_gt_mask
        n_unmatched = unmatched_gt.float().sum(dim=1)
        n_gt = gt_active_mask.float().sum(dim=1).clamp(min=1)
        recall_penalty = (n_unmatched / n_gt).mean()

        return 0.5 * overpred_loss + recall_penalty

    def _bone_length_loss(self, pred_pos, gt_pos, gt_adj, pred_matched, gt_matched, match_valid):
        """Bone length loss — vectorized edge extraction."""
        B = pred_pos.shape[0]
        device = pred_pos.device
        total = torch.tensor(0.0, device=device, requires_grad=True)
        count = 0

        for b in range(B):
            valid = match_valid[b]
            if valid.sum() < 2:
                continue

            p_idx = pred_matched[b, valid]
            g_idx = gt_matched[b, valid]
            M = p_idx.shape[0]

            # GT adjacency submatrix
            gt_sub = gt_adj[b][g_idx][:, g_idx]
            triu = torch.triu(torch.ones(M, M, device=device), diagonal=1)
            edge_mask = (gt_sub * triu) > 0.5

            if not edge_mask.any():
                continue

            edge_pairs = edge_mask.nonzero(as_tuple=False)  # [E, 2]
            E = edge_pairs.shape[0]

            # Vectorized bone length computation
            gt_starts = gt_pos[b, g_idx[edge_pairs[:, 0]]]  # [E, 3]
            gt_ends = gt_pos[b, g_idx[edge_pairs[:, 1]]]  # [E, 3]
            gt_lengths = (gt_starts - gt_ends).norm(dim=-1)  # [E]

            pred_starts = pred_pos[b, p_idx[edge_pairs[:, 0]]]  # [E, 3]
            pred_ends = pred_pos[b, p_idx[edge_pairs[:, 1]]]  # [E, 3]
            pred_lengths = (pred_starts - pred_ends).norm(dim=-1)  # [E]

            # Log-ratio loss on valid bones
            valid_bones = gt_lengths > 1e-4
            if valid_bones.any():
                ratios = pred_lengths[valid_bones] / gt_lengths[valid_bones].clamp(min=1e-4)
                log_ratios = ratios.log().abs()
                bone_loss = log_ratios.mean()
                # Extra penalty for extreme outliers (ratio > 2x or < 0.5x)
                extreme_mask = log_ratios > 0.693  # log(2) ≈ 0.693
                if extreme_mask.any():
                    bone_loss = bone_loss + log_ratios[extreme_mask].mean()
                total = total + bone_loss
                count += 1

        return total / max(count, 1)

    def _active_count_loss(self, pred_active_prob, gt_active):
        """
        Active node count supervision — undercount-biased.
        
        Penalizes underprediction more than overprediction to prevent
        the trivial "predict nothing" solution.
        """
        # Predicted count = soft sum of probabilities (differentiable)
        pred_count = pred_active_prob.sum(dim=-1)  # [B]
        gt_count = gt_active.sum(dim=-1)  # [B]

        count_error = pred_count - gt_count  # positive = overpredict, negative = underpredict

        # Asymmetric quadratic: undercount penalized more
        undercount = torch.relu(-count_error)  # how many missing
        overcount = torch.relu(count_error)  # how many extra

        loss = (
            self.count_underpredict_scale * undercount.pow(2)
            + self.count_overpredict_scale * overcount.pow(2)
        ).mean()

        # Normalize by GT count to make scale-invariant
        norm = gt_count.clamp(min=1).mean()
        return loss / norm

    def _skinning_loss(self, pred_skin_logits, gt_skin_joints, gt_skin_weights):
        """KL divergence skinning loss."""
        B, P, J = pred_skin_logits.shape
        K = gt_skin_joints.shape[-1]

        target = torch.zeros(B, P, J, device=pred_skin_logits.device)
        for k in range(K):
            idx = gt_skin_joints[:, :, k].clamp(0, J - 1).long()
            w = gt_skin_weights[:, :, k].float()
            target.scatter_add_(2, idx.unsqueeze(-1), w.unsqueeze(-1))

        target_sum = target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        target = target / target_sum

        pred_log_prob = F.log_softmax(pred_skin_logits, dim=-1)
        valid = gt_skin_weights.sum(dim=-1) > 0

        loss = F.kl_div(pred_log_prob, target, reduction="none").sum(dim=-1)
        loss = (loss * valid.float()).sum() / valid.float().sum().clamp(min=1)
        return loss
