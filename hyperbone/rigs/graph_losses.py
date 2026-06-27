"""
Graph matching losses for topology-aware rig prediction.

Uses Hungarian (optimal bipartite) matching to assign predicted nodes to GT joints,
then computes losses only on matched pairs. This handles variable skeleton topology.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@torch.no_grad()
def hungarian_match(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor,
                    gt_active: torch.Tensor, pred_active: torch.Tensor = None,
                    cost_xyz_weight: float = 1.0,
                    cost_active_weight: float = 0.5) -> list:
    """Compute Hungarian matching between predicted and GT nodes.

    Args:
        pred_xyz: [B, N_pred, 3]
        gt_xyz: [B, N_gt, 3]
        gt_active: [B, N_gt] — 1 if GT node exists
        pred_active: [B, N_pred] — predicted activity probability (optional)

    Returns:
        List of (pred_indices, gt_indices) tuples per batch element.
        Only includes matches to active GT nodes.
    """
    assert linear_sum_assignment is not None, "scipy required for Hungarian matching"

    B = pred_xyz.shape[0]
    matches = []

    for b in range(B):
        active_mask = gt_active[b] > 0.5
        n_gt = int(active_mask.sum().item())
        if n_gt == 0:
            matches.append((torch.zeros(0, dtype=torch.long),
                           torch.zeros(0, dtype=torch.long)))
            continue

        # Get active GT positions
        gt_indices = torch.where(active_mask)[0]
        gt_pos = gt_xyz[b, gt_indices]  # [n_gt, 3]

        # All predicted positions
        pred_pos = pred_xyz[b]  # [N_pred, 3]
        N_pred = pred_pos.shape[0]

        # Cost matrix: [N_pred, n_gt]
        # Position cost (L2)
        cost_xyz = torch.cdist(pred_pos, gt_pos, p=2)  # [N_pred, n_gt]

        # Activity cost: prefer assigning active predictions to GT
        if pred_active is not None:
            # Lower cost for high-activity predictions
            active_cost = (1.0 - pred_active[b]).unsqueeze(1).expand(-1, n_gt)
            cost = cost_xyz_weight * cost_xyz + cost_active_weight * active_cost
        else:
            cost = cost_xyz * cost_xyz_weight

        # Solve assignment
        cost_np = cost.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)

        # row_ind: indices into pred, col_ind: indices into gt_indices
        pred_matched = torch.tensor(row_ind, dtype=torch.long, device=pred_xyz.device)
        gt_matched = gt_indices[col_ind]  # map back to original GT indices

        matches.append((pred_matched, gt_matched))

    return matches


def compute_graph_loss(pred: dict, gt_xyz: torch.Tensor, gt_active: torch.Tensor,
                       gt_vis: torch.Tensor, gt_adj: torch.Tensor,
                       gt_types: torch.Tensor, gt_bone_lengths: torch.Tensor,
                       gt_xy: torch.Tensor = None,
                       camera_K: torch.Tensor = None,
                       camera_ext: torch.Tensor = None) -> dict:
    """
    Compute full graph-matching loss.

    Args:
        pred: model output dict
        gt_xyz: [B, N_gt, 3] ground truth 3D positions
        gt_active: [B, N_gt] which GT nodes exist
        gt_vis: [B, N_gt] which GT nodes are visible
        gt_adj: [B, N_gt, N_gt] ground truth adjacency
        gt_types: [B, N_gt] joint type indices
        gt_bone_lengths: [B, N_gt] rest bone lengths per joint
        gt_xy: [B, N_gt, 2] projected 2D positions (optional)
        camera_K: [B, 3, 3] intrinsics (optional)
        camera_ext: [B, 4, 4] extrinsics (optional)

    Returns:
        dict of losses
    """
    device = gt_xyz.device
    B = gt_xyz.shape[0]

    pred_xyz = pred["node_xyz"]
    pred_active_logits = pred["node_active_logits"]
    pred_active = pred["node_active"]
    pred_type_logits = pred["node_type_logits"]
    edge_logits = pred["edge_logits"]

    # Step 1: Hungarian matching
    matches = hungarian_match(pred_xyz, gt_xyz, gt_active, pred_active)

    # Step 2: Matched position loss (Huber)
    pos_loss = torch.tensor(0.0, device=device)
    n_matched = 0
    for b, (pred_idx, gt_idx) in enumerate(matches):
        if len(pred_idx) == 0:
            continue
        matched_pred = pred_xyz[b, pred_idx]
        matched_gt = gt_xyz[b, gt_idx]
        pos_loss = pos_loss + F.huber_loss(matched_pred, matched_gt, reduction='sum')
        n_matched += len(pred_idx)
    pos_loss = pos_loss / max(n_matched, 1)

    # Step 3: Node active loss (BCE)
    # Matched pred nodes should be active, unmatched should be inactive
    active_targets = torch.zeros_like(pred_active_logits)
    for b, (pred_idx, gt_idx) in enumerate(matches):
        if len(pred_idx) > 0:
            active_targets[b, pred_idx] = 1.0

    active_loss = F.binary_cross_entropy_with_logits(
        pred_active_logits, active_targets, reduction='mean')

    # Step 4: Node type loss (CE on matched nodes only)
    type_loss = torch.tensor(0.0, device=device)
    n_typed = 0
    for b, (pred_idx, gt_idx) in enumerate(matches):
        if len(pred_idx) == 0:
            continue
        pred_types_b = pred_type_logits[b, pred_idx]  # [M, T]
        gt_types_b = gt_types[b, gt_idx]  # [M]
        type_loss = type_loss + F.cross_entropy(pred_types_b, gt_types_b, reduction='sum')
        n_typed += len(pred_idx)
    type_loss = type_loss / max(n_typed, 1)

    # Step 5: Edge loss (BCE on matched node pairs)
    edge_loss = torch.tensor(0.0, device=device)
    n_edges = 0
    for b, (pred_idx, gt_idx) in enumerate(matches):
        if len(pred_idx) < 2:
            continue
        M = len(pred_idx)
        # Build GT adjacency for matched subset (vectorized)
        gt_adj_matched = gt_adj[b][gt_idx][:, gt_idx]  # [M, M]

        # Predicted edge logits for matched pairs
        pred_edge_matched = edge_logits[b][pred_idx][:, pred_idx]  # [M, M]
        edge_loss = edge_loss + F.binary_cross_entropy_with_logits(
            pred_edge_matched, gt_adj_matched, reduction='sum')
        n_edges += M * M
    edge_loss = edge_loss / max(n_edges, 1)

    # Step 6: Bone length consistency loss (vectorized)
    bone_loss = torch.tensor(0.0, device=device)
    n_bones = 0
    for b, (pred_idx, gt_idx) in enumerate(matches):
        if len(pred_idx) < 2:
            continue
        M = len(pred_idx)
        # Get adjacency and bone lengths for matched subset
        adj_sub = gt_adj[b][gt_idx][:, gt_idx]  # [M, M]
        bone_pairs = torch.where(adj_sub > 0.5)
        if len(bone_pairs[0]) == 0:
            continue
        bi, bj = bone_pairs
        gt_lens = gt_bone_lengths[b, gt_idx[bj]]
        valid = gt_lens > 0.01
        if not valid.any():
            continue
        bi_v, bj_v = bi[valid], bj[valid]
        gt_lens_v = gt_lens[valid]
        pred_vecs = pred_xyz[b, pred_idx[bi_v]] - pred_xyz[b, pred_idx[bj_v]]
        pred_lens = torch.norm(pred_vecs, dim=-1)
        ratios = pred_lens / gt_lens_v
        bone_loss = bone_loss + F.huber_loss(
            ratios, torch.ones_like(ratios), reduction='sum')
        n_bones += int(valid.sum().item())
    bone_loss = bone_loss / max(n_bones, 1)

    # Step 7: 2D reprojection loss (if camera available)
    reproj_loss = torch.tensor(0.0, device=device)
    if gt_xy is not None:
        n_reproj = 0
        for b, (pred_idx, gt_idx) in enumerate(matches):
            if len(pred_idx) == 0:
                continue
            # Only on visible GT joints
            vis_mask = gt_vis[b, gt_idx] > 0.5
            if not vis_mask.any():
                continue
            matched_gt_xy = gt_xy[b, gt_idx][vis_mask]
            matched_pred_xyz = pred_xyz[b, pred_idx][vis_mask]

            # Project predicted 3D to 2D if camera available
            if camera_K is not None and camera_ext is not None:
                K = camera_K[b]  # [3, 3]
                ext = camera_ext[b]  # [4, 4]
                pts_h = torch.cat([matched_pred_xyz,
                                   torch.ones(matched_pred_xyz.shape[0], 1, device=device)], dim=1)
                pts_cam = (ext @ pts_h.T).T[:, :3]
                valid = pts_cam[:, 2] > 0.01
                if valid.any():
                    proj = (K @ pts_cam[valid].T).T
                    pred_2d = proj[:, :2] / proj[:, 2:3]
                    gt_2d = matched_gt_xy[valid]
                    reproj_loss = reproj_loss + F.huber_loss(pred_2d, gt_2d, reduction='sum')
                    n_reproj += valid.sum().item()
        reproj_loss = reproj_loss / max(n_reproj, 1)

    # Total weighted loss
    total = (1.0 * pos_loss +
             0.5 * active_loss +
             0.3 * type_loss +
             0.3 * edge_loss +
             0.5 * bone_loss +
             0.2 * reproj_loss)

    return {
        "total": total,
        "pos_loss": pos_loss.item(),
        "active_loss": active_loss.item(),
        "type_loss": type_loss.item(),
        "edge_loss": edge_loss.item(),
        "bone_loss": bone_loss.item(),
        "reproj_loss": reproj_loss.item(),
        "n_matched": n_matched,
    }
