from __future__ import annotations

from typing import Optional

import torch

ROOT_STRUCTURAL_FEATURE_NAMES = [
    "norm_x",
    "norm_y",
    "norm_z",
    "dist_to_centroid",
    "norm_radius",
    "radius_percentile",
    "local_density_mean_knn_dist",
    "nn1_dist",
    "nn2_dist",
    "knn_indegree",
    "candidate_centrality",
    "mean_incoming_rank",
    "min_incoming_rank",
    "max_incoming_rank",
    "active_count_norm",
    "bbox_diagonal",
    "mean_incoming_pair_score",
    "max_incoming_pair_score",
]
ROOT_STRUCTURAL_FEATURE_DIM = len(ROOT_STRUCTURAL_FEATURE_NAMES)


def _safe_feature_tensor_like(joint_pos: torch.Tensor) -> torch.Tensor:
    B, N, _ = joint_pos.shape
    return torch.zeros(B, N, ROOT_STRUCTURAL_FEATURE_DIM, device=joint_pos.device, dtype=joint_pos.dtype)


@torch.no_grad()
def compute_root_structural_features(
    joint_pos: torch.Tensor,
    active_mask: torch.Tensor,
    candidate_indices: Optional[torch.Tensor] = None,
    candidate_mask: Optional[torch.Tensor] = None,
    pair_logits: Optional[torch.Tensor] = None,
    k: int = 12,
) -> torch.Tensor:
    """Compute structural per-node root features.

    Returns tensor [B, N, F] where F == ROOT_STRUCTURAL_FEATURE_DIM.
    If candidate indices are not provided, kNN candidates are derived from geometry.
    Any pair_logits contribution is detached by design to avoid circular gradients.
    """
    if joint_pos.dim() != 3 or joint_pos.size(-1) != 3:
        raise ValueError("joint_pos must be [B, N, 3]")

    B, N, _ = joint_pos.shape
    active_mask = active_mask.bool()

    # CPU path is much faster for rank/in-degree style scalar-heavy operations.
    joint_pos_cpu = joint_pos.detach().cpu().float()
    active_mask_cpu = active_mask.detach().cpu()
    out = torch.zeros(B, N, ROOT_STRUCTURAL_FEATURE_DIM, dtype=joint_pos_cpu.dtype)
    candidate_indices_cpu = candidate_indices.detach().cpu() if candidate_indices is not None else None
    candidate_mask_cpu = candidate_mask.detach().cpu() if candidate_mask is not None else None
    detached_pair_logits_cpu = pair_logits.detach().cpu().float() if pair_logits is not None else None

    for b in range(B):
        active_idx = torch.where(active_mask_cpu[b])[0]
        n_active = int(active_idx.numel())
        if n_active == 0:
            continue

        pos_all = joint_pos_cpu[b]
        pos_active = pos_all[active_idx]

        centroid = pos_active.mean(dim=0, keepdim=True)
        centered = pos_all - centroid

        bb_min = pos_active.min(dim=0).values
        bb_max = pos_active.max(dim=0).values
        bbox_diag = torch.norm(bb_max - bb_min).clamp(min=1e-6)

        norm_xyz = centered / bbox_diag

        dist_to_centroid = torch.norm(centered, dim=-1)
        active_dist = dist_to_centroid[active_idx]
        radius = active_dist.max().clamp(min=1e-6)
        norm_radius = dist_to_centroid / radius

        # Radius percentile among active nodes.
        radius_percentile = torch.zeros(N, dtype=pos_all.dtype)
        if n_active > 1:
            order = torch.argsort(active_dist)
            ranks = torch.empty_like(order)
            ranks[order] = torch.arange(n_active, device=order.device)
            radius_percentile_active = ranks.float() / float(max(n_active - 1, 1))
            radius_percentile[active_idx] = radius_percentile_active

        # Pairwise distances over active nodes.
        dmat = torch.cdist(pos_active, pos_active)
        dmat.fill_diagonal_(float("inf"))
        sorted_d, sorted_idx = torch.sort(dmat, dim=1)

        nn1 = torch.zeros(N, dtype=pos_all.dtype)
        nn2 = torch.zeros(N, dtype=pos_all.dtype)
        local_density = torch.zeros(N, dtype=pos_all.dtype)

        if n_active > 1:
            nn1_active = sorted_d[:, 0]
            nn2_active = sorted_d[:, 1] if n_active > 2 else sorted_d[:, 0]
            k_eff = min(max(int(k), 1), n_active - 1)
            local_density_active = sorted_d[:, :k_eff].mean(dim=1)

            nn1[active_idx] = nn1_active
            nn2[active_idx] = nn2_active
            local_density[active_idx] = local_density_active

        # Candidate graph statistics: indegree and incoming rank stats.
        indegree_active = torch.zeros(n_active, dtype=pos_all.dtype)
        rank_sum_active = torch.zeros(n_active, dtype=pos_all.dtype)
        rank_min_active = torch.full((n_active,), float(max(int(k), 1) + 1), dtype=pos_all.dtype)
        rank_max_active = torch.zeros(n_active, dtype=pos_all.dtype)
        incoming_score_sum_active = torch.zeros(n_active, dtype=pos_all.dtype)
        incoming_score_max_active = torch.full((n_active,), -1e6, dtype=pos_all.dtype)

        global_to_local = {int(g.item()): i for i, g in enumerate(active_idx)}

        for src_local, src_global_t in enumerate(active_idx):
            src_global = int(src_global_t.item())
            if candidate_indices_cpu is not None and candidate_mask_cpu is not None:
                cand_g = candidate_indices_cpu[b, src_global]
                cand_m = candidate_mask_cpu[b, src_global].bool()
                cand_list = [int(c) for c, m in zip(cand_g.tolist(), cand_m.tolist()) if m]
            else:
                if n_active <= 1:
                    cand_list = []
                else:
                    k_eff = min(max(int(k), 1), n_active - 1)
                    neigh_local = sorted_idx[src_local, :k_eff]
                    cand_list = [int(active_idx[nl].item()) for nl in neigh_local.tolist()]

            for rank0, tgt_global in enumerate(cand_list):
                tgt_local = global_to_local.get(tgt_global)
                if tgt_local is None:
                    continue
                rank = rank0 + 1
                indegree_active[tgt_local] += 1.0
                rank_sum_active[tgt_local] += float(rank)
                rank_val = float(rank)
                if rank_val < float(rank_min_active[tgt_local]):
                    rank_min_active[tgt_local] = rank_val
                if rank_val > float(rank_max_active[tgt_local]):
                    rank_max_active[tgt_local] = rank_val

                if detached_pair_logits_cpu is not None:
                    score = detached_pair_logits_cpu[b, src_global, tgt_global]
                    incoming_score_sum_active[tgt_local] += score
                    if float(score) > float(incoming_score_max_active[tgt_local]):
                        incoming_score_max_active[tgt_local] = score

        indegree = torch.zeros(N, dtype=pos_all.dtype)
        candidate_centrality = torch.zeros(N, dtype=pos_all.dtype)
        mean_in_rank = torch.zeros(N, dtype=pos_all.dtype)
        min_in_rank = torch.zeros(N, dtype=pos_all.dtype)
        max_in_rank = torch.zeros(N, dtype=pos_all.dtype)
        mean_in_pair_score = torch.zeros(N, dtype=pos_all.dtype)
        max_in_pair_score = torch.zeros(N, dtype=pos_all.dtype)

        if n_active > 0:
            indegree[active_idx] = indegree_active
            candidate_centrality[active_idx] = indegree_active / float(max(n_active, 1))

            has_in = indegree_active > 0
            default_rank = torch.full_like(indegree_active, float(max(int(k), 1) + 1))
            mean_rank_active = torch.where(has_in, rank_sum_active / indegree_active.clamp(min=1.0), default_rank)
            min_rank_active = torch.where(has_in, rank_min_active, default_rank)
            max_rank_active = torch.where(has_in, rank_max_active, default_rank)

            mean_in_rank[active_idx] = mean_rank_active
            min_in_rank[active_idx] = min_rank_active
            max_in_rank[active_idx] = max_rank_active

            if detached_pair_logits_cpu is not None:
                mean_score_active = torch.where(has_in, incoming_score_sum_active / indegree_active.clamp(min=1.0), torch.zeros_like(incoming_score_sum_active))
                max_score_active = torch.where(has_in, incoming_score_max_active, torch.zeros_like(incoming_score_max_active))
                mean_in_pair_score[active_idx] = mean_score_active
                max_in_pair_score[active_idx] = max_score_active

        active_count_norm = torch.full((N,), float(n_active) / float(max(N, 1)), dtype=pos_all.dtype)
        bbox_feat = torch.full((N,), float(bbox_diag.item()), dtype=pos_all.dtype)

        out[b, :, 0:3] = norm_xyz
        out[b, :, 3] = dist_to_centroid
        out[b, :, 4] = norm_radius
        out[b, :, 5] = radius_percentile
        out[b, :, 6] = local_density
        out[b, :, 7] = nn1
        out[b, :, 8] = nn2
        out[b, :, 9] = indegree
        out[b, :, 10] = candidate_centrality
        out[b, :, 11] = mean_in_rank
        out[b, :, 12] = min_in_rank
        out[b, :, 13] = max_in_rank
        out[b, :, 14] = active_count_norm
        out[b, :, 15] = bbox_feat
        out[b, :, 16] = mean_in_pair_score
        out[b, :, 17] = max_in_pair_score

        # Zero out inactive rows to keep feature support on valid GT nodes.
        inactive = ~active_mask_cpu[b]
        if inactive.any():
            out[b, inactive] = 0.0

    return out.to(joint_pos.device, dtype=joint_pos.dtype)
