from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def build_undirected_adjacency(gt_adj: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    """Build symmetric undirected adjacency over active nodes without self edges."""
    adj = (gt_adj > 0.5).bool()
    adj = adj | adj.transpose(1, 2)
    adj = adj & (~torch.eye(adj.shape[1], device=adj.device, dtype=torch.bool).unsqueeze(0))
    active_pair = active_mask.unsqueeze(1) & active_mask.unsqueeze(2)
    return adj & active_pair


def build_undirected_knn_candidates(
    joint_pos: torch.Tensor,
    active_mask: torch.Tensor,
    k: int,
    gt_adj: torch.Tensor | None = None,
    force_include_gt: bool = False,
) -> Dict[str, torch.Tensor]:
    """Build symmetric k-NN edge candidates and optional GT-neighbor forcing."""
    device = joint_pos.device
    batch_size, max_nodes, _ = joint_pos.shape
    candidate_mask = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.bool, device=device)

    for b in range(batch_size):
        active_idx = torch.where(active_mask[b])[0]
        n_active = int(active_idx.numel())
        if n_active <= 1:
            continue

        pos = joint_pos[b, active_idx]
        dmat = torch.cdist(pos, pos)
        dmat.fill_diagonal_(float("inf"))
        k_eff = min(max(int(k), 1), n_active - 1)
        _, knn_local = torch.topk(dmat, k=k_eff, dim=1, largest=False)

        for src_local, src_global_t in enumerate(active_idx):
            src_global = int(src_global_t.item())
            neighbors_global = active_idx[knn_local[src_local]].tolist()
            for dst_global in neighbors_global:
                if src_global == int(dst_global):
                    continue
                candidate_mask[b, src_global, int(dst_global)] = True
                candidate_mask[b, int(dst_global), src_global] = True

        if force_include_gt and gt_adj is not None:
            gt_edges = (gt_adj[b] > 0.5) & active_mask[b].unsqueeze(0) & active_mask[b].unsqueeze(1)
            candidate_mask[b] |= gt_edges
            candidate_mask[b] |= gt_edges.transpose(0, 1)

        candidate_mask[b].fill_diagonal_(False)

    coverage = compute_candidate_coverage(candidate_mask, gt_adj, active_mask) if gt_adj is not None else torch.tensor(0.0, device=device)
    return {"candidate_mask": candidate_mask, "candidate_coverage": coverage}


def compute_candidate_coverage(candidate_mask: torch.Tensor, gt_adj: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    gt_undirected = build_undirected_adjacency(gt_adj, active_mask)
    tri = torch.triu(torch.ones_like(gt_undirected, dtype=torch.bool), diagonal=1)
    gt_edges = gt_undirected & tri
    covered = gt_edges & candidate_mask & tri
    denom = gt_edges.sum().clamp(min=1)
    return covered.sum().float() / denom.float()


def symmetrize_pair_scores(pair_logits: torch.Tensor, mode: str = "average") -> torch.Tensor:
    if mode == "average":
        return 0.5 * (pair_logits + pair_logits.transpose(1, 2))
    if mode == "max":
        return torch.maximum(pair_logits, pair_logits.transpose(1, 2))
    raise ValueError(f"Unknown symmetrize mode: {mode}")


def compute_edge_pair_features(
    joint_pos: torch.Tensor,
    active_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Compute engineered pairwise features [B,N,N,F] for undirected candidate edges."""
    device = joint_pos.device
    batch_size, max_nodes, _ = joint_pos.shape
    feat_dim = 12
    feats = torch.zeros((batch_size, max_nodes, max_nodes, feat_dim), device=device, dtype=joint_pos.dtype)

    for b in range(batch_size):
        active_idx = torch.where(active_mask[b])[0]
        n_active = int(active_idx.numel())
        if n_active <= 1:
            continue

        pos = joint_pos[b, active_idx]
        dmat = torch.cdist(pos, pos)
        dmat.fill_diagonal_(float("inf"))

        k_eff = min(max(int(k), 1), n_active - 1)
        sorted_d, sorted_i = torch.sort(dmat, dim=1)
        local_density = sorted_d[:, :k_eff].mean(dim=1)
        density_norm = local_density / local_density.mean().clamp(min=1e-6)

        # Radius-like feature from centroid distance.
        centroid = pos.mean(dim=0, keepdim=True)
        radius = torch.norm(pos - centroid, dim=1)
        radius_norm = radius / radius.max().clamp(min=1e-6)

        # Candidate degree prior from candidate mask.
        cand_sub = candidate_mask[b][active_idx][:, active_idx]
        degree_prior = cand_sub.float().sum(dim=1) / float(max(n_active - 1, 1))

        # Rank maps in kNN order: lower is better.
        rank_map = torch.full((n_active, n_active), float(k_eff + 1), device=device, dtype=joint_pos.dtype)
        for i_local in range(n_active):
            neigh = sorted_i[i_local, :k_eff]
            rank_map[i_local, neigh] = torch.arange(1, k_eff + 1, device=device, dtype=joint_pos.dtype)

        radius_i = radius_norm.unsqueeze(1).expand(n_active, n_active)
        radius_j = radius_norm.unsqueeze(0).expand(n_active, n_active)
        local_i = density_norm.unsqueeze(1).expand(n_active, n_active)
        local_j = density_norm.unsqueeze(0).expand(n_active, n_active)
        degree_i = degree_prior.unsqueeze(1).expand(n_active, n_active)
        degree_j = degree_prior.unsqueeze(0).expand(n_active, n_active)

        dist = dmat.clone()
        finite_d = dist[torch.isfinite(dist)]
        dist_norm = dist / finite_d.max().clamp(min=1e-6)
        dist_percentile = torch.argsort(torch.argsort(dist, dim=1), dim=1).float() / float(max(n_active - 1, 1))

        rank_ij = rank_map / float(max(k_eff, 1))
        rank_ji = rank_map.transpose(0, 1) / float(max(k_eff, 1))
        radius_diff = (radius_i - radius_j).abs()
        midpoint_radius = 0.5 * (radius_i + radius_j)
        mutual_knn = ((rank_map <= k_eff) & (rank_map.transpose(0, 1) <= k_eff)).float()

        sub_feats = torch.stack(
            [
                dist,
                dist_norm,
                dist_percentile,
                local_i,
                local_j,
                degree_i,
                degree_j,
                rank_ij,
                rank_ji,
                radius_diff,
                midpoint_radius,
                mutual_knn,
            ],
            dim=-1,
        )

        # Keep only finite values and zero inactive diagonals.
        sub_feats = torch.nan_to_num(sub_feats, nan=0.0, posinf=0.0, neginf=0.0)
        idx = active_idx
        feats[b][idx.unsqueeze(1), idx.unsqueeze(0)] = sub_feats

    return feats


def compute_topology_edge_inputs(
    joint_pos: torch.Tensor,
    active_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    k: int,
) -> Dict[str, torch.Tensor]:
    """Build v2.16 edge-scoring inputs: rich pair features + node/local/global context."""
    device = joint_pos.device
    dtype = joint_pos.dtype
    batch_size, max_nodes, _ = joint_pos.shape

    pair_feat_dim = 23
    node_local_dim = 8
    global_dim = 4

    pair_features = torch.zeros((batch_size, max_nodes, max_nodes, pair_feat_dim), device=device, dtype=dtype)
    node_local_features = torch.zeros((batch_size, max_nodes, node_local_dim), device=device, dtype=dtype)
    global_context = torch.zeros((batch_size, global_dim), device=device, dtype=dtype)

    for b in range(batch_size):
        active_idx = torch.where(active_mask[b])[0]
        n_active = int(active_idx.numel())
        if n_active <= 1:
            continue

        pos = joint_pos[b, active_idx]
        dmat = torch.cdist(pos, pos)
        dmat.fill_diagonal_(float("inf"))

        k_eff = min(max(int(k), 1), n_active - 1)
        sorted_d, sorted_i = torch.sort(dmat, dim=1)
        knn_d = sorted_d[:, :k_eff]

        finite_d = dmat[torch.isfinite(dmat)]
        dist_max = finite_d.max().clamp(min=1e-6)

        mean_knn = knn_d.mean(dim=1)
        min_knn = knn_d[:, 0]
        med_knn = knn_d[:, (k_eff - 1) // 2]
        max_knn = knn_d[:, k_eff - 1]

        centroid = pos.mean(dim=0, keepdim=True)
        radius = torch.norm(pos - centroid, dim=1)
        radius_rank = torch.argsort(torch.argsort(radius))
        radius_pct = radius_rank.float() / float(max(n_active - 1, 1))

        density_rank = torch.argsort(torch.argsort(mean_knn, descending=True))
        density_pct = density_rank.float() / float(max(n_active - 1, 1))

        cand_sub = candidate_mask[b][active_idx][:, active_idx]
        out_deg = cand_sub.float().sum(dim=1) / float(max(n_active - 1, 1))
        in_deg = cand_sub.float().sum(dim=0) / float(max(n_active - 1, 1))

        local_sub = torch.stack(
            [
                mean_knn / dist_max,
                min_knn / dist_max,
                med_knn / dist_max,
                max_knn / dist_max,
                radius_pct,
                out_deg,
                in_deg,
                density_pct,
            ],
            dim=-1,
        )
        node_local_features[b, active_idx] = torch.nan_to_num(local_sub, nan=0.0, posinf=0.0, neginf=0.0)

        rank_map = torch.full((n_active, n_active), float(k_eff + 1), device=device, dtype=dtype)
        for i_local in range(n_active):
            neigh = sorted_i[i_local, :k_eff]
            rank_map[i_local, neigh] = torch.arange(1, k_eff + 1, device=device, dtype=dtype)

        xyz_i = pos.unsqueeze(1).expand(n_active, n_active, 3)
        xyz_j = pos.unsqueeze(0).expand(n_active, n_active, 3)
        delta = xyz_j - xyz_i
        dist = torch.norm(delta, dim=-1, keepdim=True)
        norm_dist = dist / dist_max

        rank_ij = (rank_map / float(max(k_eff, 1))).unsqueeze(-1)
        rank_ji = (rank_map.transpose(0, 1) / float(max(k_eff, 1))).unsqueeze(-1)
        local_i = density_pct.unsqueeze(1).expand(n_active, n_active).unsqueeze(-1)
        local_j = density_pct.unsqueeze(0).expand(n_active, n_active).unsqueeze(-1)
        radius_i = radius_pct.unsqueeze(1).expand(n_active, n_active).unsqueeze(-1)
        radius_j = radius_pct.unsqueeze(0).expand(n_active, n_active).unsqueeze(-1)
        midpoint = 0.5 * (xyz_i + xyz_j)
        mutual = ((rank_map <= k_eff) & (rank_map.transpose(0, 1) <= k_eff)).float().unsqueeze(-1)
        deg_i = out_deg.unsqueeze(1).expand(n_active, n_active).unsqueeze(-1)
        deg_j = out_deg.unsqueeze(0).expand(n_active, n_active).unsqueeze(-1)

        sub_pair = torch.cat(
            [
                xyz_i,
                xyz_j,
                delta,
                dist,
                norm_dist,
                rank_ij,
                rank_ji,
                local_i,
                local_j,
                radius_i,
                radius_j,
                midpoint,
                mutual,
                deg_i,
                deg_j,
            ],
            dim=-1,
        )
        sub_pair = torch.nan_to_num(sub_pair, nan=0.0, posinf=0.0, neginf=0.0)

        pair_features[b][active_idx.unsqueeze(1), active_idx.unsqueeze(0)] = sub_pair

        active_ratio = float(n_active) / float(max(max_nodes, 1))
        global_context[b] = torch.tensor(
            [
                active_ratio,
                float(mean_knn.mean().item() / dist_max.item()),
                float(radius.mean().item() / radius.max().clamp(min=1e-6).item()),
                float(cand_sub.float().sum().item() / max(float(n_active * max(n_active - 1, 1)), 1.0)),
            ],
            device=device,
            dtype=dtype,
        )

    return {
        "pair_features": pair_features,
        "node_local_features": node_local_features,
        "global_context": global_context,
    }


class EdgeScoreRefiner(nn.Module):
    """Refine base symmetric edge logits using engineered pair features."""

    def __init__(self, feature_dim: int = 12, hidden_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, base_scores: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([base_scores.unsqueeze(-1), edge_features], dim=-1)
        return self.mlp(x).squeeze(-1)


def decode_undirected_edges(
    sym_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    active_mask: torch.Tensor,
    mode: str,
    threshold: float = 0.5,
    budget_count: int | None = None,
    budget_ratio: float | None = None,
) -> torch.Tensor:
    """Decode symmetric undirected edges as [N,N] bool mask."""
    device = sym_scores.device
    n_nodes = sym_scores.shape[0]
    edge_mask = torch.zeros((n_nodes, n_nodes), dtype=torch.bool, device=device)

    valid = candidate_mask & active_mask.unsqueeze(0) & active_mask.unsqueeze(1)
    valid.fill_diagonal_(False)
    upper_valid = valid & torch.triu(torch.ones_like(valid, dtype=torch.bool), diagonal=1)

    if mode == "threshold":
        prob = torch.sigmoid(sym_scores)
        chosen = upper_valid & (prob >= float(threshold))
    elif mode == "top_e_budget":
        idx = torch.where(upper_valid)
        if idx[0].numel() == 0:
            chosen = upper_valid & False
        else:
            scores = sym_scores[idx]
            top_e = max(0, int(budget_count) if budget_count is not None else 0)
            top_e = min(top_e, scores.numel())
            chosen = upper_valid & False
            if top_e > 0:
                top_ids = torch.topk(scores, k=top_e, largest=True).indices
                chosen[idx[0][top_ids], idx[1][top_ids]] = True
    elif mode == "ratio_budget":
        idx = torch.where(upper_valid)
        if idx[0].numel() == 0:
            chosen = upper_valid & False
        else:
            scores = sym_scores[idx]
            active_count = int(active_mask.sum().item())
            ratio = float(0.0 if budget_ratio is None else budget_ratio)
            budget = max(0, int(round(ratio * max(active_count, 1))))
            budget = min(budget, scores.numel())
            chosen = upper_valid & False
            if budget > 0:
                top_ids = torch.topk(scores, k=budget, largest=True).indices
                chosen[idx[0][top_ids], idx[1][top_ids]] = True
    elif mode == "mst_forest":
        chosen = _decode_mst_forest(sym_scores, upper_valid, active_mask)
    else:
        raise ValueError(f"Unknown decode mode: {mode}")

    edge_mask |= chosen
    edge_mask |= chosen.transpose(0, 1)
    edge_mask.fill_diagonal_(False)
    return edge_mask


def _decode_mst_forest(sym_scores: torch.Tensor, upper_valid: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    """Maximum spanning forest over available candidates."""
    chosen = upper_valid & False
    active_idx = torch.where(active_mask)[0].tolist()
    if len(active_idx) <= 1:
        return chosen

    parent = {n: n for n in active_idx}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    edges = []
    idx = torch.where(upper_valid)
    for i, j in zip(idx[0].tolist(), idx[1].tolist()):
        edges.append((float(sym_scores[i, j].item()), int(i), int(j)))
    edges.sort(key=lambda x: x[0], reverse=True)

    for _, i, j in edges:
        if union(i, j):
            chosen[i, j] = True

    return chosen


def edge_prf(pred_edges: torch.Tensor, gt_adj: torch.Tensor, active_mask: torch.Tensor) -> Dict[str, float]:
    gt_edges = build_undirected_adjacency(gt_adj.unsqueeze(0), active_mask.unsqueeze(0))[0]
    tri = torch.triu(torch.ones_like(gt_edges, dtype=torch.bool), diagonal=1)
    pred_u = pred_edges & tri
    gt_u = gt_edges & tri

    tp = int((pred_u & gt_u).sum().item())
    fp = int((pred_u & (~gt_u)).sum().item())
    fn = int(((~pred_u) & gt_u).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def graph_stats(edge_mask: torch.Tensor, active_mask: torch.Tensor) -> Dict[str, float]:
    active_nodes = torch.where(active_mask)[0].tolist()
    n_active = len(active_nodes)
    if n_active == 0:
        return {"component_count": 0.0, "average_degree": 0.0, "max_degree": 0.0, "cycle_count": 0.0}

    adj = edge_mask.bool()
    degrees = adj[active_mask][:, active_mask].sum(dim=1).float()

    seen = set()
    comp_count = 0
    for n in active_nodes:
        if n in seen:
            continue
        comp_count += 1
        stack = [n]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            neighbors = torch.where(adj[cur] & active_mask)[0].tolist()
            for nei in neighbors:
                if nei not in seen:
                    stack.append(nei)

    edge_count = int((adj & torch.triu(torch.ones_like(adj, dtype=torch.bool), diagonal=1)).sum().item())
    cycle_count = max(edge_count - n_active + comp_count, 0)
    return {
        "component_count": float(comp_count),
        "average_degree": float(degrees.mean().item()) if degrees.numel() > 0 else 0.0,
        "max_degree": float(degrees.max().item()) if degrees.numel() > 0 else 0.0,
        "cycle_count": float(cycle_count),
    }
