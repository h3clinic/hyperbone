from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass(frozen=True)
class EdgeCandidate:
    i: int
    j: int
    dist: float
    score: float


def _active_indices(active_mask: torch.Tensor) -> List[int]:
    return torch.where(active_mask)[0].tolist()


def _pairwise_dist(pos: torch.Tensor) -> torch.Tensor:
    dmat = torch.cdist(pos, pos)
    dmat.fill_diagonal_(float("inf"))
    return dmat


def _build_knn_candidate_list(joint_pos: torch.Tensor, active_mask: torch.Tensor, k: int) -> Tuple[List[EdgeCandidate], Dict[int, int], torch.Tensor, torch.Tensor]:
    active_idx = _active_indices(active_mask)
    n_active = len(active_idx)
    if n_active <= 1:
        return [], {}, torch.empty(0, 0, device=joint_pos.device), torch.empty(0, dtype=torch.long, device=joint_pos.device)

    idx_t = torch.tensor(active_idx, dtype=torch.long, device=joint_pos.device)
    pos = joint_pos[idx_t]
    dmat = _pairwise_dist(pos)

    k_eff = min(max(int(k), 1), n_active - 1)
    _, knn_local = torch.topk(dmat, k=k_eff, largest=False, dim=1)

    edge_map: Dict[Tuple[int, int], EdgeCandidate] = {}
    local_of_global = {int(g): li for li, g in enumerate(active_idx)}
    for i_local in range(n_active):
        i_global = int(active_idx[i_local])
        for j_local in knn_local[i_local].tolist():
            j_global = int(active_idx[int(j_local)])
            if i_global == j_global:
                continue
            a, b = (i_global, j_global) if i_global < j_global else (j_global, i_global)
            dist = float(dmat[i_local, int(j_local)].item())
            score = -dist
            if (a, b) not in edge_map or dist < edge_map[(a, b)].dist:
                edge_map[(a, b)] = EdgeCandidate(i=a, j=b, dist=dist, score=score)

    return list(edge_map.values()), local_of_global, dmat, idx_t


def _empty_edge_mask(num_nodes: int, device: torch.device) -> torch.Tensor:
    return torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)


def _set_undirected(mask: torch.Tensor, i: int, j: int) -> None:
    mask[i, j] = True
    mask[j, i] = True


def _kruskal_max_score(
    num_nodes: int,
    active_nodes: List[int],
    edges: List[EdgeCandidate],
    max_edges: int | None = None,
    degree_cap: int | None = None,
) -> torch.Tensor:
    if max_edges is None:
        max_edges = max(len(active_nodes) - 1, 0)

    device = torch.device("cpu")
    edge_mask = _empty_edge_mask(num_nodes, device=device)
    if len(active_nodes) <= 1 or max_edges <= 0:
        return edge_mask

    parent = {n: n for n in active_nodes}
    rank = {n: 0 for n in active_nodes}
    degree = {n: 0 for n in active_nodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    selected = 0
    for e in sorted(edges, key=lambda x: (x.score, -x.dist), reverse=True):
        if selected >= max_edges:
            break
        if degree_cap is not None and (degree[e.i] >= degree_cap or degree[e.j] >= degree_cap):
            continue
        if union(e.i, e.j):
            _set_undirected(edge_mask, e.i, e.j)
            degree[e.i] += 1
            degree[e.j] += 1
            selected += 1

    if selected < min(max_edges, max(len(active_nodes) - 1, 0)):
        for e in sorted(edges, key=lambda x: (x.score, -x.dist), reverse=True):
            if selected >= max_edges:
                break
            if edge_mask[e.i, e.j]:
                continue
            if union(e.i, e.j):
                _set_undirected(edge_mask, e.i, e.j)
                degree[e.i] += 1
                degree[e.j] += 1
                selected += 1

    return edge_mask


def knn_mst(joint_pos: torch.Tensor, active_mask: torch.Tensor, k: int = 12) -> torch.Tensor:
    num_nodes = int(joint_pos.shape[0])
    edges, _, _, _ = _build_knn_candidate_list(joint_pos, active_mask, k=k)
    active_nodes = _active_indices(active_mask)
    mask = _kruskal_max_score(num_nodes, active_nodes, edges, max_edges=max(len(active_nodes) - 1, 0), degree_cap=None)
    return mask.to(joint_pos.device)


def degree_capped_mst(joint_pos: torch.Tensor, active_mask: torch.Tensor, k: int = 12, degree_cap: int = 3) -> torch.Tensor:
    num_nodes = int(joint_pos.shape[0])
    edges, _, _, _ = _build_knn_candidate_list(joint_pos, active_mask, k=k)
    active_nodes = _active_indices(active_mask)
    mask = _kruskal_max_score(
        num_nodes,
        active_nodes,
        edges,
        max_edges=max(len(active_nodes) - 1, 0),
        degree_cap=max(int(degree_cap), 1),
    )
    return mask.to(joint_pos.device)


def budgeted_knn_forest(joint_pos: torch.Tensor, active_mask: torch.Tensor, k: int = 12, budget_ratio: float = 1.0) -> torch.Tensor:
    num_nodes = int(joint_pos.shape[0])
    edges, _, _, _ = _build_knn_candidate_list(joint_pos, active_mask, k=k)
    active_nodes = _active_indices(active_mask)
    budget = int(round(float(budget_ratio) * max(len(active_nodes), 1)))
    budget = max(0, min(budget, len(edges)))
    mask = _kruskal_max_score(num_nodes, active_nodes, edges, max_edges=budget, degree_cap=None)
    return mask.to(joint_pos.device)


def density_normalized_mst(joint_pos: torch.Tensor, active_mask: torch.Tensor, k: int = 12) -> torch.Tensor:
    num_nodes = int(joint_pos.shape[0])
    edges, local_of_global, dmat, idx_t = _build_knn_candidate_list(joint_pos, active_mask, k=k)
    active_nodes = _active_indices(active_mask)
    if len(active_nodes) <= 1:
        return _empty_edge_mask(num_nodes, device=joint_pos.device)

    k_eff = min(max(int(k), 1), len(active_nodes) - 1)
    sorted_d, _ = torch.sort(dmat, dim=1)
    local_density = sorted_d[:, :k_eff].mean(dim=1)

    norm_edges: List[EdgeCandidate] = []
    for e in edges:
        li = local_of_global[e.i]
        lj = local_of_global[e.j]
        denom = float((0.5 * (local_density[li] + local_density[lj])).item())
        score = -float(e.dist / max(denom, 1e-6))
        norm_edges.append(EdgeCandidate(i=e.i, j=e.j, dist=e.dist, score=score))

    mask = _kruskal_max_score(num_nodes, active_nodes, norm_edges, max_edges=max(len(active_nodes) - 1, 0), degree_cap=None)
    return mask.to(joint_pos.device)


def mutual_knn_sparse(joint_pos: torch.Tensor, active_mask: torch.Tensor, k: int = 12) -> torch.Tensor:
    device = joint_pos.device
    num_nodes = int(joint_pos.shape[0])
    mask = _empty_edge_mask(num_nodes, device=device)

    active_nodes = _active_indices(active_mask)
    n_active = len(active_nodes)
    if n_active <= 1:
        return mask

    idx_t = torch.tensor(active_nodes, dtype=torch.long, device=device)
    pos = joint_pos[idx_t]
    dmat = _pairwise_dist(pos)
    k_eff = min(max(int(k), 1), n_active - 1)
    _, knn_local = torch.topk(dmat, k=k_eff, largest=False, dim=1)

    knn_bool = torch.zeros((n_active, n_active), dtype=torch.bool, device=device)
    for i_local in range(n_active):
        knn_bool[i_local, knn_local[i_local]] = True

    mutual = knn_bool & knn_bool.transpose(0, 1)
    mutual.fill_diagonal_(False)
    tri = torch.triu(torch.ones_like(mutual, dtype=torch.bool), diagonal=1)
    idx = torch.where(mutual & tri)
    for i_local, j_local in zip(idx[0].tolist(), idx[1].tolist()):
        i_global = int(active_nodes[i_local])
        j_global = int(active_nodes[j_local])
        _set_undirected(mask, i_global, j_global)
    return mask


def hybrid_mst_plus_branches(joint_pos: torch.Tensor, active_mask: torch.Tensor, k: int = 12, extra_ratio: float = 0.25) -> torch.Tensor:
    device = joint_pos.device
    num_nodes = int(joint_pos.shape[0])
    edges, _, _, _ = _build_knn_candidate_list(joint_pos, active_mask, k=k)
    active_nodes = _active_indices(active_mask)
    if len(active_nodes) <= 1:
        return _empty_edge_mask(num_nodes, device=device)

    base = knn_mst(joint_pos, active_mask, k=k).to(device)

    budget_extra = int(round(float(extra_ratio) * max(len(active_nodes) - 1, 0)))
    if budget_extra <= 0:
        return base

    deg = base[active_mask][:, active_mask].sum(dim=1)
    local_nodes = {int(g): li for li, g in enumerate(active_nodes)}

    candidates = []
    for e in edges:
        if base[e.i, e.j]:
            continue
        li = local_nodes[e.i]
        lj = local_nodes[e.j]
        leaf_bonus = 0.0
        if int(deg[li].item()) <= 1:
            leaf_bonus += 0.1
        if int(deg[lj].item()) <= 1:
            leaf_bonus += 0.1
        candidates.append((e.score + leaf_bonus, e))

    candidates.sort(key=lambda x: x[0], reverse=True)
    added = 0
    for _, e in candidates:
        if added >= budget_extra:
            break
        if not base[e.i, e.j]:
            _set_undirected(base, e.i, e.j)
            added += 1

    return base


OPTIMIZER_REGISTRY = {
    "knn_mst": knn_mst,
    "degree_capped_mst": degree_capped_mst,
    "budgeted_knn_forest": budgeted_knn_forest,
    "density_normalized_mst": density_normalized_mst,
    "mutual_knn_sparse": mutual_knn_sparse,
    "hybrid_mst_plus_branches": hybrid_mst_plus_branches,
}


def _build_knn_meta(joint_pos: torch.Tensor, active_mask: torch.Tensor, k: int) -> Tuple[List[int], torch.Tensor, torch.Tensor, int]:
    active_nodes = _active_indices(active_mask)
    if len(active_nodes) <= 1:
        return active_nodes, torch.empty(0, 0, device=joint_pos.device), torch.empty(0, 0, dtype=torch.bool, device=joint_pos.device), 0

    idx_t = torch.tensor(active_nodes, dtype=torch.long, device=joint_pos.device)
    pos = joint_pos[idx_t]
    dmat = _pairwise_dist(pos)
    k_eff = min(max(int(k), 1), len(active_nodes) - 1)
    _, knn_local = torch.topk(dmat, k=k_eff, largest=False, dim=1)
    knn_bool = torch.zeros((len(active_nodes), len(active_nodes)), dtype=torch.bool, device=joint_pos.device)
    for i_local in range(len(active_nodes)):
        knn_bool[i_local, knn_local[i_local]] = True
    return active_nodes, dmat, knn_bool, k_eff


def _build_hybrid_candidates(
    joint_pos: torch.Tensor,
    active_mask: torch.Tensor,
    neural_scores: torch.Tensor | None,
    k: int,
    distance_weight: float,
    neural_weight: float,
    mutual_bonus: float,
    long_edge_penalty: float,
    candidate_mask: torch.Tensor | None,
) -> List[EdgeCandidate]:
    active_nodes, dmat, knn_bool, k_eff = _build_knn_meta(joint_pos, active_mask, k)
    if len(active_nodes) <= 1:
        return []

    sorted_d, _ = torch.sort(dmat, dim=1)
    local_density = sorted_d[:, :k_eff].mean(dim=1)

    finite_d = dmat[torch.isfinite(dmat)]
    dist_max = finite_d.max().clamp(min=1e-6)
    local_of_global = {int(g): li for li, g in enumerate(active_nodes)}

    edges: List[EdgeCandidate] = []
    for ia in range(len(active_nodes)):
        gi = int(active_nodes[ia])
        for ja in range(ia + 1, len(active_nodes)):
            gj = int(active_nodes[ja])
            if candidate_mask is not None and not bool(candidate_mask[gi, gj].item()):
                continue

            dist = float(dmat[ia, ja].item())
            norm_dist = float((dmat[ia, ja] / dist_max).item())
            denom = float((0.5 * (local_density[ia] + local_density[ja])).item())
            density_norm = float(dist / max(denom, 1e-6))
            mutual = 1.0 if bool(knn_bool[ia, ja].item() and knn_bool[ja, ia].item()) else 0.0
            neural_term = 0.0 if neural_scores is None else float(neural_scores[gi, gj].item())

            det_cost = distance_weight * density_norm + long_edge_penalty * norm_dist
            score = -det_cost + neural_weight * neural_term + mutual_bonus * mutual
            edges.append(EdgeCandidate(i=gi, j=gj, dist=dist, score=score))

    return edges


def _kruskal_dynamic_score(
    num_nodes: int,
    active_nodes: List[int],
    edges: List[EdgeCandidate],
    max_edges: int,
    degree_cap: int | None,
    degree_penalty: float,
) -> torch.Tensor:
    device = torch.device("cpu")
    mask = _empty_edge_mask(num_nodes, device=device)
    if len(active_nodes) <= 1 or max_edges <= 0 or not edges:
        return mask

    parent = {n: n for n in active_nodes}
    rank = {n: 0 for n in active_nodes}
    degree = {n: 0 for n in active_nodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    selected = 0
    remaining = set(range(len(edges)))
    while selected < max_edges and remaining:
        best_idx = None
        best_util = -1e18
        for idx in remaining:
            e = edges[idx]
            if degree_cap is not None and (degree[e.i] >= degree_cap or degree[e.j] >= degree_cap):
                continue
            if find(e.i) == find(e.j):
                continue
            util = float(e.score - degree_penalty * float(degree[e.i] + degree[e.j]))
            if util > best_util:
                best_util = util
                best_idx = idx

        if best_idx is None:
            break

        e = edges[best_idx]
        union(e.i, e.j)
        _set_undirected(mask, e.i, e.j)
        degree[e.i] += 1
        degree[e.j] += 1
        selected += 1
        remaining.remove(best_idx)

    return mask


def _add_branch_edges(
    base_mask: torch.Tensor,
    active_nodes: List[int],
    edges: List[EdgeCandidate],
    extra_edges: int,
    degree_cap: int | None,
    degree_penalty: float,
    branch_bonus: float,
) -> torch.Tensor:
    if extra_edges <= 0:
        return base_mask

    out = base_mask.clone()
    degree = {n: int(out[n].sum().item()) for n in active_nodes}

    candidates = []
    for e in edges:
        if out[e.i, e.j]:
            continue
        if degree_cap is not None and (degree[e.i] >= degree_cap or degree[e.j] >= degree_cap):
            continue
        bonus = 0.0
        if degree[e.i] <= 1:
            bonus += branch_bonus
        if degree[e.j] <= 1:
            bonus += branch_bonus
        util = float(e.score + bonus - degree_penalty * float(degree[e.i] + degree[e.j]))
        candidates.append((util, e))

    candidates.sort(key=lambda x: x[0], reverse=True)
    added = 0
    for _, e in candidates:
        if added >= extra_edges:
            break
        if out[e.i, e.j]:
            continue
        if degree_cap is not None and (degree[e.i] >= degree_cap or degree[e.j] >= degree_cap):
            continue
        _set_undirected(out, e.i, e.j)
        degree[e.i] += 1
        degree[e.j] += 1
        added += 1
    return out


def hybrid_neural_cost_optimize(
    joint_pos: torch.Tensor,
    active_mask: torch.Tensor,
    neural_scores: torch.Tensor | None,
    mode: str = "density_normalized_mst",
    k: int = 12,
    distance_weight: float = 1.0,
    neural_weight: float = 1.0,
    degree_penalty: float = 0.05,
    long_edge_penalty: float = 0.25,
    mutual_bonus: float = 0.2,
    max_degree: int = 4,
    branch_extra_ratio: float = 0.25,
    branch_bonus: float = 0.15,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """v3.1 hybrid optimizer: deterministic structural cost + optional neural score bonus."""
    num_nodes = int(joint_pos.shape[0])
    active_nodes = _active_indices(active_mask)
    if len(active_nodes) <= 1:
        return _empty_edge_mask(num_nodes, device=joint_pos.device)

    edges = _build_hybrid_candidates(
        joint_pos=joint_pos,
        active_mask=active_mask,
        neural_scores=neural_scores,
        k=k,
        distance_weight=float(distance_weight),
        neural_weight=float(neural_weight),
        mutual_bonus=float(mutual_bonus),
        long_edge_penalty=float(long_edge_penalty),
        candidate_mask=candidate_mask,
    )
    if not edges:
        return _empty_edge_mask(num_nodes, device=joint_pos.device)

    tree_edges = max(len(active_nodes) - 1, 0)
    if mode == "density_normalized_mst":
        out = _kruskal_dynamic_score(
            num_nodes=num_nodes,
            active_nodes=active_nodes,
            edges=edges,
            max_edges=tree_edges,
            degree_cap=None,
            degree_penalty=float(degree_penalty),
        )
    elif mode == "degree_capped_mst":
        out = _kruskal_dynamic_score(
            num_nodes=num_nodes,
            active_nodes=active_nodes,
            edges=edges,
            max_edges=tree_edges,
            degree_cap=max(int(max_degree), 1),
            degree_penalty=float(degree_penalty),
        )
    elif mode == "hybrid_mst_plus_branches":
        base = _kruskal_dynamic_score(
            num_nodes=num_nodes,
            active_nodes=active_nodes,
            edges=edges,
            max_edges=tree_edges,
            degree_cap=max(int(max_degree), 1),
            degree_penalty=float(degree_penalty),
        )
        extra_edges = int(round(float(branch_extra_ratio) * float(max(len(active_nodes) - 1, 0))))
        out = _add_branch_edges(
            base_mask=base,
            active_nodes=active_nodes,
            edges=edges,
            extra_edges=extra_edges,
            degree_cap=max(int(max_degree), 1),
            degree_penalty=float(degree_penalty),
            branch_bonus=float(branch_bonus),
        )
    else:
        raise ValueError(f"Unknown hybrid mode: {mode}")

    return out.to(joint_pos.device)
