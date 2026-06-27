from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class ParentDecodeConfig:
    active_threshold: float = 0.5
    decode_mode: str = "parent_argmax"
    max_degree: int = 4
    max_roots: int = 2
    cycle_break_root_mode: str = "lowest_confidence"
    enforce_root_per_component: bool = False
    two_stage_root_parent: bool = False
    root_threshold: float | None = None
    root_decode_mode: str = "threshold"
    root_budget_ratio: float | None = None


def decode_parent_graph(
    positions: np.ndarray,
    active_prob: np.ndarray,
    parent_logits: np.ndarray,
    parent_offset: np.ndarray | None = None,
    edge_confidence: np.ndarray | None = None,
    config: ParentDecodeConfig | None = None,
    root_decode_bias: float = 0.0,
    root_logits: np.ndarray | None = None,
    root_decode_mode: str | None = None,
    root_budget_ratio: float | None = None,
    root_count_budget: int | None = None,
) -> dict:
    if config is None:
        config = ParentDecodeConfig()

    positions = np.asarray(positions, dtype=np.float32)
    active_prob = np.asarray(active_prob, dtype=np.float32)
    parent_logits = np.asarray(parent_logits, dtype=np.float32)
    n_nodes = positions.shape[0]
    root_class = parent_logits.shape[-1] - 1

    if root_decode_bias != 0.0:
        parent_logits = parent_logits.copy()
        parent_logits[..., root_class] += float(root_decode_bias)

    active_mask = active_prob > config.active_threshold
    if config.two_stage_root_parent:
        raise ValueError("Use decode_two_stage_root_parent_graph for two-stage decoding")
    active_idx = np.where(active_mask)[0].tolist()
    parent_choice = np.argmax(parent_logits, axis=-1).astype(np.int64)

    if root_logits is None:
        root_scores = parent_logits[:, root_class]
    else:
        root_scores = np.asarray(root_logits, dtype=np.float32)
    if root_decode_mode is None:
        root_decode_mode = getattr(config, "root_decode_mode", "threshold")
    if root_budget_ratio is None:
        root_budget_ratio = getattr(config, "root_budget_ratio", None)

    root_scores = np.clip(root_scores, -50.0, 50.0)
    root_prob = 1.0 / (1.0 + np.exp(-root_scores))
    root_mask = np.zeros(n_nodes, dtype=bool)
    if root_decode_mode == "threshold":
        root_mask = (root_prob > config.root_threshold if config.root_threshold is not None else root_prob > config.active_threshold) & active_mask
    elif root_decode_mode == "topk_budget":
        if root_count_budget is None:
            root_count_budget = max(1, int(round(float(active_mask.sum()) * float(root_budget_ratio if root_budget_ratio is not None else 0.03))))
        active_scores = root_scores[active_mask]
        if active_scores.size > 0:
            active_order = np.argsort(-active_scores)
            active_candidates = np.where(active_mask)[0]
            keep = active_candidates[active_order[: min(int(root_count_budget), active_candidates.size)]]
            root_mask[keep] = True
    elif root_decode_mode == "ratio_budget":
        ratio = float(root_budget_ratio if root_budget_ratio is not None else 0.03)
        budget = max(1, int(round(ratio * float(active_mask.sum()))))
        active_scores = root_scores[active_mask]
        if active_scores.size > 0:
            active_order = np.argsort(-active_scores)
            active_candidates = np.where(active_mask)[0]
            keep = active_candidates[active_order[: min(budget, active_candidates.size)]]
            root_mask[keep] = True
    else:
        raise ValueError(f"Unknown root decode mode: {root_decode_mode}")

    if root_mask.sum() == 0 and active_idx:
        root_mask[np.argmax(root_scores[active_mask])] = True
    if root_mask.sum() > config.max_roots:
        root_order = np.argsort(-root_scores[root_mask])
        keep = np.where(root_mask)[0][root_order[: config.max_roots]]
        new_root_mask = np.zeros_like(root_mask, dtype=bool)
        new_root_mask[keep] = True
        root_mask = new_root_mask

    # Convert class choice to parent pointer in slot space; root class => -1.
    parent_ptr = np.full(n_nodes, -1, dtype=np.int64)
    for i in active_idx:
        if root_mask[i]:
            continue
        choice = int(parent_choice[i])
        if choice != root_class and 0 <= choice < n_nodes and choice != i:
            parent_ptr[i] = choice

    if config.decode_mode == "parent_argmax":
        parent_ptr = _allow_forest(parent_ptr, active_mask, config.max_roots)
    elif config.decode_mode == "parent_argmax_acyclic":
        parent_ptr = _break_cycles(parent_ptr, active_mask, active_prob)
    elif config.decode_mode == "parent_mst_hybrid":
        parent_ptr = _parent_mst_hybrid(
            positions, active_mask, parent_logits, parent_ptr, config,
        )
    else:
        raise ValueError(f"Unknown parent decode mode: {config.decode_mode}")

    parent_ptr = _apply_degree_cap(parent_ptr, active_mask, config.max_degree)
    if config.enforce_root_per_component:
        root_scores = parent_logits[:, root_class]
        parent_ptr = _enforce_root_per_component(parent_ptr, active_mask, root_scores)
    metadata = _graph_metadata(parent_ptr, active_mask, active_prob)
    metadata["mode"] = config.decode_mode
    metadata["active_threshold"] = config.active_threshold
    metadata["root_decode_mode"] = root_decode_mode
    metadata["root_count_pred"] = int(root_mask.sum())

    edges = []
    for child, parent in enumerate(parent_ptr):
        if parent >= 0 and active_mask[child]:
            edges.append((int(parent), int(child)))

    return {
        "active_mask": active_mask,
        "root_mask": root_mask,
        "parent_ptr": parent_ptr,
        "edges": edges,
        "edge_count": len(edges),
        "metadata": metadata,
        "parent_offset": np.zeros_like(positions) if parent_offset is None else np.asarray(parent_offset, dtype=np.float32),
    }


def decode_two_stage_root_parent_graph(
    positions: np.ndarray,
    active_prob: np.ndarray,
    root_logits: np.ndarray,
    parent_candidate_logits: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_mask: np.ndarray,
    config: ParentDecodeConfig | None = None,
    root_threshold: float | None = None,
    root_decode_mode: str | None = None,
    root_budget_ratio: float | None = None,
    root_count_budget: int | None = None,
    root_count_weight: float = 1.0,
) -> dict:
    if config is None:
        config = ParentDecodeConfig(two_stage_root_parent=True)
    positions = np.asarray(positions, dtype=np.float32)
    active_prob = np.asarray(active_prob, dtype=np.float32)
    root_logits = np.asarray(root_logits, dtype=np.float32)
    parent_candidate_logits = np.asarray(parent_candidate_logits, dtype=np.float32)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    n_nodes = positions.shape[0]
    active_mask = active_prob > config.active_threshold
    active_idx = np.where(active_mask)[0].tolist()
    if root_threshold is None:
        root_threshold = config.root_threshold if config.root_threshold is not None else config.active_threshold
    if root_decode_mode is None:
        root_decode_mode = getattr(config, "root_decode_mode", "threshold")
    if root_budget_ratio is None:
        root_budget_ratio = getattr(config, "root_budget_ratio", None)

    root_logits = np.clip(root_logits, -50.0, 50.0)
    root_prob = 1.0 / (1.0 + np.exp(-root_logits))
    root_mask = np.zeros(n_nodes, dtype=bool)
    if root_decode_mode == "threshold":
        root_mask = (root_prob > root_threshold) & active_mask
    elif root_decode_mode == "topk_budget":
        if root_count_budget is None:
            root_count_budget = max(1, int(round(float(active_mask.sum()) * float(root_budget_ratio if root_budget_ratio is not None else 0.03))))
        active_scores = root_logits[active_mask]
        if active_scores.size > 0:
            active_order = np.argsort(-active_scores)
            active_candidates = np.where(active_mask)[0]
            keep = active_candidates[active_order[: min(int(root_count_budget), active_candidates.size)]]
            root_mask[keep] = True
    elif root_decode_mode == "ratio_budget":
        ratio = float(root_budget_ratio if root_budget_ratio is not None else 0.03)
        budget = max(1, int(round(ratio * float(active_mask.sum()))))
        active_scores = root_logits[active_mask]
        if active_scores.size > 0:
            active_order = np.argsort(-active_scores)
            active_candidates = np.where(active_mask)[0]
            keep = active_candidates[active_order[: min(budget, active_candidates.size)]]
            root_mask[keep] = True
    else:
        raise ValueError(f"Unknown root decode mode: {root_decode_mode}")

    if root_mask.sum() == 0 and active_idx:
        root_mask[np.argmax(root_logits[active_mask])] = True
    if root_mask.sum() > config.max_roots:
        root_order = np.argsort(-root_logits[root_mask])
        keep = np.where(root_mask)[0][root_order[: config.max_roots]]
        new_root_mask = np.zeros_like(root_mask, dtype=bool)
        new_root_mask[keep] = True
        root_mask = new_root_mask

    parent_ptr = np.full(n_nodes, -1, dtype=np.int64)
    for child in active_idx:
        if root_mask[child]:
            continue
        cand_idx = candidate_indices[child]
        cand_mask = candidate_mask[child]
        scores = parent_candidate_logits[child]
        best_parent = -1
        best_score = -1e9
        for slot, ok in enumerate(cand_mask):
            if not ok:
                continue
            parent = int(cand_idx[slot])
            if parent < 0 or parent >= n_nodes or parent == child or root_mask[parent]:
                continue
            score = float(scores[slot])
            if score > best_score:
                best_score = score
                best_parent = parent
        parent_ptr[child] = best_parent

    # Enforce at least one root per connected component.
    if config.enforce_root_per_component:
        parent_ptr = _enforce_root_per_component(parent_ptr, active_mask, root_logits)
        root_mask = active_mask & (parent_ptr < 0)

    if config.decode_mode == "parent_argmax":
        parent_ptr = _allow_forest(parent_ptr, active_mask, config.max_roots)
    elif config.decode_mode == "parent_argmax_acyclic":
        parent_ptr = _break_cycles(parent_ptr, active_mask, active_prob)
    elif config.decode_mode == "parent_mst_hybrid":
        parent_ptr = _parent_mst_hybrid(positions, active_mask, np.concatenate([parent_candidate_logits, root_logits[..., None]], axis=-1), parent_ptr, config)
    else:
        raise ValueError(f"Unknown parent decode mode: {config.decode_mode}")

    parent_ptr = _apply_degree_cap(parent_ptr, active_mask, config.max_degree)
    metadata = _graph_metadata(parent_ptr, active_mask, active_prob)
    metadata["mode"] = config.decode_mode
    metadata["active_threshold"] = config.active_threshold
    metadata["root_count_pred"] = int(root_mask.sum())
    metadata["root_decode_mode"] = root_decode_mode

    edges = []
    for child, parent in enumerate(parent_ptr):
        if parent >= 0 and active_mask[child]:
            edges.append((int(parent), int(child)))

    return {
        "active_mask": active_mask,
        "root_mask": root_mask,
        "parent_ptr": parent_ptr,
        "edges": edges,
        "edge_count": len(edges),
        "metadata": metadata,
        "parent_offset": np.zeros_like(positions),
    }


def _allow_forest(parent_ptr: np.ndarray, active_mask: np.ndarray, max_roots: int) -> np.ndarray:
    roots = [i for i in np.where(active_mask)[0] if parent_ptr[i] < 0]
    if len(roots) <= max_roots:
        return parent_ptr
    # Keep the highest-confidence roots (lowest index fallback) and attach others to root.
    return parent_ptr


def _break_cycles(parent_ptr: np.ndarray, active_mask: np.ndarray, active_prob: np.ndarray) -> np.ndarray:
    out = parent_ptr.copy()
    for node in np.where(active_mask)[0]:
        visited = set()
        cur = node
        while cur >= 0 and cur not in visited:
            visited.add(cur)
            cur = int(out[cur])
        if cur in visited:
            # cycle detected, break at lowest-confidence node in the cycle
            cycle = []
            cur = node
            while cur >= 0 and cur not in cycle:
                cycle.append(cur)
                cur = int(out[cur])
            break_node = min(cycle, key=lambda x: float(active_prob[x]))
            out[break_node] = -1
    return out


def _parent_mst_hybrid(
    positions: np.ndarray,
    active_mask: np.ndarray,
    parent_logits: np.ndarray,
    parent_ptr: np.ndarray,
    config: ParentDecodeConfig,
) -> np.ndarray:
    # Greedy parent choice with acyclicity and simple degree cap.
    n_nodes = positions.shape[0]
    root_class = parent_logits.shape[-1] - 1
    child_scores = np.max(parent_logits[:, :root_class], axis=-1) - parent_logits[:, root_class]
    order = np.argsort(-child_scores)

    out = np.full(n_nodes, -1, dtype=np.int64)
    degrees = np.zeros(n_nodes, dtype=np.int64)
    uf_parent = list(range(n_nodes))

    def find(x: int) -> int:
        while uf_parent[x] != x:
            uf_parent[x] = uf_parent[uf_parent[x]]
            x = uf_parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        uf_parent[rb] = ra
        return True

    for child in order:
        if not active_mask[child]:
            continue
        candidates = np.argsort(-parent_logits[child, :root_class])
        chosen = -1
        for cand in candidates:
            if cand == child or not active_mask[cand]:
                continue
            if degrees[cand] >= config.max_degree or degrees[child] >= config.max_degree:
                continue
            if union(child, cand):
                chosen = int(cand)
                degrees[child] += 1
                degrees[cand] += 1
                break
        out[child] = chosen
    return out


def _apply_degree_cap(parent_ptr: np.ndarray, active_mask: np.ndarray, max_degree: int) -> np.ndarray:
    if max_degree <= 0:
        return parent_ptr
    out = parent_ptr.copy()
    degrees = np.zeros_like(parent_ptr)
    for child, parent in enumerate(out):
        if parent >= 0 and active_mask[child]:
            degrees[child] += 1
            degrees[parent] += 1
    for child, parent in enumerate(out):
        if parent < 0:
            continue
        if degrees[child] > max_degree or degrees[parent] > max_degree:
            out[child] = -1
    return out


def _enforce_root_per_component(parent_ptr: np.ndarray, active_mask: np.ndarray, root_scores: np.ndarray) -> np.ndarray:
    out = parent_ptr.copy()
    active_nodes = np.where(active_mask)[0].tolist()
    if not active_nodes:
        return out

    adj = {n: set() for n in active_nodes}
    for child, parent in enumerate(out):
        if parent >= 0 and child in adj and parent in adj:
            adj[child].add(int(parent))
            adj[int(parent)].add(child)

    seen = set()
    for node in active_nodes:
        if node in seen:
            continue
        stack = [node]
        comp = []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            stack.extend(adj[cur] - seen)

        if any(out[n] < 0 for n in comp):
            continue
        promote = max(comp, key=lambda idx: float(root_scores[idx]))
        out[promote] = -1

    return out


def _graph_metadata(parent_ptr: np.ndarray, active_mask: np.ndarray, active_prob: np.ndarray) -> Dict:
    active_nodes = np.where(active_mask)[0].tolist()
    degrees = {i: 0 for i in active_nodes}
    edges = []
    for child, parent in enumerate(parent_ptr):
        if parent >= 0 and active_mask[child] and active_mask[parent]:
            degrees[child] = degrees.get(child, 0) + 1
            degrees[parent] = degrees.get(parent, 0) + 1
            edges.append((parent, child))
    comp_count, connected_ratio = _component_stats(active_nodes, edges)
    cycle_rate = _cycle_rate(parent_ptr, active_mask)
    return {
        "component_count": comp_count,
        "connected_ratio": connected_ratio,
        "cycle_rate": cycle_rate,
        "average_degree": float(np.mean(list(degrees.values()))) if degrees else 0.0,
        "max_degree": int(max(degrees.values())) if degrees else 0,
        "pred_edge_count": len(edges),
        "root_count": int(np.sum(active_mask & (parent_ptr < 0))),
    }


def _component_stats(nodes: List[int], edges: List[tuple[int, int]]) -> tuple[int, float]:
    if not nodes:
        return 0, 0.0
    adj = {n: set() for n in nodes}
    for i, j in edges:
        if i in adj and j in adj:
            adj[i].add(j)
            adj[j].add(i)
    seen = set()
    sizes = []
    for n in nodes:
        if n in seen:
            continue
        stack = [n]
        size = 0
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            size += 1
            stack.extend(adj[cur] - seen)
        sizes.append(size)
    largest = max(sizes) if sizes else 0
    return len(sizes), largest / max(len(nodes), 1)


def _cycle_rate(parent_ptr: np.ndarray, active_mask: np.ndarray) -> float:
    active_nodes = np.where(active_mask)[0]
    if active_nodes.size == 0:
        return 0.0
    cycle_nodes = 0
    for node in active_nodes:
        visited = set()
        cur = int(node)
        while cur >= 0 and cur not in visited:
            visited.add(cur)
            cur = int(parent_ptr[cur])
        if cur in visited and cur >= 0:
            cycle_nodes += 1
    return cycle_nodes / max(active_nodes.size, 1)
