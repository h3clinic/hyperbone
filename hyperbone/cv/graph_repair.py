"""Graph topology repair — bridge close disconnected components, merge close nodes.

Fixes the main cause of quality-gate rejections on SAM2 masks:
skeletonization of complex shapes produces multiple disconnected graph components.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict


def repair_graph(
    graph: Dict,
    mask_shape: Tuple[int, int] = None,
    bridge_gap_px: float = 12.0,
    bridge_gap_ratio: float = 0.025,
    merge_distance: float = 4.0,
    min_branch_length: int = 6,
    max_bridge_angle_deg: float = 60.0,
    spur_prune_length: int = 8,
) -> Dict:
    """Repair a skeleton graph by bridging close components and pruning spurs.

    Args:
        graph: Dict with "nodes" and "edges".
        mask_shape: (H, W) for adaptive bridge threshold.
        bridge_gap_px: Fixed minimum bridge gap threshold in pixels.
        bridge_gap_ratio: Ratio of max(H,W) to add to bridge threshold.
        merge_distance: Merge nodes closer than this distance.
        min_branch_length: Prune endpoint branches shorter than this.
        max_bridge_angle_deg: Max angle deviation for bridge (unused currently, reserved).
        spur_prune_length: Prune endpoint branches shorter than this after bridging.

    Returns:
        Dict with repaired graph and repair metadata.
    """
    nodes = [n.copy() for n in graph.get("nodes", [])]
    edges = [e.copy() for e in graph.get("edges", [])]

    if len(nodes) < 2:
        return {
            "graph": {"nodes": nodes, "edges": edges},
            "components_before": 1 if nodes else 0,
            "components_after": 1 if nodes else 0,
            "bridges_added": 0,
            "nodes_merged": 0,
            "spurs_pruned": 0,
            "nodes_before": len(nodes),
            "nodes_after": len(nodes),
            "edges_before": len(edges),
            "edges_after": len(edges),
        }

    # Compute adaptive bridge threshold
    if mask_shape:
        max_dim = max(mask_shape)
        bridge_threshold = max(bridge_gap_px, bridge_gap_ratio * max_dim)
    else:
        bridge_threshold = bridge_gap_px

    # Step 0: Record stats
    nodes_before = len(nodes)
    edges_before = len(edges)
    components_before = _count_components(nodes, edges)

    # Step 1: Merge very close nodes (skip for large graphs — too slow O(n²))
    nodes_merged = 0
    if len(nodes) <= 200:
        nodes, edges, nodes_merged = _merge_close_nodes(nodes, edges, merge_distance)

    # Step 2: Bridge disconnected components (two passes — tight then loose)
    bridges_added = 0
    max_iterations = min(15, components_before + 5)

    # Pass 1: bridge with primary threshold
    for _ in range(max_iterations):
        components = _find_components(nodes, edges)
        if len(components) <= 1:
            break
        best_bridge = _find_best_bridge_all(nodes, edges, components, bridge_threshold)
        if best_bridge is None:
            break
        nid_a, nid_b, dist = best_bridge
        edges.append({
            "parent": nid_a,
            "child": nid_b,
            "length_px": float(dist),
            "bridge": True,
        })
        bridges_added += 1

    # Pass 2: looser threshold (1.5x) for remaining disconnected components
    components = _find_components(nodes, edges)
    if len(components) > 1:
        loose_threshold = bridge_threshold * 1.5
        for _ in range(max_iterations):
            components = _find_components(nodes, edges)
            if len(components) <= 1:
                break
            best_bridge = _find_best_bridge_all(nodes, edges, components, loose_threshold)
            if best_bridge is None:
                break
            nid_a, nid_b, dist = best_bridge
            edges.append({
                "parent": nid_a,
                "child": nid_b,
                "length_px": float(dist),
                "bridge": True,
            })
            bridges_added += 1

    # Step 3: Prune short endpoint spurs (after bridging)
    prune_len = max(spur_prune_length, min_branch_length)
    edges, nodes, spurs_pruned = _prune_endpoint_spurs(
        edges, nodes, prune_len
    )

    # Step 4: Re-index
    nodes, edges = _reindex_graph(nodes, edges)

    components_after = _count_components(nodes, edges)

    return {
        "graph": {"nodes": nodes, "edges": edges},
        "components_before": components_before,
        "components_after": components_after,
        "bridges_added": bridges_added,
        "nodes_merged": nodes_merged,
        "spurs_pruned": spurs_pruned,
        "nodes_before": nodes_before,
        "nodes_after": len(nodes),
        "edges_before": edges_before,
        "edges_after": len(edges),
    }


def _count_components(nodes: List[Dict], edges: List[Dict]) -> int:
    """Count connected components in the graph."""
    return len(_find_components(nodes, edges))


def _find_components(nodes: List[Dict], edges: List[Dict]) -> List[Set[int]]:
    """Find connected components using union-find."""
    if not nodes:
        return []

    node_ids = {n["id"] for n in nodes}
    parent = {nid: nid for nid in node_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for edge in edges:
        p, c = edge["parent"], edge["child"]
        if p in node_ids and c in node_ids:
            union(p, c)

    components = defaultdict(set)
    for nid in node_ids:
        components[find(nid)].add(nid)

    return list(components.values())


def _merge_close_nodes(
    nodes: List[Dict], edges: List[Dict], distance: float
) -> Tuple[List[Dict], List[Dict], int]:
    """Merge nodes that are closer than `distance` pixels."""
    if not nodes:
        return nodes, edges, 0

    merged_count = 0
    merge_map = {}  # old_id -> keep_id

    # Find pairs to merge
    node_map = {n["id"]: n for n in nodes}
    node_ids = list(node_map.keys())

    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            nid_a, nid_b = node_ids[i], node_ids[j]
            if nid_a in merge_map or nid_b in merge_map:
                continue
            a = node_map[nid_a]
            b = node_map[nid_b]
            dist = _node_dist(a, b)
            if dist <= distance:
                # Keep the junction if one is a junction
                if b["type"] == "junction":
                    merge_map[nid_a] = nid_b
                else:
                    merge_map[nid_b] = nid_a
                merged_count += 1

    if not merge_map:
        return nodes, edges, 0

    # Resolve chains: if A->B and B->C, then A->C
    def resolve(nid):
        visited = set()
        while nid in merge_map:
            if nid in visited:
                break
            visited.add(nid)
            nid = merge_map[nid]
        return nid

    # Update edges
    new_edges = []
    seen_edges = set()
    for edge in edges:
        p = resolve(edge["parent"])
        c = resolve(edge["child"])
        if p == c:
            continue  # self-loop after merge
        key = (min(p, c), max(p, c))
        if key not in seen_edges:
            seen_edges.add(key)
            new_edges.append({"parent": p, "child": c, "length_px": edge["length_px"]})

    # Remove merged nodes
    removed_ids = {resolve(nid) for nid in merge_map.keys() if resolve(nid) != nid}
    # Actually: keep the targets, remove the sources
    removed_ids = set(merge_map.keys())
    new_nodes = [n for n in nodes if n["id"] not in removed_ids]

    return new_nodes, new_edges, merged_count


def _find_best_bridge(
    nodes: List[Dict],
    edges: List[Dict],
    components: List[Set[int]],
    max_distance: float,
) -> Tuple[int, int, float]:
    """Find the closest pair of nodes in different components within max_distance.

    Only considers endpoints and low-degree nodes to keep O(n²) manageable.
    """
    if len(components) < 2:
        return None

    node_map = {n["id"]: n for n in nodes}

    # Compute node degrees
    degree = defaultdict(int)
    for e in edges:
        degree[e["parent"]] += 1
        degree[e["child"]] += 1

    # Build component membership
    comp_of = {}
    for ci, comp in enumerate(components):
        for nid in comp:
            comp_of[nid] = ci

    # Only consider endpoints/low-degree nodes as bridge candidates (performance)
    candidates_per_comp = defaultdict(list)
    for n in nodes:
        nid = n["id"]
        ci = comp_of.get(nid)
        if ci is None:
            continue
        # Prefer endpoints (degree 1) and junctions at boundaries
        if degree[nid] <= 2 or n["type"] == "endpoint":
            candidates_per_comp[ci].append(nid)

    # Fallback: if a component has no candidates, use all its nodes (capped)
    for ci, comp in enumerate(components):
        if ci not in candidates_per_comp or not candidates_per_comp[ci]:
            candidates_per_comp[ci] = list(comp)[:20]
        else:
            # Cap to avoid quadratic blowup
            candidates_per_comp[ci] = candidates_per_comp[ci][:20]

    best = None
    best_dist = max_distance + 1

    comp_indices = list(range(len(components)))
    for ci in comp_indices:
        for cj in comp_indices:
            if cj <= ci:
                continue
            for nid_a in candidates_per_comp[ci]:
                if nid_a not in node_map:
                    continue
                for nid_b in candidates_per_comp[cj]:
                    if nid_b not in node_map:
                        continue
                    dist = _node_dist(node_map[nid_a], node_map[nid_b])
                    if dist < best_dist:
                        best_dist = dist
                        best = (nid_a, nid_b, dist)

    if best and best[2] <= max_distance:
        return best
    return None


def _find_best_bridge_all(
    nodes: List[Dict],
    edges: List[Dict],
    components: List[Set[int]],
    max_distance: float,
) -> Tuple[int, int, float]:
    """Find closest pair of ANY nodes in different components within max_distance.

    Considers all nodes (not just endpoints) to find the true closest pair.
    Uses spatial sorting for efficiency on larger graphs.
    """
    if len(components) < 2:
        return None

    node_map = {n["id"]: n for n in nodes}

    # Build component membership
    comp_of = {}
    for ci, comp in enumerate(components):
        for nid in comp:
            comp_of[nid] = ci

    # Gather all candidates per component (capped per component for perf)
    candidates_per_comp = defaultdict(list)
    for n in nodes:
        nid = n["id"]
        ci = comp_of.get(nid)
        if ci is not None:
            candidates_per_comp[ci].append(nid)

    # Cap large components — keep boundary nodes (endpoints first, then all)
    for ci in list(candidates_per_comp.keys()):
        cands = candidates_per_comp[ci]
        if len(cands) > 40:
            # Prioritize endpoints and low-degree
            degree = defaultdict(int)
            for e in edges:
                degree[e["parent"]] += 1
                degree[e["child"]] += 1
            # Sort: endpoints first, then by degree ascending
            cands.sort(key=lambda nid: (0 if node_map[nid]["type"] == "endpoint" else 1, degree[nid]))
            candidates_per_comp[ci] = cands[:40]

    best = None
    best_dist = max_distance + 1

    comp_indices = list(range(len(components)))
    for ci in comp_indices:
        for cj in comp_indices:
            if cj <= ci:
                continue
            for nid_a in candidates_per_comp[ci]:
                if nid_a not in node_map:
                    continue
                na = node_map[nid_a]
                for nid_b in candidates_per_comp[cj]:
                    if nid_b not in node_map:
                        continue
                    dist = _node_dist(na, node_map[nid_b])
                    if dist < best_dist:
                        best_dist = dist
                        best = (nid_a, nid_b, dist)

    if best and best[2] <= max_distance:
        return best
    return None


def _prune_endpoint_spurs(
    edges: List[Dict], nodes: List[Dict], min_length: int
) -> Tuple[List[Dict], List[Dict], int]:
    """Remove short branches that terminate at endpoints.

    Never prunes:
    - Junction-to-junction edges
    - The only edge connecting two large components
    """
    pruned = 0
    endpoint_ids = {n["id"] for n in nodes if n["type"] == "endpoint"}

    # Compute degree
    degree = defaultdict(int)
    for e in edges:
        degree[e["parent"]] += 1
        degree[e["child"]] += 1

    # Only prune edges where one end is a degree-1 endpoint
    kept_edges = []
    removed_nodes = set()

    for edge in edges:
        is_short = edge["length_px"] < min_length
        p_is_leaf = edge["parent"] in endpoint_ids and degree[edge["parent"]] == 1
        c_is_leaf = edge["child"] in endpoint_ids and degree[edge["child"]] == 1

        if is_short and (p_is_leaf or c_is_leaf):
            # Remove the leaf endpoint
            if p_is_leaf:
                removed_nodes.add(edge["parent"])
            if c_is_leaf:
                removed_nodes.add(edge["child"])
            pruned += 1
        else:
            kept_edges.append(edge)

    kept_nodes = [n for n in nodes if n["id"] not in removed_nodes]
    return kept_edges, kept_nodes, pruned


def _node_dist(a: Dict, b: Dict) -> float:
    """Euclidean distance between two nodes."""
    ax, ay = a["xy"]
    bx, by = b["xy"]
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _reindex_graph(
    nodes: List[Dict], edges: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    """Re-assign sequential IDs."""
    old_to_new = {}
    for i, node in enumerate(nodes):
        old_to_new[node["id"]] = i
        node["id"] = i

    new_edges = []
    for edge in edges:
        if edge["parent"] in old_to_new and edge["child"] in old_to_new:
            new_edge = {
                "parent": old_to_new[edge["parent"]],
                "child": old_to_new[edge["child"]],
                "length_px": edge["length_px"],
            }
            if edge.get("bridge"):
                new_edge["bridge"] = True
            new_edges.append(new_edge)

    return nodes, new_edges
