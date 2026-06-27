"""Quality scoring and rejection logic for skeleton graphs."""

import numpy as np
from typing import Dict, List, Tuple


# Default rejection thresholds (all configurable)
DEFAULT_THRESHOLDS = {
    "min_mask_area_ratio": 0.002,
    "max_mask_area_ratio": 0.80,
    "min_bbox_width": 12,
    "min_bbox_height": 12,
    "min_node_count": 2,
    "min_edge_count": 1,
    "min_largest_component_ratio": 0.75,
    "max_tiny_branch_count": 20,
    "require_connected": True,
}


def score_graph(
    mask: np.ndarray,
    graph: Dict,
    frame_shape: Tuple[int, int],
    bbox: Tuple[int, int, int, int] = None,
    thresholds: Dict = None,
) -> Dict:
    """Compute quality metrics and acceptance decision for a skeleton graph.

    Args:
        mask: Binary mask (uint8, 255=fg).
        graph: Dict with "nodes" and "edges".
        frame_shape: (height, width) of the source frame.
        bbox: (x_min, y_min, x_max, y_max) or None.
        thresholds: Override default thresholds.

    Returns:
        Quality dict with metrics, accepted bool, and reject_reasons list.
    """
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    h, w = frame_shape[:2]
    frame_area = h * w

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    # Mask metrics
    mask_area_px = int(np.count_nonzero(mask)) if mask is not None else 0
    mask_area_ratio = mask_area_px / frame_area if frame_area > 0 else 0

    # BBox metrics
    if bbox:
        bx1, by1, bx2, by2 = bbox
        bbox_width = bx2 - bx1
        bbox_height = by2 - by1
    else:
        bbox_width = 0
        bbox_height = 0
    bbox_area = bbox_width * bbox_height
    bbox_area_ratio = bbox_area / frame_area if frame_area > 0 else 0

    # Node/edge metrics
    node_count = len(nodes)
    edge_count = len(edges)
    endpoint_count = sum(1 for n in nodes if n.get("type") == "endpoint")
    junction_count = sum(1 for n in nodes if n.get("type") == "junction")

    # Graph connectivity
    components = _connected_components(nodes, edges)
    num_components = len(components)
    largest_component_size = max((len(c) for c in components), default=0)
    largest_component_ratio = largest_component_size / node_count if node_count > 0 else 0
    graph_connected = num_components <= 1

    # Edge lengths
    edge_lengths = []
    node_map = {n["id"]: n["xy"] for n in nodes}
    for edge in edges:
        p = node_map.get(edge["parent"])
        c = node_map.get(edge["child"])
        if p and c:
            dist = ((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2) ** 0.5
            edge_lengths.append(dist)
    mean_edge_length = float(np.mean(edge_lengths)) if edge_lengths else 0.0

    # Tiny branches: edges to endpoints with length < 10px
    tiny_branch_count = sum(
        1 for e in edges
        if e.get("length_px", 0) < 10
        and (e["parent"] in {n["id"] for n in nodes if n.get("type") == "endpoint"}
             or e["child"] in {n["id"] for n in nodes if n.get("type") == "endpoint"})
    )

    # Graph density
    max_edges = node_count * (node_count - 1) / 2 if node_count > 1 else 1
    graph_density = edge_count / max_edges if max_edges > 0 else 0

    # Rejection logic
    reject_reasons = []

    if mask_area_ratio < th["min_mask_area_ratio"]:
        reject_reasons.append(f"mask_too_small ({mask_area_ratio:.4f} < {th['min_mask_area_ratio']})")
    if mask_area_ratio > th["max_mask_area_ratio"]:
        reject_reasons.append(f"mask_too_large ({mask_area_ratio:.4f} > {th['max_mask_area_ratio']})")
    if bbox_width < th["min_bbox_width"]:
        reject_reasons.append(f"bbox_too_narrow ({bbox_width} < {th['min_bbox_width']})")
    if bbox_height < th["min_bbox_height"]:
        reject_reasons.append(f"bbox_too_short ({bbox_height} < {th['min_bbox_height']})")
    if node_count < th["min_node_count"]:
        reject_reasons.append(f"too_few_nodes ({node_count} < {th['min_node_count']})")
    if edge_count < th["min_edge_count"]:
        reject_reasons.append(f"too_few_edges ({edge_count} < {th['min_edge_count']})")
    if largest_component_ratio < th["min_largest_component_ratio"] and node_count > 0:
        reject_reasons.append(f"fragmented_graph ({largest_component_ratio:.2f} < {th['min_largest_component_ratio']})")
    if th["require_connected"] and not graph_connected and node_count > 1:
        reject_reasons.append(f"disconnected_graph ({num_components} components)")
    if tiny_branch_count > th["max_tiny_branch_count"]:
        reject_reasons.append(f"too_many_tiny_branches ({tiny_branch_count} > {th['max_tiny_branch_count']})")

    accepted = len(reject_reasons) == 0

    return {
        "mask_area_px": mask_area_px,
        "mask_area_ratio": round(mask_area_ratio, 6),
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area_ratio": round(bbox_area_ratio, 6),
        "skeleton_node_count": node_count,
        "skeleton_edge_count": edge_count,
        "endpoint_count": endpoint_count,
        "branch_count": junction_count,
        "largest_component_ratio": round(largest_component_ratio, 4),
        "tiny_branch_count": tiny_branch_count,
        "graph_connected": graph_connected,
        "graph_density": round(graph_density, 4),
        "mean_edge_length": round(mean_edge_length, 2),
        "accepted": accepted,
        "reject_reasons": reject_reasons,
    }


def quality_score_numeric(quality: Dict) -> float:
    """Compute a 0-1 numeric quality score from the quality dict.

    Higher = better. Used for ranking/sorting.
    """
    if not quality.get("accepted"):
        return 0.0

    score = 0.5  # base for accepted

    # Bonus for more nodes (up to 0.2)
    nc = quality.get("skeleton_node_count", 0)
    score += min(nc / 50, 0.2)

    # Bonus for connectivity (0.1)
    if quality.get("graph_connected"):
        score += 0.1

    # Bonus for low tiny branch ratio (0.1)
    tb = quality.get("tiny_branch_count", 0)
    ec = quality.get("skeleton_edge_count", 1)
    if ec > 0:
        score += max(0, 0.1 * (1 - tb / ec))

    # Bonus for reasonable density (0.1)
    d = quality.get("graph_density", 0)
    if 0.01 < d < 0.5:
        score += 0.1

    return round(min(score, 1.0), 4)


def _connected_components(nodes: List[Dict], edges: List[Dict]) -> List[set]:
    """Find connected components in the graph using union-find."""
    if not nodes:
        return []

    parent = {}
    for n in nodes:
        parent[n["id"]] = n["id"]

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
        if edge["parent"] in parent and edge["child"] in parent:
            union(edge["parent"], edge["child"])

    components = {}
    for nid in parent:
        root = find(nid)
        components.setdefault(root, set()).add(nid)

    return list(components.values())
