"""
Medial axis label extraction from binary masks.

Given a mask of any object, compute its medial axis (topological skeleton)
and extract:
- Endpoint nodes
- Branch nodes
- Ridge nodes
- Medial axis edges
- Radius (distance-to-boundary) at each node
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from hyperbone.labels.schema import (
    GraphLabel,
    HyperNodeLabel,
    HyperEdgeLabel,
    NodeType,
    EdgeType,
    LabelSource,
)


@dataclass
class MedialAxisConfig:
    """Configuration for medial axis extraction."""
    min_branch_length: int = 5          # prune short spurs (pixels)
    simplify_tolerance: float = 2.0     # Douglas-Peucker simplification
    confidence_from_mask_quality: bool = True


def extract_medial_axis_graph(
    mask: np.ndarray,
    config: MedialAxisConfig = MedialAxisConfig(),
    sample_id: str = "medial",
) -> GraphLabel:
    """
    Extract structural graph from binary mask via medial axis transform.

    Steps:
    1. Skeletonize mask
    2. Find branch points, endpoints
    3. Trace edges between special points
    4. Compute radius (distance transform) at each node
    5. Build graph with nodes + edges

    Args:
        mask: [H, W] binary mask (uint8 or bool)
        config: extraction parameters
        sample_id: identifier for this sample

    Returns:
        GraphLabel with structural nodes and medial_axis edges
    """
    try:
        from skimage.morphology import skeletonize, remove_small_objects
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        raise ImportError("scikit-image and scipy required: pip install scikit-image scipy")

    # Ensure binary
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() < 10:
        return GraphLabel(sample_id=sample_id, nodes=[], edges=[])

    # Distance transform (gives radius at each pixel)
    dist_transform = distance_transform_edt(mask)

    # Skeletonize
    skeleton = skeletonize(mask > 0).astype(np.uint8)

    # Find special points
    endpoints, branch_points = _find_skeleton_points(skeleton)

    # Prune short branches
    if config.min_branch_length > 0:
        skeleton, endpoints, branch_points = _prune_short_branches(
            skeleton, endpoints, branch_points, config.min_branch_length
        )

    # Trace paths between special points
    special_points = endpoints + branch_points
    if not special_points:
        # Single line with no branches
        ys, xs = np.where(skeleton > 0)
        if len(ys) == 0:
            return GraphLabel(sample_id=sample_id, nodes=[], edges=[])
        # Use first and last skeleton pixel as endpoints
        special_points = [(ys[0], xs[0]), (ys[-1], xs[-1])]
        endpoints = special_points[:]

    # Build graph
    nodes: list[HyperNodeLabel] = []
    edges: list[HyperEdgeLabel] = []

    # Create nodes at special points
    point_to_node_id: dict[tuple[int, int], int] = {}
    node_id = 0

    for pt in special_points:
        y, x = pt
        radius = float(dist_transform[y, x])

        if pt in endpoints:
            ntype = NodeType.ENDPOINT
        elif pt in branch_points:
            ntype = NodeType.BRANCH
        else:
            ntype = NodeType.RIDGE

        conf = 0.6
        if config.confidence_from_mask_quality:
            # Higher confidence for thicker parts (larger radius)
            conf = min(0.9, 0.4 + radius / (dist_transform.max() + 1e-8) * 0.5)

        nodes.append(HyperNodeLabel(
            id=node_id,
            node_type=ntype,
            xy=[float(x), float(y)],
            radius=radius,
            confidence=conf,
            label_sources={LabelSource.MEDIAL_AXIS.value: conf},
        ))
        point_to_node_id[pt] = node_id
        node_id += 1

    # Trace edges between adjacent special points along skeleton
    edge_id = 0
    visited_edges: set[tuple[int, int]] = set()

    for pt in special_points:
        neighbors = _skeleton_neighbors(skeleton, pt)
        for nb in neighbors:
            # Walk along skeleton until we hit another special point
            path = _trace_path(skeleton, pt, nb, set(special_points))
            if path and path[-1] in point_to_node_id:
                src_id = point_to_node_id[pt]
                tgt_id = point_to_node_id[path[-1]]
                edge_key = (min(src_id, tgt_id), max(src_id, tgt_id))
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    # Edge length = path pixel count
                    length = float(len(path))
                    edges.append(HyperEdgeLabel(
                        id=edge_id,
                        source_node_id=src_id,
                        target_node_id=tgt_id,
                        edge_type=EdgeType.MEDIAL_AXIS,
                        confidence=0.6,
                        label_sources={LabelSource.MEDIAL_AXIS.value: 0.6},
                        length=length,
                    ))
                    edge_id += 1

    return GraphLabel(
        sample_id=sample_id,
        nodes=nodes,
        edges=edges,
        metadata={
            "source": "medial_axis",
            "mask_area": int(mask.sum()),
            "skeleton_pixels": int(skeleton.sum()),
        },
    )


def _find_skeleton_points(skeleton: np.ndarray):
    """Find endpoints (1 neighbor) and branch points (3+ neighbors) in skeleton."""
    endpoints = []
    branch_points = []

    ys, xs = np.where(skeleton > 0)
    for y, x in zip(ys, xs):
        n_count = _count_neighbors(skeleton, y, x)
        if n_count == 1:
            endpoints.append((y, x))
        elif n_count >= 3:
            branch_points.append((y, x))

    return endpoints, branch_points


def _count_neighbors(skeleton: np.ndarray, y: int, x: int) -> int:
    """Count 8-connected skeleton neighbors."""
    h, w = skeleton.shape
    count = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx] > 0:
                count += 1
    return count


def _skeleton_neighbors(skeleton: np.ndarray, pt: tuple[int, int]) -> list[tuple[int, int]]:
    """Get 8-connected skeleton neighbors of a point."""
    y, x = pt
    h, w = skeleton.shape
    neighbors = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx] > 0:
                neighbors.append((ny, nx))
    return neighbors


def _trace_path(
    skeleton: np.ndarray,
    start: tuple[int, int],
    first_step: tuple[int, int],
    special_points: set[tuple[int, int]],
    max_steps: int = 10000,
) -> Optional[list[tuple[int, int]]]:
    """
    Trace a path from start through first_step until reaching another special point.
    """
    path = [first_step]
    prev = start
    current = first_step

    for _ in range(max_steps):
        if current in special_points and current != start:
            return path

        # Find next pixel (not prev, not start)
        neighbors = _skeleton_neighbors(skeleton, current)
        next_pt = None
        for nb in neighbors:
            if nb != prev and nb != start:
                next_pt = nb
                break

        if next_pt is None:
            # Dead end or back to start
            if current in special_points:
                return path
            return None

        path.append(next_pt)
        prev = current
        current = next_pt

        if current in special_points:
            return path

    return None


def _prune_short_branches(
    skeleton: np.ndarray,
    endpoints: list[tuple[int, int]],
    branch_points: list[tuple[int, int]],
    min_length: int,
):
    """Remove short branches (spurs) from skeleton."""
    special = set(branch_points)
    pruned = skeleton.copy()

    new_endpoints = []
    for ep in endpoints:
        # Trace from endpoint to nearest branch point
        path = [ep]
        prev = None
        current = ep

        for _ in range(min_length + 1):
            neighbors = _skeleton_neighbors(pruned, current)
            next_pt = None
            for nb in neighbors:
                if nb != prev:
                    next_pt = nb
                    break
            if next_pt is None:
                break
            if next_pt in special:
                # Reached branch point - check length
                if len(path) < min_length:
                    # Prune this branch
                    for pt in path:
                        pruned[pt[0], pt[1]] = 0
                else:
                    new_endpoints.append(ep)
                break
            path.append(next_pt)
            prev = current
            current = next_pt
        else:
            new_endpoints.append(ep)

    # Recompute branch points
    new_branch_points = []
    ys, xs = np.where(pruned > 0)
    for y, x in zip(ys, xs):
        if _count_neighbors(pruned, y, x) >= 3:
            new_branch_points.append((y, x))

    return pruned, new_endpoints, new_branch_points
