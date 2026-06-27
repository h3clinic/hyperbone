"""
Target generation for HyperNodeNet training.

Generates heatmaps, radius maps, and affinity fields from GraphLabel.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from hyperbone.labels.schema import GraphLabel, NodeType, EdgeType
from hyperbone.hypernodes.dataset import (
    NODE_TYPES,
    NODE_TYPE_TO_IDX,
    NUM_NODE_TYPES,
    normalize_positions,
)


def generate_node_heatmaps(
    graph: GraphLabel,
    resolution: int,
    sigma: float | None = None,
) -> np.ndarray:
    """Generate per-type heatmaps [T, H, W] with Gaussian peaks at node locations."""
    if sigma is None:
        sigma = max(2.0, resolution / 48.0)

    heatmaps = np.zeros((NUM_NODE_TYPES, resolution, resolution), dtype=np.float32)
    positions = normalize_positions(graph, resolution)

    for node in graph.nodes:
        if node.id not in positions:
            continue
        if node.node_type not in NODE_TYPE_TO_IDX:
            continue
        tidx = NODE_TYPE_TO_IDX[node.node_type]
        px = int(positions[node.id][0] * (resolution - 1))
        py = int(positions[node.id][1] * (resolution - 1))

        r = int(3 * sigma)
        x0, x1 = max(0, px - r), min(resolution, px + r + 1)
        y0, y1 = max(0, py - r), min(resolution, py + r + 1)
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                d2 = (xx - px) ** 2 + (yy - py) ** 2
                val = math.exp(-d2 / (2 * sigma * sigma))
                heatmaps[tidx, yy, xx] = max(heatmaps[tidx, yy, xx], val)

    return heatmaps


def generate_all_node_heatmap(
    graph: GraphLabel,
    resolution: int,
    sigma: float | None = None,
) -> np.ndarray:
    """Generate a single combined heatmap [1, H, W] with all nodes."""
    if sigma is None:
        sigma = max(2.0, resolution / 48.0)

    heatmap = np.zeros((1, resolution, resolution), dtype=np.float32)
    positions = normalize_positions(graph, resolution)

    for node in graph.nodes:
        if node.id not in positions:
            continue
        px = int(positions[node.id][0] * (resolution - 1))
        py = int(positions[node.id][1] * (resolution - 1))

        r = int(3 * sigma)
        x0, x1 = max(0, px - r), min(resolution, px + r + 1)
        y0, y1 = max(0, py - r), min(resolution, py + r + 1)
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                d2 = (xx - px) ** 2 + (yy - py) ** 2
                val = math.exp(-d2 / (2 * sigma * sigma))
                heatmap[0, yy, xx] = max(heatmap[0, yy, xx], val)

    return heatmap


def generate_radius_map(
    graph: GraphLabel,
    resolution: int,
) -> np.ndarray:
    """Generate radius map [1, H, W] from node radius annotations."""
    import cv2

    radius_map = np.zeros((1, resolution, resolution), dtype=np.float32)
    positions = normalize_positions(graph, resolution)

    for node in graph.nodes:
        if node.id not in positions or node.radius is None:
            continue
        px = int(positions[node.id][0] * (resolution - 1))
        py = int(positions[node.id][1] * (resolution - 1))
        r = min(node.radius, 1.0)
        rad_px = max(2, int(r * resolution * 0.1))
        cv2.circle(radius_map[0], (px, py), rad_px, float(r), -1)

    return radius_map


def generate_node_type_map(
    graph: GraphLabel,
    resolution: int,
) -> np.ndarray:
    """Generate node type index map [1, H, W], 0 = background."""
    import cv2

    type_map = np.zeros((resolution, resolution), dtype=np.uint8)
    positions = normalize_positions(graph, resolution)

    for node in graph.nodes:
        if node.id not in positions:
            continue
        if node.node_type not in NODE_TYPE_TO_IDX:
            continue
        px = int(positions[node.id][0] * (resolution - 1))
        py = int(positions[node.id][1] * (resolution - 1))
        # +1 since 0 is background
        tidx = NODE_TYPE_TO_IDX[node.node_type] + 1
        cv2.circle(type_map, (px, py), 3, int(tidx), -1)

    return type_map.astype(np.int64).reshape(1, resolution, resolution)


def generate_edge_affinity_map(
    graph: GraphLabel,
    resolution: int,
) -> np.ndarray:
    """Generate edge affinity map [1, H, W] - presence of edges."""
    import cv2

    affinity = np.zeros((1, resolution, resolution), dtype=np.float32)
    positions = normalize_positions(graph, resolution)

    for edge in graph.edges:
        if edge.source_node_id in positions and edge.target_node_id in positions:
            p1 = (positions[edge.source_node_id] * (resolution - 1)).astype(int)
            p2 = (positions[edge.target_node_id] * (resolution - 1)).astype(int)
            cv2.line(affinity[0], tuple(p1), tuple(p2), 1.0, 2, cv2.LINE_AA)

    return affinity


def generate_targets(
    graph: GraphLabel,
    resolution: int,
    max_nodes: int = 64,
    sigma: float | None = None,
) -> dict[str, np.ndarray]:
    """Generate all targets for a single graph."""
    return {
        "node_heatmaps": generate_node_heatmaps(graph, resolution, sigma),
        "all_node_heatmap": generate_all_node_heatmap(graph, resolution, sigma),
        "radius_map": generate_radius_map(graph, resolution),
        "node_type_map": generate_node_type_map(graph, resolution),
        "edge_affinity_map": generate_edge_affinity_map(graph, resolution),
    }
