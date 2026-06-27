"""
LabelForge dataset loader for HyperNodeNet training.

Loads trainable_graphs.jsonl and produces tensors for:
- input image (real or graph-rasterized)
- node heatmap targets
- graph token targets (active, xy, xyz, node_type, confidence)
- edge matrix targets
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from hyperbone.labels.schema import (
    GraphLabel,
    NodeType,
    EdgeType,
    load_graph_labels,
)

NODE_TYPES = [
    NodeType.ENDPOINT,
    NodeType.BRANCH,
    NodeType.ARTICULATION,
    NodeType.BEND,
    NodeType.RIDGE,
    NodeType.ROOT,
    NodeType.CENTER,
    NodeType.SEMANTIC_JOINT,
]
NODE_TYPE_TO_IDX = {nt: i for i, nt in enumerate(NODE_TYPES)}
NUM_NODE_TYPES = len(NODE_TYPES)

EDGE_TYPES = [
    EdgeType.BONE,
    EdgeType.BRANCH,
    EdgeType.VEIN,
    EdgeType.MEDIAL_AXIS,
    EdgeType.HINGE_LINK,
    EdgeType.RIDGE,
    EdgeType.DEFORMATION_LINK,
    EdgeType.UNKNOWN,
]
EDGE_TYPE_TO_IDX = {et: i for i, et in enumerate(EDGE_TYPES)}
NUM_EDGE_TYPES = len(EDGE_TYPES)


def rasterize_graph(
    graph: GraphLabel,
    resolution: int,
    augment: bool = False,
) -> np.ndarray:
    """Rasterize a graph into a synthetic grayscale image [H,W]."""
    img = np.zeros((resolution, resolution), dtype=np.float32)

    if not graph.nodes:
        return img

    # Collect node positions (use xy if available, else generate layout)
    positions = {}
    for node in graph.nodes:
        if node.xy is not None:
            positions[node.id] = np.array(node.xy, dtype=np.float32)

    if not positions:
        # Generate circular layout from xyz or sequential
        n = len(graph.nodes)
        for i, node in enumerate(graph.nodes):
            if node.xyz is not None:
                # Project xyz onto 2D (simple orthographic XY)
                x = node.xyz[0]
                y = node.xyz[1]
                positions[node.id] = np.array([x, y], dtype=np.float32)
            else:
                angle = 2 * math.pi * i / max(n, 1)
                positions[node.id] = np.array(
                    [0.5 + 0.3 * math.cos(angle), 0.5 + 0.3 * math.sin(angle)],
                    dtype=np.float32,
                )

    # Normalize positions to [0.05, 0.95] range
    if positions:
        pts = np.array(list(positions.values()))
        pmin = pts.min(axis=0)
        pmax = pts.max(axis=0)
        span = pmax - pmin
        span[span < 1e-6] = 1.0
        for nid in positions:
            positions[nid] = 0.05 + 0.9 * (positions[nid] - pmin) / span

    # Augmentation params
    line_width = random.uniform(1.0, 3.0) if augment else 2.0
    node_radius = random.randint(3, 6) if augment else 4
    noise_std = random.uniform(0.0, 0.05) if augment else 0.0

    # Draw edges
    for edge in graph.edges:
        if edge.source_node_id in positions and edge.target_node_id in positions:
            p1 = (positions[edge.source_node_id] * resolution).astype(int)
            p2 = (positions[edge.target_node_id] * resolution).astype(int)
            cv2.line(img, tuple(p1), tuple(p2), 0.7, int(line_width), cv2.LINE_AA)

    # Draw nodes
    for node in graph.nodes:
        if node.id in positions:
            p = (positions[node.id] * resolution).astype(int)
            cv2.circle(img, tuple(p), node_radius, 1.0, -1, cv2.LINE_AA)

    # Optional noise
    if noise_std > 0:
        img += np.random.randn(*img.shape).astype(np.float32) * noise_std

    # Optional blur
    if augment and random.random() < 0.3:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    return np.clip(img, 0.0, 1.0)


def normalize_positions(graph: GraphLabel, resolution: int) -> dict[int, np.ndarray]:
    """Get normalized [0,1] positions for all nodes."""
    positions = {}
    for node in graph.nodes:
        if node.xy is not None:
            positions[node.id] = np.array(node.xy, dtype=np.float32)

    if not positions:
        n = len(graph.nodes)
        for i, node in enumerate(graph.nodes):
            if node.xyz is not None:
                positions[node.id] = np.array(
                    [node.xyz[0], node.xyz[1]], dtype=np.float32
                )
            else:
                angle = 2 * math.pi * i / max(n, 1)
                positions[node.id] = np.array(
                    [0.5 + 0.3 * math.cos(angle), 0.5 + 0.3 * math.sin(angle)],
                    dtype=np.float32,
                )

    if positions:
        pts = np.array(list(positions.values()))
        pmin = pts.min(axis=0)
        pmax = pts.max(axis=0)
        span = pmax - pmin
        span[span < 1e-6] = 1.0
        for nid in positions:
            positions[nid] = 0.05 + 0.9 * (positions[nid] - pmin) / span

    return positions


class HyperNodeDataset(Dataset):
    """Dataset for HyperNodeNet training from LabelForge graphs."""

    def __init__(
        self,
        graphs_path: str | Path,
        resolution: int = 192,
        max_nodes: int = 64,
        augment: bool = False,
        graphs: Optional[list[GraphLabel]] = None,
    ):
        self.resolution = resolution
        self.max_nodes = max_nodes
        self.augment = augment

        if graphs is not None:
            self.graphs = graphs
        else:
            self.graphs = load_graph_labels(Path(graphs_path))

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        graph = self.graphs[idx]
        res = self.resolution
        max_n = self.max_nodes

        # --- Input image ---
        input_source = "graph_rasterized"
        image = None

        if graph.image_path and Path(graph.image_path).exists():
            img = cv2.imread(graph.image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (res, res))
                image = img.astype(np.float32) / 255.0
                input_source = "real_image"

        if image is None:
            image = rasterize_graph(graph, res, augment=self.augment)

        # [1, H, W] grayscale
        image_tensor = torch.from_numpy(image).unsqueeze(0)

        # --- Node positions normalized to [0, 1] ---
        positions = normalize_positions(graph, res)

        # --- Heatmap targets [T, H, W] ---
        heatmaps = np.zeros((NUM_NODE_TYPES, res, res), dtype=np.float32)
        sigma = max(2.0, res / 48.0)

        for node in graph.nodes:
            if node.id not in positions:
                continue
            ntype = node.node_type
            if ntype not in NODE_TYPE_TO_IDX:
                continue
            tidx = NODE_TYPE_TO_IDX[ntype]
            px = int(positions[node.id][0] * (res - 1))
            py = int(positions[node.id][1] * (res - 1))
            # Gaussian splat
            x0 = max(0, px - int(3 * sigma))
            x1 = min(res, px + int(3 * sigma) + 1)
            y0 = max(0, py - int(3 * sigma))
            y1 = min(res, py + int(3 * sigma) + 1)
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    d2 = (xx - px) ** 2 + (yy - py) ** 2
                    val = math.exp(-d2 / (2 * sigma * sigma))
                    heatmaps[tidx, yy, xx] = max(heatmaps[tidx, yy, xx], val)

        heatmap_tensor = torch.from_numpy(heatmaps)

        # --- Node tokens ---
        num_nodes = min(len(graph.nodes), max_n)
        node_active = torch.zeros(max_n)
        node_xy = torch.zeros(max_n, 2)
        node_xyz = torch.zeros(max_n, 3)
        node_type = torch.zeros(max_n, dtype=torch.long)
        node_conf = torch.zeros(max_n)

        sorted_nodes = sorted(graph.nodes, key=lambda n: n.confidence, reverse=True)
        for i, node in enumerate(sorted_nodes[:num_nodes]):
            node_active[i] = 1.0
            if node.id in positions:
                node_xy[i, 0] = float(positions[node.id][0])
                node_xy[i, 1] = float(positions[node.id][1])
            if node.xyz is not None:
                node_xyz[i] = torch.tensor(node.xyz[:3], dtype=torch.float32)
            ntype = node.node_type
            node_type[i] = NODE_TYPE_TO_IDX.get(ntype, 0)
            node_conf[i] = node.confidence

        # --- Edge matrix ---
        # Build node_id -> slot mapping
        id_to_slot = {}
        for i, node in enumerate(sorted_nodes[:num_nodes]):
            id_to_slot[node.id] = i

        edge_active = torch.zeros(max_n, max_n)
        edge_type_mat = torch.zeros(max_n, max_n, dtype=torch.long)

        for edge in graph.edges:
            si = id_to_slot.get(edge.source_node_id)
            ti = id_to_slot.get(edge.target_node_id)
            if si is not None and ti is not None:
                edge_active[si, ti] = 1.0
                edge_active[ti, si] = 1.0
                etype_idx = EDGE_TYPE_TO_IDX.get(edge.edge_type, 0)
                edge_type_mat[si, ti] = etype_idx
                edge_type_mat[ti, si] = etype_idx

        # --- Radius map ---
        radius_map = np.zeros((1, res, res), dtype=np.float32)
        for node in graph.nodes:
            if node.id not in positions or node.radius is None:
                continue
            px = int(positions[node.id][0] * (res - 1))
            py = int(positions[node.id][1] * (res - 1))
            r = min(node.radius, 1.0)
            rad_px = max(2, int(r * res * 0.1))
            cv2.circle(radius_map[0], (px, py), rad_px, float(r), -1)

        radius_tensor = torch.from_numpy(radius_map)

        return {
            "image": image_tensor,
            "heatmaps": heatmap_tensor,
            "radius_map": radius_tensor,
            "node_active": node_active,
            "node_xy": node_xy,
            "node_xyz": node_xyz,
            "node_type": node_type,
            "node_confidence": node_conf,
            "edge_active": edge_active,
            "edge_type": edge_type_mat,
            "sample_id": graph.sample_id,
            "input_source": input_source,
            "num_nodes": num_nodes,
        }


def split_train_val(
    graphs: list[GraphLabel],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[GraphLabel], list[GraphLabel]]:
    """Split graphs into train/val ensuring no duplicate sample_ids leak."""
    rng = random.Random(seed)
    ids = list({g.sample_id for g in graphs})
    rng.shuffle(ids)
    val_count = max(1, int(len(ids) * val_fraction))
    val_ids = set(ids[:val_count])
    train = [g for g in graphs if g.sample_id not in val_ids]
    val = [g for g in graphs if g.sample_id in val_ids]
    return train, val
