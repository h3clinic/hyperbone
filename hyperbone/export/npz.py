"""NPZ export for skeleton graph sequences."""

import numpy as np
from pathlib import Path
from typing import List, Dict


def export_npz(
    records: List[Dict],
    output_dir: str,
    filename: str = "skeleton_data.npz",
    max_nodes: int = 64,
) -> Path:
    """Export a list of graph records to a single .npz file.

    Stores:
        nodes_xy:    [T, N, 2]
        node_active: [T, N]
        node_type:   [T, N]  (0=none, 1=endpoint, 2=junction, 3=center)
        edges:       [T, N, N]  adjacency matrix
        confidence:  [T, N]
        frame_idx:   [T]
        timestamp:   [T]
    """
    T = len(records)
    nodes_xy = np.zeros((T, max_nodes, 2), dtype=np.float32)
    node_active = np.zeros((T, max_nodes), dtype=np.uint8)
    node_type = np.zeros((T, max_nodes), dtype=np.uint8)
    edges_mat = np.zeros((T, max_nodes, max_nodes), dtype=np.uint8)
    confidence = np.zeros((T, max_nodes), dtype=np.float32)
    frame_indices = np.zeros(T, dtype=np.int32)
    timestamps = np.zeros(T, dtype=np.float32)

    type_map = {"endpoint": 1, "junction": 2, "center": 3}

    for t, rec in enumerate(records):
        frame_indices[t] = rec.get("frame_idx", 0)
        timestamps[t] = rec.get("timestamp_sec", 0.0)

        nodes = rec.get("nodes", [])
        for i, node in enumerate(nodes[:max_nodes]):
            nodes_xy[t, i] = node["xy"]
            node_active[t, i] = 1
            node_type[t, i] = type_map.get(node.get("type", ""), 0)
            confidence[t, i] = node.get("confidence", 1.0)

        for edge in rec.get("edges", []):
            p, c = edge["parent"], edge["child"]
            if p < max_nodes and c < max_nodes:
                edges_mat[t, p, c] = 1
                edges_mat[t, c, p] = 1

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    np.savez_compressed(
        str(path),
        nodes_xy=nodes_xy,
        node_active=node_active,
        node_type=node_type,
        edges=edges_mat,
        confidence=confidence,
        frame_idx=frame_indices,
        timestamp=timestamps,
    )
    return path
