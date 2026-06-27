"""Simple graph tracking stability metrics.

Evaluates whether HyperBone graph outputs are structurally stable
across consecutive sampled frames. This is NOT semantic tracking.
It is a structural stability smoke test.
"""

import numpy as np
from typing import Dict, List, Optional


def compute_graph_metrics(graph_record: Dict) -> Dict:
    """Compute per-frame metrics from a graph record."""
    nodes = graph_record.get("nodes", [])
    edges = graph_record.get("edges", [])
    bbox = graph_record.get("bbox_xywh", [0, 0, 0, 0])

    node_count = len(nodes)
    edge_count = len(edges)

    # Graph centroid (mean of node positions)
    if nodes:
        xs = [n["xy"][0] for n in nodes]
        ys = [n["xy"][1] for n in nodes]
        centroid_x = float(np.mean(xs))
        centroid_y = float(np.mean(ys))
        graph_bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
    else:
        centroid_x = bbox[0] + bbox[2] / 2
        centroid_y = bbox[1] + bbox[3] / 2
        graph_bbox = list(bbox)

    # Bbox center
    bbox_center_x = bbox[0] + bbox[2] / 2
    bbox_center_y = bbox[1] + bbox[3] / 2

    # Skeleton length estimate (sum of edge lengths)
    skeleton_length = 0.0
    node_map = {n.get("id", i): n["xy"] for i, n in enumerate(nodes)}
    for edge in edges:
        p = node_map.get(edge.get("parent"))
        c = node_map.get(edge.get("child"))
        if p and c:
            skeleton_length += ((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2) ** 0.5

    return {
        "frame_idx": graph_record.get("frame_idx", 0),
        "accepted": graph_record.get("accepted", False),
        "node_count": node_count,
        "edge_count": edge_count,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "bbox_center_x": bbox_center_x,
        "bbox_center_y": bbox_center_y,
        "graph_bbox": graph_bbox,
        "skeleton_length": skeleton_length,
    }


def compute_tracking_metrics(graph_records: List[Dict]) -> Dict:
    """Compute temporal tracking stability metrics across a sequence.

    Args:
        graph_records: List of graph records (from graphs.jsonl), sorted by frame_idx.

    Returns:
        Dict with tracking stability metrics.
    """
    if not graph_records:
        return _empty_metrics()

    per_frame = [compute_graph_metrics(r) for r in graph_records]

    frames_with_proposal = len(per_frame)
    frames_with_graph = sum(1 for m in per_frame if m["node_count"] > 0)
    accepted_frames = sum(1 for m in per_frame if m["accepted"])
    rejected_frames = sum(1 for m in per_frame if not m["accepted"])

    graph_presence_rate = frames_with_graph / frames_with_proposal if frames_with_proposal else 0
    acceptance_rate = accepted_frames / frames_with_proposal if frames_with_proposal else 0

    # Centroid jumps between consecutive frames
    centroid_jumps = []
    for i in range(1, len(per_frame)):
        dx = per_frame[i]["centroid_x"] - per_frame[i - 1]["centroid_x"]
        dy = per_frame[i]["centroid_y"] - per_frame[i - 1]["centroid_y"]
        jump = (dx ** 2 + dy ** 2) ** 0.5
        centroid_jumps.append(jump)

    centroid_jump_mean = float(np.mean(centroid_jumps)) if centroid_jumps else 0
    centroid_jump_p95 = float(np.percentile(centroid_jumps, 95)) if centroid_jumps else 0

    # Node/edge stats
    node_counts = [m["node_count"] for m in per_frame if m["node_count"] > 0]
    edge_counts = [m["edge_count"] for m in per_frame if m["edge_count"] > 0]

    node_count_mean = float(np.mean(node_counts)) if node_counts else 0
    node_count_std = float(np.std(node_counts)) if node_counts else 0
    edge_count_mean = float(np.mean(edge_counts)) if edge_counts else 0
    edge_count_std = float(np.std(edge_counts)) if edge_counts else 0

    # Skeleton length stats
    skel_lengths = [m["skeleton_length"] for m in per_frame if m["skeleton_length"] > 0]
    skeleton_length_mean = float(np.mean(skel_lengths)) if skel_lengths else 0
    skeleton_length_std = float(np.std(skel_lengths)) if skel_lengths else 0

    # Topology stability score:
    # 1.0 - normalized variation of node_count and edge_count
    # CV = std/mean; score = 1 - (avg CV clamped to [0,1])
    cv_nodes = node_count_std / node_count_mean if node_count_mean > 0 else 1.0
    cv_edges = edge_count_std / edge_count_mean if edge_count_mean > 0 else 1.0
    avg_cv = (cv_nodes + cv_edges) / 2
    topology_stability_score = max(0.0, 1.0 - avg_cv)

    return {
        "frames_with_proposal": frames_with_proposal,
        "frames_with_graph": frames_with_graph,
        "accepted_frames": accepted_frames,
        "rejected_frames": rejected_frames,
        "graph_presence_rate": round(graph_presence_rate, 4),
        "acceptance_rate": round(acceptance_rate, 4),
        "centroid_jump_px_mean": round(centroid_jump_mean, 2),
        "centroid_jump_px_p95": round(centroid_jump_p95, 2),
        "node_count_mean": round(node_count_mean, 1),
        "node_count_std": round(node_count_std, 1),
        "edge_count_mean": round(edge_count_mean, 1),
        "edge_count_std": round(edge_count_std, 1),
        "skeleton_length_mean": round(skeleton_length_mean, 1),
        "skeleton_length_std": round(skeleton_length_std, 1),
        "topology_stability_score": round(topology_stability_score, 4),
        "per_frame_metrics": per_frame,
    }


def _empty_metrics() -> Dict:
    return {
        "frames_with_proposal": 0,
        "frames_with_graph": 0,
        "accepted_frames": 0,
        "rejected_frames": 0,
        "graph_presence_rate": 0,
        "acceptance_rate": 0,
        "centroid_jump_px_mean": 0,
        "centroid_jump_px_p95": 0,
        "node_count_mean": 0,
        "node_count_std": 0,
        "edge_count_mean": 0,
        "edge_count_std": 0,
        "skeleton_length_mean": 0,
        "skeleton_length_std": 0,
        "topology_stability_score": 0,
        "per_frame_metrics": [],
    }
