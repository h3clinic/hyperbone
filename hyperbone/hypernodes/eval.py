"""
Evaluation metrics for HyperNodeNet.

Computes node/edge precision/recall/F1, Chamfer distance,
type accuracy, and source-specific breakdowns.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from hyperbone.labels.schema import GraphLabel, NodeType, EdgeType
from hyperbone.hypernodes.dataset import NODE_TYPES, NODE_TYPE_TO_IDX


@dataclass
class MetricsResult:
    node_precision: float = 0.0
    node_recall: float = 0.0
    node_f1: float = 0.0
    edge_precision: float = 0.0
    edge_recall: float = 0.0
    edge_f1: float = 0.0
    graph_chamfer: float = 0.0
    node_type_accuracy: float = 0.0
    edge_type_accuracy: float = 0.0
    avg_pred_node_count: float = 0.0
    avg_target_node_count: float = 0.0
    invalid_graph_rate: float = 0.0
    per_node_type_f1: dict[str, float] = field(default_factory=dict)
    per_source_metrics: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "node_precision": self.node_precision,
            "node_recall": self.node_recall,
            "node_f1": self.node_f1,
            "edge_precision": self.edge_precision,
            "edge_recall": self.edge_recall,
            "edge_f1": self.edge_f1,
            "graph_chamfer": self.graph_chamfer,
            "node_type_accuracy": self.node_type_accuracy,
            "edge_type_accuracy": self.edge_type_accuracy,
            "avg_pred_node_count": self.avg_pred_node_count,
            "avg_target_node_count": self.avg_target_node_count,
            "invalid_graph_rate": self.invalid_graph_rate,
            "per_node_type_f1": self.per_node_type_f1,
            "per_source_metrics": self.per_source_metrics,
        }


def _get_node_positions(graph: GraphLabel) -> np.ndarray:
    """Extract [N, 2] array of node positions, normalized to [0, 1]."""
    from hyperbone.hypernodes.dataset import normalize_positions

    positions = normalize_positions(graph, 192)  # resolution doesn't matter for normalization
    if not positions and graph.nodes:
        # Fallback
        return np.zeros((len(graph.nodes), 2), dtype=np.float32)
    if not graph.nodes:
        return np.zeros((0, 2), dtype=np.float32)

    # Return positions in graph node order
    pts = []
    for n in graph.nodes:
        if n.id in positions:
            pts.append(positions[n.id])
        else:
            pts.append(np.array([0.0, 0.0], dtype=np.float32))
    return np.array(pts, dtype=np.float32)


def _match_nodes(
    pred_pts: np.ndarray,
    target_pts: np.ndarray,
    threshold: float = 0.05,
) -> tuple[int, int, int, list[tuple[int, int]]]:
    """Match predicted to target nodes by nearest-neighbor within threshold.
    
    Returns: (true_positives, false_positives, false_negatives, matched_pairs)
    """
    if pred_pts.shape[0] == 0 and target_pts.shape[0] == 0:
        return 0, 0, 0, []
    if pred_pts.shape[0] == 0:
        return 0, 0, target_pts.shape[0], []
    if target_pts.shape[0] == 0:
        return 0, pred_pts.shape[0], 0, []

    # Compute pairwise distances
    # [M, K]
    diff = pred_pts[:, None, :] - target_pts[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)

    matched_pred = set()
    matched_target = set()
    pairs = []

    # Greedy matching by distance
    flat_idx = np.argsort(dists.ravel())
    for idx in flat_idx:
        pi = idx // target_pts.shape[0]
        ti = idx % target_pts.shape[0]
        if pi in matched_pred or ti in matched_target:
            continue
        if dists[pi, ti] > threshold:
            break
        matched_pred.add(pi)
        matched_target.add(ti)
        pairs.append((pi, ti))

    tp = len(pairs)
    fp = pred_pts.shape[0] - tp
    fn = target_pts.shape[0] - tp
    return tp, fp, fn, pairs


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_metrics(
    predictions: list[GraphLabel],
    targets: list[GraphLabel],
    distance_threshold: float = 0.05,
    _depth: int = 0,
) -> MetricsResult:
    """Compute evaluation metrics between predicted and target graphs."""
    total_node_tp = 0
    total_node_fp = 0
    total_node_fn = 0
    total_edge_tp = 0
    total_edge_fp = 0
    total_edge_fn = 0
    chamfer_sum = 0.0
    chamfer_count = 0
    type_correct = 0
    type_total = 0
    etype_correct = 0
    etype_total = 0
    pred_node_counts = []
    target_node_counts = []
    invalid_count = 0

    # Per node type tracking
    per_type_tp = defaultdict(int)
    per_type_fp = defaultdict(int)
    per_type_fn = defaultdict(int)

    # Per source tracking
    source_preds = defaultdict(list)
    source_targets = defaultdict(list)

    for pred_g, target_g in zip(predictions, targets):
        # Check validity
        is_valid = pred_g.metadata.get("valid", True)
        if not is_valid:
            invalid_count += 1
            total_node_fn += len(target_g.nodes)
            total_edge_fn += len(target_g.edges)
            target_node_counts.append(len(target_g.nodes))
            pred_node_counts.append(0)
            continue

        pred_pts = _get_node_positions(pred_g)
        target_pts = _get_node_positions(target_g)

        pred_node_counts.append(len(pred_g.nodes))
        target_node_counts.append(len(target_g.nodes))

        # Node matching
        tp, fp, fn, pairs = _match_nodes(pred_pts, target_pts, distance_threshold)
        total_node_tp += tp
        total_node_fp += fp
        total_node_fn += fn

        # Node type accuracy on matched pairs
        for pi, ti in pairs:
            pred_type = pred_g.nodes[pi].node_type
            target_type = target_g.nodes[ti].node_type
            type_total += 1
            if pred_type == target_type:
                type_correct += 1

        # Per-node-type metrics
        for ti_idx, tnode in enumerate(target_g.nodes):
            nt = tnode.node_type.value
            matched = any(ti == ti_idx for _, ti in pairs)
            if matched:
                per_type_tp[nt] += 1
            else:
                per_type_fn[nt] += 1

        for pi_idx, pnode in enumerate(pred_g.nodes):
            nt = pnode.node_type.value
            matched = any(pi == pi_idx for pi, _ in pairs)
            if not matched:
                per_type_fp[nt] += 1

        # Chamfer distance
        if pred_pts.shape[0] > 0 and target_pts.shape[0] > 0:
            diff = pred_pts[:, None, :] - target_pts[None, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            d_p2t = dists.min(axis=1).mean()
            d_t2p = dists.min(axis=0).mean()
            chamfer_sum += (d_p2t + d_t2p) * 0.5
            chamfer_count += 1

        # Edge matching
        pred_edges = {(e.source_node_id, e.target_node_id) for e in pred_g.edges}
        pred_edges |= {(e.target_node_id, e.source_node_id) for e in pred_g.edges}
        target_edges = {(e.source_node_id, e.target_node_id) for e in target_g.edges}
        target_edges |= {(e.target_node_id, e.source_node_id) for e in target_g.edges}

        # Map matched node IDs
        pred_to_target_map = {pi: ti for pi, ti in pairs}
        mapped_pred_edges = set()
        for pe in pred_g.edges:
            src_mapped = pred_to_target_map.get(pe.source_node_id)
            tgt_mapped = pred_to_target_map.get(pe.target_node_id)
            if src_mapped is not None and tgt_mapped is not None:
                mapped_pred_edges.add((src_mapped, tgt_mapped))
                mapped_pred_edges.add((tgt_mapped, src_mapped))

        target_edge_set = set()
        for te in target_g.edges:
            target_edge_set.add((te.source_node_id, te.target_node_id))
            target_edge_set.add((te.target_node_id, te.source_node_id))

        edge_tp = len(mapped_pred_edges & target_edge_set) // 2
        edge_fp = len(pred_g.edges) - edge_tp
        edge_fn = len(target_g.edges) - edge_tp
        total_edge_tp += max(edge_tp, 0)
        total_edge_fp += max(edge_fp, 0)
        total_edge_fn += max(edge_fn, 0)

        # Edge type accuracy
        for pe in pred_g.edges:
            src_m = pred_to_target_map.get(pe.source_node_id)
            tgt_m = pred_to_target_map.get(pe.target_node_id)
            if src_m is not None and tgt_m is not None:
                # Find matching target edge
                for te in target_g.edges:
                    if (te.source_node_id == src_m and te.target_node_id == tgt_m) or \
                       (te.source_node_id == tgt_m and te.target_node_id == src_m):
                        etype_total += 1
                        if pe.edge_type == te.edge_type:
                            etype_correct += 1
                        break

        # Source-specific grouping
        source = target_g.metadata.get("source", "unknown")
        source_preds[source].append(pred_g)
        source_targets[source].append(target_g)

    # Aggregate
    n_total = len(predictions)
    node_prec = total_node_tp / max(total_node_tp + total_node_fp, 1)
    node_rec = total_node_tp / max(total_node_tp + total_node_fn, 1)
    edge_prec = total_edge_tp / max(total_edge_tp + total_edge_fp, 1)
    edge_rec = total_edge_tp / max(total_edge_tp + total_edge_fn, 1)

    # Per-type F1
    per_type_f1 = {}
    for nt in set(list(per_type_tp.keys()) + list(per_type_fn.keys()) + list(per_type_fp.keys())):
        tp = per_type_tp[nt]
        fp = per_type_fp[nt]
        fn = per_type_fn[nt]
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        per_type_f1[nt] = _f1(p, r)

    # Per-source
    per_source = {}
    if _depth == 0:
        for src in source_preds:
            s_preds = source_preds[src]
            s_targets = source_targets[src]
            s_metrics = compute_metrics(s_preds, s_targets, distance_threshold, _depth=1)
            per_source[src] = {
                "node_f1": s_metrics.node_f1,
                "edge_f1": s_metrics.edge_f1,
                "count": len(s_preds),
            }

    return MetricsResult(
        node_precision=node_prec,
        node_recall=node_rec,
        node_f1=_f1(node_prec, node_rec),
        edge_precision=edge_prec,
        edge_recall=edge_rec,
        edge_f1=_f1(edge_prec, edge_rec),
        graph_chamfer=chamfer_sum / max(chamfer_count, 1),
        node_type_accuracy=type_correct / max(type_total, 1),
        edge_type_accuracy=etype_correct / max(etype_total, 1),
        avg_pred_node_count=np.mean(pred_node_counts) if pred_node_counts else 0.0,
        avg_target_node_count=np.mean(target_node_counts) if target_node_counts else 0.0,
        invalid_graph_rate=invalid_count / max(n_total, 1),
        per_node_type_f1=per_type_f1,
        per_source_metrics=per_source,
    )
