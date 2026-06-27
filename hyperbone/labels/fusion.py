"""
Label fusion: merge candidate nodes from multiple sources into one graph.

Weights:
  manual_review: 1.00
  rig_gt: 0.95
  skinning: 0.90
  procedural: 0.90
  animal_dataset: 0.80
  motion_articulation: 0.70
  medial_axis: 0.55
  self_supervised: 0.45
  vlm_semantic: 0.20

Rules:
- Merge nearby compatible nodes (same type + close location + compatible edges)
- Preserve multiple labels if a node is both branch and articulation
- Reject low-confidence isolated nodes
- Mark conflict cases for review
- Write uncertainty score
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
    LabelFusionReport,
    SOURCE_WEIGHTS,
)


@dataclass
class FusionConfig:
    """Configuration for label fusion."""
    # Spatial merge threshold (in canonical units)
    merge_distance: float = 0.05
    # Minimum fused confidence to keep a node
    min_confidence: float = 0.30
    # Minimum sources required to keep an uncertain node
    min_sources_for_uncertain: int = 1
    # Whether to reject isolated nodes (no edges)
    reject_isolated: bool = False
    # Max nodes before triggering review
    max_nodes_before_review: int = 100


def fuse_graph_labels(
    candidates: list[GraphLabel],
    config: FusionConfig = FusionConfig(),
) -> tuple[GraphLabel, LabelFusionReport]:
    """
    Fuse multiple GraphLabels (from different sources) into one.

    Each candidate GraphLabel contributes nodes and edges.
    Nodes that are spatially close and type-compatible are merged.
    Edges are merged if they connect the same merged nodes.

    Returns:
        (fused_graph, report)
    """
    report = LabelFusionReport()

    # Collect all candidate nodes with their source graph info
    all_nodes: list[HyperNodeLabel] = []
    all_edges: list[HyperEdgeLabel] = []
    node_remap: dict[tuple[int, int], int] = {}  # (graph_idx, old_id) -> merged_id

    for gi, graph in enumerate(candidates):
        for node in graph.nodes:
            all_nodes.append(node)
        for edge in graph.edges:
            all_edges.append(edge)
        source = graph.metadata.get("source", "unknown")
        if source not in report.sources_used:
            report.sources_used.append(source)

    report.total_candidates = len(all_nodes)

    if not all_nodes:
        return GraphLabel(sample_id="fused", nodes=[], edges=[]), report

    # === STEP 1: Spatial clustering of nodes ===
    merged_nodes = _merge_nearby_nodes(all_nodes, config)
    report.merged_count = report.total_candidates - len(merged_nodes)

    # === STEP 2: Filter by confidence ===
    accepted_nodes = []
    rejected_nodes = []
    uncertain_nodes = []

    for node in merged_nodes:
        eff_conf = node.effective_confidence()
        node.confidence = eff_conf

        if eff_conf >= config.min_confidence:
            # Check for type conflicts
            if node.uncertainty_reason:
                uncertain_nodes.append(node)
            else:
                accepted_nodes.append(node)
        else:
            # Low confidence
            num_sources = sum(1 for v in node.label_sources.values() if v is not None)
            if num_sources >= config.min_sources_for_uncertain:
                node.uncertainty_reason = "low_confidence"
                uncertain_nodes.append(node)
            else:
                node.accepted = False
                rejected_nodes.append(node)

    # === STEP 3: Reject isolated if configured ===
    final_nodes = accepted_nodes + uncertain_nodes
    if config.reject_isolated:
        node_ids = {n.id for n in final_nodes}
        connected_ids = set()
        for e in all_edges:
            if e.source_node_id in node_ids:
                connected_ids.add(e.source_node_id)
            if e.target_node_id in node_ids:
                connected_ids.add(e.target_node_id)
        isolated = [n for n in final_nodes if n.id not in connected_ids]
        for n in isolated:
            n.accepted = False
            n.uncertainty_reason = "isolated_node"
            rejected_nodes.append(n)
        final_nodes = [n for n in final_nodes if n.id in connected_ids]

    # === STEP 4: Remap edges ===
    # Build id mapping from merge step
    id_map = _build_id_map(all_nodes, final_nodes, config)
    final_edges = _remap_and_merge_edges(all_edges, id_map, final_nodes)

    # === STEP 5: Reassign sequential IDs ===
    for i, node in enumerate(final_nodes):
        node.id = i
    for i, edge in enumerate(final_edges):
        edge.id = i

    # Report
    report.rejected_count = len(rejected_nodes)
    report.uncertain_count = len(uncertain_nodes)
    report.final_node_count = len(final_nodes)
    report.final_edge_count = len(final_edges)
    report.conflicts = [
        {"node_id": n.id, "reason": n.uncertainty_reason}
        for n in uncertain_nodes
    ]

    fused = GraphLabel(
        sample_id="fused",
        nodes=final_nodes,
        edges=final_edges,
        metadata={
            "fusion_config": {
                "merge_distance": config.merge_distance,
                "min_confidence": config.min_confidence,
            },
            "sources": report.sources_used,
        },
    )
    return fused, report


def _merge_nearby_nodes(
    nodes: list[HyperNodeLabel],
    config: FusionConfig,
) -> list[HyperNodeLabel]:
    """
    Merge nodes that are spatially close and type-compatible.

    Uses greedy clustering: for each node, find the nearest existing cluster
    center within merge_distance. If found and types are compatible, merge.
    Otherwise start a new cluster.
    """
    if not nodes:
        return []

    clusters: list[list[HyperNodeLabel]] = []
    cluster_centers: list[np.ndarray] = []
    cluster_types: list[NodeType] = []

    for node in nodes:
        pos = _node_position(node)
        if pos is None:
            # No spatial info — can't merge, keep as-is
            clusters.append([node])
            cluster_centers.append(np.zeros(3))
            cluster_types.append(node.node_type)
            continue

        merged = False
        best_dist = float("inf")
        best_idx = -1

        for ci, center in enumerate(cluster_centers):
            if cluster_centers[ci] is None:
                continue
            dist = np.linalg.norm(pos - center)
            if dist < config.merge_distance and dist < best_dist:
                if _types_compatible(node.node_type, cluster_types[ci]):
                    best_dist = dist
                    best_idx = ci

        if best_idx >= 0:
            clusters[best_idx].append(node)
            # Update center as weighted average
            n_in_cluster = len(clusters[best_idx])
            cluster_centers[best_idx] = (
                cluster_centers[best_idx] * (n_in_cluster - 1) + pos
            ) / n_in_cluster
            merged = True

        if not merged:
            clusters.append([node])
            cluster_centers.append(pos)
            cluster_types.append(node.node_type)

    # Merge each cluster into a single node
    merged_nodes = []
    for cluster in clusters:
        merged_nodes.append(_merge_cluster(cluster))
    return merged_nodes


def _merge_cluster(cluster: list[HyperNodeLabel]) -> HyperNodeLabel:
    """Merge a cluster of nearby compatible nodes into one."""
    if len(cluster) == 1:
        return cluster[0]

    # Use highest-weight source's position
    best_node = max(cluster, key=lambda n: n.effective_confidence())

    # Merge label sources
    merged_sources: dict[str, Optional[float]] = {}
    for node in cluster:
        for src, conf in node.label_sources.items():
            if src not in merged_sources or (conf is not None and (
                merged_sources[src] is None or conf > merged_sources[src]
            )):
                merged_sources[src] = conf

    # Determine type: use most common, or mark conflict
    types = [n.node_type for n in cluster]
    type_counts: dict[NodeType, int] = {}
    for t in types:
        type_counts[t] = type_counts.get(t, 0) + 1
    dominant_type = max(type_counts, key=type_counts.get)

    uncertainty = None
    if len(type_counts) > 1:
        uncertainty = f"type_conflict: {[t.value for t in type_counts.keys()]}"

    # Merge semantics (take non-None)
    semantics = [n.semantic for n in cluster if n.semantic]
    semantic = semantics[0] if semantics else None

    # Average position
    positions = [_node_position(n) for n in cluster if _node_position(n) is not None]
    avg_pos = np.mean(positions, axis=0).tolist() if positions else best_node.xyz

    # Average xy
    xys = [n.xy for n in cluster if n.xy is not None]
    avg_xy = np.mean(xys, axis=0).tolist() if xys else best_node.xy

    return HyperNodeLabel(
        id=best_node.id,
        node_type=dominant_type,
        xyz=avg_pos,
        xy=avg_xy,
        semantic=semantic,
        radius=best_node.radius,
        confidence=0.0,  # will be recomputed
        label_sources=merged_sources,
        accepted=None if uncertainty else True,
        uncertainty_reason=uncertainty,
        parent_id=best_node.parent_id,
    )


def _node_position(node: HyperNodeLabel) -> Optional[np.ndarray]:
    """Get 3D position of a node, or None if not available."""
    if node.xyz is not None:
        return np.array(node.xyz, dtype=np.float64)
    return None


def _types_compatible(t1: NodeType, t2: NodeType) -> bool:
    """Check if two node types can be merged."""
    if t1 == t2:
        return True
    if NodeType.UNKNOWN in (t1, t2):
        return True
    # Compatible pairs
    compat = {
        frozenset({NodeType.ARTICULATION, NodeType.SEMANTIC_JOINT}),
        frozenset({NodeType.ENDPOINT, NodeType.SEMANTIC_JOINT}),
        frozenset({NodeType.BRANCH, NodeType.ARTICULATION}),
        frozenset({NodeType.BEND, NodeType.ARTICULATION}),
    }
    return frozenset({t1, t2}) in compat


def _build_id_map(
    original_nodes: list[HyperNodeLabel],
    final_nodes: list[HyperNodeLabel],
    config: FusionConfig,
) -> dict[int, int]:
    """Map original node IDs to final merged node IDs."""
    id_map: dict[int, int] = {}
    final_positions = []
    for n in final_nodes:
        pos = _node_position(n)
        final_positions.append(pos)

    for orig in original_nodes:
        orig_pos = _node_position(orig)
        if orig_pos is None:
            continue
        best_idx = -1
        best_dist = float("inf")
        for fi, fpos in enumerate(final_positions):
            if fpos is None:
                continue
            d = np.linalg.norm(orig_pos - fpos)
            if d < best_dist:
                best_dist = d
                best_idx = fi
        if best_idx >= 0 and best_dist < config.merge_distance * 2:
            id_map[orig.id] = final_nodes[best_idx].id
    return id_map


def _remap_and_merge_edges(
    edges: list[HyperEdgeLabel],
    id_map: dict[int, int],
    final_nodes: list[HyperNodeLabel],
) -> list[HyperEdgeLabel]:
    """Remap edge endpoints and deduplicate."""
    valid_ids = {n.id for n in final_nodes}
    seen: set[tuple[int, int, str]] = set()
    result = []

    for edge in edges:
        src = id_map.get(edge.source_node_id, edge.source_node_id)
        tgt = id_map.get(edge.target_node_id, edge.target_node_id)
        if src not in valid_ids or tgt not in valid_ids:
            continue
        if src == tgt:
            continue
        key = (min(src, tgt), max(src, tgt), edge.edge_type.value)
        if key in seen:
            continue
        seen.add(key)
        result.append(HyperEdgeLabel(
            id=len(result),
            source_node_id=src,
            target_node_id=tgt,
            edge_type=edge.edge_type,
            confidence=edge.confidence,
            label_sources=edge.label_sources,
            length=edge.length,
            semantic=edge.semantic,
        ))
    return result
