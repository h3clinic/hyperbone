"""
Graph-level acceptance calibration for LabelForge.

Adds graph-level quality scoring and trainability classification:
- graph_confidence_mean / min
- low_confidence_node_ratio
- uncertain_node_ratio
- source_diversity_count
- accepted / review_needed / rejected status
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from hyperbone.labels.schema import GraphLabel, NodeType, LabelSource


@dataclass
class GraphQuality:
    """Quality metrics for a single graph."""
    graph_confidence_mean: float = 0.0
    graph_confidence_min: float = 0.0
    low_confidence_node_ratio: float = 0.0
    uncertain_node_count: int = 0
    uncertain_node_ratio: float = 0.0
    source_diversity_count: int = 0
    node_count: int = 0
    edge_count: int = 0
    is_connected: bool = True
    accepted: bool = True
    review_needed: bool = False
    reject_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class AcceptanceConfig:
    """Configuration for graph acceptance calibration."""
    min_nodes: int = 2
    min_edges: int = 1
    min_graph_confidence_mean: float = 0.45
    low_confidence_threshold: float = 0.50
    max_low_confidence_ratio: float = 0.60
    max_uncertain_ratio_for_review: float = 0.25
    # Sources that require multi-source confirmation for acceptance
    weak_single_sources: set = field(default_factory=lambda: {"medial_axis", "motion_articulation"})
    # Allow fragments for these sources
    fragment_allowed_sources: set = field(default_factory=lambda: {
        "procedural_rock", "medial_axis", "rigged_asset", "rig_extraction",
    })
    # Minimum confidence for trainable
    trainable_min_confidence: float = 0.60


def assess_graph_quality(
    graph: GraphLabel,
    config: AcceptanceConfig = AcceptanceConfig(),
) -> GraphQuality:
    """
    Compute graph-level quality metrics and acceptance decision.

    Returns GraphQuality with acceptance/review/rejection status.
    """
    q = GraphQuality()
    q.node_count = graph.node_count()
    q.edge_count = graph.edge_count()

    # === Rejection checks (hard fails) ===
    if q.node_count < config.min_nodes:
        q.accepted = False
        q.reject_reason = f"too_few_nodes ({q.node_count} < {config.min_nodes})"
        return q

    if q.edge_count < config.min_edges:
        q.accepted = False
        q.reject_reason = f"too_few_edges ({q.edge_count} < {config.min_edges})"
        return q

    # === Confidence metrics ===
    confidences = [n.confidence for n in graph.nodes]
    q.graph_confidence_mean = float(np.mean(confidences))
    q.graph_confidence_min = float(np.min(confidences))

    low_conf_count = sum(1 for c in confidences if c < config.low_confidence_threshold)
    q.low_confidence_node_ratio = low_conf_count / q.node_count

    if q.graph_confidence_mean < config.min_graph_confidence_mean:
        q.accepted = False
        q.reject_reason = f"low_mean_confidence ({q.graph_confidence_mean:.3f} < {config.min_graph_confidence_mean})"
        return q

    # === Uncertainty ===
    q.uncertain_node_count = len(graph.uncertain_nodes())
    q.uncertain_node_ratio = q.uncertain_node_count / q.node_count

    # === Source diversity ===
    all_sources = set()
    for n in graph.nodes:
        for src in n.label_sources:
            all_sources.add(src)
    q.source_diversity_count = len(all_sources)

    # === Connectivity check ===
    q.is_connected = _check_connected(graph)

    # === Review triggers ===
    source = graph.metadata.get("source", "")

    if q.uncertain_node_ratio > config.max_uncertain_ratio_for_review:
        q.review_needed = True

    if q.source_diversity_count == 1 and source in config.weak_single_sources:
        q.review_needed = True

    if not q.is_connected and source not in config.fragment_allowed_sources:
        q.review_needed = True

    if q.low_confidence_node_ratio > config.max_low_confidence_ratio:
        q.review_needed = True

    return q


def classify_graph(
    graph: GraphLabel,
    quality: GraphQuality,
    config: AcceptanceConfig = AcceptanceConfig(),
) -> str:
    """
    Classify graph into: 'trainable', 'review', 'rejected'.

    Trainable: accepted, not review_needed, high confidence.
    Review: accepted but needs human review.
    Rejected: failed quality checks.
    """
    if not quality.accepted:
        return "rejected"
    if quality.review_needed:
        return "review"
    if quality.graph_confidence_mean < config.trainable_min_confidence:
        return "review"
    return "trainable"


def _check_connected(graph: GraphLabel) -> bool:
    """Check if the graph is connected (all nodes reachable via edges)."""
    if graph.node_count() <= 1:
        return True
    if graph.edge_count() == 0:
        return False

    node_ids = {n.id for n in graph.nodes}
    adj: dict[int, set] = {nid: set() for nid in node_ids}
    for e in graph.edges:
        if e.source_node_id in adj and e.target_node_id in adj:
            adj[e.source_node_id].add(e.target_node_id)
            adj[e.target_node_id].add(e.source_node_id)

    # BFS from first node
    start = next(iter(node_ids))
    visited = set()
    queue = [start]
    while queue:
        nid = queue.pop()
        if nid in visited:
            continue
        visited.add(nid)
        queue.extend(adj[nid] - visited)

    return len(visited) == len(node_ids)
