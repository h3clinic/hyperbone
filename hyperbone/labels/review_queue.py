"""
Review queue for uncertain/conflicting label candidates.

Writes cases that need human review to a JSONL queue file.
Prioritizes cases by:
- Source disagreement
- Low confidence
- Abnormal node count
- Motion says articulation but medial graph disagrees
- Missing expected joints for known animal schema
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from hyperbone.labels.schema import (
    GraphLabel,
    HyperNodeLabel,
    NodeType,
    LabelSource,
)


@dataclass
class ReviewItem:
    """One item in the review queue."""
    sample_id: str
    node_id: int
    reason: str
    priority: float                      # higher = more urgent [0, 1]
    node_type: str = ""
    semantic: Optional[str] = None
    confidence: float = 0.0
    label_sources: dict = field(default_factory=dict)
    suggested_action: str = ""           # "accept", "reject", "edit_position", "edit_type"
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ReviewItem":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ReviewQueueConfig:
    """Configuration for review queue generation."""
    # Confidence thresholds
    low_confidence_threshold: float = 0.50
    # Source disagreement: if top two sources disagree by this much
    source_disagreement_threshold: float = 0.40
    # Expected semantic joints for animal schema
    expected_animal_joints: list[str] = field(default_factory=lambda: [
        "head", "neck", "tail_base",
        "front_left_shoulder", "front_left_hoof",
        "front_right_shoulder", "front_right_hoof",
        "rear_left_hip", "rear_left_hoof",
        "rear_right_hip", "rear_right_hoof",
    ])
    # Maximum expected node count per graph
    max_normal_nodes: int = 50
    # Minimum expected node count
    min_normal_nodes: int = 3


def generate_review_queue(
    graph: GraphLabel,
    config: ReviewQueueConfig = ReviewQueueConfig(),
) -> list[ReviewItem]:
    """
    Analyze a fused graph and generate review items for uncertain cases.

    Returns list of ReviewItems sorted by priority (descending).
    """
    items: list[ReviewItem] = []

    # Check abnormal node count
    if graph.node_count() > config.max_normal_nodes:
        items.append(ReviewItem(
            sample_id=graph.sample_id,
            node_id=-1,
            reason="abnormal_node_count_high",
            priority=0.6,
            context={"node_count": graph.node_count(), "max": config.max_normal_nodes},
            suggested_action="review_graph",
        ))
    elif graph.node_count() < config.min_normal_nodes and graph.node_count() > 0:
        items.append(ReviewItem(
            sample_id=graph.sample_id,
            node_id=-1,
            reason="abnormal_node_count_low",
            priority=0.4,
            context={"node_count": graph.node_count(), "min": config.min_normal_nodes},
            suggested_action="review_graph",
        ))

    # Per-node checks
    for node in graph.nodes:
        # Low confidence
        if node.confidence < config.low_confidence_threshold:
            items.append(ReviewItem(
                sample_id=graph.sample_id,
                node_id=node.id,
                reason="low_confidence",
                priority=0.7 * (1.0 - node.confidence),
                node_type=node.node_type.value,
                semantic=node.semantic,
                confidence=node.confidence,
                label_sources=node.label_sources,
                suggested_action="accept" if node.confidence > 0.3 else "reject",
            ))

        # Source disagreement
        if len(node.label_sources) >= 2:
            confs = [v for v in node.label_sources.values() if v is not None]
            if len(confs) >= 2:
                confs_sorted = sorted(confs, reverse=True)
                disagreement = confs_sorted[0] - confs_sorted[-1]
                if disagreement > config.source_disagreement_threshold:
                    items.append(ReviewItem(
                        sample_id=graph.sample_id,
                        node_id=node.id,
                        reason="source_disagreement",
                        priority=0.8 * disagreement,
                        node_type=node.node_type.value,
                        semantic=node.semantic,
                        confidence=node.confidence,
                        label_sources=node.label_sources,
                        suggested_action="edit_type",
                        context={"disagreement": round(disagreement, 3)},
                    ))

        # Explicit uncertainty from fusion
        if node.uncertainty_reason:
            items.append(ReviewItem(
                sample_id=graph.sample_id,
                node_id=node.id,
                reason=node.uncertainty_reason,
                priority=0.75,
                node_type=node.node_type.value,
                semantic=node.semantic,
                confidence=node.confidence,
                label_sources=node.label_sources,
                suggested_action="edit_type",
            ))

        # Motion says articulation but is type mismatch
        motion_conf = node.label_sources.get(LabelSource.MOTION_ARTICULATION.value)
        if motion_conf and motion_conf > 0.5:
            if node.node_type not in (NodeType.ARTICULATION, NodeType.SEMANTIC_JOINT, NodeType.BEND):
                items.append(ReviewItem(
                    sample_id=graph.sample_id,
                    node_id=node.id,
                    reason="motion_type_mismatch",
                    priority=0.65,
                    node_type=node.node_type.value,
                    semantic=node.semantic,
                    confidence=node.confidence,
                    label_sources=node.label_sources,
                    suggested_action="edit_type",
                    context={"motion_conf": motion_conf},
                ))

    # Check missing expected joints (for animal-like graphs)
    if _looks_like_animal(graph):
        present_semantics = {n.semantic for n in graph.nodes if n.semantic}
        for expected in config.expected_animal_joints:
            if expected not in present_semantics:
                items.append(ReviewItem(
                    sample_id=graph.sample_id,
                    node_id=-1,
                    reason="missing_expected_joint",
                    priority=0.5,
                    semantic=expected,
                    suggested_action="add_node",
                    context={"expected_joint": expected},
                ))

    # Sort by priority descending
    items.sort(key=lambda x: x.priority, reverse=True)
    return items


def _looks_like_animal(graph: GraphLabel) -> bool:
    """Heuristic: does this graph contain animal-like semantic joints?"""
    animal_keywords = {"shoulder", "hip", "knee", "elbow", "hoof", "paw", "neck", "head", "tail"}
    for node in graph.nodes:
        if node.semantic and any(kw in node.semantic for kw in animal_keywords):
            return True
    return False


def save_review_queue(items: list[ReviewItem], path: Path) -> None:
    """Write review queue as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item.to_dict()) + "\n")


def load_review_queue(path: Path) -> list[ReviewItem]:
    """Load review queue from JSONL."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(ReviewItem.from_dict(json.loads(line)))
    return items
