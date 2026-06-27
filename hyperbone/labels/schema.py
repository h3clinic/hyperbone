"""
HyperBone LabelForge schema.

Defines the canonical graph label format that fuses multiple label sources
into one unified training target.

Node types span biological joints, structural skeletons, and functional articulations.
Edge types span bones, branches, veins, medial axes, and deformation links.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


class NodeType(str, Enum):
    ROOT = "root"
    CENTER = "center"
    ENDPOINT = "endpoint"
    BRANCH = "branch"
    ARTICULATION = "articulation"
    BEND = "bend"
    RIDGE = "ridge"
    SEMANTIC_JOINT = "semantic_joint"
    UNKNOWN = "unknown"


class EdgeType(str, Enum):
    BONE = "bone"
    BRANCH = "branch"
    VEIN = "vein"
    MEDIAL_AXIS = "medial_axis"
    HINGE_LINK = "hinge_link"
    RIDGE = "ridge"
    DEFORMATION_LINK = "deformation_link"
    UNKNOWN = "unknown"


class LabelSource(str, Enum):
    MANUAL_REVIEW = "manual_review"
    RIG_GT = "rig_gt"
    SKINNING = "skinning"
    PROCEDURAL = "procedural"
    ANIMAL_DATASET = "animal_dataset"
    MOTION_ARTICULATION = "motion_articulation"
    MEDIAL_AXIS = "medial_axis"
    SELF_SUPERVISED = "self_supervised"
    VLM_SEMANTIC = "vlm_semantic"


# Priority weights for fusion
SOURCE_WEIGHTS: dict[LabelSource, float] = {
    LabelSource.MANUAL_REVIEW: 1.00,
    LabelSource.RIG_GT: 0.95,
    LabelSource.SKINNING: 0.90,
    LabelSource.PROCEDURAL: 0.90,
    LabelSource.ANIMAL_DATASET: 0.80,
    LabelSource.MOTION_ARTICULATION: 0.70,
    LabelSource.MEDIAL_AXIS: 0.55,
    LabelSource.SELF_SUPERVISED: 0.45,
    LabelSource.VLM_SEMANTIC: 0.20,
}


@dataclass
class HyperNodeLabel:
    """A single labeled node in the graph."""
    id: int
    node_type: NodeType
    xyz: Optional[list[float]] = None          # 3D canonical coords [x, y, z]
    xy: Optional[list[float]] = None           # 2D image coords [u, v] in pixels
    semantic: Optional[str] = None             # e.g. "front_left_knee", "vein_fork"
    radius: Optional[float] = None             # influence radius / thickness
    confidence: float = 0.0                    # fused confidence [0, 1]
    label_sources: dict[str, Optional[float]] = field(default_factory=dict)
    accepted: Optional[bool] = None            # None=unreviewed, True/False=human decision
    uncertainty_reason: Optional[str] = None   # why this node is uncertain
    parent_id: Optional[int] = None            # hierarchy parent

    def effective_confidence(self) -> float:
        """Weighted confidence from all contributing sources."""
        if not self.label_sources:
            return 0.0
        total_w = 0.0
        weighted_conf = 0.0
        for src_name, src_conf in self.label_sources.items():
            if src_conf is None:
                continue
            try:
                src = LabelSource(src_name)
            except ValueError:
                continue
            w = SOURCE_WEIGHTS.get(src, 0.1)
            weighted_conf += w * src_conf
            total_w += w
        return weighted_conf / total_w if total_w > 0 else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["node_type"] = self.node_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "HyperNodeLabel":
        d = dict(d)
        d["node_type"] = NodeType(d["node_type"])
        return cls(**d)


@dataclass
class HyperEdgeLabel:
    """A single labeled edge in the graph."""
    id: int
    source_node_id: int
    target_node_id: int
    edge_type: EdgeType
    confidence: float = 0.0
    label_sources: dict[str, Optional[float]] = field(default_factory=dict)
    length: Optional[float] = None             # 3D length if known
    semantic: Optional[str] = None             # e.g. "upper_arm", "midrib"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["edge_type"] = self.edge_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "HyperEdgeLabel":
        d = dict(d)
        d["edge_type"] = EdgeType(d["edge_type"])
        return cls(**d)


@dataclass
class GraphLabel:
    """Complete graph label for one frame/sample."""
    sample_id: str
    image_path: Optional[str] = None
    depth_path: Optional[str] = None
    mask_path: Optional[str] = None
    nodes: list[HyperNodeLabel] = field(default_factory=list)
    edges: list[HyperEdgeLabel] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)

    def nodes_by_type(self, ntype: NodeType) -> list[HyperNodeLabel]:
        return [n for n in self.nodes if n.node_type == ntype]

    def accepted_nodes(self) -> list[HyperNodeLabel]:
        return [n for n in self.nodes if n.accepted is not False]

    def uncertain_nodes(self) -> list[HyperNodeLabel]:
        return [n for n in self.nodes if n.uncertainty_reason is not None]

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "depth_path": self.depth_path,
            "mask_path": self.mask_path,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GraphLabel":
        nodes = [HyperNodeLabel.from_dict(n) for n in d.get("nodes", [])]
        edges = [HyperEdgeLabel.from_dict(e) for e in d.get("edges", [])]
        return cls(
            sample_id=d["sample_id"],
            image_path=d.get("image_path"),
            depth_path=d.get("depth_path"),
            mask_path=d.get("mask_path"),
            nodes=nodes,
            edges=edges,
            metadata=d.get("metadata", {}),
        )

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_jsonl_line(cls, line: str) -> "GraphLabel":
        return cls.from_dict(json.loads(line))


@dataclass
class LabelFusionReport:
    """Report from label fusion process."""
    total_candidates: int = 0
    merged_count: int = 0
    rejected_count: int = 0
    uncertain_count: int = 0
    final_node_count: int = 0
    final_edge_count: int = 0
    sources_used: list[str] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def save_graph_labels(labels: list[GraphLabel], path: Path) -> None:
    """Write graph labels as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for gl in labels:
            f.write(gl.to_jsonl_line() + "\n")


def load_graph_labels(path: Path) -> list[GraphLabel]:
    """Read graph labels from JSONL."""
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(GraphLabel.from_jsonl_line(line))
    return labels
