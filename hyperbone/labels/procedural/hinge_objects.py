"""
Procedural hinge/articulated object generator.

Generates simple articulated objects (hinges, levers, robot arms, doors)
where articulation points are known by construction.
"""
from __future__ import annotations

import math
import random
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
class HingeParams:
    """Parameters for procedural hinge object generation."""
    num_segments: int = 3           # number of rigid segments
    segment_length: float = 1.0
    joint_angles: Optional[list[float]] = None  # radians, one per hinge
    seed: Optional[int] = None


def generate_hinge_graph(params: HingeParams = HingeParams()) -> GraphLabel:
    """
    Generate a chain of rigid segments connected by hinges.

    Like a robot arm or multi-jointed mechanism.
    Each hinge is an ARTICULATION node; segment endpoints are ENDPOINT nodes.
    """
    if params.seed is not None:
        random.seed(params.seed)
        np.random.seed(params.seed)

    nodes: list[HyperNodeLabel] = []
    edges: list[HyperEdgeLabel] = []
    node_id = 0
    edge_id = 0

    def add_node(xyz, ntype, parent_id=None, semantic=None):
        nonlocal node_id
        n = HyperNodeLabel(
            id=node_id,
            node_type=ntype,
            xyz=xyz.tolist(),
            confidence=1.0,
            label_sources={LabelSource.PROCEDURAL.value: 1.0},
            accepted=True,
            parent_id=parent_id,
            semantic=semantic,
        )
        nodes.append(n)
        node_id += 1
        return n.id

    def add_edge(src, tgt, etype, length=None):
        nonlocal edge_id
        if length is None:
            s_xyz = np.array(nodes[src].xyz)
            t_xyz = np.array(nodes[tgt].xyz)
            length = float(np.linalg.norm(t_xyz - s_xyz))
        edges.append(HyperEdgeLabel(
            id=edge_id,
            source_node_id=src,
            target_node_id=tgt,
            edge_type=etype,
            confidence=1.0,
            label_sources={LabelSource.PROCEDURAL.value: 1.0},
            length=length,
        ))
        edge_id += 1

    # Generate joint angles if not provided
    if params.joint_angles is None:
        angles = [random.uniform(-math.pi / 3, math.pi / 3)
                  for _ in range(params.num_segments - 1)]
    else:
        angles = params.joint_angles

    # Build chain
    current_pos = np.array([0.0, 0.0, 0.0])
    current_angle = 0.0  # cumulative angle in XY plane

    # Base/root endpoint
    base_id = add_node(current_pos, NodeType.ROOT, semantic="base_mount")

    prev_id = base_id
    for seg_i in range(params.num_segments):
        # Segment direction
        direction = np.array([math.cos(current_angle), math.sin(current_angle), 0.0])
        end_pos = current_pos + direction * params.segment_length

        if seg_i < params.num_segments - 1:
            # Hinge at end of segment
            hinge_id = add_node(end_pos, NodeType.ARTICULATION, parent_id=prev_id,
                                semantic=f"hinge_{seg_i}")
            add_edge(prev_id, hinge_id, EdgeType.HINGE_LINK)

            # Apply angle for next segment
            if seg_i < len(angles):
                current_angle += angles[seg_i]

            current_pos = end_pos
            prev_id = hinge_id
        else:
            # Terminal endpoint
            tip_id = add_node(end_pos, NodeType.ENDPOINT, parent_id=prev_id,
                              semantic="end_effector")
            add_edge(prev_id, tip_id, EdgeType.HINGE_LINK)

    return GraphLabel(
        sample_id=f"proc_hinge_s{params.seed or 0}_seg{params.num_segments}",
        nodes=nodes,
        edges=edges,
        metadata={
            "source": "procedural_hinge",
            "num_segments": params.num_segments,
            "joint_angles_deg": [math.degrees(a) for a in angles],
            "seed": params.seed,
        },
    )


@dataclass
class RopeParams:
    """Parameters for procedural rope/chain generation."""
    num_nodes: int = 10
    segment_length: float = 0.3
    gravity_sag: float = 0.5        # how much the rope sags
    seed: Optional[int] = None


def generate_rope_graph(params: RopeParams = RopeParams()) -> GraphLabel:
    """
    Generate a rope/chain hanging under gravity (catenary-like).

    All internal nodes are BEND type (deformation, not rigid articulation).
    """
    if params.seed is not None:
        random.seed(params.seed)
        np.random.seed(params.seed)

    nodes: list[HyperNodeLabel] = []
    edges: list[HyperEdgeLabel] = []
    node_id = 0
    edge_id = 0

    def add_node(xyz, ntype, parent_id=None, semantic=None):
        nonlocal node_id
        n = HyperNodeLabel(
            id=node_id,
            node_type=ntype,
            xyz=xyz.tolist(),
            confidence=1.0,
            label_sources={LabelSource.PROCEDURAL.value: 1.0},
            accepted=True,
            parent_id=parent_id,
            semantic=semantic,
        )
        nodes.append(n)
        node_id += 1
        return n.id

    # Generate catenary-like shape
    total_span = (params.num_nodes - 1) * params.segment_length
    positions = []
    for i in range(params.num_nodes):
        t = i / (params.num_nodes - 1)  # 0..1
        x = t * total_span
        # Catenary-like sag
        y = -params.gravity_sag * 4 * t * (1 - t)
        # Add noise
        y += random.gauss(0, 0.02)
        x += random.gauss(0, 0.01)
        positions.append(np.array([x, y, 0.0]))

    # Create nodes
    for i, pos in enumerate(positions):
        if i == 0:
            ntype = NodeType.ENDPOINT
            sem = "rope_anchor_left"
        elif i == params.num_nodes - 1:
            ntype = NodeType.ENDPOINT
            sem = "rope_anchor_right"
        else:
            ntype = NodeType.BEND
            sem = f"rope_node_{i}"

        parent = i - 1 if i > 0 else None
        nid = add_node(pos, ntype, parent_id=parent, semantic=sem)

    # Create edges
    for i in range(params.num_nodes - 1):
        length = float(np.linalg.norm(positions[i + 1] - positions[i]))
        edges.append(HyperEdgeLabel(
            id=edge_id,
            source_node_id=i,
            target_node_id=i + 1,
            edge_type=EdgeType.DEFORMATION_LINK,
            confidence=1.0,
            label_sources={LabelSource.PROCEDURAL.value: 1.0},
            length=length,
        ))
        edge_id += 1

    return GraphLabel(
        sample_id=f"proc_rope_s{params.seed or 0}",
        nodes=nodes,
        edges=edges,
        metadata={
            "source": "procedural_rope",
            "num_nodes": params.num_nodes,
            "seed": params.seed,
        },
    )
