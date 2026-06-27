"""
Procedural rock/ridge generator.

Generates rock-like objects where the structural skeleton is the medial axis
of ridges and protrusions. Not semantic joints — ridge nodes.
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
class RockParams:
    """Parameters for procedural rock generation."""
    num_ridges: int = 4
    ridge_nodes_per_ridge: int = 3
    center_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 1.0
    seed: Optional[int] = None


def generate_rock_graph(params: RockParams = RockParams()) -> GraphLabel:
    """
    Generate a rock-like structure with ridges radiating from center.

    Nodes are ridge points (not biological joints).
    Edges are ridge connections.
    """
    if params.seed is not None:
        random.seed(params.seed)
        np.random.seed(params.seed)

    nodes: list[HyperNodeLabel] = []
    edges: list[HyperEdgeLabel] = []
    node_id = 0
    edge_id = 0

    def add_node(xyz, ntype, parent_id=None, semantic=None, radius=None):
        nonlocal node_id
        n = HyperNodeLabel(
            id=node_id,
            node_type=ntype,
            xyz=xyz.tolist(),
            radius=radius,
            confidence=0.75,  # procedural ridges are approximate
            label_sources={LabelSource.PROCEDURAL.value: 0.75},
            accepted=True,
            parent_id=parent_id,
            semantic=semantic,
        )
        nodes.append(n)
        node_id += 1
        return n.id

    # Center node
    center_id = add_node(params.center_position, NodeType.CENTER, semantic="rock_center",
                         radius=params.radius * 0.3)

    # Generate ridges radiating outward
    for ri in range(params.num_ridges):
        # Random direction on sphere
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0.2, math.pi - 0.2)  # avoid poles

        direction = np.array([
            math.sin(phi) * math.cos(theta),
            math.sin(phi) * math.sin(theta),
            math.cos(phi),
        ])

        prev_id = center_id
        prev_pos = params.center_position.copy()

        for ni in range(params.ridge_nodes_per_ridge):
            t = (ni + 1) / params.ridge_nodes_per_ridge
            # Position along ridge with some noise
            pos = params.center_position + direction * params.radius * t
            pos += np.random.randn(3) * 0.05  # noise

            if ni == params.ridge_nodes_per_ridge - 1:
                ntype = NodeType.ENDPOINT
                sem = f"ridge_{ri}_tip"
            else:
                ntype = NodeType.RIDGE
                sem = f"ridge_{ri}_node_{ni}"

            nid = add_node(pos, ntype, parent_id=prev_id, semantic=sem,
                           radius=params.radius * 0.1 * (1 - t * 0.5))
            edges.append(HyperEdgeLabel(
                id=edge_id,
                source_node_id=prev_id,
                target_node_id=nid,
                edge_type=EdgeType.RIDGE,
                confidence=0.75,
                label_sources={LabelSource.PROCEDURAL.value: 0.75},
                length=float(np.linalg.norm(pos - prev_pos)),
            ))
            edge_id += 1
            prev_id = nid
            prev_pos = pos

    return GraphLabel(
        sample_id=f"proc_rock_s{params.seed or 0}",
        nodes=nodes,
        edges=edges,
        metadata={
            "source": "procedural_rock",
            "num_ridges": params.num_ridges,
            "seed": params.seed,
        },
    )
