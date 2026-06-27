"""
Procedural branch/tree generator with known graph labels.

Generates Y-branch structures, tree-like branching patterns, and random
articulated structures where the ground-truth graph is known by construction.
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
class BranchParams:
    """Parameters for procedural branch generation."""
    max_depth: int = 4
    branch_prob: float = 0.7        # probability of branching at each node
    min_length: float = 0.3
    max_length: float = 1.5
    angle_range: tuple[float, float] = (20.0, 60.0)  # degrees
    radius_decay: float = 0.7       # child radius = parent * decay
    initial_radius: float = 0.1
    seed: Optional[int] = None


def generate_branch_graph(params: BranchParams = BranchParams()) -> GraphLabel:
    """
    Generate a procedural branching structure (tree/Y-branch/coral-like).

    Returns a GraphLabel with exact node positions, types, and edges.
    """
    if params.seed is not None:
        random.seed(params.seed)
        np.random.seed(params.seed)

    nodes: list[HyperNodeLabel] = []
    edges: list[HyperEdgeLabel] = []
    node_id = 0
    edge_id = 0

    def add_node(xyz, ntype, radius, parent_id=None, semantic=None):
        nonlocal node_id
        n = HyperNodeLabel(
            id=node_id,
            node_type=ntype,
            xyz=xyz.tolist(),
            radius=radius,
            confidence=1.0,
            label_sources={LabelSource.PROCEDURAL.value: 1.0},
            accepted=True,
            parent_id=parent_id,
            semantic=semantic,
        )
        nodes.append(n)
        node_id += 1
        return n.id

    def add_edge(src, tgt, length):
        nonlocal edge_id
        edges.append(HyperEdgeLabel(
            id=edge_id,
            source_node_id=src,
            target_node_id=tgt,
            edge_type=EdgeType.BRANCH,
            confidence=1.0,
            label_sources={LabelSource.PROCEDURAL.value: 1.0},
            length=length,
        ))
        edge_id += 1

    # Start with root at origin
    root_pos = np.array([0.0, 0.0, 0.0])
    root_id = add_node(root_pos, NodeType.ROOT, params.initial_radius, semantic="trunk_base")

    def grow(parent_id, pos, direction, depth, radius):
        if depth >= params.max_depth:
            # Terminal endpoint
            length = random.uniform(params.min_length, params.max_length) * 0.5
            end_pos = pos + direction * length
            eid = add_node(end_pos, NodeType.ENDPOINT, radius * params.radius_decay, parent_id)
            add_edge(parent_id, eid, length)
            return

        # Extend main branch
        length = random.uniform(params.min_length, params.max_length)
        new_pos = pos + direction * length

        # Decide if branch point
        if random.random() < params.branch_prob:
            # Branch node
            bid = add_node(new_pos, NodeType.BRANCH, radius, parent_id, semantic=f"fork_d{depth}")
            add_edge(parent_id, bid, length)

            # Generate 2-3 child branches
            n_children = random.choice([2, 2, 3])
            for _ in range(n_children):
                angle = math.radians(random.uniform(*params.angle_range))
                # Random rotation around the branch direction
                perp = _random_perpendicular(direction)
                child_dir = _rotate_around_axis(direction, perp, angle)
                # Add some random twist
                twist = random.uniform(0, 2 * math.pi)
                child_dir = _rotate_around_axis(child_dir, direction, twist)
                child_dir = child_dir / (np.linalg.norm(child_dir) + 1e-8)

                grow(bid, new_pos, child_dir, depth + 1, radius * params.radius_decay)
        else:
            # Bend/continuation node
            bend_id = add_node(new_pos, NodeType.BEND, radius, parent_id)
            add_edge(parent_id, bend_id, length)

            # Continue with slight angle change
            angle = math.radians(random.uniform(5, 20))
            perp = _random_perpendicular(direction)
            new_dir = _rotate_around_axis(direction, perp, angle)
            new_dir = new_dir / (np.linalg.norm(new_dir) + 1e-8)

            grow(bend_id, new_pos, new_dir, depth + 1, radius * params.radius_decay)

    # Grow upward
    grow(root_id, root_pos, np.array([0.0, 1.0, 0.0]), 0, params.initial_radius)

    return GraphLabel(
        sample_id=f"proc_branch_s{params.seed or 0}",
        nodes=nodes,
        edges=edges,
        metadata={
            "source": "procedural_branch",
            "max_depth": params.max_depth,
            "seed": params.seed,
        },
    )


def _random_perpendicular(v: np.ndarray) -> np.ndarray:
    """Find a random vector perpendicular to v."""
    if abs(v[0]) < 0.9:
        perp = np.cross(v, np.array([1, 0, 0]))
    else:
        perp = np.cross(v, np.array([0, 1, 0]))
    return perp / (np.linalg.norm(perp) + 1e-8)


def _rotate_around_axis(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vector v around axis by angle (radians) using Rodrigues formula."""
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return v * cos_a + np.cross(axis, v) * sin_a + axis * np.dot(axis, v) * (1 - cos_a)
