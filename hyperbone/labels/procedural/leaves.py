"""
Procedural leaf generator with vein graph labels.

Generates leaf shapes with midrib, secondary veins, and tertiary branches.
The vein graph is the ground-truth structural skeleton of the leaf.
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
class LeafParams:
    """Parameters for procedural leaf generation."""
    midrib_length: float = 3.0
    num_secondary_veins: int = 6       # pairs
    secondary_angle: float = 45.0      # degrees from midrib
    secondary_length_ratio: float = 0.6
    tertiary_prob: float = 0.4
    tertiary_length_ratio: float = 0.3
    midrib_segments: int = 8
    seed: Optional[int] = None


def generate_leaf_graph(params: LeafParams = LeafParams()) -> GraphLabel:
    """
    Generate a leaf vein graph.

    Structure:
    - Midrib: root → tip (series of bend nodes)
    - Secondary veins: branch off midrib at regular intervals
    - Tertiary veins: optional branches off secondary veins
    """
    if params.seed is not None:
        random.seed(params.seed)
        np.random.seed(params.seed)

    nodes: list[HyperNodeLabel] = []
    edges: list[HyperEdgeLabel] = []
    node_id = 0
    edge_id = 0

    def add_node(xy, ntype, parent_id=None, semantic=None):
        nonlocal node_id
        n = HyperNodeLabel(
            id=node_id,
            node_type=ntype,
            xyz=[xy[0], xy[1], 0.0],  # 2D leaf, z=0
            xy=xy.tolist(),
            confidence=1.0,
            label_sources={LabelSource.PROCEDURAL.value: 1.0},
            accepted=True,
            parent_id=parent_id,
            semantic=semantic,
        )
        nodes.append(n)
        node_id += 1
        return n.id

    def add_edge(src, tgt, etype=EdgeType.VEIN):
        nonlocal edge_id
        s_xyz = np.array(nodes[src].xyz[:2])
        t_xyz = np.array(nodes[tgt].xyz[:2])
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

    # Midrib: from petiole (base) to leaf tip
    seg_len = params.midrib_length / params.midrib_segments
    midrib_nodes = []

    # Root (petiole attachment)
    base_pos = np.array([0.0, 0.0])
    root_id = add_node(base_pos, NodeType.ROOT, semantic="petiole")
    midrib_nodes.append(root_id)

    # Midrib segments
    for i in range(1, params.midrib_segments):
        # Slight random curve
        x_offset = random.gauss(0, 0.02)
        pos = np.array([x_offset, i * seg_len])
        ntype = NodeType.BEND
        sem = f"midrib_{i}"
        nid = add_node(pos, ntype, parent_id=midrib_nodes[-1], semantic=sem)
        add_edge(midrib_nodes[-1], nid)
        midrib_nodes.append(nid)

    # Tip
    tip_pos = np.array([0.0, params.midrib_length])
    tip_id = add_node(tip_pos, NodeType.ENDPOINT, parent_id=midrib_nodes[-1], semantic="leaf_tip")
    add_edge(midrib_nodes[-1], tip_id)
    midrib_nodes.append(tip_id)

    # Secondary veins branching off midrib
    vein_spacing = params.midrib_segments // (params.num_secondary_veins + 1)
    for vi in range(params.num_secondary_veins):
        midrib_idx = (vi + 1) * vein_spacing
        if midrib_idx >= len(midrib_nodes) - 1:
            break

        parent_nid = midrib_nodes[midrib_idx]
        parent_pos = np.array(nodes[parent_nid].xyz[:2])

        # Mark parent as branch node
        nodes[parent_nid].node_type = NodeType.BRANCH
        nodes[parent_nid].semantic = f"vein_fork_{vi}"

        # Generate left and right secondary veins
        for side in [-1, 1]:
            angle = math.radians(params.secondary_angle * side)
            # Secondary vein curves outward and upward
            sec_length = params.secondary_length_ratio * params.midrib_length * random.uniform(0.7, 1.0)
            direction = np.array([math.sin(angle), math.cos(angle)])
            end_pos = parent_pos + direction * sec_length

            side_name = "left" if side == -1 else "right"

            # Check if this vein should branch (tertiary)
            if random.random() < params.tertiary_prob:
                # Midpoint becomes a branch
                mid_pos = parent_pos + direction * sec_length * 0.5
                mid_id = add_node(mid_pos, NodeType.BRANCH, parent_id=parent_nid,
                                  semantic=f"sec_vein_{vi}_{side_name}_fork")
                add_edge(parent_nid, mid_id)

                # Continue to endpoint
                end_id = add_node(end_pos, NodeType.ENDPOINT, parent_id=mid_id,
                                  semantic=f"sec_vein_{vi}_{side_name}_tip")
                add_edge(mid_id, end_id)

                # Tertiary branch
                tert_angle = angle + math.radians(30 * side)
                tert_dir = np.array([math.sin(tert_angle), math.cos(tert_angle)])
                tert_len = params.tertiary_length_ratio * sec_length
                tert_pos = mid_pos + tert_dir * tert_len
                tert_id = add_node(tert_pos, NodeType.ENDPOINT, parent_id=mid_id,
                                   semantic=f"tert_vein_{vi}_{side_name}")
                add_edge(mid_id, tert_id)
            else:
                # Simple secondary vein to endpoint
                end_id = add_node(end_pos, NodeType.ENDPOINT, parent_id=parent_nid,
                                  semantic=f"sec_vein_{vi}_{side_name}_tip")
                add_edge(parent_nid, end_id)

    return GraphLabel(
        sample_id=f"proc_leaf_s{params.seed or 0}",
        nodes=nodes,
        edges=edges,
        metadata={
            "source": "procedural_leaf",
            "num_secondary_veins": params.num_secondary_veins,
            "seed": params.seed,
        },
    )
