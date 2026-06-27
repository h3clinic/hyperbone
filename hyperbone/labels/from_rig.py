"""
Extract graph labels from rigged 3D assets (GLB/FBX).

Given a rigged mesh with armature, extract:
- joint positions (rest pose + animated)
- bone edges (parent-child hierarchy)
- joint names mapped to HyperNode semantic labels
- projected 2D joints for rendered frames
- skinning weight associations

Supports GLB via trimesh/pygltflib.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    from pygltflib import GLTF2
except ImportError:
    GLTF2 = None

from hyperbone.labels.schema import (
    GraphLabel,
    HyperNodeLabel,
    HyperEdgeLabel,
    NodeType,
    EdgeType,
    LabelSource,
)


@dataclass
class JointInfo:
    """Raw joint extracted from armature."""
    name: str
    index: int
    parent_index: Optional[int]
    local_translation: np.ndarray   # [3]
    local_rotation: np.ndarray      # quaternion [x,y,z,w]
    world_position: np.ndarray      # [3] rest pose world coords
    children_indices: list[int] = field(default_factory=list)


@dataclass
class RigExtractionConfig:
    """Configuration for rig label extraction."""
    # Mapping from armature joint names to semantic labels
    joint_name_map: dict[str, str] = field(default_factory=dict)
    # Node type assignments (semantic_name -> NodeType)
    node_type_map: dict[str, NodeType] = field(default_factory=dict)
    # Camera intrinsics for 2D projection
    focal_length: float = 500.0
    image_width: int = 512
    image_height: int = 512
    # Camera extrinsics (world-to-camera)
    camera_position: np.ndarray = field(default_factory=lambda: np.array([0, 1.0, 3.0]))
    camera_target: np.ndarray = field(default_factory=lambda: np.array([0, 0.5, 0]))
    camera_up: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))


def extract_joints_from_gltf(gltf_path: Path) -> list[JointInfo]:
    """
    Extract joint hierarchy from a GLB/GLTF file.

    Reads the node tree, identifies skin joints, and computes world positions.
    """
    if GLTF2 is None:
        raise ImportError("pygltflib required: pip install pygltflib")

    gltf = GLTF2().load(str(gltf_path))
    joints: list[JointInfo] = []

    # Find skin and its joints
    if not gltf.skins:
        raise ValueError(f"No skins (armatures) found in {gltf_path}")

    skin = gltf.skins[0]
    joint_node_indices = skin.joints

    # Build parent map from node children
    parent_map: dict[int, int] = {}
    for ni, node in enumerate(gltf.nodes):
        if node.children:
            for child_idx in node.children:
                parent_map[child_idx] = ni

    # Extract each joint
    for local_idx, node_idx in enumerate(joint_node_indices):
        node = gltf.nodes[node_idx]
        translation = np.array(node.translation or [0, 0, 0], dtype=np.float64)
        rotation = np.array(node.rotation or [0, 0, 0, 1], dtype=np.float64)

        # Find parent in joint list
        parent_node_idx = parent_map.get(node_idx)
        parent_local_idx = None
        if parent_node_idx is not None and parent_node_idx in joint_node_indices:
            parent_local_idx = joint_node_indices.index(parent_node_idx)

        children = []
        if node.children:
            for c in node.children:
                if c in joint_node_indices:
                    children.append(joint_node_indices.index(c))

        joints.append(JointInfo(
            name=node.name or f"joint_{local_idx}",
            index=local_idx,
            parent_index=parent_local_idx,
            local_translation=translation,
            local_rotation=rotation,
            world_position=np.zeros(3),  # computed below
            children_indices=children,
        ))

    # Compute world positions via forward kinematics
    _compute_world_positions(joints)

    return joints


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x,y,z,w] to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def _compute_world_positions(joints: list[JointInfo]) -> None:
    """Forward kinematics: compute world position for each joint."""
    world_transforms: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # idx -> (rot3x3, pos)

    def get_world(idx: int) -> tuple[np.ndarray, np.ndarray]:
        if idx in world_transforms:
            return world_transforms[idx]

        j = joints[idx]
        local_rot = _quat_to_matrix(j.local_rotation)
        local_pos = j.local_translation

        if j.parent_index is None:
            world_rot = local_rot
            world_pos = local_pos.copy()
        else:
            parent_rot, parent_pos = get_world(j.parent_index)
            world_rot = parent_rot @ local_rot
            world_pos = parent_pos + parent_rot @ local_pos

        world_transforms[idx] = (world_rot, world_pos)
        return world_rot, world_pos

    for i in range(len(joints)):
        _, pos = get_world(i)
        joints[i].world_position = pos


def project_joints_to_2d(
    joints: list[JointInfo],
    config: RigExtractionConfig,
) -> np.ndarray:
    """
    Project 3D joint positions to 2D image coordinates.

    Returns: [N, 2] array of (u, v) pixel coordinates.
    """
    # Build view matrix (look-at)
    forward = config.camera_target - config.camera_position
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, config.camera_up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)

    view_rot = np.stack([right, up, -forward], axis=0)  # [3, 3]

    positions = np.array([j.world_position for j in joints])  # [N, 3]
    cam_coords = (positions - config.camera_position) @ view_rot.T  # [N, 3]

    # Perspective projection
    fx = fy = config.focal_length
    cx = config.image_width / 2
    cy = config.image_height / 2

    z = cam_coords[:, 2]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)

    u = fx * cam_coords[:, 0] / (-z) + cx
    v = fy * cam_coords[:, 1] / (-z) + cy

    return np.stack([u, v], axis=1)


def joints_to_graph_label(
    joints: list[JointInfo],
    config: RigExtractionConfig,
    sample_id: str,
    image_path: Optional[str] = None,
) -> GraphLabel:
    """
    Convert extracted joints to a GraphLabel with nodes and edges.
    """
    # Project to 2D
    xy_coords = project_joints_to_2d(joints, config)

    nodes = []
    edges = []

    for j in joints:
        # Determine semantic name
        semantic = config.joint_name_map.get(j.name, j.name)

        # Determine node type
        node_type = config.node_type_map.get(semantic, NodeType.SEMANTIC_JOINT)

        # Root detection
        if j.parent_index is None:
            node_type = NodeType.ROOT

        # Endpoint detection (no children)
        if not j.children_indices and node_type == NodeType.SEMANTIC_JOINT:
            node_type = NodeType.ENDPOINT

        xy = xy_coords[j.index].tolist()

        nodes.append(HyperNodeLabel(
            id=j.index,
            node_type=node_type,
            xyz=j.world_position.tolist(),
            xy=xy,
            semantic=semantic,
            confidence=1.0,  # GT from rig
            label_sources={LabelSource.RIG_GT.value: 1.0},
            accepted=True,
            parent_id=j.parent_index,
        ))

    # Create edges from parent-child relationships
    edge_id = 0
    for j in joints:
        if j.parent_index is not None:
            edges.append(HyperEdgeLabel(
                id=edge_id,
                source_node_id=j.parent_index,
                target_node_id=j.index,
                edge_type=EdgeType.BONE,
                confidence=1.0,
                label_sources={LabelSource.RIG_GT.value: 1.0},
                length=float(np.linalg.norm(
                    joints[j.index].world_position - joints[j.parent_index].world_position
                )),
            ))
            edge_id += 1

    return GraphLabel(
        sample_id=sample_id,
        image_path=image_path,
        nodes=nodes,
        edges=edges,
        metadata={
            "source": "rig_extraction",
            "joint_count": len(joints),
        },
    )


def extract_rig_labels(
    gltf_path: Path,
    config: RigExtractionConfig,
    sample_id_prefix: str = "rig",
) -> GraphLabel:
    """
    Full pipeline: extract joints from GLB → build graph label.
    """
    joints = extract_joints_from_gltf(gltf_path)
    sample_id = f"{sample_id_prefix}_{Path(gltf_path).stem}_rest"
    return joints_to_graph_label(joints, config, sample_id=sample_id)
