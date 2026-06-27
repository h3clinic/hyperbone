"""
Topology utilities for variable-skeleton Anymate assets.

Provides per-joint structural features independent of raw index:
- depth_from_root
- child_count
- is_root, is_leaf, is_branch
- normalized position within hierarchy
- topology signature for clustering
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch


def build_hierarchy(conns: np.ndarray, n_joints: int) -> Dict:
    """Build hierarchy from parent-index array.

    Args:
        conns: [N] int array where conns[j] = parent of joint j
        n_joints: actual number of joints (conns may be padded)

    Returns dict with children, parents, roots, depths, etc.
    """
    children = defaultdict(list)
    parents = {}

    for j in range(n_joints):
        p = int(conns[j])
        parents[j] = p
        if p != j and 0 <= p < n_joints:
            children[p].append(j)

    # Find roots (self-parent or invalid parent)
    roots = [j for j in range(n_joints)
             if int(conns[j]) == j or int(conns[j]) >= n_joints or int(conns[j]) < 0]
    if not roots:
        roots = [0]  # fallback

    # BFS to compute depths
    depths = np.full(n_joints, -1, dtype=int)
    queue = list(roots)
    for r in roots:
        depths[r] = 0

    while queue:
        node = queue.pop(0)
        for child in children[node]:
            if depths[child] == -1:
                depths[child] = depths[node] + 1
                queue.append(child)

    # Fix any unreachable nodes
    depths[depths == -1] = 0

    return {
        "children": dict(children),
        "parents": parents,
        "roots": roots,
        "depths": depths,
        "n_joints": n_joints,
    }


def compute_joint_features(joints: np.ndarray, conns: np.ndarray,
                           n_joints: int) -> np.ndarray:
    """Compute per-joint structural features for topology-aware training.

    Returns: [N, F] feature matrix where F includes:
        0: depth_from_root (normalized)
        1: child_count (normalized)
        2: is_root (0/1)
        3: is_leaf (0/1)
        4: is_branch (0/1)
        5: relative_x (position normalized to [-1,1])
        6: relative_y
        7: relative_z
        8: distance_from_centroid (normalized)
        9: bone_length_to_parent (normalized)
    """
    hier = build_hierarchy(conns, n_joints)
    children = hier["children"]
    depths = hier["depths"]
    roots = hier["roots"]

    max_depth = max(depths.max(), 1)
    features = np.zeros((n_joints, 10), dtype=np.float32)

    # Centroid
    centroid = joints[:n_joints].mean(axis=0)
    extent = joints[:n_joints].max(axis=0) - joints[:n_joints].min(axis=0)
    scale = max(extent.max(), 1e-6)

    for j in range(n_joints):
        # Structural features
        features[j, 0] = depths[j] / max_depth
        features[j, 1] = len(children.get(j, [])) / max(4, 1)  # normalize by typical max
        features[j, 2] = 1.0 if j in roots else 0.0
        features[j, 3] = 1.0 if j not in children else 0.0
        features[j, 4] = 1.0 if len(children.get(j, [])) > 1 else 0.0

        # Position features (normalized)
        rel_pos = (joints[j] - centroid) / scale
        features[j, 5:8] = rel_pos

        # Distance from centroid
        features[j, 8] = np.linalg.norm(joints[j] - centroid) / scale

        # Bone length to parent
        p = int(conns[j])
        if p != j and 0 <= p < n_joints:
            features[j, 9] = np.linalg.norm(joints[j] - joints[p]) / scale
        else:
            features[j, 9] = 0.0

    return features


def compute_adjacency(conns: np.ndarray, n_joints: int) -> np.ndarray:
    """Compute adjacency matrix from parent indices.

    Returns: [N, N] binary adjacency matrix (symmetric).
    """
    adj = np.zeros((n_joints, n_joints), dtype=np.float32)
    for j in range(n_joints):
        p = int(conns[j])
        if p != j and 0 <= p < n_joints:
            adj[j, p] = 1.0
            adj[p, j] = 1.0
    return adj


def infer_joint_type(joint_idx: int, conns: np.ndarray, n_joints: int,
                     depths: np.ndarray, children: dict) -> str:
    """Infer structural type of a joint based on hierarchy position."""
    is_root = (int(conns[joint_idx]) == joint_idx or
               int(conns[joint_idx]) >= n_joints)
    n_children = len(children.get(joint_idx, []))
    depth = int(depths[joint_idx])

    if is_root:
        return "root"
    elif n_children == 0:
        return "endpoint"
    elif n_children > 1:
        return "branch"
    elif depth <= 2:
        return "spine"
    else:
        return "limb"


JOINT_TYPE_NAMES = ["root", "spine", "branch", "limb", "endpoint", "unknown"]
JOINT_TYPE_TO_IDX = {name: i for i, name in enumerate(JOINT_TYPE_NAMES)}
N_JOINT_TYPES = len(JOINT_TYPE_NAMES)


def compute_joint_types(conns: np.ndarray, n_joints: int) -> np.ndarray:
    """Compute type index for each joint. Returns [N] int array."""
    hier = build_hierarchy(conns, n_joints)
    types = np.zeros(n_joints, dtype=np.int64)
    for j in range(n_joints):
        type_name = infer_joint_type(j, conns, n_joints, hier["depths"], hier["children"])
        types[j] = JOINT_TYPE_TO_IDX.get(type_name, JOINT_TYPE_TO_IDX["unknown"])
    return types


def compute_rest_bone_lengths(joints: np.ndarray, conns: np.ndarray,
                              n_joints: int) -> np.ndarray:
    """Compute rest-pose bone lengths for each joint-parent pair.

    Returns: [N] array where entry j = distance(joint_j, parent_j).
    """
    lengths = np.zeros(n_joints, dtype=np.float32)
    for j in range(n_joints):
        p = int(conns[j])
        if p != j and 0 <= p < n_joints:
            lengths[j] = np.linalg.norm(joints[j] - joints[p])
    return lengths
