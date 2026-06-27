"""
Canonical animal skeleton normalization.

Normalizes all animal rigs into a consistent root-relative coordinate system
for comparison and evaluation across different assets and scales.
"""
from __future__ import annotations

import math
import numpy as np
from typing import List, Optional, Tuple, Dict
from .schema import PoseFrame3D, Joint3D


def find_root_joint(frame: PoseFrame3D) -> Optional[Joint3D]:
    """Find the root joint (pelvis/root/hips, or first parentless joint)."""
    # Priority: look for known root-type joints
    for j in frame.joints:
        if j.type == "root":
            return j

    # Look by name
    for j in frame.joints:
        name = j.name.lower()
        if any(k in name for k in ("root", "pelvis", "hips", "hip")):
            return j

    # Fallback: first joint with no parent
    for j in frame.joints:
        if j.parent_id is None:
            return j

    return frame.joints[0] if frame.joints else None


def compute_torso_length(frame: PoseFrame3D) -> float:
    """Compute torso length from root to neck/head for scale normalization."""
    root = find_root_joint(frame)
    if not root:
        return 1.0

    # Find neck or head joint
    neck = None
    for j in frame.joints:
        if j.type in ("neck", "head"):
            neck = j
            break

    if neck is None:
        # Fallback: use distance from root to farthest joint
        root_pos = np.array(root.world_xyz)
        max_dist = 0.0
        for j in frame.joints:
            d = np.linalg.norm(np.array(j.world_xyz) - root_pos)
            max_dist = max(max_dist, d)
        return max_dist if max_dist > 0 else 1.0

    root_pos = np.array(root.world_xyz)
    neck_pos = np.array(neck.world_xyz)
    length = float(np.linalg.norm(neck_pos - root_pos))
    return length if length > 0 else 1.0


def estimate_forward_axis(frame: PoseFrame3D) -> np.ndarray:
    """Estimate the animal's forward direction from root→neck/head."""
    root = find_root_joint(frame)
    if not root:
        return np.array([1.0, 0.0, 0.0])

    # Find head/neck
    target = None
    for j in frame.joints:
        if j.type in ("head", "neck"):
            target = j
            break

    if target is None:
        return np.array([1.0, 0.0, 0.0])

    root_pos = np.array(root.world_xyz)
    target_pos = np.array(target.world_xyz)
    forward = target_pos - root_pos
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0])
    return forward / norm


def canonicalize_frame(
    frame: PoseFrame3D,
    scale_factor: Optional[float] = None,
    up_axis: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> Dict:
    """
    Normalize a pose frame to canonical animal coordinates.

    Returns dict with:
      - canonical_joints: list of {id, name, canonical_xyz}
      - canonical_root_joint: root joint name
      - scale_factor: applied scale
      - forward_axis: estimated forward direction
      - up_axis: up direction used
      - normalization_version: "v1"
    """
    root = find_root_joint(frame)
    if not root:
        return {
            "canonical_joints": [],
            "canonical_root_joint": None,
            "scale_factor": 1.0,
            "forward_axis": [1, 0, 0],
            "up_axis": list(up_axis),
            "normalization_version": "v1",
        }

    root_pos = np.array(root.world_xyz)

    # Scale factor
    if scale_factor is None:
        scale_factor = compute_torso_length(frame)

    # Forward axis for optional rotation alignment
    forward = estimate_forward_axis(frame)

    # Normalize: translate to root, divide by scale
    canonical_joints = []
    for j in frame.joints:
        pos = np.array(j.world_xyz)
        canonical_xyz = (pos - root_pos) / scale_factor
        canonical_joints.append({
            "id": j.id,
            "name": j.name,
            "type": j.type,
            "canonical_xyz": canonical_xyz.tolist(),
            "visible": j.visible,
            "confidence": j.confidence,
        })

    return {
        "canonical_joints": canonical_joints,
        "canonical_root_joint": root.name,
        "scale_factor": float(scale_factor),
        "forward_axis": forward.tolist(),
        "up_axis": up_axis.tolist(),
        "normalization_version": "v1",
    }


def compute_bone_lengths(frame: PoseFrame3D) -> Dict[str, float]:
    """Compute length of each bone in world coordinates."""
    joint_map = {j.id: j for j in frame.joints}
    lengths = {}
    for j in frame.joints:
        if j.parent_id is not None and j.parent_id in joint_map:
            parent = joint_map[j.parent_id]
            pos = np.array(j.world_xyz)
            parent_pos = np.array(parent.world_xyz)
            length = float(np.linalg.norm(pos - parent_pos))
            bone_name = f"{parent.name}_to_{j.name}"
            lengths[bone_name] = length
    return lengths


def project_joint_to_image(
    world_xyz: Tuple[float, float, float],
    camera_K: List[List[float]],
    camera_extrinsic: List[List[float]],
) -> Tuple[float, float]:
    """Project a 3D world point to 2D image coordinates using camera matrices."""
    K = np.array(camera_K)       # 3x3
    E = np.array(camera_extrinsic)  # 4x4

    # Transform to camera coordinates
    p_world = np.array([*world_xyz, 1.0])
    p_cam = E @ p_world  # 4x1

    # Project
    if abs(p_cam[2]) < 1e-8:
        return (0.0, 0.0)

    p_img = K @ p_cam[:3]
    u = p_img[0] / p_img[2]
    v = p_img[1] / p_img[2]
    return (float(u), float(v))
