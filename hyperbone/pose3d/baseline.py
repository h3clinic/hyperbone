"""
HyperBone3D baseline predictor (non-learning placeholder).

Input: bbox, rest_skeleton, camera metadata
Output: root-centered rest pose placed inside bbox

This is NOT a trained model. It is a dummy baseline for evaluation
infrastructure testing. It places the rest-pose skeleton at the bbox center.
"""
from __future__ import annotations

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from .schema import PoseFrame3D, Joint3D, Bone3D


def load_rest_skeleton(path: str) -> Dict:
    """Load rest skeleton JSON."""
    with open(path) as f:
        return json.load(f)


def rest_pose_baseline(
    rest_skeleton: Dict,
    bbox_xywh: Tuple[int, int, int, int],
    resolution: Tuple[int, int] = (640, 480),
    camera_K: Optional[List[List[float]]] = None,
    frame_idx: int = 0,
    asset_id: str = "unknown",
    animation_name: str = "rest",
) -> PoseFrame3D:
    """
    Non-learning baseline: place rest skeleton centered at bbox.

    Strategy:
    - Map rest bone positions to canonical [-0.5, 0.5] space
    - Scale and translate to fit within bbox
    - Project to image coordinates
    - z=0 for all joints (flat canonical)
    """
    bones_data = rest_skeleton.get("bones", [])
    if not bones_data:
        return PoseFrame3D(
            asset_id=asset_id,
            frame_idx=frame_idx,
            timestamp_sec=0.0,
            animation_name=animation_name,
        )

    # Build joint positions from rest bone heads
    # Build name -> idx mapping
    name_to_idx = {}
    for idx, bone in enumerate(bones_data):
        name_to_idx[bone["name"]] = idx

    # Get 3D positions (use head_local if available, otherwise zeros)
    positions_3d = []
    for bone in bones_data:
        if "head_local" in bone:
            positions_3d.append(bone["head_local"])
        else:
            positions_3d.append([0.0, 0.0, 0.0])
    positions_3d = np.array(positions_3d)

    # Normalize to canonical space
    if positions_3d.shape[0] > 0:
        center = positions_3d.mean(axis=0)
        extent = positions_3d.max(axis=0) - positions_3d.min(axis=0)
        max_extent = max(extent.max(), 1e-6)
        canonical = (positions_3d - center) / max_extent  # [-0.5, 0.5]
    else:
        canonical = positions_3d

    # Map canonical to bbox image coordinates
    bx, by, bw, bh = bbox_xywh
    joints = []
    for idx, bone in enumerate(bones_data):
        cx, cy, cz = canonical[idx]
        # Map to image: x in [bx, bx+bw], y in [by, by+bh]
        img_x = bx + (cx + 0.5) * bw
        img_y = by + (0.5 - cy) * bh  # Flip Y

        parent_idx = name_to_idx.get(bone.get("parent")) if bone.get("parent") else None

        joints.append(Joint3D(
            id=idx,
            name=bone["name"],
            parent_id=parent_idx,
            type=bone.get("type", "unknown"),
            world_xyz=(float(canonical[idx][0]), float(canonical[idx][1]), float(canonical[idx][2])),
            camera_xyz=(float(canonical[idx][0]), float(canonical[idx][1]), 0.0),
            local_xyz=tuple(positions_3d[idx].tolist()),
            image_xy=(float(img_x), float(img_y)),
            visible=True,
            confidence=0.5,  # Low confidence - this is a baseline
        ))

    # Build bones
    bone_list = []
    for idx, bone_data in enumerate(bones_data):
        parent_name = bone_data.get("parent")
        if parent_name and parent_name in name_to_idx:
            parent_idx = name_to_idx[parent_name]
            length = float(np.linalg.norm(canonical[idx] - canonical[parent_idx]))
            bone_list.append(Bone3D(
                parent_id=parent_idx,
                child_id=idx,
                name=f"{parent_name}_to_{bone_data['name']}",
                length=length,
                rest_length=bone_data.get("length", 0.0),
            ))

    return PoseFrame3D(
        asset_id=asset_id,
        frame_idx=frame_idx,
        timestamp_sec=frame_idx / 24.0,
        animation_name=animation_name,
        joints=joints,
        bones=bone_list,
        camera_K=camera_K,
        resolution=resolution,
        bbox_xywh=bbox_xywh,
        coord_systems={
            "world": "canonical_rest",
            "camera": "canonical_rest",
            "image": "pixel_xy",
            "canonical": "rest_pose_centered",
        },
    )
