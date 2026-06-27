"""
HyperBone Internal 3D Pose Schema.

Defines the canonical data structures for representing animal skeletal pose
in 3D. These are internal armature joints — NOT 2D contour skeletons.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Tuple
from pathlib import Path


class JointType(str, Enum):
    ROOT = "root"
    SPINE = "spine"
    NECK = "neck"
    HEAD = "head"
    SHOULDER = "shoulder"
    ELBOW = "elbow"
    WRIST = "wrist"
    HIP = "hip"
    KNEE = "knee"
    ANKLE = "ankle"
    PAW = "paw"
    TAIL = "tail"
    EAR = "ear"
    UNKNOWN = "unknown"


@dataclass
class Joint3D:
    """A single 3D joint in an animal armature."""
    id: int
    name: str
    parent_id: Optional[int]
    type: str = JointType.UNKNOWN.value
    world_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    image_xy: Tuple[float, float] = (0.0, 0.0)
    visible: bool = True
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "parent_id": self.parent_id,
            "type": self.type,
            "world_xyz": list(self.world_xyz),
            "camera_xyz": list(self.camera_xyz),
            "local_xyz": list(self.local_xyz),
            "image_xy": list(self.image_xy),
            "visible": self.visible,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Joint3D":
        return cls(
            id=d["id"],
            name=d["name"],
            parent_id=d.get("parent_id"),
            type=d.get("type", JointType.UNKNOWN.value),
            world_xyz=tuple(d.get("world_xyz", [0, 0, 0])),
            camera_xyz=tuple(d.get("camera_xyz", [0, 0, 0])),
            local_xyz=tuple(d.get("local_xyz", [0, 0, 0])),
            image_xy=tuple(d.get("image_xy", [0, 0])),
            visible=d.get("visible", True),
            confidence=d.get("confidence", 1.0),
        )


@dataclass
class Bone3D:
    """A bone connecting two joints."""
    parent_id: int
    child_id: int
    name: str = ""
    length: float = 0.0
    rest_length: float = 0.0
    local_rotation_quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    world_rotation_quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def to_dict(self) -> dict:
        return {
            "parent_id": self.parent_id,
            "child_id": self.child_id,
            "name": self.name,
            "length": self.length,
            "rest_length": self.rest_length,
            "local_rotation_quat": list(self.local_rotation_quat),
            "world_rotation_quat": list(self.world_rotation_quat),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Bone3D":
        return cls(
            parent_id=d["parent_id"],
            child_id=d["child_id"],
            name=d.get("name", ""),
            length=d.get("length", 0.0),
            rest_length=d.get("rest_length", 0.0),
            local_rotation_quat=tuple(d.get("local_rotation_quat", [0, 0, 0, 1])),
            world_rotation_quat=tuple(d.get("world_rotation_quat", [0, 0, 0, 1])),
        )


@dataclass
class PoseFrame3D:
    """One frame of 3D animal pose data."""
    asset_id: str
    frame_idx: int
    timestamp_sec: float
    animation_name: str
    joints: List[Joint3D] = field(default_factory=list)
    bones: List[Bone3D] = field(default_factory=list)
    camera_K: Optional[List[List[float]]] = None  # 3x3 intrinsics
    camera_extrinsic: Optional[List[List[float]]] = None  # 4x4 world-to-camera
    resolution: Tuple[int, int] = (640, 480)
    bbox_xywh: Optional[Tuple[int, int, int, int]] = None
    mask_path: Optional[str] = None
    rgb_path: Optional[str] = None
    depth_path: Optional[str] = None
    coord_systems: dict = field(default_factory=lambda: {
        "world": "blender_world",
        "camera": "blender_camera",
        "image": "pixel_xy",
        "canonical": "animal_root_normalized",
    })

    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "frame_idx": self.frame_idx,
            "timestamp_sec": self.timestamp_sec,
            "animation_name": self.animation_name,
            "joints": [j.to_dict() for j in self.joints],
            "bones": [b.to_dict() for b in self.bones],
            "camera_K": self.camera_K,
            "camera_extrinsic": self.camera_extrinsic,
            "resolution": list(self.resolution),
            "bbox_xywh": list(self.bbox_xywh) if self.bbox_xywh else None,
            "mask_path": self.mask_path,
            "rgb_path": self.rgb_path,
            "depth_path": self.depth_path,
            "coord_systems": self.coord_systems,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PoseFrame3D":
        return cls(
            asset_id=d["asset_id"],
            frame_idx=d["frame_idx"],
            timestamp_sec=d["timestamp_sec"],
            animation_name=d["animation_name"],
            joints=[Joint3D.from_dict(j) for j in d.get("joints", [])],
            bones=[Bone3D.from_dict(b) for b in d.get("bones", [])],
            camera_K=d.get("camera_K"),
            camera_extrinsic=d.get("camera_extrinsic"),
            resolution=tuple(d.get("resolution", [640, 480])),
            bbox_xywh=tuple(d["bbox_xywh"]) if d.get("bbox_xywh") else None,
            mask_path=d.get("mask_path"),
            rgb_path=d.get("rgb_path"),
            depth_path=d.get("depth_path"),
            coord_systems=d.get("coord_systems", {
                "world": "blender_world",
                "camera": "blender_camera",
                "image": "pixel_xy",
                "canonical": "animal_root_normalized",
            }),
        )

    def joint_by_name(self, name: str) -> Optional[Joint3D]:
        for j in self.joints:
            if j.name == name:
                return j
        return None

    def joint_by_id(self, joint_id: int) -> Optional[Joint3D]:
        for j in self.joints:
            if j.id == joint_id:
                return j
        return None

    def parent_child_pairs(self) -> List[Tuple[int, int]]:
        return [(j.parent_id, j.id) for j in self.joints if j.parent_id is not None]


def write_pose3d_jsonl(frames: List[PoseFrame3D], path: str):
    """Write pose frames to JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for frame in frames:
            f.write(json.dumps(frame.to_dict()) + "\n")


def read_pose3d_jsonl(path: str) -> List[PoseFrame3D]:
    """Read pose frames from JSONL file."""
    frames = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                frames.append(PoseFrame3D.from_dict(json.loads(line)))
    return frames
