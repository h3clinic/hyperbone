"""
Anymate-compatible rig schema for HyperBone.

Defines data structures for:
- AssetRig: complete rigged asset metadata
- Skeleton: joint hierarchy
- Joint: single joint with transforms
- Bone: parent-child bone connection
- SkinningWeights: per-vertex joint weights
- AnimationClip: action metadata
- FramePoseLabel: per-frame evaluated pose label (the gold label format)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Dict


class MotionSource(str, Enum):
    """How the animation was obtained."""
    ORIGINAL = "original"               # Asset shipped with this animation
    RETARGET = "retarget"               # Retargeted from a compatible skeleton
    SYNTHETIC = "synthetic_rig_motion"  # Procedurally generated


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
    FOOT = "foot"
    TAIL = "tail"
    WING = "wing"
    FIN = "fin"
    TENTACLE = "tentacle"
    HINGE = "hinge"
    TIP = "tip"
    UNKNOWN = "unknown"


@dataclass
class Joint:
    """A single joint in a skeleton hierarchy."""
    id: int
    name: str
    parent_id: Optional[int]
    type: str = JointType.UNKNOWN.value

    # Rest pose (bind pose)
    rest_world_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rest_local_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Per-frame evaluated pose (filled at render time)
    world_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    image_xy: Tuple[float, float] = (0.0, 0.0)
    visible: bool = True
    confidence: float = 1.0

    # Rotations (quaternion wxyz)
    world_rotation_quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    local_rotation_quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Joint":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Bone:
    """A bone connecting two joints."""
    parent_id: int
    child_id: int
    length: float = 0.0
    type: str = "bone"

    # Per-frame evaluated rotation
    world_rotation_quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    local_rotation_quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Bone":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Skeleton:
    """Complete skeleton hierarchy."""
    joints: List[Joint] = field(default_factory=list)
    bones: List[Bone] = field(default_factory=list)

    @property
    def joint_count(self) -> int:
        return len(self.joints)

    @property
    def bone_count(self) -> int:
        return len(self.bones)

    @property
    def root(self) -> Optional[Joint]:
        for j in self.joints:
            if j.parent_id is None:
                return j
        return None

    def get_joint(self, joint_id: int) -> Optional[Joint]:
        for j in self.joints:
            if j.id == joint_id:
                return j
        return None

    def children_of(self, joint_id: int) -> List[Joint]:
        return [j for j in self.joints if j.parent_id == joint_id]

    def to_dict(self) -> dict:
        return {
            "joints": [j.to_dict() for j in self.joints],
            "bones": [b.to_dict() for b in self.bones],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Skeleton":
        return cls(
            joints=[Joint.from_dict(j) for j in d.get("joints", [])],
            bones=[Bone.from_dict(b) for b in d.get("bones", [])],
        )


@dataclass
class SkinningWeights:
    """Per-vertex skinning weight data."""
    vertex_count: int = 0
    joint_count: int = 0
    weights_path: Optional[str] = None  # .npz with sparse weights matrix
    influence_maps_dir: Optional[str] = None  # per-joint heatmap images

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnimationClip:
    """Metadata about a single animation action."""
    name: str
    duration_sec: float = 3.0
    fps: int = 24
    frame_count: int = 72
    motion_source: str = MotionSource.ORIGINAL.value
    motion_type: str = "unknown"  # walk_like, idle_sway, etc.

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AnimationClip":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Camera:
    """Camera intrinsics and extrinsics for a rendered view."""
    K: List[List[float]] = field(default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    extrinsic: List[List[float]] = field(default_factory=lambda: [
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
    ])
    resolution: Tuple[int, int] = (512, 512)
    view_name: str = "front"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FramePoseLabel:
    """
    The gold label format: per-frame evaluated pose.
    
    CRITICAL: joints must be exported AFTER animation evaluation at this exact frame.
    frame rendered timestamp == exported skeleton timestamp.
    """
    asset_id: str = ""
    animation_id: str = ""
    frame_idx: int = 0
    timestamp_sec: float = 0.0

    # Paths to rendered data
    rgb_path: str = ""
    mask_path: str = ""
    depth_path: str = ""

    # Camera
    camera: Optional[Camera] = None

    # Evaluated skeleton at this frame
    joints: List[Joint] = field(default_factory=list)
    bones: List[Bone] = field(default_factory=list)

    # Skinning reference
    skinning: Optional[SkinningWeights] = None

    def to_dict(self) -> dict:
        d = {
            "asset_id": self.asset_id,
            "animation_id": self.animation_id,
            "frame_idx": self.frame_idx,
            "timestamp_sec": self.timestamp_sec,
            "rgb_path": self.rgb_path,
            "mask_path": self.mask_path,
            "depth_path": self.depth_path,
            "camera": self.camera.to_dict() if self.camera else None,
            "joints": [j.to_dict() for j in self.joints],
            "bones": [b.to_dict() for b in self.bones],
            "skinning": self.skinning.to_dict() if self.skinning else None,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FramePoseLabel":
        return cls(
            asset_id=d.get("asset_id", ""),
            animation_id=d.get("animation_id", ""),
            frame_idx=d.get("frame_idx", 0),
            timestamp_sec=d.get("timestamp_sec", 0.0),
            rgb_path=d.get("rgb_path", ""),
            mask_path=d.get("mask_path", ""),
            depth_path=d.get("depth_path", ""),
            camera=Camera.from_dict(d["camera"]) if d.get("camera") else None,
            joints=[Joint.from_dict(j) for j in d.get("joints", [])],
            bones=[Bone.from_dict(b) for b in d.get("bones", [])],
            skinning=SkinningWeights(**d["skinning"]) if d.get("skinning") else None,
        )


@dataclass
class AssetRig:
    """Complete rigged asset metadata for the Anymate pipeline."""
    asset_id: str = ""
    mesh_path: str = ""
    rig_path: str = ""
    has_skeleton: bool = False
    has_skinning: bool = False
    has_animation: bool = False
    joint_count: int = 0
    bone_count: int = 0
    vertex_count: int = 0
    category: str = "unknown"
    source: str = "anymate"  # anymate, objaverse, custom

    # Rest skeleton
    rest_skeleton: Optional[Skeleton] = None

    # Available animations
    animations: List[AnimationClip] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Don't include rest_skeleton in the lightweight index
        d.pop("rest_skeleton", None)
        d["animations"] = [a.to_dict() for a in self.animations]
        return d

    def to_index_row(self) -> dict:
        """Lightweight row for assets.jsonl index."""
        return {
            "asset_id": self.asset_id,
            "mesh_path": self.mesh_path,
            "rig_path": self.rig_path,
            "has_skeleton": self.has_skeleton,
            "has_skinning": self.has_skinning,
            "has_animation": self.has_animation,
            "joint_count": self.joint_count,
            "bone_count": self.bone_count,
            "vertex_count": self.vertex_count,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AssetRig":
        animations = [AnimationClip.from_dict(a) for a in d.get("animations", [])]
        return cls(
            asset_id=d.get("asset_id", ""),
            mesh_path=d.get("mesh_path", ""),
            rig_path=d.get("rig_path", ""),
            has_skeleton=d.get("has_skeleton", False),
            has_skinning=d.get("has_skinning", False),
            has_animation=d.get("has_animation", False),
            joint_count=d.get("joint_count", 0),
            bone_count=d.get("bone_count", 0),
            vertex_count=d.get("vertex_count", 0),
            category=d.get("category", "unknown"),
            source=d.get("source", "anymate"),
            animations=animations,
        )


def write_jsonl(path: Path, records: List[dict]):
    """Write records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    """Read records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
