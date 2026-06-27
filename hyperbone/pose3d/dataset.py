"""
Pose3D dataset for HyperBone3D training.

Loads RGB + mask + depth + GT armature from the rendered dataset structure.
Returns tensors suitable for supervised training.
"""
from __future__ import annotations

import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .joint_map import (
    QUADRUPED_JOINTS, QUADRUPED_BONES, NUM_JOINTS,
    remap_joints_to_canonical,
)
from .canonicalize import find_root_joint, compute_torso_length
from .schema import PoseFrame3D, Joint3D


class Pose3DDataset(Dataset):
    """
    Dataset for HyperBone3D supervised training.

    Expected directory structure:
        <root>/
            rgb/frame_0000.png ...
            masks/frame_0000.png ...
            depth/frame_0000.png ...  (optional)
            pose3d_gt.jsonl
            rest_skeleton.json
            dataset_manifest.json (optional)
    """

    def __init__(
        self,
        root_dir: str,
        resolution: Tuple[int, int] = (256, 256),
        use_depth: bool = True,
        asset_id: str = "",
    ):
        self.root = Path(root_dir)
        self.resolution = resolution
        self.use_depth = use_depth
        self.asset_id = asset_id

        # Load GT pose records
        gt_path = self.root / "pose3d_gt.jsonl"
        self.records: List[Dict] = []
        with open(gt_path) as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

        # Load rest skeleton if available
        rest_path = self.root / "rest_skeleton.json"
        self.rest_skeleton = None
        if rest_path.exists():
            with open(rest_path) as f:
                self.rest_skeleton = json.load(f)

        # Detect asset_id from records if not provided
        if not self.asset_id and self.records:
            self.asset_id = self.records[0].get("asset_id", "")

        # Index RGB/mask/depth files
        self.rgb_dir = self.root / "rgb"
        self.mask_dir = self.root / "masks"
        self.depth_dir = self.root / "depth"

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: Path, channels: int = 3) -> Optional[np.ndarray]:
        """Load image and resize."""
        if not path.exists():
            return None
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        h, w = self.resolution
        img = cv2.resize(img, (w, h))
        return img

    def _find_frame_file(self, directory: Path, frame_idx: int) -> Optional[Path]:
        """Find frame file with various naming conventions."""
        patterns = [
            f"frame_{frame_idx:04d}.png",
            f"frame_{frame_idx:04d}.jpg",
            f"{frame_idx:04d}.png",
            f"{frame_idx:06d}.png",
        ]
        for p in patterns:
            path = directory / p
            if path.exists():
                return path
        return None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        frame_idx = record["frame_idx"]
        h, w = self.resolution

        # Load RGB
        rgb_path = self._find_frame_file(self.rgb_dir, frame_idx)
        if rgb_path is not None:
            rgb = self._load_image(rgb_path, 3)
            if rgb is not None and len(rgb.shape) == 3:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            else:
                rgb = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Load mask
        mask_path = self._find_frame_file(self.mask_dir, frame_idx)
        if mask_path is not None:
            mask = self._load_image(mask_path, 1)
            if mask is not None:
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]
            else:
                mask = np.zeros((h, w), dtype=np.uint8)
        else:
            mask = np.ones((h, w), dtype=np.uint8) * 255

        # Load depth
        if self.use_depth:
            depth_path = self._find_frame_file(self.depth_dir, frame_idx)
            if depth_path is not None:
                depth = self._load_image(depth_path, 1)
                if depth is not None:
                    if len(depth.shape) == 3:
                        depth = depth[:, :, 0]
                else:
                    depth = np.zeros((h, w), dtype=np.uint8)
            else:
                depth = np.zeros((h, w), dtype=np.uint8)
        else:
            depth = np.zeros((h, w), dtype=np.uint8)

        # Convert to tensors [C, H, W], normalized to [0, 1]
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # [3,H,W]
        mask_t = torch.from_numpy(mask).unsqueeze(0).float() / 255.0     # [1,H,W]
        depth_t = torch.from_numpy(depth).unsqueeze(0).float() / 255.0   # [1,H,W]

        # Remap joints to canonical quadruped schema
        source_joints = record.get("joints", [])
        xyz_list, visible_list = remap_joints_to_canonical(source_joints, self.asset_id)

        # Canonicalize: root-relative, scale-normalized
        xyz_arr = np.array(xyz_list, dtype=np.float32)
        root_xyz = xyz_arr[0]  # root is index 0
        scale = np.linalg.norm(xyz_arr[3] - xyz_arr[0]) if visible_list[3] else 1.0  # root→neck distance
        scale = max(scale, 1e-3)

        canonical_xyz = (xyz_arr - root_xyz) / scale

        joint_xyz = torch.from_numpy(canonical_xyz)        # [J, 3]
        joint_visible = torch.tensor(visible_list, dtype=torch.float32)  # [J]

        # Bone edges
        bone_edges = torch.tensor(QUADRUPED_BONES, dtype=torch.long)  # [E, 2]

        # BBox
        bbox = record.get("bbox_xywh", [0, 0, w, h])
        if bbox is None:
            bbox = [0, 0, w, h]

        # Camera
        camera_K = record.get("camera_K")
        camera_ext = record.get("camera_extrinsic")

        return {
            "rgb": rgb_t,                    # [3, H, W]
            "mask": mask_t,                  # [1, H, W]
            "depth": depth_t,                # [1, H, W]
            "bbox_xywh": torch.tensor(bbox, dtype=torch.float32),
            "joint_xyz_canonical": joint_xyz,  # [J, 3]
            "joint_visible": joint_visible,    # [J]
            "bone_edges": bone_edges,          # [E, 2]
            "asset_id": self.asset_id,
            "frame_idx": frame_idx,
            "scale": torch.tensor(scale, dtype=torch.float32),
            "root_xyz": torch.from_numpy(root_xyz),
        }


class Pose3DDatasetFromVideo(Dataset):
    """
    Dataset variant that loads frames directly from a video + GT JSONL.
    No pre-rendered RGB/mask/depth directory needed.
    """

    def __init__(
        self,
        video_path: str,
        gt_jsonl_path: str,
        resolution: Tuple[int, int] = (256, 256),
        asset_id: str = "",
    ):
        self.video_path = video_path
        self.resolution = resolution
        self.asset_id = asset_id

        # Load GT
        self.records: List[Dict] = []
        with open(gt_jsonl_path) as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

        # Pre-load all video frames
        cap = cv2.VideoCapture(video_path)
        self.frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
        cap.release()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        frame_idx = record["frame_idx"]
        h, w = self.resolution

        # Get video frame
        if frame_idx < len(self.frames):
            frame = self.frames[frame_idx]
            frame = cv2.resize(frame, (w, h))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Generate mask from background diff (dark bg)
        gray = cv2.cvtColor(cv2.resize(self.frames[frame_idx] if frame_idx < len(self.frames)
                                        else np.zeros((h, w, 3), dtype=np.uint8), (w, h)),
                            cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        # No depth available from video
        depth = np.zeros((h, w), dtype=np.uint8)

        # Tensors
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        depth_t = torch.from_numpy(depth).unsqueeze(0).float() / 255.0

        # Remap joints
        source_joints = record.get("joints", [])
        xyz_list, visible_list = remap_joints_to_canonical(source_joints, self.asset_id)

        xyz_arr = np.array(xyz_list, dtype=np.float32)
        root_xyz = xyz_arr[0]
        scale = np.linalg.norm(xyz_arr[3] - xyz_arr[0]) if visible_list[3] else 1.0
        scale = max(scale, 1e-3)
        canonical_xyz = (xyz_arr - root_xyz) / scale

        joint_xyz = torch.from_numpy(canonical_xyz)
        joint_visible = torch.tensor(visible_list, dtype=torch.float32)
        bone_edges = torch.tensor(QUADRUPED_BONES, dtype=torch.long)

        bbox = record.get("bbox_xywh", [0, 0, w, h])
        if bbox is None:
            bbox = [0, 0, w, h]

        return {
            "rgb": rgb_t,
            "mask": mask_t,
            "depth": depth_t,
            "bbox_xywh": torch.tensor(bbox, dtype=torch.float32),
            "joint_xyz_canonical": joint_xyz,
            "joint_visible": joint_visible,
            "bone_edges": bone_edges,
            "asset_id": self.asset_id,
            "frame_idx": frame_idx,
            "scale": torch.tensor(scale, dtype=torch.float32),
            "root_xyz": torch.from_numpy(root_xyz),
        }
