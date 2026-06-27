"""
Crop-based dataset for HyperBonePose3D-Joint training.

Loads labels that include:
- image path
- bbox for cropping
- 3D canonical joints (primary target)
- 2D projected joints (auxiliary target)
- visibility
- camera metadata when available

Produces:
- cropped image tensor
- 3D joint targets
- 2D joint targets (normalized to crop)
- Gaussian heatmaps (auxiliary)
- visibility mask
"""
from __future__ import annotations

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from ..pose2d.quadruped_schema import QUADRUPED_JOINTS_2D, NUM_JOINTS_2D, JOINT_ID


def generate_heatmap(h: int, w: int, cx: float, cy: float, sigma: float = 3.0) -> np.ndarray:
    """Gaussian heatmap peaked at (cx, cy) in pixel coords."""
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))


class Pose3DJointDataset(Dataset):
    """
    Dataset for HyperBonePose3D-Joint.

    Label format (JSONL):
    {
        "image_path": "...",
        "frame_idx": 0,
        "bbox_xywh": [x, y, w, h],
        "joints_3d": {
            "root": {"xyz": [x,y,z], "visible": true},
            "neck": {"xyz": [x,y,z], "visible": true},
            ...
        },
        "joints_2d": {
            "root": {"xy": [u,v], "visible": true},
            ...
        },
        "canonical_scale": 1.0,
        "canonical_root": [x,y,z],
        "asset_id": "Fox.glb",
        "species": "fox",
        "camera": {...}  // optional
    }
    """

    def __init__(
        self,
        labels_path: str,
        resolution: int = 192,
        heatmap_resolution: int = 48,
        sigma: float = 2.0,
        augment: bool = False,
        bbox_expand: float = 1.25,
        input_channels: int = 3,
    ):
        self.resolution = resolution
        self.heatmap_resolution = heatmap_resolution
        self.sigma = sigma
        self.augment = augment
        self.bbox_expand = bbox_expand
        self.input_channels = input_channels

        self.records: List[Dict] = []
        with open(labels_path) as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

        self.labels_dir = Path(labels_path).parent

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, img_path: str) -> str:
        if Path(img_path).is_absolute():
            return img_path
        return str(self.labels_dir / img_path)

    def _crop_image(self, img: np.ndarray, bbox_xywh: List[float]) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Crop image to bbox (expanded, made square)."""
        H, W = img.shape[:2]
        bx, by, bw, bh = bbox_xywh

        # Expand and make square
        cx, cy = bx + bw / 2, by + bh / 2
        side = max(bw, bh) * self.bbox_expand

        x1 = max(0, int(cx - side / 2))
        y1 = max(0, int(cy - side / 2))
        x2 = min(W, int(cx + side / 2))
        y2 = min(H, int(cy + side / 2))

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            crop = img
            x1, y1, x2, y2 = 0, 0, W, H

        crop = cv2.resize(crop, (self.resolution, self.resolution))
        return crop, (x1, y1, x2 - x1, y2 - y1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        res = self.resolution
        hm_res = self.heatmap_resolution

        # Load image
        img_path = self._resolve_path(record["image_path"])
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((res, res, 3), dtype=np.uint8)
            actual_bbox = (0.0, 0.0, float(res), float(res))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bbox = record.get("bbox_xywh", [0, 0, img.shape[1], img.shape[0]])
            img, actual_bbox = self._crop_image(img, bbox)

        # Image tensor
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [3, H, W]

        # If input_channels > 3, pad with zeros (mask/depth not available)
        if self.input_channels > 3:
            extra = torch.zeros(self.input_channels - 3, res, res)
            img_t = torch.cat([img_t, extra], dim=0)

        # Parse 3D joints (PRIMARY TARGET)
        joints_3d_data = record.get("joints_3d", {})
        joint_xyz = np.zeros((NUM_JOINTS_2D, 3), dtype=np.float32)
        visibility = np.zeros(NUM_JOINTS_2D, dtype=np.float32)

        for joint_name, jinfo in joints_3d_data.items():
            if joint_name not in JOINT_ID:
                continue
            jid = JOINT_ID[joint_name]
            if not jinfo.get("visible", True):
                continue
            joint_xyz[jid] = jinfo["xyz"]
            visibility[jid] = 1.0

        # Parse 2D joints (AUXILIARY - for reprojection loss)
        joints_2d_data = record.get("joints_2d", {})
        joints_2d = np.zeros((NUM_JOINTS_2D, 2), dtype=np.float32)
        heatmaps = np.zeros((NUM_JOINTS_2D, hm_res, hm_res), dtype=np.float32)

        bx, by, bw, bh = actual_bbox

        for joint_name, jinfo in joints_2d_data.items():
            if joint_name not in JOINT_ID:
                continue
            jid = JOINT_ID[joint_name]
            if not jinfo.get("visible", True):
                continue
            if visibility[jid] < 0.5:
                continue  # Only if 3D is also visible

            abs_x, abs_y = jinfo["xy"]
            # Convert to crop-relative normalized [0, 1]
            crop_x = (abs_x - bx) / max(bw, 1)
            crop_y = (abs_y - by) / max(bh, 1)

            if 0 <= crop_x <= 1 and 0 <= crop_y <= 1:
                joints_2d[jid] = [crop_x, crop_y]
                # Generate heatmap at heatmap resolution
                hm_x = crop_x * hm_res
                hm_y = crop_y * hm_res
                heatmaps[jid] = generate_heatmap(hm_res, hm_res, hm_x, hm_y, self.sigma)

        # Augmentation: horizontal flip
        if self.augment and np.random.rand() > 0.5:
            img_t = img_t.flip(2)
            heatmaps = heatmaps[:, :, ::-1].copy()
            joints_2d[:, 0] = 1.0 - joints_2d[:, 0]
            joint_xyz[:, 0] = -joint_xyz[:, 0]  # Flip x in canonical space
            # Swap left/right
            for ln, rn in _LR_PAIRS:
                li, ri = JOINT_ID[ln], JOINT_ID[rn]
                joint_xyz[[li, ri]] = joint_xyz[[ri, li]]
                joints_2d[[li, ri]] = joints_2d[[ri, li]]
                visibility[[li, ri]] = visibility[[ri, li]]
                heatmaps[[li, ri]] = heatmaps[[ri, li]]

        return {
            "image": img_t,                                             # [C, H, W]
            "joint_xyz_canonical": torch.from_numpy(joint_xyz),         # [J, 3] PRIMARY
            "joint_visible": torch.from_numpy(visibility),              # [J]
            "joints_2d": torch.from_numpy(joints_2d),                   # [J, 2] auxiliary
            "heatmaps": torch.from_numpy(heatmaps),                     # [J, Hm, Hm] auxiliary
            "bbox_xywh": torch.tensor(actual_bbox, dtype=torch.float32),
            "frame_idx": record.get("frame_idx", idx),
            "asset_id": record.get("asset_id", ""),
            "species": record.get("species", ""),
        }


_LR_PAIRS = [
    ("front_left_shoulder", "front_right_shoulder"),
    ("front_left_elbow", "front_right_elbow"),
    ("front_left_hoof", "front_right_hoof"),
    ("rear_left_hip", "rear_right_hip"),
    ("rear_left_knee", "rear_right_knee"),
    ("rear_left_hoof", "rear_right_hoof"),
]
