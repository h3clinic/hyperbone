"""
Pose2D dataset: loads animal crops with 2D joint labels and generates heatmaps.

Supports:
- Synthetic labels from 3D GT projection (export_pose2d_from_pose3d_gt.py)
- Manual 2D labels in JSONL format

Output per sample:
- image crop [3, H, W]
- heatmaps [J, H, W] - Gaussian peaks at labeled joints
- visibility [J] - binary
- joints_xy [J, 2] - normalized to [0,1] within crop
- bbox_xywh
"""
from __future__ import annotations

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .quadruped_schema import QUADRUPED_JOINTS_2D, NUM_JOINTS_2D, JOINT_ID


def generate_heatmap(
    height: int,
    width: int,
    center_x: float,
    center_y: float,
    sigma: float = 3.0,
) -> np.ndarray:
    """Generate a single Gaussian heatmap peaked at (center_x, center_y).

    center_x, center_y are in pixel coordinates within [0, width) and [0, height).
    """
    xx = np.arange(width, dtype=np.float32)
    yy = np.arange(height, dtype=np.float32)
    xx, yy = np.meshgrid(xx, yy)
    heatmap = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2))
    return heatmap


class Pose2DDataset(Dataset):
    """
    Dataset for 2D animal pose estimation.

    Label format (JSONL lines):
    {
        "image_path": "path/to/frame.png",
        "frame_idx": 0,
        "bbox_xywh": [x, y, w, h],
        "joints": {
            "front_left_shoulder": {"xy": [u, v], "visible": true},
            "head": {"xy": [u, v], "visible": true},
            ...
        },
        "asset_id": "Fox.glb",
        "species": "fox"
    }

    xy coords in joints are absolute image coordinates.
    The dataset crops to bbox and converts to crop-relative coords.
    """

    def __init__(
        self,
        labels_path: str,
        resolution: int = 192,
        sigma: float = 3.0,
        augment: bool = False,
        bbox_expand: float = 1.2,
    ):
        self.resolution = resolution
        self.sigma = sigma
        self.augment = augment
        self.bbox_expand = bbox_expand

        # Load labels
        self.records: List[Dict] = []
        with open(labels_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    self.records.append(rec)

        # Resolve image paths relative to labels file
        self.labels_dir = Path(labels_path).parent

    def __len__(self) -> int:
        return len(self.records)

    def _load_and_crop(self, record: Dict) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Load image and crop to bbox."""
        img_path = record["image_path"]
        if not Path(img_path).is_absolute():
            img_path = str(self.labels_dir / img_path)

        img = cv2.imread(img_path)
        if img is None:
            # Fallback: black image
            img = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
            return img, (0, 0, self.resolution, self.resolution)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        # Get bbox
        bbox = record.get("bbox_xywh")
        if bbox is None:
            bbox = [0, 0, W, H]

        bx, by, bw, bh = bbox

        # Expand bbox
        cx, cy = bx + bw / 2, by + bh / 2
        bw *= self.bbox_expand
        bh *= self.bbox_expand
        # Make square
        side = max(bw, bh)
        bx = cx - side / 2
        by = cy - side / 2
        bw = bh = side

        # Clamp to image bounds
        x1 = max(0, int(bx))
        y1 = max(0, int(by))
        x2 = min(W, int(bx + bw))
        y2 = min(H, int(by + bh))

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            crop = img  # fallback to full image
            x1, y1 = 0, 0
            x2, y2 = W, H

        crop = cv2.resize(crop, (self.resolution, self.resolution))
        actual_bbox = (x1, y1, x2 - x1, y2 - y1)
        return crop, actual_bbox

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        res = self.resolution

        # Load and crop image
        crop, actual_bbox = self._load_and_crop(record)
        bx, by, bw, bh = actual_bbox

        # Image tensor [3, H, W]
        img_t = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0

        # Parse joints
        joints_data = record.get("joints", {})
        joints_xy = np.zeros((NUM_JOINTS_2D, 2), dtype=np.float32)
        visibility = np.zeros(NUM_JOINTS_2D, dtype=np.float32)
        heatmaps = np.zeros((NUM_JOINTS_2D, res, res), dtype=np.float32)

        for joint_name, jinfo in joints_data.items():
            if joint_name not in JOINT_ID:
                continue
            jid = JOINT_ID[joint_name]
            if not jinfo.get("visible", True):
                continue

            # Absolute image coords -> crop-relative coords
            abs_x, abs_y = jinfo["xy"]
            # Convert to crop coordinates
            crop_x = (abs_x - bx) / max(bw, 1) * res
            crop_y = (abs_y - by) / max(bh, 1) * res

            # Only mark visible if within crop bounds
            if 0 <= crop_x < res and 0 <= crop_y < res:
                joints_xy[jid] = [crop_x / res, crop_y / res]  # normalized [0,1]
                visibility[jid] = 1.0
                heatmaps[jid] = generate_heatmap(res, res, crop_x, crop_y, self.sigma)

        # Augmentation (simple horizontal flip)
        if self.augment and np.random.rand() > 0.5:
            img_t = img_t.flip(2)  # flip W
            heatmaps = heatmaps[:, :, ::-1].copy()
            # Swap left/right joints
            for left_name, right_name in _LR_PAIRS:
                li, ri = JOINT_ID[left_name], JOINT_ID[right_name]
                joints_xy[[li, ri]] = joints_xy[[ri, li]]
                visibility[[li, ri]] = visibility[[ri, li]]
                heatmaps[[li, ri]] = heatmaps[[ri, li]]
            # Flip x coordinate
            mask = visibility > 0
            joints_xy[mask, 0] = 1.0 - joints_xy[mask, 0]

        return {
            "image": img_t,                                          # [3, H, W]
            "heatmaps": torch.from_numpy(heatmaps),                  # [J, H, W]
            "visibility": torch.from_numpy(visibility),              # [J]
            "joints_xy": torch.from_numpy(joints_xy),                # [J, 2]
            "bbox_xywh": torch.tensor(actual_bbox, dtype=torch.float32),
            "frame_idx": record.get("frame_idx", idx),
            "asset_id": record.get("asset_id", ""),
            "species": record.get("species", ""),
        }


# Left-right joint pairs for augmentation
_LR_PAIRS = [
    ("front_left_shoulder", "front_right_shoulder"),
    ("front_left_elbow", "front_right_elbow"),
    ("front_left_hoof", "front_right_hoof"),
    ("rear_left_hip", "rear_right_hip"),
    ("rear_left_knee", "rear_right_knee"),
    ("rear_left_hoof", "rear_right_hoof"),
]
