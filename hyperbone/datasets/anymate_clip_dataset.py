"""
Anymate Clip Dataset — PyTorch dataset for rendered clips with skeleton labels.

Supports:
- clip_len=1: single-frame mode (for per-frame pose prediction)
- clip_len=8/16: temporal window mode (for temporal training)

Returns RGB/mask/depth tensors and joint targets per frame.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import cv2
except ImportError:
    cv2 = None


class AnymateClipDataset(Dataset):
    """
    Dataset of rendered Anymate clips with per-frame skeleton labels.
    
    Args:
        index_path: path to dataset_index.jsonl (or train.jsonl / val.jsonl)
        clip_len: number of frames per sample (1 = single frame, >1 = temporal)
        stride: frame stride when sampling windows (only for clip_len > 1)
        resolution: resize images to this resolution
        max_joints: maximum joints to pad to (for batching variable skeletons)
        use_mask: include mask channel
        use_depth: include depth channel
        root_dir: base directory for resolving relative paths
    """

    def __init__(
        self,
        index_path: str,
        clip_len: int = 1,
        stride: int = 1,
        resolution: int = 256,
        max_joints: int = 128,
        use_mask: bool = True,
        use_depth: bool = True,
        root_dir: Optional[str] = None,
    ):
        self.clip_len = clip_len
        self.stride = stride
        self.resolution = resolution
        self.max_joints = max_joints
        self.use_mask = use_mask
        self.use_depth = use_depth

        index_path = Path(index_path)
        self.root_dir = Path(root_dir) if root_dir else index_path.parent

        # Load frame labels
        self.frame_labels = []
        with open(index_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.frame_labels.append(json.loads(line))

        # Group by clip (asset_id + animation_id)
        self.clips: Dict[str, List[int]] = {}
        for i, label in enumerate(self.frame_labels):
            clip_key = f"{label.get('asset_id', '')}_{label.get('animation_id', '')}"
            self.clips.setdefault(clip_key, []).append(i)

        # Sort frames within each clip by frame_idx
        for key in self.clips:
            self.clips[key].sort(key=lambda i: self.frame_labels[i].get("frame_idx", 0))

        # Build sample index
        self.samples = self._build_samples()

    def _build_samples(self) -> List[Tuple[str, int]]:
        """Build list of (clip_key, start_offset) for windowed sampling."""
        samples = []
        for clip_key, frame_indices in self.clips.items():
            n_frames = len(frame_indices)
            if self.clip_len == 1:
                # Every frame is a sample
                for offset in range(n_frames):
                    samples.append((clip_key, offset))
            else:
                # Sliding window
                window_frames = self.clip_len * self.stride
                for start in range(0, max(1, n_frames - window_frames + 1), self.stride):
                    samples.append((clip_key, start))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clip_key, start_offset = self.samples[idx]
        frame_indices = self.clips[clip_key]

        # Gather frame indices for this window
        selected = []
        for t in range(self.clip_len):
            fi = start_offset + t * self.stride
            fi = min(fi, len(frame_indices) - 1)  # Clamp
            selected.append(frame_indices[fi])

        # Load data for each frame in window
        rgb_frames = []
        mask_frames = []
        depth_frames = []
        joint_xyz = []  # [T, J, 3]
        joint_xy = []   # [T, J, 2]
        joint_vis = []  # [T, J]
        joint_active = []  # [T, J] — 1 if joint exists

        for frame_global_idx in selected:
            label = self.frame_labels[frame_global_idx]

            # Load RGB
            rgb = self._load_image(label.get("rgb_path", ""), channels=3)
            rgb_frames.append(rgb)

            # Load mask
            if self.use_mask:
                mask = self._load_image(label.get("mask_path", ""), channels=1)
                mask_frames.append(mask)

            # Load depth
            if self.use_depth:
                depth = self._load_depth(label.get("depth_path", ""))
                depth_frames.append(depth)

            # Parse joints
            joints = label.get("joints", [])
            xyz = np.zeros((self.max_joints, 3), dtype=np.float32)
            xy = np.zeros((self.max_joints, 2), dtype=np.float32)
            vis = np.zeros(self.max_joints, dtype=np.float32)
            active = np.zeros(self.max_joints, dtype=np.float32)

            for j in joints:
                ji = j.get("id", 0)
                if ji >= self.max_joints:
                    continue
                active[ji] = 1.0
                # Support both camera_xyz and world_xyz (pilot uses world_xyz)
                xyz[ji] = j.get("camera_xyz", j.get("world_xyz", [0, 0, 0]))[:3]
                xy[ji] = j.get("image_xy", [0, 0])[:2]
                vis[ji] = 1.0 if j.get("visible", True) else 0.0

            joint_xyz.append(xyz)
            joint_xy.append(xy)
            joint_vis.append(vis)
            joint_active.append(active)

        # Stack into tensors
        rgb_tensor = torch.stack(rgb_frames, dim=0)  # [T, 3, H, W]
        result = {"rgb": rgb_tensor}

        if self.use_mask:
            result["mask"] = torch.stack(mask_frames, dim=0)  # [T, 1, H, W]
        if self.use_depth:
            result["depth"] = torch.stack(depth_frames, dim=0)  # [T, 1, H, W]

        result["joint_xyz"] = torch.tensor(np.array(joint_xyz), dtype=torch.float32)  # [T, J, 3]
        result["joint_xy"] = torch.tensor(np.array(joint_xy), dtype=torch.float32)  # [T, J, 2]
        result["joint_vis"] = torch.tensor(np.array(joint_vis), dtype=torch.float32)  # [T, J]
        result["joint_active"] = torch.tensor(np.array(joint_active), dtype=torch.float32)  # [T, J]

        # Squeeze temporal dim for single-frame mode
        if self.clip_len == 1:
            result = {k: v.squeeze(0) for k, v in result.items()}

        # Metadata (not tensors)
        result["clip_key"] = clip_key
        result["frame_idx"] = self.frame_labels[selected[0]].get("frame_idx", 0)

        return result

    def _load_image(self, rel_path: str, channels: int = 3) -> torch.Tensor:
        """Load and resize an image to a normalized tensor."""
        full_path = self.root_dir / rel_path
        if cv2 is not None and full_path.exists():
            if channels == 1:
                img = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (self.resolution, self.resolution))
                    return torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
            else:
                img = cv2.imread(str(full_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.resolution, self.resolution))
                    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

        # Return zeros if file missing
        return torch.zeros(channels, self.resolution, self.resolution, dtype=torch.float32)

    def _load_depth(self, rel_path: str) -> torch.Tensor:
        """Load depth map (EXR, NPZ, NPY, or PNG)."""
        full_path = self.root_dir / rel_path

        if full_path.exists():
            ext = full_path.suffix.lower()
            try:
                if ext == ".npy":
                    depth = np.load(str(full_path))
                elif ext == ".npz":
                    data = np.load(str(full_path))
                    depth = data[list(data.keys())[0]]
                elif ext == ".exr" and cv2 is not None:
                    depth = cv2.imread(str(full_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
                elif ext == ".png" and cv2 is not None:
                    depth = cv2.imread(str(full_path), cv2.IMREAD_ANYDEPTH)
                    if depth is not None:
                        depth = depth.astype(np.float32) / 65535.0  # 16-bit to [0,1]
                else:
                    return torch.zeros(1, self.resolution, self.resolution, dtype=torch.float32)

                if depth is not None:
                    depth = cv2.resize(depth.astype(np.float32), (self.resolution, self.resolution))
                    # Normalize: clip and scale
                    valid = depth[depth < 1e9]
                    if len(valid) > 0:
                        depth = np.clip(depth, 0, valid.max())
                        depth = depth / (valid.max() + 1e-8)
                    return torch.from_numpy(depth).unsqueeze(0)
            except Exception:
                pass

        return torch.zeros(1, self.resolution, self.resolution, dtype=torch.float32)

    @property
    def num_clips(self) -> int:
        return len(self.clips)

    @property
    def total_frames(self) -> int:
        return len(self.frame_labels)
