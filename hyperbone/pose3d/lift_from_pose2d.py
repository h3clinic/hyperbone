"""
3D lifting from 2D keypoints - placeholder module.

Status: experimental_lift
NOT the main pipeline. HyperBonePose3D-Joint already predicts 3D directly.
This module exists only for future use if we need to lift external 2D detections.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple

STATUS = "experimental_lift"


def lift_2d_to_3d(
    joints_2d: np.ndarray,          # [J, 2]
    confidences: np.ndarray,        # [J]
    bbox_xywh: Tuple[float, float, float, float],
    depth_map: Optional[np.ndarray] = None,  # [H, W]
) -> Dict[str, np.ndarray]:
    """
    Placeholder: lift 2D keypoints to approximate 3D.

    This is NOT the main pose prediction path.
    HyperBonePose3D-Joint predicts 3D directly from crops.

    This function exists for:
    - lifting external 2D detections (DeepLabCut, SLEAP, etc.)
    - depth-based z estimation when depth maps are available
    """
    J = joints_2d.shape[0]

    # Placeholder: normalize 2D to canonical-like coords
    bx, by, bw, bh = bbox_xywh
    norm_x = (joints_2d[:, 0] - bx - bw/2) / max(bw, 1)
    norm_y = -(joints_2d[:, 1] - by - bh/2) / max(bh, 1)  # flip y

    # Z estimation from depth map if available
    if depth_map is not None:
        H, W = depth_map.shape
        z_vals = np.zeros(J)
        for j in range(J):
            if confidences[j] > 0.3:
                px = int(np.clip(joints_2d[j, 0], 0, W-1))
                py = int(np.clip(joints_2d[j, 1], 0, H-1))
                z_vals[j] = depth_map[py, px] / 255.0  # normalized
        z_est = z_vals
    else:
        z_est = np.zeros(J)

    xyz_canonical = np.stack([norm_x, norm_y, z_est], axis=-1)

    return {
        "joint_xyz_canonical": xyz_canonical,  # [J, 3]
        "status": STATUS,
        "method": "placeholder_lift",
    }
