"""Frame saving utilities."""

import cv2
import numpy as np
from pathlib import Path


def save_frame(frame: np.ndarray, output_dir: str, frame_idx: int) -> Path:
    """Save a BGR frame as PNG."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"frame_{frame_idx:06d}.png"
    cv2.imwrite(str(path), frame)
    return path
