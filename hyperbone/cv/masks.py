"""Mask generation — pluggable backend interface.

Backends:
- "threshold": Adaptive threshold + morphology. Fast, low quality.
- "sam2": SAM2 automatic mask generation. High quality, requires GPU.
- "noop": Returns no masks. For testing pipeline without segmentation.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional


class MaskBackend:
    """Base class for mask generation backends.

    All backends must implement generate() and return a list of mask records.
    """

    def generate(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Generate masks for a single frame.

        Returns list of mask records:
        {
            "mask": np.ndarray (uint8, 255=fg),
            "bbox": [x, y, w, h],
            "area": int,
            "confidence": float,
            "source": str,
            "object_id": int,
            "touches_edge": bool,
            "metadata": dict,
        }
        """
        raise NotImplementedError


class ThresholdMaskBackend(MaskBackend):
    """Adaptive threshold + morphology mask generator.

    Not meant to be accurate — only provides pipeline structure.
    """

    def __init__(self, min_area: int = 500, max_objects: int = 10,
                 min_area_ratio: float = 0.002, max_area_ratio: float = 0.80):
        self.min_area = min_area
        self.max_objects = max_objects
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def generate(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        h, w = frame.shape[:2]
        frame_area = h * w
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5
        )

        combined = cv2.bitwise_or(edges, thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

        records = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue

            area_ratio = area / frame_area
            if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                continue

            mask = np.zeros((h, w), dtype=np.uint8)
            mask[labels == i] = 255

            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])

            touches_edge = _mask_touches_edge(mask, h, w)

            records.append({
                "mask": mask,
                "bbox": [x, y, bw, bh],
                "area": area,
                "confidence": 0.5,  # threshold has no confidence
                "source": "threshold",
                "object_id": len(records),
                "touches_edge": touches_edge,
                "metadata": {},
            })

            if len(records) >= self.max_objects:
                break

        # Fallback: Otsu if nothing found
        if not records:
            _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            area = int(np.count_nonzero(otsu))
            area_ratio = area / frame_area
            if self.min_area_ratio <= area_ratio <= self.max_area_ratio:
                records.append({
                    "mask": otsu,
                    "bbox": [0, 0, w, h],
                    "area": area,
                    "confidence": 0.3,
                    "source": "threshold",
                    "object_id": 0,
                    "touches_edge": True,
                    "metadata": {"fallback": "otsu"},
                })

        return records


class NoopMaskBackend(MaskBackend):
    """Returns no masks. For testing pipeline plumbing without segmentation."""

    def generate(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        return []


def _mask_touches_edge(mask: np.ndarray, h: int, w: int, margin: int = 2) -> bool:
    """Check if mask has foreground pixels within margin of frame edge."""
    if mask[:margin, :].any():
        return True
    if mask[-margin:, :].any():
        return True
    if mask[:, :margin].any():
        return True
    if mask[:, -margin:].any():
        return True
    return False


def get_mask_generator(backend: str = "threshold", **kwargs) -> MaskBackend:
    """Factory: returns the appropriate mask generator.

    backend: "threshold" (default) | "sam2" | "grounded_sam2" | "noop"
    """
    if backend == "threshold":
        valid_keys = {"min_area", "max_objects", "min_area_ratio", "max_area_ratio"}
        filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
        return ThresholdMaskBackend(**filtered)
    elif backend == "sam2":
        from hyperbone.cv.sam2_adapter import SAM2MaskBackend
        return SAM2MaskBackend(**kwargs)
    elif backend == "grounded_sam2":
        from hyperbone.cv.grounded_sam2_adapter import GroundedSAM2Backend
        return GroundedSAM2Backend(**kwargs)
    elif backend == "noop":
        return NoopMaskBackend()
    else:
        raise ValueError(f"Unknown mask backend: {backend}")


def generate_masks(frame: np.ndarray, backend: str = "threshold", **kwargs) -> List[Dict[str, Any]]:
    """Convenience function: generate masks for a frame using the specified backend.

    Returns list of mask records with keys:
        mask, bbox, area, confidence, source, object_id, touches_edge, metadata
    """
    gen = get_mask_generator(backend, **kwargs)
    return gen.generate(frame)


# Backward compat aliases
ThresholdMaskGenerator = ThresholdMaskBackend
BaseMaskGenerator = MaskBackend


def save_mask(mask: np.ndarray, output_dir: str, frame_idx: int, object_id: int) -> Path:
    """Save a binary mask as PNG."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"mask_{frame_idx:06d}_obj{object_id:03d}.png"
    cv2.imwrite(str(path), mask)
    return path


def save_mask(mask: np.ndarray, output_dir: str, frame_idx: int, object_id: int) -> Path:
    """Save a binary mask as PNG."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"mask_{frame_idx:06d}_obj{object_id:03d}.png"
    cv2.imwrite(str(path), mask)
    return path
