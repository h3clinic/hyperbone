"""Bbox crop utilities for HyperBone custom mapper.

Handles cropping to detection boxes, resizing for speed,
and coordinate mapping between crop and frame space.
"""

import numpy as np
from typing import Dict, Tuple


def crop_to_bbox(frame: np.ndarray, bbox_xywh: Tuple[int, int, int, int],
                 pad_px: int = 8) -> Dict:
    """Crop a frame to a bounding box with padding.

    Args:
        frame: BGR image (H, W, 3) or grayscale (H, W).
        bbox_xywh: (x, y, w, h) bounding box.
        pad_px: Padding around the box in pixels.

    Returns:
        Dict with: crop, crop_bbox_xywh, offset_xy, scale (always 1.0 here).
    """
    h, w = frame.shape[:2]
    bx, by, bw, bh = bbox_xywh

    # Apply padding and clamp
    x1 = max(0, bx - pad_px)
    y1 = max(0, by - pad_px)
    x2 = min(w, bx + bw + pad_px)
    y2 = min(h, by + bh + pad_px)

    crop = frame[y1:y2, x1:x2].copy()

    return {
        "crop": crop,
        "crop_bbox_xywh": (x1, y1, x2 - x1, y2 - y1),
        "offset_xy": (x1, y1),
        "scale": 1.0,
        "original_shape": (h, w),
    }


def crop_mask_to_bbox(mask: np.ndarray, bbox_xywh: Tuple[int, int, int, int],
                      pad_px: int = 8) -> Dict:
    """Crop a mask to a bounding box with padding."""
    return crop_to_bbox(mask, bbox_xywh, pad_px)


def resize_crop_max_side(crop: np.ndarray, max_side: int = 384) -> Dict:
    """Resize a crop so its longest side is max_side pixels.

    Returns:
        Dict with: resized, scale, original_shape.
    """
    import cv2

    h, w = crop.shape[:2]
    if max(h, w) <= max_side:
        return {"resized": crop, "scale": 1.0, "original_shape": (h, w)}

    scale = max_side / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)

    return {"resized": resized, "scale": scale, "original_shape": (h, w)}


def map_crop_point_to_frame(point_xy: Tuple[int, int], crop_meta: Dict) -> Tuple[int, int]:
    """Map a point from crop coordinates back to frame coordinates.

    Args:
        point_xy: (x, y) in crop space.
        crop_meta: Dict with offset_xy and scale.

    Returns:
        (x, y) in frame coordinates.
    """
    ox, oy = crop_meta["offset_xy"]
    scale = crop_meta.get("scale", 1.0)

    # If the crop was resized, undo the resize first
    x = int(point_xy[0] / scale) + ox
    y = int(point_xy[1] / scale) + oy
    return (x, y)


def map_frame_point_to_crop(point_xy: Tuple[int, int], crop_meta: Dict) -> Tuple[int, int]:
    """Map a point from frame coordinates to crop coordinates.

    Args:
        point_xy: (x, y) in frame space.
        crop_meta: Dict with offset_xy and scale.

    Returns:
        (x, y) in crop coordinates.
    """
    ox, oy = crop_meta["offset_xy"]
    scale = crop_meta.get("scale", 1.0)

    x = int((point_xy[0] - ox) * scale)
    y = int((point_xy[1] - oy) * scale)
    return (x, y)
