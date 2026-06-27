"""HyperBone custom mask approximation — no SAM2 dependency.

Generates a foreground mask from a bbox crop using classical CV:
    edge detection → adaptive threshold → flood-fill → cleanup

Not expected to beat SAM2. Exists to make the full skeleton path independent.
"""

import cv2
import numpy as np
from typing import Dict


def mask_from_bbox_crop(crop: np.ndarray, method: str = "combined") -> Dict:
    """Generate a binary foreground mask from a bbox crop.

    Args:
        crop: BGR image crop (H, W, 3).
        method: "edges", "threshold", "grabcut_lite", or "combined" (default).

    Returns:
        Dict with: mask (uint8, 255=fg), method, metadata.
    """
    if crop.ndim == 2:
        gray = crop
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]

    if method == "edges":
        mask = _mask_from_edges(gray, h, w)
    elif method == "threshold":
        mask = _mask_from_threshold(gray, h, w)
    elif method == "grabcut_lite":
        mask = _mask_grabcut_lite(crop, h, w)
    else:  # combined
        mask = _mask_combined(crop, gray, h, w)

    # Cleanup
    mask = _cleanup_mask(mask, h, w)

    return {
        "mask": mask,
        "method": method,
        "metadata": {"crop_shape": (h, w)},
    }


def _mask_from_edges(gray: np.ndarray, h: int, w: int) -> np.ndarray:
    """Edge-based mask: Canny → dilate → flood fill interior."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Flood fill from center to find interior
    mask = np.zeros((h, w), dtype=np.uint8)
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    seed = (w // 2, h // 2)

    # Invert: edges=barrier, fill the interior
    inv_edges = cv2.bitwise_not(edges)
    cv2.floodFill(inv_edges, flood_mask, seed, 128)
    mask[inv_edges == 128] = 255

    return mask


def _mask_from_threshold(gray: np.ndarray, h: int, w: int) -> np.ndarray:
    """Adaptive threshold mask."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5
    )

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closed


def _mask_grabcut_lite(crop: np.ndarray, h: int, w: int) -> np.ndarray:
    """Simplified GrabCut: use center rectangle as foreground hint."""
    if crop.ndim == 2:
        crop3 = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    else:
        crop3 = crop

    mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    # Define rect: inner 80% of crop
    margin_x = max(1, w // 10)
    margin_y = max(1, h // 10)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    try:
        cv2.grabCut(crop3, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        # Convert GrabCut mask to binary
        result = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        # Fallback to threshold if GrabCut fails
        gray = cv2.cvtColor(crop3, cv2.COLOR_BGR2GRAY)
        result = _mask_from_threshold(gray, h, w)

    return result


def _mask_combined(crop: np.ndarray, gray: np.ndarray, h: int, w: int) -> np.ndarray:
    """Combined approach: edges + threshold + center flood fill.

    Takes the intersection of multiple cues for robustness.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Adaptive threshold (finds dark/light regions)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5
    )

    # 2. Sobel gradient magnitude (edge strength)
    sx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    mag = (mag / mag.max() * 255).astype(np.uint8) if mag.max() > 0 else np.zeros_like(gray)
    _, edge_mask = cv2.threshold(mag, 30, 255, cv2.THRESH_BINARY)

    # 3. Otsu on the crop
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Combine: at least 2 of 3 agree
    combined = np.zeros((h, w), dtype=np.uint8)
    vote = (thresh > 0).astype(np.uint8) + (edge_mask > 0).astype(np.uint8) + (otsu > 0).astype(np.uint8)
    combined[vote >= 2] = 255

    # Morphological closing to connect regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    return combined


def _cleanup_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """Remove tiny islands and fill small holes."""
    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(16, int(h * w * 0.01))  # at least 1% of crop

    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255

    # Fill small holes
    inv = 255 - clean
    num_labels_h, labels_h, stats_h, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    max_hole = max(16, int(h * w * 0.05))
    for i in range(1, num_labels_h):
        area = stats_h[i, cv2.CC_STAT_AREA]
        if area < max_hole:
            # Check it doesn't touch border (true hole vs background)
            x = stats_h[i, cv2.CC_STAT_LEFT]
            y = stats_h[i, cv2.CC_STAT_TOP]
            bw = stats_h[i, cv2.CC_STAT_WIDTH]
            bh = stats_h[i, cv2.CC_STAT_HEIGHT]
            if x > 0 and y > 0 and (x + bw) < w and (y + bh) < h:
                clean[labels_h == i] = 255

    return clean
