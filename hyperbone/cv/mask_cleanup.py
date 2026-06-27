"""Mask cleanup — preprocessing binary masks before skeletonization.

Improves skeleton connectivity by:
- Filling small holes
- Removing tiny isolated islands
- Morphological closing to bridge narrow gaps
- Optional: keep only the largest connected component
- Contour smoothing to reduce jagged edges
"""

import cv2
import numpy as np
from typing import Dict, Any


def clean_mask_for_skeleton(
    mask: np.ndarray,
    min_component_area: int = 64,
    close_kernel: int = 5,
    fill_holes: bool = True,
    keep_largest: bool = True,
    smooth_contour: bool = True,
    dilate_before_thin: int = 0,
) -> Dict[str, Any]:
    """Clean a binary mask to improve skeleton extraction.

    Args:
        mask: Binary mask (uint8, 255=foreground).
        min_component_area: Remove components smaller than this (pixels).
        close_kernel: Morphological closing kernel size (0=skip).
        fill_holes: Fill internal holes in the mask.
        keep_largest: Keep only the largest connected component.
        smooth_contour: Apply contour approximation to smooth edges.
        dilate_before_thin: Pixels to dilate before returning (helps bridge narrow gaps).

    Returns:
        Dict with clean_mask and metadata.
    """
    h, w = mask.shape[:2]
    binary = (mask > 127).astype(np.uint8)
    original_area = int(binary.sum())

    components_removed = 0
    holes_filled = 0

    # 1. Morphological closing — bridge narrow gaps
    if close_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 2. Fill holes (flood fill exterior background, remainder = internal holes)
    if fill_holes:
        inv = (1 - binary).astype(np.uint8)  # bg+holes=1, fg=0
        # Only proceed if (0,0) is actually background
        if inv[0, 0] == 1:
            flood = inv.copy()
            ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            cv2.floodFill(flood, ff_mask, (0, 0), 0)
            # After flood: exterior bg=0, internal holes=1, fg=0
            holes_filled = int(flood.sum())
            binary = binary | flood

    # 3. Remove tiny components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels > 2:  # more than just background + one component
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_component_area:
                binary[labels == i] = 0
                components_removed += 1

    # 4. Keep largest component only
    largest_kept = False
    if keep_largest:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if num_labels > 2:
            # Find largest foreground component
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = np.argmax(areas) + 1
            new_binary = np.zeros_like(binary)
            new_binary[labels == largest_label] = 1
            components_removed += (num_labels - 2)  # all others removed
            binary = new_binary
            largest_kept = True

    # 5. Smooth contour
    if smooth_contour:
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            smooth_mask = np.zeros_like(binary)
            for cnt in contours:
                # Approximate contour to remove jaggedness
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                cv2.drawContours(smooth_mask, [approx], -1, 1, -1)
            binary = smooth_mask

    # 6. Optional dilation to help bridge very narrow gaps
    if dilate_before_thin > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_before_thin * 2 + 1, dilate_before_thin * 2 + 1)
        )
        binary = cv2.dilate(binary, k, iterations=1)

    clean_mask = binary * 255

    return {
        "clean_mask": clean_mask,
        "components_removed": components_removed,
        "holes_filled": holes_filled,
        "cleanup_applied": True,
        "largest_kept": largest_kept,
        "cleanup_metadata": {
            "original_area": original_area,
            "clean_area": int((clean_mask > 0).sum()),
            "close_kernel": close_kernel,
            "min_component_area": min_component_area,
            "fill_holes": fill_holes,
            "keep_largest": keep_largest,
            "smooth_contour": smooth_contour,
            "dilate_before_thin": dilate_before_thin,
        },
    }
