"""Skeletonization — medial axis extraction from binary masks."""

import cv2
import numpy as np


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Thin a binary mask to a 1-pixel-wide skeleton using morphological thinning.

    Input: uint8 mask (255=foreground).
    Output: uint8 skeleton (255=skeleton pixel, 0=background).
    """
    # Ensure binary
    binary = (mask > 127).astype(np.uint8)

    # Zhang-Suen thinning via OpenCV ximgproc if available, else manual
    try:
        skeleton = cv2.ximgproc.thinning(binary * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except AttributeError:
        # Fallback: iterative morphological skeleton
        skeleton = _morphological_skeleton(binary)

    return skeleton


def _morphological_skeleton(binary: np.ndarray) -> np.ndarray:
    """Iterative morphological skeletonization fallback."""
    img = binary.copy()
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = cv2.erode(img, element)
        if cv2.countNonZero(img) == 0:
            break

    return skel * 255
