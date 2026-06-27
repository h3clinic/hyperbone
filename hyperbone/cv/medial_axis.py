"""Distance-transform medial axis prototype.

Computes skeleton via ridges of the distance transform.
Stores radius_norm per skeleton pixel for future use.
Not default yet — research path for HyperBone v1.
"""

import cv2
import numpy as np
from typing import Dict, Tuple


def distance_transform_skeleton(mask: np.ndarray, ridge_threshold: float = 0.4) -> Dict:
    """Compute medial axis via distance transform ridges.

    Args:
        mask: Binary mask (uint8, 255=foreground or 1=foreground).
        ridge_threshold: Ratio of max distance to consider as ridge.
            Lower = thicker skeleton, higher = thinner.

    Returns:
        Dict with:
            skeleton: uint8 (255=skeleton pixel)
            distance_map: float32 distance transform
            radius_at_pixel: distance value at each skeleton pixel
    """
    binary = (mask > 0).astype(np.uint8)

    # Compute Euclidean distance transform
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    if dist.max() == 0:
        return {
            "skeleton": np.zeros_like(binary, dtype=np.uint8),
            "distance_map": dist,
            "radius_at_pixel": np.zeros_like(dist),
        }

    # Normalize distance map
    dist_norm = dist / dist.max()

    # Find ridge pixels: local maxima in distance map
    # A pixel is on the medial axis if it's a local maximum
    # Use Laplacian of distance as ridge indicator
    laplacian = cv2.Laplacian(dist, cv2.CV_64F, ksize=3)

    # Ridge candidates: distance above threshold AND negative Laplacian (local max)
    skeleton = np.zeros_like(binary, dtype=np.uint8)
    ridge_mask = (dist_norm >= ridge_threshold) & (laplacian < 0) & (binary == 1)
    skeleton[ridge_mask] = 1

    # Alternative: non-maximum suppression approach
    # Check if pixel is local maximum in its 3x3 neighborhood
    kernel = np.ones((3, 3), dtype=np.float32)
    local_max = cv2.dilate(dist, kernel, iterations=1)
    is_local_max = (dist == local_max) & (binary == 1) & (dist > 0)

    # Combine: ridges OR strong local maxima
    skeleton_combined = skeleton | is_local_max.astype(np.uint8)

    # Thin to 1px using simple iterative erosion of thick regions
    # Only keep pixels that won't disconnect the skeleton
    skeleton_final = _thin_ridges(skeleton_combined)

    return {
        "skeleton": skeleton_final * 255,
        "distance_map": dist,
        "radius_at_pixel": dist * skeleton_final,
    }


def _thin_ridges(skeleton: np.ndarray) -> np.ndarray:
    """Thin ridge-based skeleton to approximately 1 pixel width.

    Uses morphological thinning (minimal, just to clean up thick ridges).
    """
    from hyperbone.cv.custom_thinning import zhang_suen_thin
    return zhang_suen_thin(skeleton, max_iterations=20)
