"""Tests for mask cleanup module."""

import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.cv.mask_cleanup import clean_mask_for_skeleton


def test_fills_small_holes():
    """Mask with internal holes gets filled."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (80, 80), 255, -1)
    # Punch a small hole
    cv2.circle(mask, (50, 50), 5, 0, -1)

    result = clean_mask_for_skeleton(mask, fill_holes=True)
    clean = result["clean_mask"]

    # The hole should be filled
    assert clean[50, 50] > 0, "Hole should be filled"
    assert result["holes_filled"] > 0
    print("  PASS: fills small holes")


def test_removes_tiny_islands():
    """Tiny isolated components are removed."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Large object
    cv2.rectangle(mask, (50, 50), (150, 150), 255, -1)
    # Tiny island far from main object (outside closing reach)
    mask[5, 5] = 255
    mask[5, 6] = 255
    mask[6, 5] = 255

    # Disable closing and smoothing so island removal is isolated
    result = clean_mask_for_skeleton(
        mask, min_component_area=64, keep_largest=False,
        close_kernel=0, smooth_contour=False, fill_holes=False
    )
    clean = result["clean_mask"]

    # Tiny island removed
    assert clean[5, 5] == 0, "Tiny island should be removed"
    # Large object remains
    assert clean[100, 100] > 0, "Large object should remain"
    assert result["components_removed"] > 0
    print("  PASS: removes tiny islands")


def test_keep_largest_component():
    """With keep_largest=True, only biggest component survives."""
    mask = np.zeros((300, 300), dtype=np.uint8)
    # Large component
    cv2.rectangle(mask, (10, 10), (100, 100), 255, -1)
    # Medium component (well separated)
    cv2.rectangle(mask, (200, 200), (280, 280), 255, -1)

    result = clean_mask_for_skeleton(mask, keep_largest=True, min_component_area=10, close_kernel=0, fill_holes=False)
    clean = result["clean_mask"]

    # Large should remain
    assert clean[50, 50] > 0, "Largest component should remain"
    # Medium removed
    assert clean[240, 240] == 0, "Non-largest component should be removed"
    assert result["largest_kept"] is True
    print("  PASS: keep largest component")


def test_morphological_closing():
    """Closing bridges narrow gaps."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Two rectangles with 3px gap
    cv2.rectangle(mask, (20, 40), (45, 60), 255, -1)
    cv2.rectangle(mask, (48, 40), (80, 60), 255, -1)

    result = clean_mask_for_skeleton(mask, close_kernel=7, keep_largest=False, fill_holes=False)
    clean = result["clean_mask"]

    # The gap should be bridged
    assert clean[50, 46] > 0 or clean[50, 47] > 0, "Gap should be closed"
    print("  PASS: morphological closing bridges gaps")


def test_cleanup_metadata():
    """Cleanup returns proper metadata."""
    mask = np.ones((50, 50), dtype=np.uint8) * 255
    result = clean_mask_for_skeleton(mask)

    assert result["cleanup_applied"] is True
    assert "clean_mask" in result
    assert "components_removed" in result
    assert "holes_filled" in result
    assert "cleanup_metadata" in result
    assert result["cleanup_metadata"]["close_kernel"] == 5
    print("  PASS: cleanup metadata present")


def run_all():
    print("[HyperBone Mask Cleanup Tests]")
    test_fills_small_holes()
    test_removes_tiny_islands()
    test_keep_largest_component()
    test_morphological_closing()
    test_cleanup_metadata()
    print(f"\nAll 5 tests passed.")


if __name__ == "__main__":
    run_all()
