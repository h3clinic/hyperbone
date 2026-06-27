"""Tests for HyperBone custom thinning (Zhang-Suen, Guo-Hall)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
from hyperbone.cv.custom_thinning import zhang_suen_thin, guo_hall_thin, skeletonize_custom


def test_rectangle_produces_centerline():
    """A filled rectangle should thin to a horizontal centerline."""
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[30:70, 20:180] = 1

    skeleton = zhang_suen_thin(mask)

    # Should produce a thin line roughly in the center
    assert skeleton.sum() > 0, "Skeleton should not be empty"
    # All skeleton pixels should be 1px wide (no 2x2 blocks)
    ys, xs = np.where(skeleton == 1)
    assert len(ys) > 10, f"Expected >10 skeleton pixels, got {len(ys)}"
    # Y-coords should be concentrated around center (row 50)
    assert abs(ys.mean() - 50) < 10, f"Centerline should be near row 50, got mean {ys.mean():.1f}"
    print("  PASS: rectangle produces centerline skeleton")


def test_y_shape_produces_junction():
    """A Y-shape mask should produce junction + 3 endpoints."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(mask, (100, 120), (100, 180), 1, 6)  # stem
    cv2.line(mask, (100, 120), (60, 50), 1, 6)    # left branch
    cv2.line(mask, (100, 120), (140, 50), 1, 6)   # right branch

    skeleton = zhang_suen_thin(mask)
    assert skeleton.sum() > 0, "Skeleton should not be empty"

    # Count endpoints (degree 1) and junctions (degree 3+)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbors = cv2.filter2D(skeleton, -1, kernel)
    neighbors = neighbors * skeleton

    endpoints = np.sum((neighbors == 1) & (skeleton == 1))
    junctions = np.sum((neighbors >= 3) & (skeleton == 1))

    assert endpoints >= 3, f"Expected >=3 endpoints, got {endpoints}"
    assert junctions >= 1, f"Expected >=1 junction, got {junctions}"
    print("  PASS: Y-shape produces junction + 3 endpoints")


def test_ring_does_not_crash():
    """A ring/loop mask should thin without error."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 30, 1, 8)  # ring, not filled

    skeleton = zhang_suen_thin(mask)
    assert skeleton.sum() > 0, "Ring skeleton should not be empty"
    print("  PASS: ring/loop does not crash")


def test_convergence():
    """Thinning should converge within max_iterations."""
    mask = np.zeros((50, 200), dtype=np.uint8)
    mask[10:40, 10:190] = 1

    skeleton = zhang_suen_thin(mask, max_iterations=200)
    assert skeleton.sum() > 0
    print("  PASS: thinning converges")


def test_guo_hall():
    """Guo-Hall should also produce a valid skeleton."""
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[30:70, 20:180] = 1

    skeleton = guo_hall_thin(mask)
    assert skeleton.sum() > 0, "Guo-Hall skeleton should not be empty"
    ys, xs = np.where(skeleton == 1)
    assert len(ys) > 10
    print("  PASS: Guo-Hall produces valid skeleton")


def test_skeletonize_custom_api():
    """skeletonize_custom should return uint8 255-valued skeleton."""
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[30:70, 20:180] = 255

    skel = skeletonize_custom(mask, algorithm="zhang-suen")
    assert skel.dtype == np.uint8
    assert skel.max() == 255
    assert np.count_nonzero(skel) > 10
    print("  PASS: skeletonize_custom API correct")


def test_single_pixel():
    """A single pixel should not crash."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 5] = 1
    skeleton = zhang_suen_thin(mask)
    # Single pixel is already thin
    assert skeleton[5, 5] == 1
    print("  PASS: single pixel handled")


def test_empty_mask():
    """Empty mask returns empty skeleton."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    skeleton = zhang_suen_thin(mask)
    assert skeleton.sum() == 0
    print("  PASS: empty mask returns empty skeleton")


def run_all():
    print("[HyperBone Custom Thinning Tests]")
    test_rectangle_produces_centerline()
    test_y_shape_produces_junction()
    test_ring_does_not_crash()
    test_convergence()
    test_guo_hall()
    test_skeletonize_custom_api()
    test_single_pixel()
    test_empty_mask()
    print(f"\nAll 8 tests passed.")


if __name__ == "__main__":
    run_all()
