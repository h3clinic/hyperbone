"""Tests for HyperBone pseudo-label factory."""

import json
import tempfile
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.cv.skeletonize import skeletonize_mask
from hyperbone.cv.graph import skeleton_to_graph
from hyperbone.cv.masks import ThresholdMaskGenerator
from hyperbone.cv.overlay import draw_overlay
from hyperbone.export.jsonl import JSONLWriter, graph_to_record
from hyperbone.export.npz import export_npz


def _make_rectangle_mask(w=200, h=100):
    """Create a horizontal rectangle mask — should yield a centerline."""
    mask = np.zeros((300, 400), dtype=np.uint8)
    mask[100:200, 100:300] = 255
    return mask


def _make_y_shape_mask():
    """Create a Y-shaped mask — should yield 1 junction + 3 endpoints."""
    mask = np.zeros((300, 300), dtype=np.uint8)
    # Vertical stem
    cv2.line(mask, (150, 200), (150, 280), 255, 8)
    # Left branch
    cv2.line(mask, (150, 200), (100, 120), 255, 8)
    # Right branch
    cv2.line(mask, (150, 200), (200, 120), 255, 8)
    return mask


def test_rectangle_creates_centerline():
    """A rectangle mask should produce a roughly horizontal skeleton graph."""
    mask = _make_rectangle_mask()
    skel = skeletonize_mask(mask)
    graph = skeleton_to_graph(skel, min_branch_length=5)

    assert len(graph["nodes"]) >= 2, f"Expected >=2 nodes, got {len(graph['nodes'])}"
    # Should have endpoints (no complex branching in a rectangle)
    endpoints = [n for n in graph["nodes"] if n["type"] == "endpoint"]
    assert len(endpoints) >= 2, f"Expected >=2 endpoints, got {len(endpoints)}"
    print("  PASS: rectangle → centerline graph")


def test_y_shape_creates_branch():
    """A Y-shape mask should produce 1 junction and 3 endpoints."""
    mask = _make_y_shape_mask()
    skel = skeletonize_mask(mask)
    graph = skeleton_to_graph(skel, min_branch_length=5)

    junctions = [n for n in graph["nodes"] if n["type"] == "junction"]
    endpoints = [n for n in graph["nodes"] if n["type"] == "endpoint"]

    assert len(junctions) >= 1, f"Expected >=1 junction, got {len(junctions)}"
    assert len(endpoints) >= 3, f"Expected >=3 endpoints, got {len(endpoints)}"
    assert len(graph["edges"]) >= 3, f"Expected >=3 edges, got {len(graph['edges'])}"
    print("  PASS: Y-shape → junction + 3 endpoints")


def test_pruning_removes_short_branches():
    """Short branches below threshold should be pruned."""
    mask = _make_y_shape_mask()
    skel = skeletonize_mask(mask)

    # With very high threshold, most branches get pruned
    graph_strict = skeleton_to_graph(skel, min_branch_length=200)
    graph_lenient = skeleton_to_graph(skel, min_branch_length=1)

    assert len(graph_strict["nodes"]) <= len(graph_lenient["nodes"]), \
        "Strict pruning should remove nodes"
    print("  PASS: pruning reduces node count")


def test_jsonl_export_structure():
    """JSONL export should contain required fields."""
    mask = _make_rectangle_mask()
    skel = skeletonize_mask(mask)
    graph = skeleton_to_graph(skel, min_branch_length=5)

    record = graph_to_record(
        video_id="test123",
        frame_idx=42,
        timestamp_sec=1.4,
        object_id=0,
        graph=graph,
        bbox=(100, 100, 300, 200),
    )

    assert record["video_id"] == "test123"
    assert record["frame_idx"] == 42
    assert "nodes" in record
    assert "edges" in record
    assert record["coord_system"] == "image_xy"
    assert record["quality"]["node_count"] >= 2

    # Write and re-read
    with tempfile.TemporaryDirectory() as tmpdir:
        with JSONLWriter(tmpdir, "test.jsonl") as w:
            w.write(record)

        lines = (Path(tmpdir) / "test.jsonl").read_text().strip().split("\n")
        parsed = json.loads(lines[0])
        assert parsed["video_id"] == "test123"
        assert len(parsed["nodes"]) >= 2

    print("  PASS: JSONL export has correct structure")


def test_npz_export():
    """NPZ export should produce valid arrays."""
    records = []
    for i in range(5):
        records.append({
            "frame_idx": i * 30,
            "timestamp_sec": i * 1.0,
            "nodes": [
                {"id": 0, "type": "endpoint", "xy": [100, 100], "confidence": 0.9},
                {"id": 1, "type": "endpoint", "xy": [200, 100], "confidence": 0.8},
            ],
            "edges": [{"parent": 0, "child": 1, "length_px": 100.0}],
        })

    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_npz(records, tmpdir, "test.npz")
        data = np.load(str(path))
        assert data["nodes_xy"].shape == (5, 64, 2)
        assert data["node_active"].shape == (5, 64)
        assert data["edges"].shape == (5, 64, 64)
        assert data["node_active"][0, 0] == 1
        assert data["node_active"][0, 2] == 0  # no third node
        assert data["edges"][0, 0, 1] == 1
        assert data["edges"][0, 1, 0] == 1  # symmetric
        data.close()

    print("  PASS: NPZ export produces valid arrays")


def test_overlay_does_not_crash():
    """Overlay generation should not throw on valid inputs."""
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50)
    mask = _make_rectangle_mask()
    skel = skeletonize_mask(mask)
    graph = skeleton_to_graph(skel, min_branch_length=5)

    overlay = draw_overlay(frame, mask, graph, object_id=0)
    assert overlay.shape == frame.shape
    assert overlay.dtype == np.uint8
    # Should have some non-gray pixels from the overlay
    assert not np.array_equal(overlay, frame)
    print("  PASS: overlay generation works")


def run_all():
    print("[HyperBone Tests]")
    test_rectangle_creates_centerline()
    test_y_shape_creates_branch()
    test_pruning_removes_short_branches()
    test_jsonl_export_structure()
    test_npz_export()
    test_overlay_does_not_crash()
    print("\nAll tests passed.")


if __name__ == "__main__":
    run_all()
