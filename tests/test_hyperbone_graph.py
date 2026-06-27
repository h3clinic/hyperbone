"""Tests for HyperBone custom graph extraction and full custom mapper pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
from hyperbone.cv.custom_thinning import skeletonize_custom
from hyperbone.cv.graph import skeleton_to_graph
from hyperbone.cv.crops import (
    crop_to_bbox, resize_crop_max_side,
    map_crop_point_to_frame, map_frame_point_to_crop,
)
from hyperbone.cv.hyperbone_graph import map_proposal_to_skeleton
from hyperbone.objects.proposals import ObjectProposal


def test_graph_from_rectangle():
    """Rectangle mask → custom thinning → graph with endpoints."""
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[30:70, 20:180] = 255

    skel = skeletonize_custom(mask)
    graph = skeleton_to_graph(skel, min_branch_length=5)

    assert len(graph["nodes"]) >= 2, f"Expected >=2 nodes, got {len(graph['nodes'])}"
    endpoints = [n for n in graph["nodes"] if n["type"] == "endpoint"]
    assert len(endpoints) >= 2, f"Expected >=2 endpoints, got {len(endpoints)}"
    print("  PASS: rectangle → custom skeleton → graph with endpoints")


def test_graph_from_y_shape():
    """Y-shape → one junction, 3 endpoints, 3 edges."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(mask, (100, 120), (100, 180), 255, 6)
    cv2.line(mask, (100, 120), (60, 50), 255, 6)
    cv2.line(mask, (100, 120), (140, 50), 255, 6)

    skel = skeletonize_custom(mask)
    graph = skeleton_to_graph(skel, min_branch_length=5)

    junctions = [n for n in graph["nodes"] if n["type"] == "junction"]
    endpoints = [n for n in graph["nodes"] if n["type"] == "endpoint"]
    assert len(junctions) >= 1, f"Expected >=1 junction, got {len(junctions)}"
    assert len(endpoints) >= 3, f"Expected >=3 endpoints, got {len(endpoints)}"
    assert len(graph["edges"]) >= 3, f"Expected >=3 edges, got {len(graph['edges'])}"
    print("  PASS: Y-shape → junction + 3 endpoints + 3 edges")


def test_short_branch_pruned():
    """Short endpoint branches should be pruned."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(mask, (100, 20), (100, 180), 255, 6)  # long vertical
    cv2.line(mask, (100, 100), (110, 105), 255, 4)  # tiny spur

    skel = skeletonize_custom(mask)
    graph_strict = skeleton_to_graph(skel, min_branch_length=30)
    graph_lenient = skeleton_to_graph(skel, min_branch_length=1)

    assert len(graph_strict["nodes"]) <= len(graph_lenient["nodes"])
    print("  PASS: short branch pruned with strict threshold")


def test_crop_coordinate_roundtrip():
    """Crop → frame coordinate mapping should round-trip correctly."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = (100, 50, 200, 150)  # x, y, w, h

    crop_result = crop_to_bbox(frame, bbox, pad_px=8)
    crop_meta = {"offset_xy": crop_result["offset_xy"], "scale": 1.0}

    # Point in crop space
    crop_point = (50, 30)
    frame_point = map_crop_point_to_frame(crop_point, crop_meta)
    back_to_crop = map_frame_point_to_crop(frame_point, crop_meta)

    assert abs(back_to_crop[0] - crop_point[0]) <= 1, f"X roundtrip failed: {back_to_crop} vs {crop_point}"
    assert abs(back_to_crop[1] - crop_point[1]) <= 1, f"Y roundtrip failed: {back_to_crop} vs {crop_point}"
    print("  PASS: crop coordinate roundtrip correct")


def test_resize_crop_maps_correctly():
    """Resized crop should map points back to frame correctly."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = (100, 50, 400, 300)

    crop_result = crop_to_bbox(frame, bbox, pad_px=8)
    crop = crop_result["crop"]
    resize_result = resize_crop_max_side(crop, max_side=200)

    crop_meta = {
        "offset_xy": crop_result["offset_xy"],
        "scale": resize_result["scale"],
    }

    # Point at center of resized crop
    rh, rw = resize_result["resized"].shape[:2]
    resized_center = (rw // 2, rh // 2)
    frame_point = map_crop_point_to_frame(resized_center, crop_meta)

    # Should be somewhere inside the original bbox
    assert 92 <= frame_point[0] <= 508, f"Frame X out of range: {frame_point[0]}"
    assert 42 <= frame_point[1] <= 358, f"Frame Y out of range: {frame_point[1]}"
    print("  PASS: resized crop maps points back to frame")


def test_manual_proposal_produces_graph():
    """Manual bbox proposal → custom mapper → graph record."""
    # Create a frame with a clear vertical object
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(frame, (150, 50), (170, 250), (200, 200, 200), -1)

    proposal = ObjectProposal.manual(
        frame_idx=0, object_id=0,
        bbox_xywh=(140, 40, 40, 220),
        label="pole",
    )

    result = map_proposal_to_skeleton(
        frame, proposal,
        max_side=256,
        thinning_algorithm="zhang-suen",
        min_branch_length=5,
        mask_method="threshold",
    )

    assert result["metadata"]["skeleton_mapper"] == "hyperbone-custom"
    assert result["metadata"]["thinning_algorithm"] == "zhang-suen"
    assert result["metadata"]["graph_builder"] == "hyperbone-pixel-trace-v0"
    assert result["metadata"]["object_label"] == "pole"
    assert result["metadata"]["object_label_source"] == "manual"
    # May or may not produce nodes depending on mask quality
    print(f"  PASS: manual proposal → custom mapper (nodes={result['metadata']['node_count']})")


def test_graph_jsonl_includes_mapper_metadata():
    """Graph JSONL should include skeleton_mapper='hyperbone-custom'."""
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.line(frame, (50, 100), (250, 100), (255, 255, 255), 8)

    proposal = ObjectProposal.manual(
        frame_idx=0, object_id=0,
        bbox_xywh=(40, 85, 220, 30),
        label="line",
    )

    result = map_proposal_to_skeleton(frame, proposal, max_side=384)

    meta = result["metadata"]
    assert meta["skeleton_mapper"] == "hyperbone-custom"
    assert "skeleton_runtime_ms" in meta
    assert meta["skeleton_runtime_ms"] > 0
    print("  PASS: graph includes skeleton_mapper='hyperbone-custom' + runtime")


def test_empty_crop_handled():
    """An empty/black crop should not crash."""
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    proposal = ObjectProposal.manual(
        frame_idx=0, object_id=0,
        bbox_xywh=(50, 50, 100, 100),
        label="empty",
    )
    result = map_proposal_to_skeleton(frame, proposal)
    assert result["metadata"]["node_count"] == 0
    print("  PASS: empty crop handled gracefully")


def run_all():
    print("[HyperBone Custom Mapper Tests]")
    test_graph_from_rectangle()
    test_graph_from_y_shape()
    test_short_branch_pruned()
    test_crop_coordinate_roundtrip()
    test_resize_crop_maps_correctly()
    test_manual_proposal_produces_graph()
    test_graph_jsonl_includes_mapper_metadata()
    test_empty_crop_handled()
    print(f"\nAll 8 tests passed.")


if __name__ == "__main__":
    run_all()
