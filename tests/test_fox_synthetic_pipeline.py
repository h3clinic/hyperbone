"""Tests for Fox synthetic tracking pipeline.

These tests do NOT require Blender. They test:
- Asset path resolution
- Proposal generation logic
- Proposal schema validation
- Pipeline integration
- Tracking metrics computation
- Overlay handling
"""

import sys, json, tempfile, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

from hyperbone.objects.proposals import ObjectProposal
from hyperbone.tracking.simple_graph_track import compute_graph_metrics, compute_tracking_metrics


def _make_synthetic_video(out_path: str, width=640, height=480, fps=24, duration_sec=2):
    """Create a synthetic video with a fox-like shape on dark background."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    for i in range(int(fps * duration_sec)):
        frame = np.full((height, width, 3), (38, 38, 51), dtype=np.uint8)  # Dark bg
        # Draw a moving elongated shape (simulates fox body)
        cx = 200 + int(50 * np.sin(i * 0.1))
        cy = 240
        # Body
        cv2.ellipse(frame, (cx, cy), (120, 40), 0, 0, 360, (180, 120, 60), -1)
        # Head
        cv2.ellipse(frame, (cx + 100, cy - 20), (30, 25), -15, 0, 360, (200, 140, 70), -1)
        # Tail
        pts = np.array([[cx - 120, cy], [cx - 160, cy - 50], [cx - 140, cy - 10]], np.int32)
        cv2.fillPoly(frame, [pts], (180, 120, 60))
        # Legs
        for lx in [cx - 60, cx - 20, cx + 30, cx + 70]:
            cv2.line(frame, (lx, cy + 35), (lx, cy + 80), (160, 100, 50), 6)
        writer.write(frame)
    writer.release()


def test_fox_asset_path_resolution():
    """Fox asset path can be constructed correctly."""
    from scripts.fetch_gltf_sample_asset import ASSET_META
    assert "Fox" in ASSET_META
    meta = ASSET_META["Fox"]
    assert "glb_subpath" in meta
    expected_path = Path("assets/gltf_samples/Fox") / meta["glb_subpath"]
    assert str(expected_path).endswith("Fox.glb")
    print("  PASS: Fox asset path resolution works")


def test_proposal_bbox_inside_frame():
    """Generated proposals have bboxes inside frame bounds."""
    from scripts.make_fox_proposals import estimate_bbox_background_diff

    frame = np.full((480, 640, 3), (38, 38, 51), dtype=np.uint8)
    cv2.ellipse(frame, (320, 240), (150, 60), 0, 0, 360, (180, 120, 60), -1)

    result = estimate_bbox_background_diff(frame)
    x, y, w, h = result["bbox_xywh"]
    assert x >= 0, f"x={x} < 0"
    assert y >= 0, f"y={y} < 0"
    assert x + w <= 640, f"x+w={x+w} > 640"
    assert y + h <= 480, f"y+h={y+h} > 480"
    assert w > 0 and h > 0
    print("  PASS: proposal bbox stays inside frame")


def test_proposal_jsonl_schema():
    """Proposal JSONL schema is valid."""
    from scripts.make_fox_proposals import estimate_bbox_background_diff

    frame = np.full((480, 640, 3), (38, 38, 51), dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (440, 330), (180, 120, 60), -1)

    result = estimate_bbox_background_diff(frame)

    proposal = {
        "frame_idx": 0,
        "timestamp_sec": 0.0,
        "object_id": 0,
        "label": "fox",
        "label_confidence": 1.0,
        "bbox_xywh": result["bbox_xywh"],
        "prompt": "fox",
        "proposal_method": result["proposal_method"],
    }

    # Verify all required fields
    required = ["frame_idx", "object_id", "label", "bbox_xywh"]
    for field in required:
        assert field in proposal, f"Missing field: {field}"

    assert len(proposal["bbox_xywh"]) == 4
    assert proposal["proposal_method"] in ("synthetic_background_bbox", "central_fallback")
    print("  PASS: proposal JSONL schema is valid")


def test_manual_synthetic_works_without_dino():
    """Manual synthetic proposals work without DINO installed."""
    tmpdir = tempfile.mkdtemp()
    try:
        video_path = str(Path(tmpdir) / "fox.mp4")
        _make_synthetic_video(video_path, duration_sec=1)

        from scripts.make_fox_proposals import make_proposals
        proposals = make_proposals(video_path, label="fox", sample_fps=5.0)

        assert len(proposals) >= 1
        assert proposals[0]["label"] == "fox"
        assert proposals[0]["proposal_method"] in ("synthetic_background_bbox", "central_fallback")
        print("  PASS: manual synthetic proposal source works without DINO")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_graph_record_includes_fox_label():
    """Graph records from the pipeline include object_label='fox'."""
    tmpdir = tempfile.mkdtemp()
    try:
        video_path = str(Path(tmpdir) / "fox.mp4")
        proposals_path = str(Path(tmpdir) / "proposals.jsonl")
        out_dir = str(Path(tmpdir) / "output")

        _make_synthetic_video(video_path, duration_sec=1)

        # Generate proposals
        from scripts.make_fox_proposals import make_proposals
        proposals = make_proposals(video_path, label="fox", sample_fps=5.0,
                                   output_path=proposals_path)

        # Run pipeline
        from hyperbone.pipelines.proposal_skeleton import run_proposal_skeleton
        stats = run_proposal_skeleton(
            video_path=video_path,
            output_dir=out_dir,
            proposals_path=proposals_path,
            proposal_source="manual",
            sample_fps=5.0,
            max_side=256,
        )

        graphs_jsonl = Path(out_dir) / "graphs" / "graphs.jsonl"
        assert graphs_jsonl.exists()
        with open(graphs_jsonl) as f:
            records = [json.loads(l) for l in f if l.strip()]
        assert len(records) >= 1
        assert records[0]["object_label"] == "fox"
        print("  PASS: graph record includes object_label='fox'")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_graph_record_includes_hyperbone_custom():
    """Graph records include skeleton_mapper='hyperbone-custom'."""
    tmpdir = tempfile.mkdtemp()
    try:
        video_path = str(Path(tmpdir) / "fox.mp4")
        proposals_path = str(Path(tmpdir) / "proposals.jsonl")
        out_dir = str(Path(tmpdir) / "output")

        _make_synthetic_video(video_path, duration_sec=1)

        from scripts.make_fox_proposals import make_proposals
        make_proposals(video_path, label="fox", sample_fps=5.0, output_path=proposals_path)

        from hyperbone.pipelines.proposal_skeleton import run_proposal_skeleton
        run_proposal_skeleton(
            video_path=video_path,
            output_dir=out_dir,
            proposals_path=proposals_path,
            proposal_source="manual",
            sample_fps=5.0,
            max_side=256,
        )

        graphs_jsonl = Path(out_dir) / "graphs" / "graphs.jsonl"
        with open(graphs_jsonl) as f:
            records = [json.loads(l) for l in f if l.strip()]
        for r in records:
            assert r["skeleton_mapper"] == "hyperbone-custom", \
                f"Expected hyperbone-custom, got {r['skeleton_mapper']}"
        print("  PASS: graph record includes skeleton_mapper='hyperbone-custom'")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_tracking_metrics_from_mock_sequence():
    """Tracking metrics compute correctly from mock graph sequence."""
    # Mock records simulating consistent fox skeleton across frames
    mock_records = []
    for i in range(10):
        nodes = [
            {"id": 0, "xy": [100 + i * 2, 200], "type": "endpoint"},
            {"id": 1, "xy": [200 + i * 2, 200], "type": "junction"},
            {"id": 2, "xy": [300 + i * 2, 200], "type": "endpoint"},
            {"id": 3, "xy": [200 + i * 2, 150], "type": "endpoint"},
        ]
        edges = [
            {"parent": 0, "child": 1},
            {"parent": 1, "child": 2},
            {"parent": 1, "child": 3},
        ]
        mock_records.append({
            "frame_idx": i * 24,
            "accepted": True,
            "nodes": nodes,
            "edges": edges,
            "bbox_xywh": [80 + i * 2, 130, 240, 90],
            "node_count": 4,
            "edge_count": 3,
        })

    metrics = compute_tracking_metrics(mock_records)

    assert metrics["frames_with_proposal"] == 10
    assert metrics["accepted_frames"] == 10
    assert metrics["topology_stability_score"] > 0.9, \
        f"Expected high stability, got {metrics['topology_stability_score']}"
    assert metrics["centroid_jump_px_mean"] > 0
    assert metrics["node_count_mean"] == 4.0
    assert metrics["edge_count_mean"] == 3.0
    print("  PASS: tracking metrics compute from mock graph sequence")


def test_overlay_handles_missing_rejected():
    """Overlay script handles frames with no rejected graphs without crashing."""
    tmpdir = tempfile.mkdtemp()
    try:
        video_path = str(Path(tmpdir) / "fox.mp4")
        graphs_path = str(Path(tmpdir) / "graphs.jsonl")
        overlay_path = str(Path(tmpdir) / "overlay.mp4")

        _make_synthetic_video(video_path, width=320, height=240, duration_sec=1)

        # Write graph records for only the first frame (not all)
        with open(graphs_path, "w") as f:
            record = {
                "frame_idx": 0,
                "accepted": True,
                "nodes": [
                    {"id": 0, "xy": [100, 120], "type": "endpoint"},
                    {"id": 1, "xy": [200, 120], "type": "endpoint"},
                ],
                "edges": [{"parent": 0, "child": 1}],
                "bbox_xywh": [80, 100, 140, 40],
                "node_count": 2,
                "edge_count": 1,
                "reject_reasons": [],
            }
            f.write(json.dumps(record) + "\n")

        from scripts.make_fox_tracking_overlay import make_tracking_overlay
        make_tracking_overlay(video_path, graphs_path, overlay_path, sample_fps=5.0)

        assert Path(overlay_path).exists()
        print("  PASS: overlay handles missing rejected graphs without crashing")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_all():
    print("[HyperBone Fox Synthetic Pipeline Tests]")
    test_fox_asset_path_resolution()
    test_proposal_bbox_inside_frame()
    test_proposal_jsonl_schema()
    test_manual_synthetic_works_without_dino()
    test_graph_record_includes_fox_label()
    test_graph_record_includes_hyperbone_custom()
    test_tracking_metrics_from_mock_sequence()
    test_overlay_handles_missing_rejected()
    print(f"\nAll 8 tests passed.")


if __name__ == "__main__":
    run_all()
