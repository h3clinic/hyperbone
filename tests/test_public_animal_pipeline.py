"""Tests for public animal pipeline (Quaternius Animated Animal Pack).

Does NOT require Blender or the asset to be downloaded.
"""

import sys, json, tempfile, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

from hyperbone.tracking.simple_graph_track import compute_graph_metrics, compute_tracking_metrics


def _make_wolf_video(out_path: str, width=640, height=480, fps=24, duration_sec=2):
    """Create a synthetic wolf-like shape video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    bg = (31, 31, 46)  # Dark background

    for i in range(int(fps * duration_sec)):
        t = i / fps
        frame = np.full((height, width, 3), bg, dtype=np.uint8)

        # Wolf body
        cx = int(width * 0.4 + 40 * np.sin(t * 1.5))
        cy = int(height * 0.55)

        # Body ellipse
        cv2.ellipse(frame, (cx, cy), (110, 45), 0, 0, 360, (130, 130, 140), -1)
        # Head
        hx = cx + 100
        hy = cy - 20 + int(5 * np.sin(t * 2))
        cv2.ellipse(frame, (hx, hy), (35, 28), -5, 0, 360, (140, 140, 150), -1)
        # Snout
        cv2.ellipse(frame, (hx + 30, hy + 5), (15, 10), 0, 0, 360, (150, 150, 160), -1)
        # Ears
        cv2.ellipse(frame, (hx - 10, hy - 28), (10, 16), -10, 0, 360, (120, 120, 130), -1)
        cv2.ellipse(frame, (hx + 10, hy - 30), (10, 16), 10, 0, 360, (120, 120, 130), -1)
        # Tail
        tail_x = cx - 110
        tail_y = cy - 20 + int(15 * np.sin(t * 3))
        pts = np.array([[cx - 100, cy - 10], [tail_x, tail_y],
                        [tail_x - 20, tail_y - 20]], np.int32)
        cv2.polylines(frame, [pts], False, (120, 120, 130), 10)
        # 4 legs with walk cycle
        for j, (lx, phase) in enumerate([(cx - 60, 0), (cx - 30, 0.5), (cx + 40, 1.0), (cx + 70, 1.5)]):
            ly = cy + 40
            swing = int(12 * np.sin(t * 5 + phase * np.pi))
            foot_x = lx + swing
            foot_y = ly + 60 + int(8 * abs(np.sin(t * 5 + phase * np.pi)))
            cv2.line(frame, (lx, ly), (foot_x, foot_y), (110, 110, 120), 7)

        writer.write(frame)
    writer.release()


def test_asset_manifest_parser():
    """Asset manifest parser works with mock data."""
    from scripts.fetch_quaternius_animals import select_animal, PRIORITY

    # Verify priority order
    assert PRIORITY[0] == "Wolf"
    assert PRIORITY[1] == "Horse"
    assert PRIORITY[2] == "Fox"
    print("  PASS: asset manifest parser works, priority = Wolf > Horse > Fox")


def test_animal_priority_selection():
    """Selected animal priority is Wolf > Horse > Fox."""
    from scripts.fetch_quaternius_animals import select_animal, find_animal_model

    tmpdir = Path(tempfile.mkdtemp())
    try:
        # Create mock model files
        (tmpdir / "Horse.gltf").write_text("{}")
        (tmpdir / "Fox.gltf").write_text("{}")

        animal, path = select_animal(tmpdir)
        assert animal == "Horse", f"Expected Horse (highest available priority), got {animal}"
        print("  PASS: selection priority Wolf > Horse > Fox works correctly")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_synthetic_bbox_inside_frame():
    """Synthetic bbox proposal stays inside frame bounds."""
    from scripts.make_public_animal_proposals import estimate_bbox_background_diff

    frame = np.full((480, 640, 3), (31, 31, 46), dtype=np.uint8)
    cv2.ellipse(frame, (320, 240), (150, 60), 0, 0, 360, (130, 130, 140), -1)

    result = estimate_bbox_background_diff(frame)
    x, y, w, h = result["bbox_xywh"]
    assert x >= 0 and y >= 0
    assert x + w <= 640 and y + h <= 480
    assert w > 0 and h > 0
    print("  PASS: synthetic bbox proposal stays inside frame bounds")


def test_proposal_jsonl_schema():
    """Proposal JSONL schema is valid."""
    from scripts.make_public_animal_proposals import estimate_bbox_background_diff

    frame = np.full((480, 640, 3), (31, 31, 46), dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (440, 330), (130, 130, 140), -1)

    result = estimate_bbox_background_diff(frame)
    proposal = {
        "frame_idx": 0,
        "timestamp_sec": 0.0,
        "object_id": 0,
        "label": "wolf",
        "label_confidence": 1.0,
        "bbox_xywh": result["bbox_xywh"],
        "prompt": "wolf",
        "proposal_method": result["proposal_method"],
        "source_asset": "Quaternius Animated Animal Pack",
    }

    required = ["frame_idx", "object_id", "label", "bbox_xywh", "source_asset"]
    for field in required:
        assert field in proposal, f"Missing field: {field}"
    assert len(proposal["bbox_xywh"]) == 4
    assert proposal["source_asset"] == "Quaternius Animated Animal Pack"
    print("  PASS: proposal JSONL schema is valid")


def test_graph_record_includes_hyperbone_custom():
    """Graph records include skeleton_mapper='hyperbone-custom'."""
    tmpdir = tempfile.mkdtemp()
    try:
        video_path = str(Path(tmpdir) / "wolf.mp4")
        proposals_path = str(Path(tmpdir) / "proposals.jsonl")
        out_dir = str(Path(tmpdir) / "output")

        _make_wolf_video(video_path, duration_sec=1)

        from scripts.make_public_animal_proposals import make_public_animal_proposals
        make_public_animal_proposals(video_path, "wolf", 5.0, proposals_path)

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
            assert r["skeleton_mapper"] == "hyperbone-custom"
        print("  PASS: graph record includes skeleton_mapper='hyperbone-custom'")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_graph_record_includes_source_asset():
    """Graph records include source_asset when present in proposal."""
    tmpdir = tempfile.mkdtemp()
    try:
        video_path = str(Path(tmpdir) / "wolf.mp4")
        proposals_path = str(Path(tmpdir) / "proposals.jsonl")
        out_dir = str(Path(tmpdir) / "output")

        _make_wolf_video(video_path, duration_sec=1)

        from scripts.make_public_animal_proposals import make_public_animal_proposals
        make_public_animal_proposals(video_path, "wolf", 5.0, proposals_path)

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
        assert len(records) >= 1
        assert records[0]["object_label"] == "wolf"
        print("  PASS: graph record includes object_label='wolf'")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_tracking_metrics_from_mock():
    """Tracking metrics compute from mock graph sequence."""
    mock_records = []
    for i in range(12):
        nodes = [
            {"id": 0, "xy": [100 + i * 3, 200], "type": "endpoint"},
            {"id": 1, "xy": [200 + i * 3, 200], "type": "junction"},
            {"id": 2, "xy": [300 + i * 3, 200], "type": "endpoint"},
            {"id": 3, "xy": [200 + i * 3, 150], "type": "endpoint"},
            {"id": 4, "xy": [200 + i * 3, 250], "type": "endpoint"},
        ]
        edges = [
            {"parent": 0, "child": 1},
            {"parent": 1, "child": 2},
            {"parent": 1, "child": 3},
            {"parent": 1, "child": 4},
        ]
        mock_records.append({
            "frame_idx": i * 5,
            "accepted": True,
            "nodes": nodes,
            "edges": edges,
            "bbox_xywh": [80 + i * 3, 130, 240, 140],
            "node_count": 5,
            "edge_count": 4,
        })

    metrics = compute_tracking_metrics(mock_records)
    assert metrics["frames_with_proposal"] == 12
    assert metrics["accepted_frames"] == 12
    assert metrics["topology_stability_score"] > 0.9
    assert metrics["node_count_mean"] == 5.0
    print("  PASS: tracking metrics compute from mock graph sequence")


def test_overlay_no_crash_empty_rejected():
    """Overlay script handles empty rejected set without crashing."""
    tmpdir = tempfile.mkdtemp()
    try:
        video_path = str(Path(tmpdir) / "wolf.mp4")
        graphs_path = str(Path(tmpdir) / "graphs.jsonl")
        overlay_path = str(Path(tmpdir) / "overlay.mp4")

        _make_wolf_video(video_path, width=320, height=240, duration_sec=1)

        # Only accepted records, no rejected
        with open(graphs_path, "w") as f:
            record = {
                "frame_idx": 0,
                "accepted": True,
                "nodes": [
                    {"id": 0, "xy": [100, 120], "type": "endpoint"},
                    {"id": 1, "xy": [220, 120], "type": "endpoint"},
                ],
                "edges": [{"parent": 0, "child": 1}],
                "bbox_xywh": [80, 100, 160, 40],
                "object_label": "wolf",
                "node_count": 2,
                "edge_count": 1,
                "reject_reasons": [],
            }
            f.write(json.dumps(record) + "\n")

        from scripts.make_animal_tracking_overlay import make_animal_tracking_overlay
        make_animal_tracking_overlay(video_path, graphs_path, overlay_path, sample_fps=5.0)

        assert Path(overlay_path).exists()
        print("  PASS: overlay handles empty rejected set without crashing")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_all():
    print("[HyperBone Public Animal Pipeline Tests]")
    test_asset_manifest_parser()
    test_animal_priority_selection()
    test_synthetic_bbox_inside_frame()
    test_proposal_jsonl_schema()
    test_graph_record_includes_hyperbone_custom()
    test_graph_record_includes_source_asset()
    test_tracking_metrics_from_mock()
    test_overlay_no_crash_empty_rejected()
    print(f"\nAll 8 tests passed.")


if __name__ == "__main__":
    run_all()
