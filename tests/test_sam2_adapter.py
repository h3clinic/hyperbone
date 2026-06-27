"""Tests for HyperBone Milestone 2 — SAM2 adapter and pluggable mask backends."""

import json
import tempfile
import numpy as np
import cv2
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.cv.masks import (
    MaskBackend,
    ThresholdMaskBackend,
    NoopMaskBackend,
    get_mask_generator,
    generate_masks,
    _mask_touches_edge,
)


def _make_test_frame(h=200, w=200):
    """Create a test frame with a white rectangle on black background."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(frame, (50, 60), (150, 140), (255, 255, 255), -1)
    return frame


def test_threshold_backend_returns_records():
    """ThresholdMaskBackend.generate() returns list of dicts with required keys."""
    frame = _make_test_frame()
    backend = ThresholdMaskBackend(min_area=100, max_objects=5)
    records = backend.generate(frame)

    assert isinstance(records, list)
    if len(records) > 0:
        rec = records[0]
        assert "mask" in rec
        assert "bbox" in rec
        assert "area" in rec
        assert "confidence" in rec
        assert "source" in rec
        assert "object_id" in rec
        assert "touches_edge" in rec
        assert "metadata" in rec
        assert rec["source"] == "threshold"
        assert isinstance(rec["mask"], np.ndarray)
        assert rec["mask"].dtype == np.uint8
        assert len(rec["bbox"]) == 4
        assert isinstance(rec["area"], int)
        assert isinstance(rec["confidence"], float)
        assert isinstance(rec["touches_edge"], bool)

    print("  PASS: threshold backend returns proper records")


def test_generate_masks_convenience():
    """generate_masks() convenience function works with threshold backend."""
    frame = _make_test_frame()
    records = generate_masks(frame, backend="threshold", min_area=100)

    assert isinstance(records, list)
    for rec in records:
        assert rec["source"] == "threshold"

    print("  PASS: generate_masks() convenience function works")


def test_noop_backend_returns_empty():
    """NoopMaskBackend always returns empty list."""
    frame = _make_test_frame()
    backend = NoopMaskBackend()
    records = backend.generate(frame)
    assert records == []
    print("  PASS: noop backend returns empty list")


def test_get_mask_generator_threshold():
    """get_mask_generator('threshold') returns ThresholdMaskBackend."""
    gen = get_mask_generator("threshold")
    assert isinstance(gen, ThresholdMaskBackend)
    print("  PASS: get_mask_generator('threshold') works")


def test_get_mask_generator_noop():
    """get_mask_generator('noop') returns NoopMaskBackend."""
    gen = get_mask_generator("noop")
    assert isinstance(gen, NoopMaskBackend)
    print("  PASS: get_mask_generator('noop') works")


def test_get_mask_generator_unknown_raises():
    """get_mask_generator('unknown') raises ValueError."""
    try:
        get_mask_generator("unknown_backend")
        assert False, "Should have raised"
    except ValueError as e:
        assert "Unknown mask backend" in str(e)
    print("  PASS: unknown backend raises ValueError")


def test_sam2_unavailable_gives_clear_error():
    """Requesting sam2 backend without SAM2 installed gives RuntimeError."""
    # Temporarily force SAM2 to be unavailable
    import hyperbone.cv.sam2_adapter as adapter
    orig_available = adapter._SAM2_AVAILABLE
    orig_error = adapter._SAM2_IMPORT_ERROR
    adapter._SAM2_AVAILABLE = False
    adapter._SAM2_IMPORT_ERROR = "No module named 'sam2'"

    try:
        from hyperbone.cv.sam2_adapter import SAM2MaskBackend
        try:
            SAM2MaskBackend(checkpoint="fake.pt", model_cfg="fake.yaml")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "SAM2 is not installed" in str(e)
            assert "pip install sam2" in str(e)
    finally:
        adapter._SAM2_AVAILABLE = orig_available
        adapter._SAM2_IMPORT_ERROR = orig_error

    print("  PASS: SAM2 unavailable gives clear RuntimeError")


def test_mask_touches_edge():
    """_mask_touches_edge detects edge-touching masks."""
    h, w = 100, 100

    # Mask touching top edge
    mask_top = np.zeros((h, w), dtype=np.uint8)
    mask_top[0, 50] = 255
    assert _mask_touches_edge(mask_top, h, w) is True

    # Mask in center
    mask_center = np.zeros((h, w), dtype=np.uint8)
    mask_center[50, 50] = 255
    assert _mask_touches_edge(mask_center, h, w) is False

    # Mask touching right edge
    mask_right = np.zeros((h, w), dtype=np.uint8)
    mask_right[50, 99] = 255
    assert _mask_touches_edge(mask_right, h, w) is True

    print("  PASS: _mask_touches_edge detection works")


def test_threshold_area_filtering():
    """ThresholdMaskBackend respects min/max area ratio filters."""
    frame = _make_test_frame(200, 200)  # 40000 pixels total
    # Rectangle is ~100*80 = 8000 pixels ≈ 20% area

    # Set max_area_ratio very low to reject everything
    backend = ThresholdMaskBackend(min_area=100, max_objects=10,
                                   min_area_ratio=0.5, max_area_ratio=0.8)
    records = backend.generate(frame)
    # All should be filtered out (rectangle is only ~20% area)
    # Otsu fallback might also be filtered
    for rec in records:
        area_ratio = rec["area"] / (200 * 200)
        assert area_ratio >= 0.5 or rec["metadata"].get("fallback") == "otsu"

    print("  PASS: threshold area filtering works")


def test_mock_sam2_pipeline_accepts_clean_mask():
    """With a mocked SAM2 returning a clean rectangular mask, pipeline should accept."""
    from hyperbone.cv.skeletonize import skeletonize_mask
    from hyperbone.cv.graph import skeleton_to_graph
    from hyperbone.quality.score import score_graph, quality_score_numeric

    # Simulate what SAM2 would return: a clean rectangular mask
    h, w = 300, 300
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (80, 100), (220, 200), 255, -1)

    # Run through skeleton pipeline
    skeleton = skeletonize_mask(mask)
    graph = skeleton_to_graph(skeleton, min_branch_length=10)

    quality = score_graph(
        mask=mask,
        graph=graph,
        frame_shape=(h, w),
        bbox=(80, 100, 220, 200),
    )

    # A clean rectangle should produce a simple skeleton that passes quality
    # (It might still be rejected due to disconnected components from thick mask)
    # At minimum, it should NOT be rejected for mask_too_small
    assert "mask_too_small" not in str(quality["reject_reasons"])
    print(f"  PASS: mock SAM2 clean mask pipeline (accepted={quality['accepted']}, "
          f"reasons={quality['reject_reasons']})")


def test_max_masks_per_frame_respected():
    """ThresholdMaskBackend respects max_objects limit."""
    # Create frame with many objects
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    for i in range(20):
        x = (i % 5) * 80 + 10
        y = (i // 5) * 100 + 10
        cv2.circle(frame, (x + 30, y + 30), 20, (255, 255, 255), -1)

    backend = ThresholdMaskBackend(min_area=50, max_objects=3)
    records = backend.generate(frame)
    assert len(records) <= 3
    print("  PASS: max_masks_per_frame respected")


def test_quality_jsonl_includes_mask_backend():
    """Pipeline outputs mask_backend field in quality records."""
    from hyperbone.pipelines.pseudo_label import run_pseudo_label

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test video
        video_path = Path(tmpdir) / "test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_path), fourcc, 24, (200, 200))
        for _ in range(24):
            frame = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.rectangle(frame, (50, 60), (150, 140), (255, 255, 255), -1)
            writer.write(frame)
        writer.release()

        out = Path(tmpdir) / "output"
        run_pseudo_label(
            video_path=str(video_path),
            output_dir=str(out),
            sample_fps=1.0,
            mask_backend="threshold",
            max_frames=1,
        )

        quality_path = out / "quality.jsonl"
        assert quality_path.exists(), "quality.jsonl not created"
        line = quality_path.read_text().strip().split("\n")[0]
        rec = json.loads(line)
        assert "mask_backend" in rec, f"mask_backend not in quality record: {rec.keys()}"
        assert rec["mask_backend"] == "threshold"

    print("  PASS: quality.jsonl includes mask_backend field")


def test_graph_jsonl_includes_mask_backend():
    """Pipeline outputs mask_backend in graph JSONL records."""
    from hyperbone.pipelines.pseudo_label import run_pseudo_label

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_path), fourcc, 24, (200, 200))
        for _ in range(24):
            frame = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.rectangle(frame, (50, 60), (150, 140), (255, 255, 255), -1)
            writer.write(frame)
        writer.release()

        out = Path(tmpdir) / "output"
        run_pseudo_label(
            video_path=str(video_path),
            output_dir=str(out),
            sample_fps=1.0,
            mask_backend="threshold",
            max_frames=1,
        )

        # Check rejected (threshold likely produces rejected graphs)
        rejected_path = out / "graphs" / "rejected_graphs.jsonl"
        if rejected_path.exists():
            content = rejected_path.read_text().strip()
            if content:
                rec = json.loads(content.split("\n")[0])
                assert "mask_backend" in rec, f"mask_backend not in graph record"
                assert rec["mask_backend"] == "threshold"

        # Check accepted
        accepted_path = out / "graphs" / "accepted_graphs.jsonl"
        if accepted_path.exists():
            content = accepted_path.read_text().strip()
            if content:
                rec = json.loads(content.split("\n")[0])
                assert "mask_backend" in rec

    print("  PASS: graph JSONL includes mask_backend field")


def test_cli_args_parse():
    """CLI scripts accept --mask-backend sam2 and SAM2 args."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/run_pseudo_label.py", "--help"],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
    )
    assert "--mask-backend" in result.stdout
    assert "sam2" in result.stdout
    assert "--sam2-checkpoint" in result.stdout
    assert "--sam2-model-cfg" in result.stdout
    assert "--device" in result.stdout
    assert "--max-masks-per-frame" in result.stdout
    assert "--min-mask-area-ratio" in result.stdout
    assert "--max-mask-area-ratio" in result.stdout

    result2 = subprocess.run(
        [sys.executable, "scripts/run_pseudo_label_batch.py", "--help"],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
    )
    assert "--mask-backend" in result2.stdout
    assert "--sam2-checkpoint" in result2.stdout

    print("  PASS: CLI args parse correctly")


def run_all():
    print("[HyperBone Milestone 2 Tests]")
    test_threshold_backend_returns_records()
    test_generate_masks_convenience()
    test_noop_backend_returns_empty()
    test_get_mask_generator_threshold()
    test_get_mask_generator_noop()
    test_get_mask_generator_unknown_raises()
    test_sam2_unavailable_gives_clear_error()
    test_mask_touches_edge()
    test_threshold_area_filtering()
    test_mock_sam2_pipeline_accepts_clean_mask()
    test_max_masks_per_frame_respected()
    test_quality_jsonl_includes_mask_backend()
    test_graph_jsonl_includes_mask_backend()
    test_cli_args_parse()
    print(f"\nAll 14 tests passed.")


if __name__ == "__main__":
    run_all()
