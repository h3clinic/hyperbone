"""Tests for manual proposal loading."""
import sys, json, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.objects.manual_loader import load_proposals_jsonl, clip_proposal_to_frame
from hyperbone.objects.proposals import ObjectProposal


def test_load_valid_proposals():
    """Valid JSONL loads correctly."""
    fixture = Path(__file__).parent / "fixtures" / "manual_proposals.jsonl"
    proposals = load_proposals_jsonl(str(fixture))
    assert len(proposals) == 30, f"Expected 30 proposals, got {len(proposals)}"
    assert proposals[0].frame_idx == 14400
    assert proposals[0].label == "person"
    assert proposals[0].label_source == "manual"
    assert proposals[0].bbox_xywh == (200, 50, 180, 400)
    print("  PASS: valid JSONL loads correctly")


def test_missing_file_raises():
    """Missing file raises FileNotFoundError."""
    try:
        load_proposals_jsonl("nonexistent_file.jsonl")
        assert False, "Should have raised"
    except FileNotFoundError:
        pass
    print("  PASS: missing file raises FileNotFoundError")


def test_invalid_bbox_rejected():
    """Zero-dimension bbox raises ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"frame_idx": 0, "bbox_xywh": [10, 10, 0, 50]}) + "\n")
        f.flush()
        try:
            load_proposals_jsonl(f.name)
            assert False, "Should have raised ValueError for w=0"
        except ValueError as e:
            assert "positive" in str(e).lower() or "invalid" in str(e).lower()
    print("  PASS: invalid bbox (w=0) raises ValueError")


def test_negative_dimension_rejected():
    """Negative dimension bbox raises ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"frame_idx": 0, "bbox_xywh": [10, 10, -5, 50]}) + "\n")
        f.flush()
        try:
            load_proposals_jsonl(f.name)
            assert False, "Should have raised ValueError for w=-5"
        except ValueError as e:
            assert "positive" in str(e).lower() or "invalid" in str(e).lower()
    print("  PASS: negative bbox dimension raises ValueError")


def test_missing_frame_idx_rejected():
    """Missing frame_idx raises ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"bbox_xywh": [10, 10, 50, 50]}) + "\n")
        f.flush()
        try:
            load_proposals_jsonl(f.name)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "frame_idx" in str(e)
    print("  PASS: missing frame_idx raises ValueError")


def test_missing_bbox_rejected():
    """Missing bbox_xywh raises ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"frame_idx": 0}) + "\n")
        f.flush()
        try:
            load_proposals_jsonl(f.name)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "bbox_xywh" in str(e)
    print("  PASS: missing bbox_xywh raises ValueError")


def test_clip_inside_frame():
    """Proposal inside frame stays unchanged."""
    p = ObjectProposal.manual(frame_idx=0, object_id=0, bbox_xywh=(50, 50, 100, 100))
    clipped = clip_proposal_to_frame(p, 640, 480)
    assert clipped is not None
    assert clipped.bbox_xywh == (50, 50, 100, 100)
    print("  PASS: inside-frame proposal unchanged")


def test_clip_outside_frame():
    """Proposal partially outside frame is clipped."""
    p = ObjectProposal.manual(frame_idx=0, object_id=0, bbox_xywh=(600, 400, 200, 200))
    clipped = clip_proposal_to_frame(p, 640, 480)
    assert clipped is not None
    x, y, w, h = clipped.bbox_xywh
    assert x + w <= 640, f"Clipped extends beyond frame width: x={x}, w={w}"
    assert y + h <= 480, f"Clipped extends beyond frame height: y={y}, h={h}"
    print("  PASS: outside-frame proposal clipped safely")


def test_clip_too_small_returns_none():
    """Proposal that clips to < 4px returns None."""
    p = ObjectProposal.manual(frame_idx=0, object_id=0, bbox_xywh=(638, 478, 100, 100))
    clipped = clip_proposal_to_frame(p, 640, 480)
    # Only 2x2 remains after clip
    assert clipped is None
    print("  PASS: too-small clipped proposal returns None")


def test_optional_fields_default():
    """Proposals with missing optional fields get defaults."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"frame_idx": 5, "bbox_xywh": [10, 10, 50, 50]}) + "\n")
        f.flush()
        proposals = load_proposals_jsonl(f.name)
    assert len(proposals) == 1
    p = proposals[0]
    assert p.label == "unknown"
    assert p.label_confidence == 1.0
    assert p.label_source == "manual"
    print("  PASS: optional fields default correctly")


def test_comments_and_blank_lines_skipped():
    """Comments and blank lines in JSONL are skipped."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("# This is a comment\n")
        f.write("\n")
        f.write(json.dumps({"frame_idx": 0, "bbox_xywh": [10, 10, 50, 50]}) + "\n")
        f.write("\n")
        f.flush()
        proposals = load_proposals_jsonl(f.name)
    assert len(proposals) == 1
    print("  PASS: comments and blank lines skipped")


def run_all():
    print("[HyperBone Manual Proposal Tests]")
    test_load_valid_proposals()
    test_missing_file_raises()
    test_invalid_bbox_rejected()
    test_negative_dimension_rejected()
    test_missing_frame_idx_rejected()
    test_missing_bbox_rejected()
    test_clip_inside_frame()
    test_clip_outside_frame()
    test_clip_too_small_returns_none()
    test_optional_fields_default()
    test_comments_and_blank_lines_skipped()
    print(f"\nAll 11 tests passed.")


if __name__ == "__main__":
    run_all()
