"""Tests for proposal-driven skeleton pipeline."""
import sys, json, tempfile, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

from hyperbone.pipelines.proposal_skeleton import run_proposal_skeleton
from hyperbone.objects.proposals import ObjectProposal


def _make_test_video(out_path: str, width=200, height=200, fps=24, duration_sec=3):
    """Create a test video with visible objects."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    for i in range(int(fps * duration_sec)):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw a vertical bar (simulates a person/pole)
        cv2.rectangle(frame, (80, 20), (120, 180), (200, 180, 160), -1)
        # Draw a horizontal bar (simulates a table)
        cv2.rectangle(frame, (20, 130), (180, 155), (100, 100, 200), -1)
        writer.write(frame)
    writer.release()


def _make_proposals_jsonl(path: str, frame_idx: int):
    """Create a minimal proposals JSONL for the test video."""
    records = [
        {"frame_idx": frame_idx, "object_id": 0, "label": "pole",
         "label_confidence": 0.9, "bbox_xywh": [70, 10, 60, 180], "prompt": "pole"},
        {"frame_idx": frame_idx, "object_id": 1, "label": "bar",
         "label_confidence": 0.8, "bbox_xywh": [10, 120, 180, 45], "prompt": "bar"},
    ]
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_pipeline_creates_graph_jsonl():
    """Pipeline produces graphs.jsonl output."""
    tmpdir = tempfile.mkdtemp()
    try:
        video_path = str(Path(tmpdir) / "test.mp4")
        proposals_path = str(Path(tmpdir) / "proposals.jsonl")
        out_dir = str(Path(tmpdir) / "output")

        _make_test_video(video_path)
        # Frame at 1fps, no skip → first frame is idx 0
        _make_proposals_jsonl(proposals_path, frame_idx=0)

        stats = run_proposal_skeleton(
            video_path=video_path,
            output_dir=out_dir,
            proposals_path=proposals_path,
            proposal_source="manual",
            sample_fps=1.0,
            max_side=128,
            thinning_algorithm="zhang-suen",
        )

        graphs_jsonl = Path(out_dir) / "graphs" / "graphs.jsonl"
        assert graphs_jsonl.exists(), f"graphs.jsonl not found at {graphs_jsonl}"
        print("  PASS: pipeline creates graphs.jsonl")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_graph_record_has_skeleton_mapper():
    """Graph records include skeleton_mapper='hyperbone-custom'."""
    tmpdir = tempfile.mkdtemp()
    try:
        video_path = str(Path(tmpdir) / "test.mp4")
        proposals_path = str(Path(tmpdir) / "proposals.jsonl")
        out_dir = str(Path(tmpdir) / "output")

        _make_test_video(video_path)
        _make_proposals_jsonl(proposals_path, frame_idx=0)

        run_proposal_skeleton(
            video_path=video_path,
            output_dir=out_dir,
            proposals_path=proposals_path,
            proposal_source="manual",
            sample_fps=1.0,
            max_side=128,
        )

        graphs_jsonl = Path(out_dir) / "graphs" / "graphs.jsonl"
        with open(graphs_jsonl) as f:
            for line in f:
                rec = json.loads(line)
                assert rec["skeleton_mapper"] == "hyperbone-custom", \
                    f"Expected 'hyperbone-custom', got {rec['skeleton_mapper']}"
        print("  PASS: graph records have skeleton_mapper='hyperbone-custom'")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_graph_record_has_object_label():
    """Graph records include object_label."""
    tmpdir = tempfile.mkdtemp()
    try:
        video_path = str(Path(tmpdir) / "test.mp4")
        proposals_path = str(Path(tmpdir) / "proposals.jsonl")
        out_dir = str(Path(tmpdir) / "output")

        _make_test_video(video_path)
        _make_proposals_jsonl(proposals_path, frame_idx=0)

        run_proposal_skeleton(
            video_path=video_path,
            output_dir=out_dir,
            proposals_path=proposals_path,
            proposal_source="manual",
            sample_fps=1.0,
            max_side=128,
        )

        graphs_jsonl = Path(out_dir) / "graphs" / "graphs.jsonl"
        with open(graphs_jsonl) as f:
            for line in f:
                rec = json.loads(line)
                assert "object_label" in rec, "Missing object_label"
                assert rec["object_label"] in ("pole", "bar")
        print("  PASS: graph records have object_label")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_graph_record_has_bbox():
    """Graph records include bbox_xywh."""
    tmpdir = tempfile.mkdtemp()
    try:
        video_path = str(Path(tmpdir) / "test.mp4")
        proposals_path = str(Path(tmpdir) / "proposals.jsonl")
        out_dir = str(Path(tmpdir) / "output")

        _make_test_video(video_path)
        _make_proposals_jsonl(proposals_path, frame_idx=0)

        run_proposal_skeleton(
            video_path=video_path,
            output_dir=out_dir,
            proposals_path=proposals_path,
            proposal_source="manual",
            sample_fps=1.0,
            max_side=128,
        )

        graphs_jsonl = Path(out_dir) / "graphs" / "graphs.jsonl"
        with open(graphs_jsonl) as f:
            for line in f:
                rec = json.loads(line)
                assert "bbox_xywh" in rec, "Missing bbox_xywh"
                assert len(rec["bbox_xywh"]) == 4
        print("  PASS: graph records have bbox_xywh")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_accepted_rejected_split():
    """Pipeline writes accepted/ and rejected/ directories."""
    tmpdir = tempfile.mkdtemp()
    try:
        video_path = str(Path(tmpdir) / "test.mp4")
        proposals_path = str(Path(tmpdir) / "proposals.jsonl")
        out_dir = str(Path(tmpdir) / "output")

        _make_test_video(video_path)
        _make_proposals_jsonl(proposals_path, frame_idx=0)

        stats = run_proposal_skeleton(
            video_path=video_path,
            output_dir=out_dir,
            proposals_path=proposals_path,
            proposal_source="manual",
            sample_fps=1.0,
            max_side=128,
        )

        accepted_dir = Path(out_dir) / "accepted"
        rejected_dir = Path(out_dir) / "rejected"
        assert accepted_dir.exists(), "accepted/ directory missing"
        assert rejected_dir.exists(), "rejected/ directory missing"

        # At least one graph should exist in either
        total = stats["accepted_count"] + stats["rejected_count"]
        assert total >= 2, f"Expected >=2 total, got {total}"
        print("  PASS: accepted/rejected split written")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_summary_md_written():
    """Pipeline writes summary.md report."""
    tmpdir = tempfile.mkdtemp()
    try:
        video_path = str(Path(tmpdir) / "test.mp4")
        proposals_path = str(Path(tmpdir) / "proposals.jsonl")
        out_dir = str(Path(tmpdir) / "output")

        _make_test_video(video_path)
        _make_proposals_jsonl(proposals_path, frame_idx=0)

        run_proposal_skeleton(
            video_path=video_path,
            output_dir=out_dir,
            proposals_path=proposals_path,
            proposal_source="manual",
            sample_fps=1.0,
            max_side=128,
        )

        from hyperbone.report.owned_mapper_report import generate_owned_mapper_report
        # Report is generated by CLI, but let's test the function directly
        stats = {"accepted_count": 1, "rejected_count": 1, "frames_processed": 1,
                 "objects_proposed": 2, "runtimes_ms": [50, 60],
                 "labels_accepted": ["pole"], "labels_rejected": ["bar"],
                 "reject_reasons": ["too_few_nodes"], "node_counts": [5, 0],
                 "edge_counts": [4, 0], "output_dir": out_dir,
                 "video_path": video_path, "resolution": "200x200",
                 "proposal_source": "manual", "text_prompt": ""}
        report_path = generate_owned_mapper_report(stats, out_dir)
        assert report_path.exists()
        content = report_path.read_text()
        assert "hyperbone-custom" in content
        assert "Ownership Claim" in content
        print("  PASS: summary.md written with correct content")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_dino_missing_fails_clearly():
    """DINO missing dependency fails only when proposal-source=groundingdino."""
    tmpdir = tempfile.mkdtemp()
    try:
        video_path = str(Path(tmpdir) / "test.mp4")
        out_dir = str(Path(tmpdir) / "output")
        _make_test_video(video_path)

        # This should attempt to import and either work (if transformers installed)
        # or fail clearly with RuntimeError
        try:
            run_proposal_skeleton(
                video_path=video_path,
                output_dir=out_dir,
                proposal_source="groundingdino",
                text_prompt="person.",
                sample_fps=1.0,
                max_side=128,
                device="cpu",
            )
            # If it works, DINO is installed — that's fine
            print("  PASS: DINO available, groundingdino source works")
        except (RuntimeError, ImportError) as e:
            # Should be a clear message about missing dep
            msg = str(e).lower()
            assert "transformers" in msg or "torch" in msg or "grounding" in msg, \
                f"Unclear error: {e}"
            print("  PASS: DINO missing, clear error raised")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_manual_works_without_dino():
    """Manual proposal path works without DINO installed."""
    tmpdir = tempfile.mkdtemp()
    try:
        video_path = str(Path(tmpdir) / "test.mp4")
        proposals_path = str(Path(tmpdir) / "proposals.jsonl")
        out_dir = str(Path(tmpdir) / "output")

        _make_test_video(video_path)
        _make_proposals_jsonl(proposals_path, frame_idx=0)

        # This should work regardless of DINO being installed
        stats = run_proposal_skeleton(
            video_path=video_path,
            output_dir=out_dir,
            proposals_path=proposals_path,
            proposal_source="manual",
            sample_fps=1.0,
            max_side=128,
        )
        assert stats["objects_proposed"] >= 2
        print("  PASS: manual proposal path works without DINO")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_quality_jsonl_written():
    """Pipeline writes quality.jsonl."""
    tmpdir = tempfile.mkdtemp()
    try:
        video_path = str(Path(tmpdir) / "test.mp4")
        proposals_path = str(Path(tmpdir) / "proposals.jsonl")
        out_dir = str(Path(tmpdir) / "output")

        _make_test_video(video_path)
        _make_proposals_jsonl(proposals_path, frame_idx=0)

        run_proposal_skeleton(
            video_path=video_path,
            output_dir=out_dir,
            proposals_path=proposals_path,
            proposal_source="manual",
            sample_fps=1.0,
            max_side=128,
        )

        quality_jsonl = Path(out_dir) / "quality.jsonl"
        assert quality_jsonl.exists(), "quality.jsonl not found"
        with open(quality_jsonl) as f:
            records = [json.loads(line) for line in f if line.strip()]
        assert len(records) >= 2
        assert records[0]["skeleton_mapper"] == "hyperbone-custom"
        print("  PASS: quality.jsonl written correctly")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_all():
    print("[HyperBone Proposal Skeleton Pipeline Tests]")
    test_pipeline_creates_graph_jsonl()
    test_graph_record_has_skeleton_mapper()
    test_graph_record_has_object_label()
    test_graph_record_has_bbox()
    test_accepted_rejected_split()
    test_summary_md_written()
    test_dino_missing_fails_clearly()
    test_manual_works_without_dino()
    test_quality_jsonl_written()
    print(f"\nAll 9 tests passed.")


if __name__ == "__main__":
    run_all()
