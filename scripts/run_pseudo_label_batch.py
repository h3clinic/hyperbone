"""Batch pseudo-label runner — process a folder of videos."""

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.pipelines.pseudo_label import run_pseudo_label
from hyperbone.io.dataset_index import DatasetIndexWriter, build_index_row
from hyperbone.report.summary import generate_summary


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def find_videos(video_dir: str, max_videos: int = None):
    """Recursively find video files in a directory."""
    videos = []
    for path in sorted(Path(video_dir).rglob("*")):
        if path.suffix.lower() in VIDEO_EXTENSIONS and path.is_file():
            videos.append(path)
            if max_videos and len(videos) >= max_videos:
                break
    return videos


def run_batch(
    video_dir: str,
    output_dir: str,
    sample_fps: float = 1.0,
    mask_backend: str = "threshold",
    min_branch_length: int = 10,
    max_videos: int = None,
    max_frames_per_video: int = None,
    mask_backend_kwargs: dict = None,
    enable_mask_cleanup: bool = None,
    enable_graph_repair: bool = None,
    bridge_gap_px: float = 12.0,
    cleanup_close_kernel: int = 5,
    keep_largest_component: bool = True,
    node_merge_radius: float = 4.0,
    spur_prune_length: int = 8,
    skip_start_sec: float = 0.0,
    skip_end_sec: float = 0.0,
):
    """Process all videos in a directory, writing a unified batch output."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    videos = find_videos(video_dir, max_videos)
    print(f"[HyperBone Batch] Found {len(videos)} videos in {video_dir}")
    print(f"[HyperBone Batch] Output: {out}")
    print()

    # Batch-level accumulators
    batch_stats = {
        "videos_processed": 0,
        "videos_failed": 0,
        "frames_sampled": 0,
        "objects_detected": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "reject_reasons": [],
        "node_counts": [],
        "edge_counts": [],
        "errors": [],
        "output_dir": str(out),
    }

    # Batch-level writers
    accepted_path = out / "accepted_graphs.jsonl"
    rejected_path = out / "rejected_graphs.jsonl"
    accepted_f = open(accepted_path, "w", encoding="utf-8")
    rejected_f = open(rejected_path, "w", encoding="utf-8")

    index_writer = DatasetIndexWriter(str(out), "dataset_index.jsonl")

    video_manifests = []

    for i, video_path in enumerate(videos):
        video_name = video_path.stem
        video_out = out / "videos" / video_name

        print(f"━━━ [{i+1}/{len(videos)}] {video_path.name} ━━━")

        try:
            stats = run_pseudo_label(
                video_path=str(video_path),
                output_dir=str(video_out),
                sample_fps=sample_fps,
                mask_backend=mask_backend,
                min_branch_length=min_branch_length,
                max_frames=max_frames_per_video,
                save_frames=True,
                save_masks=True,
                save_overlays=True,
                mask_backend_kwargs=mask_backend_kwargs,
                enable_mask_cleanup=enable_mask_cleanup,
                enable_graph_repair=enable_graph_repair,
                bridge_gap_px=bridge_gap_px,
                cleanup_close_kernel=cleanup_close_kernel,
                keep_largest_component=keep_largest_component,
                node_merge_radius=node_merge_radius,
                spur_prune_length=spur_prune_length,
                skip_start_sec=skip_start_sec,
                skip_end_sec=skip_end_sec,
            )

            batch_stats["videos_processed"] += 1
            batch_stats["frames_sampled"] += stats["frames_processed"]
            batch_stats["objects_detected"] += stats["objects_found"]
            batch_stats["accepted_count"] += stats["accepted_count"]
            batch_stats["rejected_count"] += stats["rejected_count"]
            batch_stats["reject_reasons"].extend(stats["reject_reasons"])
            batch_stats["node_counts"].extend(stats["node_counts"])
            batch_stats["edge_counts"].extend(stats["edge_counts"])

            # Copy accepted/rejected to batch-level files
            accepted_file = video_out / "graphs" / "accepted_graphs.jsonl"
            rejected_file = video_out / "graphs" / "rejected_graphs.jsonl"

            if accepted_file.exists():
                for line in accepted_file.read_text(encoding="utf-8").strip().split("\n"):
                    if line:
                        accepted_f.write(line + "\n")
                        # Write index row
                        rec = json.loads(line)
                        row = build_index_row(
                            video_id=rec.get("video_id", ""),
                            frame_idx=rec.get("frame_idx", 0),
                            timestamp_sec=rec.get("timestamp_sec", 0),
                            object_id=rec.get("object_id", 0),
                            frame_path=str(video_out / "frames" / f"frame_{rec.get('frame_idx', 0):06d}.png"),
                            mask_path=str(video_out / "masks" / f"mask_{rec.get('frame_idx', 0):06d}_obj{rec.get('object_id', 0):03d}.png"),
                            overlay_path=str(video_out / "overlays" / f"overlay_{rec.get('frame_idx', 0):06d}_obj{rec.get('object_id', 0):03d}.png"),
                            graph_path=str(accepted_file),
                            npz_path=str(video_out / "npz" / "skeleton_data.npz"),
                            quality_score=rec.get("quality", {}).get("quality_score", 0),
                        )
                        index_writer.write(row)

            if rejected_file.exists():
                for line in rejected_file.read_text(encoding="utf-8").strip().split("\n"):
                    if line:
                        rejected_f.write(line + "\n")

            video_manifests.append({
                "video": str(video_path),
                "output": str(video_out),
                "status": "ok",
                "stats": {
                    "frames": stats["frames_processed"],
                    "accepted": stats["accepted_count"],
                    "rejected": stats["rejected_count"],
                },
            })

        except Exception as e:
            batch_stats["videos_failed"] += 1
            err_msg = f"{video_path.name}: {type(e).__name__}: {e}"
            batch_stats["errors"].append(err_msg)
            print(f"  ERROR: {err_msg}")
            traceback.print_exc()
            video_manifests.append({
                "video": str(video_path),
                "status": "error",
                "error": err_msg,
            })

        print()

    # Close batch writers
    accepted_f.close()
    rejected_f.close()
    index_writer.close()

    # Batch manifest
    batch_manifest = {
        "video_dir": str(video_dir),
        "output_dir": str(out),
        "videos": video_manifests,
        "stats": {
            "videos_processed": batch_stats["videos_processed"],
            "videos_failed": batch_stats["videos_failed"],
            "frames_sampled": batch_stats["frames_sampled"],
            "objects_detected": batch_stats["objects_detected"],
            "accepted_count": batch_stats["accepted_count"],
            "rejected_count": batch_stats["rejected_count"],
        },
    }
    (out / "batch_manifest.json").write_text(
        json.dumps(batch_manifest, indent=2, default=str), encoding="utf-8"
    )

    # Summary report
    summary_path = generate_summary(batch_stats, str(out))
    print(f"[HyperBone Batch] Summary: {summary_path}")

    # Final stats
    total = batch_stats["accepted_count"] + batch_stats["rejected_count"]
    rate = batch_stats["accepted_count"] / total * 100 if total > 0 else 0
    print()
    print(f"[HyperBone Batch] COMPLETE")
    print(f"  Videos: {batch_stats['videos_processed']} ok, {batch_stats['videos_failed']} failed")
    print(f"  Graphs: {batch_stats['accepted_count']} accepted, {batch_stats['rejected_count']} rejected")
    print(f"  Acceptance rate: {rate:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="HyperBone Batch Pseudo-Label Factory"
    )
    parser.add_argument(
        "--video-dir", required=True,
        help="Directory containing video files (searched recursively)"
    )
    parser.add_argument(
        "--out", default="outputs/batch_run",
        help="Output directory (default: outputs/batch_run)"
    )
    parser.add_argument(
        "--sample-fps", type=float, default=1.0,
        help="Frame sampling rate in FPS (default: 1.0)"
    )
    parser.add_argument(
        "--mask-backend", default="threshold",
        choices=["threshold", "sam2", "grounded_sam2", "noop"],
        help="Mask generation backend (default: threshold)"
    )
    parser.add_argument(
        "--min-branch", type=int, default=10,
        help="Minimum skeleton branch length in pixels (default: 10)"
    )
    parser.add_argument(
        "--max-videos", type=int, default=None,
        help="Maximum number of videos to process"
    )
    parser.add_argument(
        "--max-frames-per-video", type=int, default=None,
        help="Maximum frames per video"
    )
    parser.add_argument(
        "--sam2-checkpoint", default="checkpoints/sam2.1_hiera_tiny.pt",
        help="Path to SAM2 checkpoint (used with --mask-backend sam2)"
    )
    parser.add_argument(
        "--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml",
        help="Path to SAM2 model config (used with --mask-backend sam2)"
    )
    parser.add_argument(
        "--device", default="cuda",
        choices=["cuda", "cpu"],
        help="Device for SAM2 inference (default: cuda)"
    )
    parser.add_argument(
        "--max-masks-per-frame", type=int, default=10,
        help="Maximum masks per frame (default: 10)"
    )
    parser.add_argument(
        "--min-mask-area-ratio", type=float, default=0.002,
        help="Minimum mask area as ratio of frame area (default: 0.002)"
    )
    parser.add_argument(
        "--max-mask-area-ratio", type=float, default=0.80,
        help="Maximum mask area as ratio of frame area (default: 0.80)"
    )
    parser.add_argument(
        "--mask-cleanup-mode", default="auto",
        choices=["off", "auto", "light", "medium"],
        help="Mask cleanup mode: off=disabled, auto=backend-dependent, light/medium=explicit (default: auto)"
    )
    parser.add_argument(
        "--graph-repair-mode", default="auto",
        choices=["off", "auto", "on"],
        help="Graph repair mode: off=disabled, auto=backend-dependent, on=always (default: auto)"
    )
    # Legacy aliases (kept for backward compat)
    parser.add_argument(
        "--enable-mask-cleanup", "--mask-cleanup", action="store_true", default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--enable-graph-repair", "--repair-graph", action="store_true", default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--bridge-gap-px", type=float, default=12.0,
        help="Max pixel gap to bridge between graph components (default: 12)"
    )
    parser.add_argument(
        "--node-merge-radius", type=float, default=4.0,
        help="Merge nodes closer than this distance in pixels (default: 4)"
    )
    parser.add_argument(
        "--spur-prune-length", type=int, default=8,
        help="Prune endpoint branches shorter than this in px (default: 8)"
    )
    parser.add_argument(
        "--cleanup-close-kernel", type=int, default=5,
        help="Morphological closing kernel size for mask cleanup (default: 5)"
    )
    parser.add_argument(
        "--keep-largest-component", action="store_true", default=True,
        help="Keep only the largest mask component during cleanup"
    )
    parser.add_argument(
        "--skip-start", type=float, default=0.0,
        help="Skip this many seconds from the start of each video (default: 0)"
    )
    parser.add_argument(
        "--skip-end", type=float, default=0.0,
        help="Skip this many seconds from the end of each video (default: 0)"
    )
    parser.add_argument(
        "--text-prompt",
        default="person. hand. arm. leg. head. torso. body. face. foot. knee. elbow. shoulder.",
        help="Text prompt for grounded_sam2 backend (object classes to detect)"
    )
    parser.add_argument(
        "--grounding-model", default="IDEA-Research/grounding-dino-tiny",
        help="HuggingFace model ID for Grounding DINO (used with grounded_sam2 backend)"
    )

    args = parser.parse_args()

    if not Path(args.video_dir).exists():
        print(f"Error: Directory not found: {args.video_dir}", file=sys.stderr)
        sys.exit(1)

    # Build mask backend kwargs
    mask_backend_kwargs = {
        "min_area_ratio": args.min_mask_area_ratio,
        "max_area_ratio": args.max_mask_area_ratio,
    }
    if args.mask_backend == "sam2":
        mask_backend_kwargs["checkpoint"] = args.sam2_checkpoint
        mask_backend_kwargs["model_cfg"] = args.sam2_model_cfg
        mask_backend_kwargs["device"] = args.device
        mask_backend_kwargs["max_masks_per_frame"] = args.max_masks_per_frame
    elif args.mask_backend == "grounded_sam2":
        mask_backend_kwargs["checkpoint"] = args.sam2_checkpoint
        mask_backend_kwargs["model_cfg"] = args.sam2_model_cfg
        mask_backend_kwargs["device"] = args.device
        mask_backend_kwargs["max_masks_per_frame"] = args.max_masks_per_frame
        mask_backend_kwargs["text_prompt"] = args.text_prompt
        mask_backend_kwargs["grounding_model"] = args.grounding_model
    else:
        mask_backend_kwargs["max_objects"] = args.max_masks_per_frame

    # Resolve cleanup/repair modes
    # New --mask-cleanup-mode / --graph-repair-mode take priority over legacy flags
    if args.mask_cleanup_mode == "off":
        enable_cleanup = False
    elif args.mask_cleanup_mode in ("light", "medium"):
        enable_cleanup = True
    elif args.enable_mask_cleanup is not None:
        enable_cleanup = args.enable_mask_cleanup  # legacy flag
    else:
        enable_cleanup = None  # auto

    if args.graph_repair_mode == "off":
        enable_repair = False
    elif args.graph_repair_mode == "on":
        enable_repair = True
    elif args.enable_graph_repair is not None:
        enable_repair = args.enable_graph_repair  # legacy flag
    else:
        enable_repair = None  # auto

    run_batch(
        video_dir=args.video_dir,
        output_dir=args.out,
        sample_fps=args.sample_fps,
        mask_backend=args.mask_backend,
        min_branch_length=args.min_branch,
        max_videos=args.max_videos,
        max_frames_per_video=args.max_frames_per_video,
        mask_backend_kwargs=mask_backend_kwargs,
        enable_mask_cleanup=enable_cleanup,
        enable_graph_repair=enable_repair,
        bridge_gap_px=args.bridge_gap_px,
        cleanup_close_kernel=args.cleanup_close_kernel,
        keep_largest_component=args.keep_largest_component,
        node_merge_radius=args.node_merge_radius,
        spur_prune_length=args.spur_prune_length,
        skip_start_sec=args.skip_start,
        skip_end_sec=args.skip_end,
    )


if __name__ == "__main__":
    main()
