"""CLI runner for the HyperBone pseudo-label factory."""

import argparse
import sys
from pathlib import Path

# Add parent to path so hyperbone package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.pipelines.pseudo_label import run_pseudo_label


def main():
    parser = argparse.ArgumentParser(
        description="HyperBone Pseudo-Label Factory — extract skeleton graphs from video"
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--out", default="outputs/run",
        help="Output directory (default: outputs/run)"
    )
    parser.add_argument(
        "--sample-fps", type=float, default=1.0,
        help="Frame sampling rate in FPS (default: 1.0)"
    )
    parser.add_argument(
        "--mask-backend", default="threshold",
        choices=["threshold", "sam2", "noop"],
        help="Mask generation backend (default: threshold)"
    )
    parser.add_argument(
        "--min-branch", type=int, default=10,
        help="Minimum skeleton branch length in pixels (default: 10)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum number of frames to process (default: all)"
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
        "--no-frames", action="store_true",
        help="Skip saving extracted frame PNGs"
    )
    parser.add_argument(
        "--no-masks", action="store_true",
        help="Skip saving mask PNGs"
    )
    parser.add_argument(
        "--no-overlays", action="store_true",
        help="Skip saving overlay debug PNGs"
    )
    parser.add_argument(
        "--enable-mask-cleanup", action="store_true", default=None,
        help="Enable mask cleanup before skeletonization (auto for sam2)"
    )
    parser.add_argument(
        "--enable-graph-repair", action="store_true", default=None,
        help="Enable graph component bridging/repair (auto for sam2)"
    )
    parser.add_argument(
        "--bridge-gap-px", type=float, default=8.0,
        help="Max pixel gap to bridge between graph components (default: 8)"
    )
    parser.add_argument(
        "--cleanup-close-kernel", type=int, default=5,
        help="Morphological closing kernel size for mask cleanup (default: 5)"
    )
    parser.add_argument(
        "--keep-largest-component", action="store_true", default=True,
        help="Keep only the largest mask component during cleanup"
    )

    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    # Build mask backend kwargs
    mask_backend_kwargs = {
        "max_masks_per_frame" if args.mask_backend == "sam2" else "max_objects": args.max_masks_per_frame,
        "min_area_ratio": args.min_mask_area_ratio,
        "max_area_ratio": args.max_mask_area_ratio,
    }
    if args.mask_backend == "sam2":
        mask_backend_kwargs["checkpoint"] = args.sam2_checkpoint
        mask_backend_kwargs["model_cfg"] = args.sam2_model_cfg
        mask_backend_kwargs["device"] = args.device

    run_pseudo_label(
        video_path=args.video,
        output_dir=args.out,
        sample_fps=args.sample_fps,
        mask_backend=args.mask_backend,
        min_branch_length=args.min_branch,
        save_frames=not args.no_frames,
        save_masks=not args.no_masks,
        save_overlays=not args.no_overlays,
        max_frames=args.max_frames,
        mask_backend_kwargs=mask_backend_kwargs,
        enable_mask_cleanup=args.enable_mask_cleanup,
        enable_graph_repair=args.enable_graph_repair,
        bridge_gap_px=args.bridge_gap_px,
        cleanup_close_kernel=args.cleanup_close_kernel,
        keep_largest_component=args.keep_largest_component,
    )


if __name__ == "__main__":
    main()
