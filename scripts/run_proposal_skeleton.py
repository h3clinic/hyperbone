"""CLI: Run HyperBone proposal-driven skeleton pipeline.

Usage:
  Manual proposals:
    python scripts/run_proposal_skeleton.py \
      --video HyperVid/HyperVid.mp4 \
      --proposals path/to/proposals.jsonl \
      --out outputs/proposal_custom_mapper

  GroundingDINO proposals:
    python scripts/run_proposal_skeleton.py \
      --video HyperVid/HyperVid.mp4 \
      --proposal-source groundingdino \
      --prompts "person. face. hand. arm. leg." \
      --out outputs/dino_custom_mapper \
      --device cuda
"""

import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.pipelines.proposal_skeleton import run_proposal_skeleton
from hyperbone.report.owned_mapper_report import generate_owned_mapper_report


def main():
    parser = argparse.ArgumentParser(
        description="HyperBone proposal-driven skeleton pipeline"
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--proposals", default=None,
                        help="Path to manual proposals JSONL (for --proposal-source manual)")
    parser.add_argument("--proposal-source", default="manual",
                        choices=["manual", "groundingdino"],
                        help="Proposal source: manual JSONL or GroundingDINO")
    parser.add_argument("--prompts", default="person. hand. arm. leg. head. torso.",
                        help="DINO text prompt (for --proposal-source groundingdino)")
    parser.add_argument("--out", default="outputs/proposal_skeleton",
                        help="Output directory")
    parser.add_argument("--sample-fps", type=float, default=1.0,
                        help="Frame sampling rate (fps)")
    parser.add_argument("--skip-start", type=float, default=0.0,
                        help="Skip this many seconds from start")
    parser.add_argument("--skip-end", type=float, default=0.0,
                        help="Skip this many seconds from end")
    parser.add_argument("--skeleton-max-side", type=int, default=384,
                        help="Max side for crop resize")
    parser.add_argument("--thinning-backend", default="zhang-suen",
                        choices=["zhang-suen", "guo-hall"],
                        help="Thinning algorithm")
    parser.add_argument("--min-branch-length", type=int, default=10,
                        help="Min branch length for pruning")
    parser.add_argument("--mask-method", default="combined",
                        choices=["combined", "edges", "threshold", "grabcut_lite"],
                        help="Custom mask method")
    parser.add_argument("--device", default="cuda",
                        help="Device for DINO (cuda/cpu)")
    parser.add_argument("--grounding-model",
                        default="IDEA-Research/grounding-dino-tiny",
                        help="GroundingDINO model ID")
    parser.add_argument("--box-threshold", type=float, default=0.25,
                        help="DINO box confidence threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25,
                        help="DINO text confidence threshold")
    parser.add_argument("--max-proposals", type=int, default=20,
                        help="Max proposals per frame from DINO")

    args = parser.parse_args()

    stats = run_proposal_skeleton(
        video_path=args.video,
        output_dir=args.out,
        proposals_path=args.proposals,
        proposal_source=args.proposal_source,
        text_prompt=args.prompts,
        sample_fps=args.sample_fps,
        skip_start_sec=args.skip_start,
        skip_end_sec=args.skip_end,
        max_side=args.skeleton_max_side,
        thinning_algorithm=args.thinning_backend,
        min_branch_length=args.min_branch_length,
        mask_method=args.mask_method,
        device=args.device,
        grounding_model=args.grounding_model,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        max_proposals_per_frame=args.max_proposals,
    )

    # Generate report
    report_path = generate_owned_mapper_report(stats, args.out)
    print(f"\n[HyperBone] Report: {report_path}")


if __name__ == "__main__":
    main()
