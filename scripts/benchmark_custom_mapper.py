"""Benchmark HyperBone custom mapper performance."""
import sys, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from hyperbone.io.video import get_video_info, sample_frames
from hyperbone.objects.proposals import ObjectProposal
from hyperbone.cv.hyperbone_graph import map_proposal_to_skeleton


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="HyperVid/HyperVid.mp4")
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--max-objects", type=int, default=5)
    parser.add_argument("--max-side", type=int, default=384)
    parser.add_argument("--skip-start", type=float, default=600)
    args = parser.parse_args()

    info = get_video_info(args.video)
    print(f"[Benchmark] Video: {info['path']}")
    print(f"[Benchmark] Resolution: {info['width']}x{info['height']}")
    print(f"[Benchmark] Max side: {args.max_side}px")
    print()

    # Use simple grid-based proposals (no DINO needed)
    all_timings = []
    frame_count = 0

    for frame_idx, ts, frame in sample_frames(
        args.video, 1.0, skip_start_sec=args.skip_start
    ):
        if frame_count >= args.max_frames:
            break

        h, w = frame.shape[:2]
        # Generate evenly-spaced bbox proposals
        proposals = []
        box_w, box_h = w // 3, h // 2
        for i in range(min(args.max_objects, 6)):
            x = (i % 3) * box_w
            y = (i // 3) * box_h
            proposals.append(ObjectProposal.manual(
                frame_idx=frame_idx, object_id=i,
                bbox_xywh=(x, y, box_w, box_h),
                label=f"region_{i}",
            ))

        for p in proposals[:args.max_objects]:
            result = map_proposal_to_skeleton(
                frame, p,
                max_side=args.max_side,
                thinning_algorithm="zhang-suen",
                min_branch_length=10,
            )
            all_timings.append(result["timings"])

        frame_count += 1

    # Report
    n = len(all_timings)
    print(f"[Benchmark] Processed {n} objects across {frame_count} frames\n")

    keys = ["crop_ms", "resize_ms", "mask_ms", "thinning_ms", "graph_ms", "total_ms"]
    print(f"{'Stage':<15} {'Mean':>8} {'P50':>8} {'P95':>8} {'Max':>8}")
    print("-" * 50)
    for key in keys:
        vals = [t.get(key, 0) for t in all_timings]
        if vals:
            arr = np.array(vals)
            print(f"{key:<15} {arr.mean():>7.1f}ms {np.percentile(arr, 50):>7.1f}ms "
                  f"{np.percentile(arr, 95):>7.1f}ms {arr.max():>7.1f}ms")

    total_vals = [t.get("total_ms", 0) for t in all_timings]
    avg_total = np.mean(total_vals)
    print(f"\n[Benchmark] Avg total per object: {avg_total:.1f}ms")
    if avg_total < 500:
        print("[Benchmark] PASS: under 500ms per object")
    else:
        print(f"[Benchmark] WARN: {avg_total:.0f}ms exceeds 500ms target")


if __name__ == "__main__":
    main()
