"""Render a demo video showing skeleton tracking overlaid on source footage.

Processes video through SAM2 + mask cleanup + graph repair, then composites
all detected skeletons onto each frame and encodes as MP4.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from hyperbone.io.video import get_video_info, sample_frames
from hyperbone.cv.masks import get_mask_generator
from hyperbone.cv.mask_cleanup import clean_mask_for_skeleton
from hyperbone.cv.skeletonize import skeletonize_mask
from hyperbone.cv.graph import skeleton_to_graph
from hyperbone.cv.graph_repair import repair_graph
from hyperbone.quality.score import score_graph


def draw_all_skeletons(frame, objects):
    """Draw all detected skeletons on a single frame."""
    overlay = frame.copy()

    # Semi-transparent mask overlay
    mask_layer = np.zeros_like(frame)

    colors = [
        (0, 255, 0), (255, 100, 0), (0, 100, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 255, 0), (0, 128, 255), (255, 0, 128),
        (128, 0, 255),
    ]

    for i, obj in enumerate(objects):
        color = colors[i % len(colors)]
        mask = obj["mask"]
        graph = obj["graph"]
        accepted = obj["accepted"]

        # Draw mask tint
        mask_layer[mask > 0] = color

        # Draw mask contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 1)

        # Draw edges
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        node_map = {n["id"]: tuple(n["xy"]) for n in nodes}

        edge_color = color if accepted else (80, 80, 80)
        for edge in edges:
            p1 = node_map.get(edge["parent"])
            p2 = node_map.get(edge["child"])
            if p1 and p2:
                cv2.line(overlay, p1, p2, edge_color, 2, cv2.LINE_AA)

        # Draw nodes
        for node in nodes:
            x, y = node["xy"]
            ntype = node.get("type", "")
            if ntype == "junction":
                cv2.circle(overlay, (x, y), 4, (0, 0, 255), -1)
            elif ntype == "endpoint":
                cv2.circle(overlay, (x, y), 3, (255, 0, 255), -1)

    # Blend mask layer
    overlay = cv2.addWeighted(overlay, 1.0, mask_layer, 0.15, 0)

    # Draw frame stats
    n_accepted = sum(1 for o in objects if o["accepted"])
    n_total = len(objects)
    cv2.putText(overlay, f"Objects: {n_total}  Accepted: {n_accepted}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"Objects: {n_total}  Accepted: {n_accepted}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 1, cv2.LINE_AA)

    return overlay


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Render skeleton tracking demo video")
    parser.add_argument("--video", default="HyperVid/HyperVid.mp4")
    parser.add_argument("--out", default="outputs/skeleton_demo.mp4")
    parser.add_argument("--start-sec", type=float, default=600)
    parser.add_argument("--duration-sec", type=float, default=300)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--output-fps", type=float, default=2.0,
                        help="Output video FPS (stretches 1fps samples for smoother playback)")
    parser.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--model-cfg",
                        default=r"C:\Users\ritayan\miniconda3\Lib\site-packages\sam2\configs\sam2.1\sam2.1_hiera_t.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    end_sec = args.start_sec + args.duration_sec

    info = get_video_info(args.video)
    print(f"[Demo] Video: {info['path']}")
    print(f"[Demo] Duration: {info['duration_sec']:.1f}s")
    print(f"[Demo] Rendering {args.start_sec:.0f}s - {end_sec:.0f}s at {args.sample_fps} fps")
    print(f"[Demo] Output: {args.out}")
    print()

    # Load SAM2
    print("[Demo] Loading SAM2...", flush=True)
    mask_gen = get_mask_generator("sam2",
                                  checkpoint=args.checkpoint,
                                  model_cfg=args.model_cfg,
                                  device=args.device)
    print("[Demo] SAM2 ready.", flush=True)

    # Set up video writer
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    frame_count = 0
    t_start = time.time()

    for frame_idx, timestamp, frame in sample_frames(
        args.video, args.sample_fps,
        skip_start_sec=args.start_sec,
        skip_end_sec=max(0, info["duration_sec"] - end_sec),
    ):
        # Init writer on first frame
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(args.out, fourcc, args.output_fps, (w, h))

        # Generate masks
        mask_records = mask_gen.generate(frame)

        objects = []
        for rec in mask_records:
            mask = rec["mask"]
            # Cleanup
            cleanup_result = clean_mask_for_skeleton(mask, close_kernel=5, keep_largest=True)
            skel_mask = cleanup_result["clean_mask"]
            # Skeletonize
            skeleton = skeletonize_mask(skel_mask)
            # Graph
            graph = skeleton_to_graph(skeleton, min_branch_length=10)
            if not graph["nodes"]:
                continue
            # Repair
            repair_result = repair_graph(graph, mask_shape=frame.shape[:2], bridge_gap_px=8.0)
            graph = repair_result["graph"]
            # Quality
            ys, xs = mask.nonzero()
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(ys) > 0 else None
            quality = score_graph(mask=mask, graph=graph, frame_shape=frame.shape[:2], bbox=bbox)

            objects.append({
                "mask": mask,
                "graph": graph,
                "accepted": quality["accepted"],
            })

        # Render composite overlay
        overlay_frame = draw_all_skeletons(frame, objects)
        writer.write(overlay_frame)
        frame_count += 1

        elapsed = time.time() - t_start
        fps_rate = frame_count / elapsed if elapsed > 0 else 0
        eta = (args.duration_sec - (timestamp - args.start_sec)) / fps_rate if fps_rate > 0 else 0
        print(f"  [{frame_count:3d}] t={timestamp:.0f}s  objects={len(objects):2d}  "
              f"accepted={sum(1 for o in objects if o['accepted']):2d}  "
              f"({fps_rate:.2f} fps, ETA {eta:.0f}s)", flush=True)

    if writer:
        writer.release()

    print(f"\n[Demo] DONE: {frame_count} frames -> {args.out}")
    print(f"[Demo] Total time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
