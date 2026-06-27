"""
Run HyperBone3D inference on an arbitrary video (no GT needed).
Produces:
  - predictions JSONL
  - overlay video with predicted skeleton
  - per-frame confidence stats

Usage:
  python scripts/infer_hyperbone3d_video.py \
    --model outputs/models/hyperbone3d_v0_fox_split/model.pt \
    --config outputs/models/hyperbone3d_v0_fox_split/config.json \
    --video assets/pig/pig_360.mp4 \
    --out outputs/inference/pig_360/ \
    --device cuda \
    --skip-frames 0 \
    --max-frames 150
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import cv2
import numpy as np
from pathlib import Path

import torch

from hyperbone.models.hyperbone3d import HyperBone3D
from hyperbone.pose3d.joint_map import QUADRUPED_JOINTS, QUADRUPED_BONES, NUM_JOINTS


def prepare_frame(frame_bgr: np.ndarray, resolution: int):
    """Prepare a video frame for model input (RGB + mask + depth tensors)."""
    h, w = resolution, resolution
    # Resize
    resized = cv2.resize(frame_bgr, (w, h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Generate mask via simple background segmentation
    # Use adaptive threshold on grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # Otsu threshold to separate foreground
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # If mostly white (background is light), invert
    if np.mean(mask) > 127:
        mask = 255 - mask

    # No depth from real video - use zeros
    depth = np.zeros((h, w), dtype=np.uint8)

    # To tensors [C, H, W], [0, 1]
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0   # [3,H,W]
    mask_t = torch.from_numpy(mask).unsqueeze(0).float() / 255.0      # [1,H,W]
    depth_t = torch.from_numpy(depth).unsqueeze(0).float() / 255.0    # [1,H,W]

    x = torch.cat([rgb_t, mask_t, depth_t], dim=0)  # [5, H, W]
    return x


def draw_skeleton_on_frame(frame_bgr: np.ndarray, pred_xyz: np.ndarray,
                           pred_vis: np.ndarray, vis_thresh: float = 0.5,
                           show_names: bool = False):
    """Draw predicted canonical skeleton as a 2D projection on the frame.

    Since we only have canonical 3D coords (root-relative, scale-normalized),
    we project using a simple orthographic projection onto the image:
      x_img = cx + pred_x * scale_px
      y_img = cy - pred_y * scale_px  (y-up in 3D, y-down in image)
    """
    H, W = frame_bgr.shape[:2]
    cx, cy = W // 2, H // 2
    # Scale: map canonical coords to pixels (canonical range is roughly [-1, 1])
    scale_px = min(W, H) * 0.35

    pts = {}
    for j in range(NUM_JOINTS):
        if pred_vis[j] > vis_thresh:
            x_img = int(cx + pred_xyz[j, 0] * scale_px)
            y_img = int(cy - pred_xyz[j, 1] * scale_px)
            pts[j] = (x_img, y_img)

    # Draw bones
    for pi, ci in QUADRUPED_BONES:
        if pi in pts and ci in pts:
            cv2.line(frame_bgr, pts[pi], pts[ci], (0, 200, 0), 2)

    # Draw joints
    for j, pt in pts.items():
        cv2.circle(frame_bgr, pt, 3, (0, 80, 255), -1)
        if show_names:
            cv2.putText(frame_bgr, QUADRUPED_JOINTS[j][:8], (pt[0]+4, pt[1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

    return frame_bgr


def main():
    parser = argparse.ArgumentParser(description="Run HyperBone3D on arbitrary video")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-frames", type=int, default=0, help="Skip N initial black frames")
    parser.add_argument("--max-frames", type=int, default=0, help="Process max N frames (0=all)")
    parser.add_argument("--vis-thresh", type=float, default=0.3, help="Visibility threshold for display")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load config + model
    with open(args.config) as f:
        config = json.load(f)

    res = config.get("resolution", 128)
    model = HyperBone3D(
        num_joints=config.get("num_joints", NUM_JOINTS),
        input_channels=5,
        base_channels=config.get("base_channels", 32),
        hidden_dim=config.get("hidden_dim", 128),
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded: {args.model} ({sum(p.numel() for p in model.parameters()):,} params)")

    # Open video
    cap = cv2.VideoCapture(args.video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {args.video} ({W}x{H} @ {fps}fps, {n_total} frames)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Video writer
    overlay_path = out_dir / "overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(overlay_path), fourcc, fps, (W, H))

    # Predictions
    predictions = []
    vis_stats = []

    frame_idx = 0
    processed = 0
    print(f"Running inference (skip={args.skip_frames}, max={args.max_frames or 'all'})...")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx < args.skip_frames:
                frame_idx += 1
                continue

            if args.max_frames > 0 and processed >= args.max_frames:
                break

            # Prepare input
            x = prepare_frame(frame, res).unsqueeze(0).to(device)

            # Inference
            pred = model(x)
            pred_xyz = pred["joint_xyz_canonical"][0].cpu().numpy()   # [J, 3]
            pred_vis = torch.sigmoid(pred["joint_visibility_logits"][0]).cpu().numpy()  # [J]

            # Store prediction
            predictions.append({
                "frame_idx": frame_idx,
                "pred_xyz": pred_xyz.tolist(),
                "pred_vis": pred_vis.tolist(),
            })

            n_visible = int((pred_vis > args.vis_thresh).sum())
            avg_conf = float(pred_vis.mean())
            vis_stats.append({"frame": frame_idx, "n_visible": n_visible, "avg_conf": avg_conf})

            # Draw overlay
            overlay = frame.copy()
            draw_skeleton_on_frame(overlay, pred_xyz, pred_vis, vis_thresh=args.vis_thresh)

            # Add info text
            cv2.putText(overlay, f"F{frame_idx} vis={n_visible}/{NUM_JOINTS} conf={avg_conf:.2f}",
                        (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            writer.write(overlay)

            processed += 1
            if processed % 100 == 0:
                print(f"  {processed} frames processed...")

            frame_idx += 1

    cap.release()
    writer.release()

    # Write predictions JSONL
    pred_path = out_dir / "predictions.jsonl"
    with open(pred_path, 'w') as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    # Write summary
    if vis_stats:
        avg_vis = np.mean([s["n_visible"] for s in vis_stats])
        avg_conf_all = np.mean([s["avg_conf"] for s in vis_stats])
    else:
        avg_vis = avg_conf_all = 0

    summary = {
        "video": args.video,
        "model": args.model,
        "frames_processed": processed,
        "skip_frames": args.skip_frames,
        "resolution": res,
        "vis_threshold": args.vis_thresh,
        "avg_visible_joints": float(avg_vis),
        "avg_confidence": float(avg_conf_all),
        "overlay": str(overlay_path),
    }
    summary_path = out_dir / "inference_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone! {processed} frames processed.")
    print(f"  Avg visible joints: {avg_vis:.1f}/{NUM_JOINTS}")
    print(f"  Avg confidence: {avg_conf_all:.3f}")
    print(f"  Overlay: {overlay_path}")
    print(f"  Predictions: {pred_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
