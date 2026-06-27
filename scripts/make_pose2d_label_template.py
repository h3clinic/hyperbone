"""
Generate a pose2d label template from a video for manual annotation.

Samples frames at given fps and writes empty/template joint labels.

Usage:
  python scripts/make_pose2d_label_template.py \
    --video assets/pig/pig_trimmed.mp4 \
    --out labels/pig_pose2d_template.jsonl \
    --sample-fps 2 \
    --species pig
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import cv2
from pathlib import Path

from hyperbone.pose2d.quadruped_schema import QUADRUPED_JOINTS_2D


def main():
    parser = argparse.ArgumentParser(description="Generate pose2d label template for manual annotation")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True, help="Output JSONL template")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Sample rate in fps")
    parser.add_argument("--species", default="unknown")
    parser.add_argument("--frames-dir", default="", help="Save sampled frames here")
    parser.add_argument("--max-frames", type=int, default=200)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_interval = max(1, int(fps / args.sample_fps))
    print(f"Video: {W}x{H} @ {fps:.1f}fps, {n_total} frames")
    print(f"Sampling every {sample_interval} frames (~{args.sample_fps}fps)")

    frames_dir = None
    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

    records = []
    fi = 0
    while fi < n_total and len(records) < args.max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break

        # Skip very dark frames
        if frame.mean() < 20:
            fi += sample_interval
            continue

        # Save frame if requested
        img_path = f"frame_{fi:04d}.png"
        if frames_dir:
            cv2.imwrite(str(frames_dir / img_path), frame)
            img_path = str(frames_dir / img_path)

        # Template: all joints empty
        joints = {}
        for name in QUADRUPED_JOINTS_2D:
            joints[name] = {"xy": [0, 0], "visible": False}

        record = {
            "image_path": img_path,
            "frame_idx": fi,
            "bbox_xywh": [0, 0, W, H],
            "joints": joints,
            "asset_id": "",
            "species": args.species,
            "img_size": [W, H],
            "labeled": False,
        }
        records.append(record)
        fi += sample_interval

    cap.release()

    with open(out_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nTemplate written: {out_path}")
    print(f"  Frames sampled: {len(records)}")
    print(f"  Joints per frame: {len(QUADRUPED_JOINTS_2D)} (all marked not visible)")
    print(f"  Edit the JSONL to fill in xy coords and set visible=true")


if __name__ == "__main__":
    main()
