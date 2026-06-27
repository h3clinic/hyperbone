"""
Create overlay video showing projected internal 3D armature joints on RGB frames.

Proves that the exported 3D armature aligns with the rendered video.

Usage:
  python scripts/make_pose3d_projection_overlay.py \
    --rgb-dir outputs/pose3d/wolf_dataset/rgb \
    --pose3d outputs/pose3d/wolf_dataset/pose3d_gt.jsonl \
    --out outputs/pose3d/wolf_projection_overlay.mp4
"""
from __future__ import annotations

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_pose3d_jsonl(path: str) -> List[Dict]:
    """Load pose3d ground-truth JSONL."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def draw_armature_overlay(img: np.ndarray, record: Dict) -> np.ndarray:
    """Draw projected 3D armature joints and bones on an image."""
    out = img.copy()
    joints = record.get("joints", record.get("joints_3d", []))
    bones = record.get("bones", [])

    # Build joint id -> image_xy map
    joint_xy = {}
    for j in joints:
        jid = j["id"]
        xy = j.get("image_xy", None)
        if xy and j.get("visible", True):
            joint_xy[jid] = (int(xy[0]), int(xy[1]))

    # Draw bones
    for bone in bones:
        pid = bone.get("parent", bone.get("parent_id"))
        cid = bone.get("child", bone.get("child_id"))
        if pid in joint_xy and cid in joint_xy:
            cv2.line(out, joint_xy[pid], joint_xy[cid], (255, 200, 0), 2)

    # Draw joints
    for jid, (px, py) in joint_xy.items():
        cv2.circle(out, (px, py), 4, (0, 255, 255), -1)
        cv2.circle(out, (px, py), 5, (0, 128, 255), 1)

    # Draw joint labels (only if few joints)
    if len(joint_xy) <= 30:
        for j in joints:
            jid = j["id"]
            if jid in joint_xy:
                name = j.get("name", "")
                if name:
                    px, py = joint_xy[jid]
                    cv2.putText(out, name[:8], (px + 6, py - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

    # Frame info
    frame_idx = record.get("frame_idx", 0)
    n_joints = len(joint_xy)
    cv2.putText(out, f"Frame {frame_idx}  Joints: {n_joints}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(out, "Projected 3D Armature (GT)",
                (10, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Draw bbox if available
    bbox = record.get("bbox_xywh")
    if bbox:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 1)

    return out


def main():
    parser = argparse.ArgumentParser(description="Pose3D projection overlay video")
    parser.add_argument("--rgb-dir", required=True, help="Directory with RGB frames (frame_NNNN.png)")
    parser.add_argument("--pose3d", required=True, help="Pose3D ground-truth JSONL")
    parser.add_argument("--out", required=True, help="Output video path")
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    rgb_dir = Path(args.rgb_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load pose data
    records = load_pose3d_jsonl(args.pose3d)
    print(f"Loaded {len(records)} pose3d records from {args.pose3d}")

    # Index by frame
    record_by_frame = {r["frame_idx"]: r for r in records}

    # Find RGB frames
    frame_files = sorted(rgb_dir.glob("frame_*.png"))
    if not frame_files:
        frame_files = sorted(rgb_dir.glob("*.png"))
    if not frame_files:
        frame_files = sorted(rgb_dir.glob("*.jpg"))

    if not frame_files:
        print("ERROR: No frame images found in", rgb_dir)
        return

    # Read first frame for dimensions
    sample = cv2.imread(str(frame_files[0]))
    h, w = sample.shape[:2]
    print(f"RGB frames: {len(frame_files)}, resolution: {w}x{h}")

    # Write video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w, h))

    for fi, fpath in enumerate(frame_files):
        img = cv2.imread(str(fpath))
        if fi in record_by_frame:
            img = draw_armature_overlay(img, record_by_frame[fi])
        writer.write(img)

    writer.release()
    print(f"Overlay video saved: {out_path} ({len(frame_files)} frames)")


if __name__ == "__main__":
    main()
