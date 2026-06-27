"""
Export 2D pose labels from existing 3D GT armature data.

Projects 3D armature joints into image space using the same camera parameters
used for rendering, producing proper 2D keypoint labels for HyperBonePose2D training.

Usage:
  python scripts/export_pose2d_from_pose3d_gt.py \
    --gt output/fox3d/fox_armature_gt.jsonl \
    --video output/fox3d/fox3d_walk.mp4 \
    --out outputs/pose2d/fox_pose2d_labels.jsonl \
    --asset-id Fox.glb
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from hyperbone.pose2d.quadruped_schema import (
    QUADRUPED_JOINTS_2D, NUM_JOINTS_2D, JOINT_ID, POSE3D_TO_POSE2D,
)


# Fox.glb joint name -> canonical pose2d joint name
FOX_JOINT_TO_POSE2D = {
    "b_Root_00": "root",
    "b_Spine01_00": "spine_1",
    "b_Spine02_00": "spine_2",
    "b_Neck_00": "neck",
    "b_Head_00": "head",
    "b_Tail01_00": "tail_base",
    "b_Tail02_00": "tail_tip",
    "b_LeftUpperArm_00": "front_left_shoulder",
    "b_LeftForeArm_00": "front_left_elbow",
    "b_LeftHand_00": "front_left_hoof",
    "b_RightUpperArm_00": "front_right_shoulder",
    "b_RightForeArm_00": "front_right_elbow",
    "b_RightHand_00": "front_right_hoof",
    "b_LeftLeg_00": "rear_left_hip",
    "b_LeftLeg01_00": "rear_left_knee",
    "b_LeftFoot_00": "rear_left_hoof",
    "b_RightLeg_00": "rear_right_hip",
    "b_RightLeg01_00": "rear_right_knee",
    "b_RightFoot_00": "rear_right_hoof",
}

# Reverse
POSE2D_TO_FOX = {v: k for k, v in FOX_JOINT_TO_POSE2D.items()}


def perspective_project(xyz, fov_deg=45, w=640, h=480,
                       cam_pos=(100.0, 50.0, 0.0),
                       cam_target=(0.0, 35.0, 0.0)):
    """Project 3D point to 2D image coordinates using perspective camera."""
    fov = np.radians(fov_deg)
    aspect = w / h
    f = 1.0 / np.tan(fov / 2)

    cam_pos = np.array(cam_pos)
    cam_target = np.array(cam_target)
    cam_up = np.array([0.0, 1.0, 0.0])

    forward = cam_target - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, cam_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    view = np.eye(4)
    view[0, :3] = right
    view[1, :3] = up
    view[2, :3] = -forward
    view[:3, 3] = -view[:3, :3] @ cam_pos

    p_homo = np.array([*xyz, 1.0])
    p_cam = view @ p_homo

    x = p_cam[0] * f / aspect
    y = p_cam[1] * f
    z = -p_cam[2]
    z = max(z, 0.01)

    px = (x / z * 0.5 + 0.5) * w
    py = (1.0 - (y / z * 0.5 + 0.5)) * h

    return (round(px, 2), round(py, 2))


def compute_bbox_from_joints(
    joints_2d: Dict[str, Tuple[float, float]],
    img_w: int,
    img_h: int,
    expand: float = 1.3,
) -> Tuple[float, float, float, float]:
    """Compute tight bbox around visible 2D joints, expanded by factor."""
    if not joints_2d:
        return (0, 0, img_w, img_h)

    xs = [xy[0] for xy in joints_2d.values()]
    ys = [xy[1] for xy in joints_2d.values()]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = (x_max - x_min) * expand
    h = (y_max - y_min) * expand
    side = max(w, h)

    bx = cx - side / 2
    by = cy - side / 2

    # Clamp
    bx = max(0, bx)
    by = max(0, by)
    bw = min(side, img_w - bx)
    bh = min(side, img_h - by)

    return (round(bx, 1), round(by, 1), round(bw, 1), round(bh, 1))


def main():
    parser = argparse.ArgumentParser(description="Export 2D pose labels from 3D GT")
    parser.add_argument("--gt", required=True, help="Pose3D GT JSONL")
    parser.add_argument("--video", required=True, help="Source video (for frame extraction paths)")
    parser.add_argument("--out", required=True, help="Output pose2d labels JSONL")
    parser.add_argument("--asset-id", default="Fox.glb")
    parser.add_argument("--species", default="fox")
    parser.add_argument("--frames-dir", default="", help="If set, save frame PNGs here")
    parser.add_argument("--img-width", type=int, default=640)
    parser.add_argument("--img-height", type=int, default=480)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Optionally extract frames
    frames_dir = None
    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

    # Load video to get actual frame dimensions + save frames
    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Save all frames if requested
    frame_paths = {}
    if frames_dir:
        fi = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            path = frames_dir / f"frame_{fi:04d}.png"
            cv2.imwrite(str(path), frame)
            frame_paths[fi] = str(path)
            fi += 1
        cap.release()
    else:
        cap.release()

    # Load 3D GT
    gt_records = []
    with open(args.gt) as f:
        for line in f:
            if line.strip():
                gt_records.append(json.loads(line))
    print(f"Loaded {len(gt_records)} 3D GT records")

    # Process each frame
    labels = []
    mapped_count = 0

    for record in gt_records:
        frame_idx = record["frame_idx"]

        # Project each joint to 2D
        joints_2d = {}
        source_joints = record.get("joints", [])

        for jinfo in source_joints:
            raw_name = jinfo.get("name", jinfo.get("id", ""))
            # Map to canonical pose2d name
            canonical = FOX_JOINT_TO_POSE2D.get(raw_name)
            if canonical is None:
                continue
            if not jinfo.get("visible", True):
                continue

            world_xyz = jinfo.get("world_xyz") or jinfo.get("xyz")
            if world_xyz is None:
                continue

            # Project to 2D
            img_xy = perspective_project(world_xyz, w=W, h=H)

            # Check bounds
            px, py = img_xy
            if 0 <= px < W and 0 <= py < H:
                joints_2d[canonical] = {"xy": [px, py], "visible": True}

        if not joints_2d:
            continue

        mapped_count += 1

        # Compute bbox from projected joints
        joint_pts = {k: v["xy"] for k, v in joints_2d.items()}
        bbox = compute_bbox_from_joints(joint_pts, W, H)

        # Image path
        if frames_dir and frame_idx in frame_paths:
            img_path = frame_paths[frame_idx]
        else:
            # Store relative video path + frame index
            img_path = f"frames/frame_{frame_idx:04d}.png"

        label = {
            "image_path": img_path,
            "frame_idx": frame_idx,
            "bbox_xywh": list(bbox),
            "joints": joints_2d,
            "asset_id": args.asset_id,
            "species": args.species,
            "source": "pose3d_projection",
            "img_size": [W, H],
        }
        labels.append(label)

    # Write output
    with open(out_path, 'w') as f:
        for label in labels:
            f.write(json.dumps(label) + "\n")

    # Stats
    all_joints = set()
    for l in labels:
        all_joints.update(l["joints"].keys())

    print(f"\nExported {len(labels)} pose2d labels -> {out_path}")
    print(f"  Mapped joints: {sorted(all_joints)}")
    print(f"  Joint coverage: {len(all_joints)}/{NUM_JOINTS_2D}")
    print(f"  Frames with labels: {mapped_count}/{len(gt_records)}")


if __name__ == "__main__":
    main()
