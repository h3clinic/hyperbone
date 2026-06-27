"""
Export combined 3D+2D pose labels from existing Fox armature GT.

Creates the dataset format needed by HyperBonePose3D-Joint:
- 3D canonical joints (primary target)
- 2D projected joints (auxiliary reprojection target)
- bbox from joint projections
- frame images

Usage:
  python scripts/export_pose3d_joint_labels.py \
    --gt output/fox3d/fox_armature_gt.jsonl \
    --video output/fox3d/fox3d_walk.mp4 \
    --out outputs/pose3d_joint/fox_walk/ \
    --asset-id Fox.glb \
    --species fox
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import cv2
import numpy as np
from pathlib import Path

from hyperbone.pose2d.quadruped_schema import (
    QUADRUPED_JOINTS_2D, NUM_JOINTS_2D, JOINT_ID,
)


# Fox.glb bone names -> canonical pose2d/3d joint names
# Actual names from GT: b_Root_00, b_Hip_01, b_Spine01_02, etc.
FOX_JOINT_MAP = {
    "b_Root_00": "root",
    "b_Hip_01": "root",           # hip = root (use whichever is available)
    "b_Spine01_02": "spine_1",
    "b_Spine02_03": "spine_2",
    "b_Neck_04": "neck",
    "b_Head_05": "head",
    "b_Tail01_012": "tail_base",
    "b_Tail02_013": "tail_tip",
    "b_LeftUpperArm_09": "front_left_shoulder",
    "b_LeftForeArm_010": "front_left_elbow",
    "b_LeftHand_011": "front_left_hoof",
    "b_RightUpperArm_06": "front_right_shoulder",
    "b_RightForeArm_07": "front_right_elbow",
    "b_RightHand_08": "front_right_hoof",
    "b_LeftLeg01_015": "rear_left_hip",
    "b_LeftLeg02_016": "rear_left_knee",
    "b_LeftFoot01_017": "rear_left_hoof",
    "b_RightLeg01_019": "rear_right_hip",
    "b_RightLeg02_020": "rear_right_knee",
    "b_RightFoot01_021": "rear_right_hoof",
}


def perspective_project(xyz, fov_deg=45, w=640, h=480,
                       cam_pos=(100.0, 50.0, 0.0),
                       cam_target=(0.0, 35.0, 0.0)):
    """Project 3D world point to 2D image coordinates."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Fox armature GT JSONL")
    parser.add_argument("--video", required=True, help="Source video")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--asset-id", default="Fox.glb")
    parser.add_argument("--species", default="fox")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Load GT
    gt_records = []
    with open(args.gt) as f:
        for line in f:
            if line.strip():
                gt_records.append(json.loads(line))
    print(f"Loaded {len(gt_records)} GT records")

    # Extract video frames
    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {W}x{H}, {n_frames} frames")

    # Save frames
    frame_paths = {}
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        path = frames_dir / f"frame_{fi:04d}.png"
        cv2.imwrite(str(path), frame)
        # Store path relative to labels file (which will be in out_dir)
        frame_paths[fi] = f"frames/frame_{fi:04d}.png"
        fi += 1
    cap.release()
    print(f"Saved {len(frame_paths)} frames")

    # Process GT records into joint labels
    labels = []
    for record in gt_records:
        frame_idx = record["frame_idx"]
        if frame_idx not in frame_paths:
            continue

        joints = record.get("joints", [])

        # Map raw joints to canonical
        joints_3d = {}
        joints_2d = {}
        world_positions = {}

        for jinfo in joints:
            raw_name = jinfo.get("name", jinfo.get("id", ""))
            canonical = FOX_JOINT_MAP.get(raw_name)
            if canonical is None:
                continue

            world_xyz = jinfo.get("world_xyz") or jinfo.get("xyz")
            if world_xyz is None:
                continue

            visible = jinfo.get("visible", True)
            world_positions[canonical] = world_xyz

            # Project to 2D
            img_xy = perspective_project(world_xyz, w=W, h=H)
            px, py = img_xy

            in_bounds = 0 <= px < W and 0 <= py < H
            if visible and in_bounds:
                joints_2d[canonical] = {"xy": [px, py], "visible": True}

        if not world_positions:
            continue

        # Canonicalize 3D: root-relative, scale by root->neck distance
        root_xyz = np.array(world_positions.get("root", [0, 0, 0]), dtype=np.float32)
        neck_xyz = np.array(world_positions.get("neck", root_xyz), dtype=np.float32)
        scale = float(np.linalg.norm(neck_xyz - root_xyz))
        scale = max(scale, 1e-3)

        for canonical, world_xyz in world_positions.items():
            canonical_xyz = ((np.array(world_xyz, dtype=np.float32) - root_xyz) / scale).tolist()
            joints_3d[canonical] = {
                "xyz": canonical_xyz,
                "visible": canonical in joints_2d,
            }

        # Compute bbox from 2D joint projections
        if joints_2d:
            xs = [j["xy"][0] for j in joints_2d.values()]
            ys = [j["xy"][1] for j in joints_2d.values()]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            bw = (x_max - x_min) * 1.3
            bh = (y_max - y_min) * 1.3
            side = max(bw, bh)
            bbox = [round(cx - side/2, 1), round(cy - side/2, 1),
                    round(side, 1), round(side, 1)]
        else:
            bbox = [0, 0, W, H]

        label = {
            "image_path": frame_paths[frame_idx],
            "frame_idx": frame_idx,
            "bbox_xywh": bbox,
            "joints_3d": joints_3d,
            "joints_2d": joints_2d,
            "canonical_scale": round(scale, 6),
            "canonical_root": root_xyz.tolist(),
            "asset_id": args.asset_id,
            "species": args.species,
            "img_size": [W, H],
        }
        labels.append(label)

    # Write labels
    labels_path = out_dir / "labels.jsonl"
    with open(labels_path, 'w') as f:
        for l in labels:
            f.write(json.dumps(l) + "\n")

    # Stats
    all_joints_3d = set()
    all_joints_2d = set()
    for l in labels:
        all_joints_3d.update(l["joints_3d"].keys())
        all_joints_2d.update(l["joints_2d"].keys())

    print(f"\nExported {len(labels)} labels -> {labels_path}")
    print(f"  3D joint coverage: {len(all_joints_3d)}/{NUM_JOINTS_2D} ({sorted(all_joints_3d)})")
    print(f"  2D joint coverage: {len(all_joints_2d)}/{NUM_JOINTS_2D}")
    print(f"  Frames dir: {frames_dir}")
    print(f"  Canonical scale (first): {labels[0]['canonical_scale']:.4f}" if labels else "")


if __name__ == "__main__":
    main()
