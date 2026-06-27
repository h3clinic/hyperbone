"""
Overlay video showing GT internal 3D armature vs HyperBone3D predicted armature.

Usage:
  python scripts/make_hyperbone3d_prediction_overlay.py \
    --model outputs/models/hyperbone3d_v0_fox/model.pt \
    --config outputs/models/hyperbone3d_v0_fox/config.json \
    --video output/fox3d/fox3d_walk.mp4 \
    --gt output/fox3d/fox_armature_gt.jsonl \
    --asset-id Fox.glb \
    --out outputs/models/hyperbone3d_v0_fox/prediction_overlay.mp4
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
from hyperbone.pose3d.dataset import Pose3DDatasetFromVideo
from hyperbone.pose3d.joint_map import QUADRUPED_JOINTS, QUADRUPED_BONES, NUM_JOINTS


def project_canonical_to_image(xyz_canonical, root_xyz, scale, bbox_xywh, resolution):
    """
    Approximate projection of canonical xyz back to image coords.

    canonical = (world - root) / scale
    world = canonical * scale + root

    Then use the same perspective_project as GT export.
    """
    from scripts.export_fox_armature_gt_nobl import perspective_project

    world_xyz = xyz_canonical * scale + root_xyz
    image_pts = []
    for j in range(len(world_xyz)):
        img_xy, _ = perspective_project(world_xyz[j].tolist(), w=resolution[0], h=resolution[1])
        image_pts.append(img_xy)
    return image_pts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--asset-id", default="Fox.glb")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
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

    # Load dataset (for GT + input tensors)
    dataset = Pose3DDatasetFromVideo(
        video_path=args.video,
        gt_jsonl_path=args.gt,
        resolution=(res, res),
        asset_id=args.asset_id,
    )

    # Load GT records for image_xy
    gt_records = []
    with open(args.gt) as f:
        for line in f:
            if line.strip():
                gt_records.append(json.loads(line))
    gt_by_frame = {r["frame_idx"]: r for r in gt_records}

    # Open video for full-res frames
    cap = cv2.VideoCapture(args.video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    print(f"Video: {W}x{H} @ {fps}fps, {n_frames} frames")
    print(f"Model: {args.model}")
    print(f"Generating prediction overlay...")

    # Run predictions on all dataset samples
    pred_by_frame = {}
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            x = torch.cat([sample["rgb"], sample["mask"], sample["depth"]], dim=0)
            x = x.unsqueeze(0).to(device)
            pred = model(x)
            pred_xyz = pred["joint_xyz_canonical"][0].cpu().numpy()
            pred_vis = torch.sigmoid(pred["joint_visibility_logits"][0]).cpu().numpy()

            # Project predicted joints back to image space
            root_xyz = sample["root_xyz"].numpy()
            scale = sample["scale"].item()
            pred_world = pred_xyz * scale + root_xyz

            from scripts.export_fox_armature_gt_nobl import perspective_project
            pred_image_pts = []
            for j in range(NUM_JOINTS):
                if pred_vis[j] > 0.5:
                    img_xy, _ = perspective_project(pred_world[j].tolist(), w=W, h=H)
                    pred_image_pts.append((j, img_xy))
                else:
                    pred_image_pts.append((j, None))

            pred_by_frame[sample["frame_idx"]] = {
                "joints": pred_image_pts,
                "vis": pred_vis,
            }

    # Write overlay video
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw GT armature (cyan)
        if fi in gt_by_frame:
            gt = gt_by_frame[fi]
            joints = gt.get("joints", [])
            joint_xy = {}
            for j in joints:
                if j.get("visible", True) and j.get("image_xy"):
                    joint_xy[j["id"]] = (int(j["image_xy"][0]), int(j["image_xy"][1]))
            for bone in gt.get("bones", []):
                pid, cid = bone.get("parent_id"), bone.get("child_id")
                if pid in joint_xy and cid in joint_xy:
                    cv2.line(frame, joint_xy[pid], joint_xy[cid], (255, 200, 0), 2)
            for jid, pt in joint_xy.items():
                cv2.circle(frame, pt, 4, (0, 255, 255), -1)

        # Draw predicted armature (green/magenta)
        if fi in pred_by_frame:
            pred = pred_by_frame[fi]
            pred_pts = {}
            for j_idx, xy in pred["joints"]:
                if xy is not None:
                    pred_pts[j_idx] = (int(xy[0]), int(xy[1]))

            for pi, ci in QUADRUPED_BONES:
                if pi in pred_pts and ci in pred_pts:
                    cv2.line(frame, pred_pts[pi], pred_pts[ci], (0, 200, 0), 2)
            for j_idx, pt in pred_pts.items():
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)

            # Compute MPJPE for display
            if fi in gt_by_frame:
                sample_idx = [i for i, r in enumerate(dataset.records) if r["frame_idx"] == fi]
                if sample_idx:
                    s = dataset[sample_idx[0]]
                    with torch.no_grad():
                        x = torch.cat([s["rgb"], s["mask"], s["depth"]], dim=0).unsqueeze(0).to(device)
                        p = model(x)
                        err = torch.norm(p["joint_xyz_canonical"][0].cpu() - s["joint_xyz_canonical"], dim=-1)
                        vis_mask = s["joint_visible"] > 0.5
                        if vis_mask.any():
                            mpjpe = err[vis_mask].mean().item()
                            cv2.putText(frame, f"MPJPE={mpjpe:.4f}",
                                        (10, H - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Legend
        cv2.putText(frame, f"Frame {fi}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.circle(frame, (W - 160, 15), 4, (0, 255, 255), -1)
        cv2.putText(frame, "GT Armature", (W - 148, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        cv2.circle(frame, (W - 160, 32), 4, (0, 0, 255), -1)
        cv2.putText(frame, "HyperBone3D Pred", (W - 148, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.putText(frame, "HyperBone3D v0 — Internal 3D Pose",
                    (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        writer.write(frame)
        fi += 1

    cap.release()
    writer.release()
    print(f"Overlay saved: {out_path} ({fi} frames)")


if __name__ == "__main__":
    main()
