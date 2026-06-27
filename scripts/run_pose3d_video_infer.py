"""
HyperBonePose3D-Joint video inference.

Pipeline:
  bbox/proposal -> crop animal -> predict 3D canonical joints
    -> project 3D to image -> overlay -> confidence gating

Usage:
  python scripts/run_pose3d_video_infer.py \
    --video assets/pig/pig_trimmed.mp4 \
    --model outputs/models/hyperbone_pose3d_joint_fox/model.pt \
    --config outputs/models/hyperbone_pose3d_joint_fox/config.json \
    --out outputs/inference/pig_pose3d_joint/ \
    --device cuda \
    --proposal-source manual \
    --manual-bbox "20,10,220,130"

  # Or with GroundingDINO:
  python scripts/run_pose3d_video_infer.py \
    --video assets/pig/pig_trimmed.mp4 \
    --model outputs/models/hyperbone_pose3d_joint_fox/model.pt \
    --config outputs/models/hyperbone_pose3d_joint_fox/config.json \
    --out outputs/inference/pig_pose3d_joint/ \
    --device cuda \
    --proposal-source groundingdino \
    --prompts "pig. hog. animal."
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from hyperbone.models.hyperbone_pose3d_joint import (
    HyperBonePose3DJoint,
    project_canonical_to_crop,
    project_crop_to_fullframe,
)
from hyperbone.pose2d.quadruped_schema import (
    QUADRUPED_JOINTS_2D, QUADRUPED_BONES_2D, NUM_JOINTS_2D,
    validate_pose, joints_inside_bbox_ratio,
)


def crop_and_prepare(
    frame_bgr: np.ndarray,
    bbox_xywh: Tuple[float, float, float, float],
    resolution: int,
    expand: float = 1.25,
) -> Tuple[torch.Tensor, Tuple[float, float, float, float]]:
    """Crop frame to bbox, resize, return tensor + actual bbox used."""
    H, W = frame_bgr.shape[:2]
    bx, by, bw, bh = bbox_xywh

    # Expand and make square
    cx, cy = bx + bw / 2, by + bh / 2
    side = max(bw, bh) * expand

    x1 = max(0, int(cx - side / 2))
    y1 = max(0, int(cy - side / 2))
    x2 = min(W, int(cx + side / 2))
    y2 = min(H, int(cy + side / 2))

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame_bgr
        x1, y1, x2, y2 = 0, 0, W, H

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, (resolution, resolution))

    tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
    actual_bbox = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
    return tensor, actual_bbox


def project_pred_to_frame(
    pred_xy_norm: np.ndarray,  # [J, 2] normalized [0,1] from soft-argmax
    bbox_xywh: Tuple[float, float, float, float],
) -> np.ndarray:
    """
    Project predicted 2D joint locations (from heatmap soft-argmax) to full frame.

    These are the model's actual localization predictions, not a generic projection.
    pred_xy_norm comes from the heatmap soft-argmax, normalized [0,1] within the crop.
    """
    bx, by, bw, bh = bbox_xywh
    frame_x = pred_xy_norm[:, 0] * bw + bx
    frame_y = pred_xy_norm[:, 1] * bh + by
    return np.stack([frame_x, frame_y], axis=-1)  # [J, 2]


def strict_pose_validation(
    confidence: np.ndarray,     # [J]
    visibility: np.ndarray,     # [J]
    joints_2d: np.ndarray,      # [J, 2] full-frame
    bbox_xywh: Tuple[float, float, float, float],
    min_confidence: float = 0.65,
    min_confident_joints: int = 8,
    min_head_conf: float = 0.50,
    min_bbox_overlap: float = 0.80,
) -> Tuple[bool, str, int]:
    """
    Strict pose validation. Returns (valid, reason, n_confident).

    Rules:
    - mean confidence of visible joints >= min_confidence
    - at least min_confident_joints joints above threshold
    - head or neck must be >= min_head_conf
    - >= min_bbox_overlap of confident joints inside bbox
    - at least 2 limb chains (shoulder+elbow+hoof) with all joints confident
    """
    from hyperbone.pose2d.quadruped_schema import JOINT_ID

    conf_thresh = 0.5  # per-joint threshold
    confident_mask = (confidence > conf_thresh) & (visibility > 0.5)
    n_confident = int(confident_mask.sum())

    # Rule 1: enough confident joints
    if n_confident < min_confident_joints:
        return False, f"insufficient_confident_joints ({n_confident}/{min_confident_joints})", n_confident

    # Rule 2: mean confidence of confident joints
    if n_confident > 0:
        mean_conf = float(confidence[confident_mask].mean())
    else:
        mean_conf = 0.0
    if mean_conf < min_confidence:
        return False, f"low_joint_confidence (mean={mean_conf:.2f}<{min_confidence})", n_confident

    # Rule 3: head or neck confidence
    head_idx = JOINT_ID.get("head", 4)
    neck_idx = JOINT_ID.get("neck", 3)
    head_neck_conf = max(confidence[head_idx], confidence[neck_idx])
    if head_neck_conf < min_head_conf:
        return False, f"weak_head_neck (best={head_neck_conf:.2f}<{min_head_conf})", n_confident

    # Rule 4: confident joints inside bbox
    bx, by, bw, bh = bbox_xywh
    confident_pts = joints_2d[confident_mask]
    if len(confident_pts) > 0:
        inside = (
            (confident_pts[:, 0] >= bx) & (confident_pts[:, 0] <= bx + bw) &
            (confident_pts[:, 1] >= by) & (confident_pts[:, 1] <= by + bh)
        )
        ratio = inside.sum() / len(confident_pts)
        if ratio < min_bbox_overlap:
            return False, f"projection_mismatch (in_bbox={ratio:.0%}<{min_bbox_overlap:.0%})", n_confident

    # Rule 5: at least 2 complete limb chains
    limb_chains = [
        ("front_left_shoulder", "front_left_elbow", "front_left_hoof"),
        ("front_right_shoulder", "front_right_elbow", "front_right_hoof"),
        ("rear_left_hip", "rear_left_knee", "rear_left_hoof"),
        ("rear_right_hip", "rear_right_knee", "rear_right_hoof"),
    ]
    complete_limbs = 0
    for chain in limb_chains:
        chain_confident = all(
            confidence[JOINT_ID[j]] > conf_thresh for j in chain if j in JOINT_ID
        )
        if chain_confident:
            complete_limbs += 1
    if complete_limbs < 2:
        return False, f"missing_limb_chains ({complete_limbs}/2 complete)", n_confident

    return True, "ok", n_confident


def draw_pose_overlay(
    frame: np.ndarray,
    joints_2d: np.ndarray,  # [J, 2]
    visibility: np.ndarray,  # [J]
    confidence: np.ndarray,  # [J]
    bbox_xywh: Tuple[float, float, float, float],
    vis_thresh: float = 0.3,
    conf_thresh: float = 0.3,
    pose_valid: bool = True,
    reject_reason: str = "",
):
    """Draw 3D-projected pose skeleton on frame."""
    bx, by, bw, bh = bbox_xywh

    # Draw bbox
    color_bbox = (0, 255, 0) if pose_valid else (0, 0, 255)
    cv2.rectangle(frame, (int(bx), int(by)), (int(bx+bw), int(by+bh)), color_bbox, 2)

    if not pose_valid:
        cv2.putText(frame, f"REJECTED: {reject_reason}", (int(bx), int(by)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        return

    # Draw bones
    for pi, ci in QUADRUPED_BONES_2D:
        if (visibility[pi] > vis_thresh and confidence[pi] > conf_thresh and
            visibility[ci] > vis_thresh and confidence[ci] > conf_thresh):
            p1 = (int(joints_2d[pi, 0]), int(joints_2d[pi, 1]))
            p2 = (int(joints_2d[ci, 0]), int(joints_2d[ci, 1]))
            avg_conf = (confidence[pi] + confidence[ci]) / 2
            # Color by confidence: green=high, yellow=medium
            g = int(200 * avg_conf)
            b_c = int(100 * (1 - avg_conf))
            cv2.line(frame, p1, p2, (b_c, g, 0), 2)

    # Draw joints
    for j in range(NUM_JOINTS_2D):
        if visibility[j] > vis_thresh and confidence[j] > conf_thresh:
            pt = (int(joints_2d[j, 0]), int(joints_2d[j, 1]))
            # Color by confidence
            r = int(255 * (1 - confidence[j]))
            g = int(255 * confidence[j])
            cv2.circle(frame, pt, 3, (0, g, r), -1)
            # Show confidence value
            cv2.putText(frame, f"{confidence[j]:.1f}", (pt[0]+3, pt[1]-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1)


def get_bbox_groundingdino(frame_bgr, prompts: str, device: str) -> Optional[Tuple[float, float, float, float]]:
    """Get animal bbox using GroundingDINO."""
    try:
        from hyperbone.objects.dino_adapter import DINOProposalAdapter
        adapter = DINOProposalAdapter(device=device)
        detections = adapter.detect(frame_bgr, prompts)
        if not detections:
            return None
        # Take highest confidence detection
        best = max(detections, key=lambda d: d.get("confidence", 0))
        return tuple(best["bbox_xywh"])
    except Exception as e:
        print(f"  GroundingDINO failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="HyperBonePose3D-Joint video inference")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--proposal-source", default="manual",
                       choices=["manual", "groundingdino"])
    parser.add_argument("--manual-bbox", default="", help="x,y,w,h for manual bbox")
    parser.add_argument("--prompts", default="animal. quadruped.")
    parser.add_argument("--vis-thresh", type=float, default=0.3)
    parser.add_argument("--conf-thresh", type=float, default=0.3)
    parser.add_argument("--min-joints", type=int, default=5)
    parser.add_argument("--skip-frames", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--bbox-expand", type=float, default=1.25)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    with open(args.config) as f:
        config = json.load(f)

    res = config.get("resolution", 192)
    model = HyperBonePose3DJoint(
        num_joints=config.get("num_joints", NUM_JOINTS_2D),
        input_channels=config.get("input_channels", 3),
        base_channels=config.get("base_channels", 48),
        hidden_dim=config.get("hidden_dim", 256),
        use_aux_heatmaps=True,
        heatmap_resolution=config.get("heatmap_resolution", 48),
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()
    print(f"Model: {args.model} ({model.count_parameters():,} params)")

    # Parse manual bbox
    manual_bbox = None
    if args.manual_bbox:
        parts = [float(x) for x in args.manual_bbox.split(",")]
        manual_bbox = tuple(parts)
        print(f"Manual bbox: {manual_bbox}")

    # Open video
    cap = cv2.VideoCapture(args.video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {W}x{H} @ {fps}fps, {n_total} frames")

    # Output video
    writer = cv2.VideoWriter(
        str(out_dir / "overlay.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H)
    )

    predictions = []
    stats = {"accepted": 0, "rejected": 0, "no_bbox": 0}
    frame_idx = 0
    processed = 0

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

            # Get bbox
            bbox = manual_bbox
            if bbox is None and args.proposal_source == "groundingdino":
                bbox = get_bbox_groundingdino(frame, args.prompts, args.device)

            if bbox is None:
                # No bbox -> no skeleton. Write frame as-is.
                cv2.putText(frame, "NO BBOX - no pose", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                writer.write(frame)
                stats["no_bbox"] += 1
                frame_idx += 1
                processed += 1
                continue

            # Crop and prepare input
            img_tensor, actual_bbox = crop_and_prepare(frame, bbox, res, args.bbox_expand)
            x = img_tensor.unsqueeze(0).to(device)

            # If model expects more channels, pad
            if config.get("input_channels", 3) > 3:
                extra = torch.zeros(1, config["input_channels"] - 3, res, res, device=device)
                x = torch.cat([x, extra], dim=1)

            # Inference
            pred = model(x)
            pred_xyz = pred["joint_xyz_canonical"][0].cpu().numpy()   # [J, 3]
            pred_vis = torch.sigmoid(pred["joint_visibility_logits"][0]).cpu().numpy()
            pred_conf = pred["joint_confidence"][0].cpu().numpy()
            pred_2d_norm = pred["projected_joint_xy"][0].cpu().numpy()  # [J, 2] from heatmap soft-argmax

            # Project model's 2D localizations to full frame
            joints_2d = project_pred_to_frame(pred_2d_norm, actual_bbox)

            # STRICT validation
            pose_valid, reason, n_conf = strict_pose_validation(
                pred_conf, pred_vis, joints_2d, actual_bbox,
                min_confidence=0.65,
                min_confident_joints=8,
                min_head_conf=0.50,
                min_bbox_overlap=0.80,
            )

            # Draw overlay
            overlay = frame.copy()
            draw_pose_overlay(
                overlay, joints_2d, pred_vis, pred_conf,
                actual_bbox, args.vis_thresh, args.conf_thresh,
                pose_valid, reason,
            )

            # Stats text
            avg_conf = float(pred_conf[pred_conf > 0.5].mean()) if (pred_conf > 0.5).any() else 0
            status = "OK" if pose_valid else "REJ"
            reason_short = reason[:40] if not pose_valid else ""
            cv2.putText(overlay, f"F{frame_idx} {status} conf_joints={n_conf}/{NUM_JOINTS_2D} mean={avg_conf:.2f}",
                        (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            if not pose_valid:
                cv2.putText(overlay, reason_short, (5, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (100, 100, 255), 1)

            writer.write(overlay)

            # Record prediction
            predictions.append({
                "frame_idx": frame_idx,
                "pred_xyz": pred_xyz.tolist(),
                "pred_vis": pred_vis.tolist(),
                "pred_conf": pred_conf.tolist(),
                "projected_2d_norm": pred_2d_norm.tolist(),
                "joints_2d_fullframe": joints_2d.tolist(),
                "bbox_used": list(actual_bbox),
                "pose_valid": pose_valid,
                "reject_reason": reason if not pose_valid else "",
                "n_confident_joints": n_conf,
                "mean_confidence": float(avg_conf),
            })

            if pose_valid:
                stats["accepted"] += 1
            else:
                stats["rejected"] += 1

            processed += 1
            if processed % 50 == 0:
                print(f"  {processed} frames (accepted={stats['accepted']}, "
                      f"rejected={stats['rejected']}, no_bbox={stats['no_bbox']})")

            frame_idx += 1

    cap.release()
    writer.release()

    # Write outputs
    with open(out_dir / "predictions.jsonl", 'w') as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    summary = {
        "video": args.video,
        "model": args.model,
        "frames_processed": processed,
        "proposal_source": args.proposal_source,
        "bbox_used": args.manual_bbox or "groundingdino",
        "stats": stats,
        "vis_threshold": args.vis_thresh,
        "conf_threshold": args.conf_thresh,
        "min_joints": args.min_joints,
    }
    with open(out_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone! {processed} frames.")
    print(f"  Accepted: {stats['accepted']}")
    print(f"  Rejected: {stats['rejected']}")
    print(f"  No bbox: {stats['no_bbox']}")
    print(f"  Overlay: {out_dir / 'overlay.mp4'}")


if __name__ == "__main__":
    main()
