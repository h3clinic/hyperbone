"""
3D pose evaluation metrics.

Compare predicted 3D pose against ground-truth armature pose.
"""
from __future__ import annotations

import numpy as np
from typing import List, Dict, Optional, Tuple
from .schema import PoseFrame3D, Joint3D


def mpjpe(pred_joints: List[Joint3D], gt_joints: List[Joint3D], use_canonical: bool = False) -> float:
    """
    Mean Per-Joint Position Error (MPJPE).

    Computes average Euclidean distance between corresponding joints.
    If use_canonical=False, uses world_xyz. Otherwise requires canonical_xyz
    to be set on joints (not supported by default schema - use world_xyz).
    """
    if not pred_joints or not gt_joints:
        return float('inf')

    # Match by ID
    gt_map = {j.id: j for j in gt_joints}
    errors = []
    for pj in pred_joints:
        if pj.id in gt_map:
            gj = gt_map[pj.id]
            pred_pos = np.array(pj.world_xyz)
            gt_pos = np.array(gj.world_xyz)
            errors.append(float(np.linalg.norm(pred_pos - gt_pos)))

    return float(np.mean(errors)) if errors else float('inf')


def mpjpe_by_name(pred_joints: List[Joint3D], gt_joints: List[Joint3D]) -> float:
    """MPJPE matching joints by name instead of ID."""
    gt_map = {j.name: j for j in gt_joints}
    errors = []
    for pj in pred_joints:
        if pj.name in gt_map:
            gj = gt_map[pj.name]
            pred_pos = np.array(pj.world_xyz)
            gt_pos = np.array(gj.world_xyz)
            errors.append(float(np.linalg.norm(pred_pos - gt_pos)))
    return float(np.mean(errors)) if errors else float('inf')


def pck_3d(
    pred_joints: List[Joint3D],
    gt_joints: List[Joint3D],
    threshold: float = 0.1,
) -> float:
    """
    Percentage of Correct Keypoints in 3D (PCK-3D).

    A joint is correct if its error is below threshold (in same units as coordinates).
    """
    gt_map = {j.id: j for j in gt_joints}
    correct = 0
    total = 0

    for pj in pred_joints:
        if pj.id in gt_map:
            gj = gt_map[pj.id]
            dist = np.linalg.norm(np.array(pj.world_xyz) - np.array(gj.world_xyz))
            if dist < threshold:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def bone_length_error(pred_frame: PoseFrame3D, gt_frame: PoseFrame3D) -> Dict[str, float]:
    """Compute per-bone length error between prediction and GT."""
    from .canonicalize import compute_bone_lengths

    pred_lengths = compute_bone_lengths(pred_frame)
    gt_lengths = compute_bone_lengths(gt_frame)

    errors = {}
    for bone_name, gt_len in gt_lengths.items():
        if bone_name in pred_lengths:
            errors[bone_name] = abs(pred_lengths[bone_name] - gt_len)

    return errors


def mean_bone_length_error(pred_frame: PoseFrame3D, gt_frame: PoseFrame3D) -> float:
    """Mean absolute bone length error."""
    errors = bone_length_error(pred_frame, gt_frame)
    if not errors:
        return float('inf')
    return float(np.mean(list(errors.values())))


def root_position_error(pred_frame: PoseFrame3D, gt_frame: PoseFrame3D) -> float:
    """Error of root joint position."""
    from .canonicalize import find_root_joint

    pred_root = find_root_joint(pred_frame)
    gt_root = find_root_joint(gt_frame)

    if pred_root is None or gt_root is None:
        return float('inf')

    return float(np.linalg.norm(
        np.array(pred_root.world_xyz) - np.array(gt_root.world_xyz)
    ))


def reprojection_error(
    pred_joints: List[Joint3D],
    gt_joints: List[Joint3D],
) -> float:
    """
    Mean reprojection error in pixels.
    Compares image_xy between predicted and GT joints.
    """
    gt_map = {j.id: j for j in gt_joints}
    errors = []

    for pj in pred_joints:
        if pj.id in gt_map:
            gj = gt_map[pj.id]
            pred_px = np.array(pj.image_xy)
            gt_px = np.array(gj.image_xy)
            errors.append(float(np.linalg.norm(pred_px - gt_px)))

    return float(np.mean(errors)) if errors else float('inf')


def temporal_acceleration_error(
    frames: List[PoseFrame3D],
    gt_frames: List[PoseFrame3D],
) -> float:
    """
    Temporal acceleration error: compares joint acceleration profiles.
    High values indicate jittery predictions.
    """
    if len(frames) < 3 or len(gt_frames) < 3:
        return 0.0

    def compute_accelerations(frame_list):
        """Compute per-joint acceleration across frames."""
        accels = []
        for i in range(1, len(frame_list) - 1):
            for j in frame_list[i].joints:
                # Find same joint in adjacent frames
                prev_j = frame_list[i-1].joint_by_id(j.id)
                next_j = frame_list[i+1].joint_by_id(j.id)
                if prev_j and next_j:
                    pos_prev = np.array(prev_j.world_xyz)
                    pos_curr = np.array(j.world_xyz)
                    pos_next = np.array(next_j.world_xyz)
                    accel = pos_prev + pos_next - 2 * pos_curr
                    accels.append(np.linalg.norm(accel))
        return accels

    pred_accels = compute_accelerations(frames)
    gt_accels = compute_accelerations(gt_frames)

    if not pred_accels or not gt_accels:
        return 0.0

    # Compare mean acceleration magnitude
    return abs(float(np.mean(pred_accels)) - float(np.mean(gt_accels)))


def joint_visibility_coverage(pred_joints: List[Joint3D], gt_joints: List[Joint3D]) -> float:
    """Fraction of visible GT joints that have a corresponding prediction."""
    visible_gt = [j for j in gt_joints if j.visible]
    if not visible_gt:
        return 1.0

    pred_ids = {j.id for j in pred_joints}
    covered = sum(1 for j in visible_gt if j.id in pred_ids)
    return covered / len(visible_gt)


def evaluate_pose3d(
    pred_frame: PoseFrame3D,
    gt_frame: PoseFrame3D,
    pck_thresholds: List[float] = [0.05, 0.10, 0.20],
) -> Dict:
    """
    Full evaluation of a single predicted pose frame against GT.

    Returns dict of all metrics.
    """
    results = {
        "frame_idx": gt_frame.frame_idx,
        "mpjpe": mpjpe(pred_frame.joints, gt_frame.joints),
        "root_position_error": root_position_error(pred_frame, gt_frame),
        "mean_bone_length_error": mean_bone_length_error(pred_frame, gt_frame),
        "reprojection_error_px": reprojection_error(pred_frame.joints, gt_frame.joints),
        "joint_visibility_coverage": joint_visibility_coverage(pred_frame.joints, gt_frame.joints),
    }

    for thresh in pck_thresholds:
        results[f"pck3d_{thresh:.2f}"] = pck_3d(pred_frame.joints, gt_frame.joints, thresh)

    return results


def evaluate_sequence(
    pred_frames: List[PoseFrame3D],
    gt_frames: List[PoseFrame3D],
    pck_thresholds: List[float] = [0.05, 0.10, 0.20],
) -> Dict:
    """Evaluate a full sequence of predicted poses against GT."""
    # Match frames by frame_idx
    gt_map = {f.frame_idx: f for f in gt_frames}

    per_frame = []
    for pf in pred_frames:
        if pf.frame_idx in gt_map:
            metrics = evaluate_pose3d(pf, gt_map[pf.frame_idx], pck_thresholds)
            per_frame.append(metrics)

    if not per_frame:
        return {"error": "no matching frames", "n_evaluated": 0}

    # Aggregate
    all_mpjpe = [m["mpjpe"] for m in per_frame if m["mpjpe"] != float('inf')]
    all_reproj = [m["reprojection_error_px"] for m in per_frame if m["reprojection_error_px"] != float('inf')]

    summary = {
        "n_evaluated": len(per_frame),
        "mpjpe_mean": float(np.mean(all_mpjpe)) if all_mpjpe else float('inf'),
        "mpjpe_p95": float(np.percentile(all_mpjpe, 95)) if all_mpjpe else float('inf'),
        "reprojection_error_mean_px": float(np.mean(all_reproj)) if all_reproj else float('inf'),
        "reprojection_error_p95_px": float(np.percentile(all_reproj, 95)) if all_reproj else float('inf'),
    }

    for thresh in pck_thresholds:
        key = f"pck3d_{thresh:.2f}"
        vals = [m[key] for m in per_frame]
        summary[f"{key}_mean"] = float(np.mean(vals))

    # Temporal smoothness
    if len(pred_frames) >= 3 and len(gt_frames) >= 3:
        summary["temporal_acceleration_error"] = temporal_acceleration_error(pred_frames, gt_frames)

    summary["per_frame"] = per_frame
    return summary
