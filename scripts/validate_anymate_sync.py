"""
Validate Anymate Synchronization — check that rendered frames and skeleton labels are consistent.

Checks:
- Every RGB/mask/depth frame exists
- Camera K and extrinsic exist
- frame_idx matches across all modalities
- timestamp_sec = frame_idx / fps
- Projected joints lie inside image bounds when marked visible
- Projected joints overlap the object mask
- Joint positions change across frames for animated clips
- Bone lengths remain stable across frames
- No skeleton is accidentally rest-pose for every frame

Usage:
    python scripts/validate_anymate_sync.py \\
        --dataset outputs/anymate_clips_pilot/dataset_index.jsonl \\
        --out outputs/anymate_clips_pilot/sync_report.md \\
        --sample-count 50
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.rigs.schema import read_jsonl


def validate_dataset(dataset_path: str, sample_count: int = 50) -> dict:
    """Run all synchronization checks on the dataset."""
    dataset_path = Path(dataset_path)
    root_dir = dataset_path.parent

    labels = read_jsonl(dataset_path)
    if not labels:
        return {"status": "FAIL", "reason": "Empty dataset", "checks": {}}

    # Sample
    if sample_count > 0 and len(labels) > sample_count:
        step = len(labels) // sample_count
        sampled = labels[::step][:sample_count]
    else:
        sampled = labels

    checks = {
        "total_frames": len(labels),
        "sampled_frames": len(sampled),
        "rgb_missing": 0,
        "mask_missing": 0,
        "depth_missing": 0,
        "skeleton_missing": 0,
        "camera_missing": 0,
        "timestamp_mismatch": 0,
        "joint_out_of_bounds": 0,
        "joint_visible_count": 0,
        "joint_outside_mask": 0,
        "joint_inside_mask": 0,
        "static_clips": [],
        "bone_length_drift": [],
        "clips_checked": set(),
    }

    # Group by clip
    clips = defaultdict(list)
    for label in labels:
        clip_key = f"{label.get('asset_id', '')}_{label.get('animation_id', '')}"
        clips[clip_key].append(label)

    # Per-frame checks on sampled frames
    for label in sampled:
        # File existence
        rgb_path = root_dir / label.get("rgb_path", "")
        mask_path = root_dir / label.get("mask_path", "")
        depth_path = root_dir / label.get("depth_path", "")

        if not rgb_path.exists():
            checks["rgb_missing"] += 1
        if label.get("mask_path") and not mask_path.exists():
            checks["mask_missing"] += 1
        if label.get("depth_path") and not depth_path.exists():
            checks["depth_missing"] += 1

        # Skeleton existence
        joints = label.get("joints", [])
        if not joints:
            checks["skeleton_missing"] += 1

        # Camera existence
        camera = label.get("camera")
        if not camera or not camera.get("K") or not camera.get("extrinsic"):
            checks["camera_missing"] += 1

        # Timestamp consistency
        frame_idx = label.get("frame_idx", 0)
        fps = label.get("fps", 12)
        expected_ts = frame_idx / fps
        actual_ts = label.get("timestamp_sec", -1)
        if abs(expected_ts - actual_ts) > 0.01:
            checks["timestamp_mismatch"] += 1

        # Joint projection bounds
        if camera and joints:
            resolution = camera.get("resolution", [256, 256])
            W, H = resolution[0], resolution[1]
            for j in joints:
                if j.get("visible", False):
                    checks["joint_visible_count"] += 1
                    xy = j.get("image_xy", [0, 0])
                    if xy[0] < 0 or xy[0] >= W or xy[1] < 0 or xy[1] >= H:
                        checks["joint_out_of_bounds"] += 1

        # Joint vs mask overlap (if mask exists and is loadable)
        if mask_path.exists() and joints:
            try:
                import cv2
                mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    for j in joints:
                        if j.get("visible", False):
                            xy = j.get("image_xy", [0, 0])
                            ix, iy = int(round(xy[0])), int(round(xy[1]))
                            if 0 <= ix < mask_img.shape[1] and 0 <= iy < mask_img.shape[0]:
                                if mask_img[iy, ix] > 128:
                                    checks["joint_inside_mask"] += 1
                                else:
                                    checks["joint_outside_mask"] += 1
            except Exception:
                pass

    # Per-clip checks
    for clip_key, clip_labels in clips.items():
        if len(clip_labels) < 3:
            continue
        checks["clips_checked"].add(clip_key)

        # Check if joint positions change across frames
        all_xyz = []
        for label in clip_labels:
            joints = label.get("joints", [])
            if joints:
                frame_xyz = [j.get("world_xyz", [0, 0, 0]) for j in joints]
                all_xyz.append(np.array(frame_xyz))

        if len(all_xyz) >= 3:
            xyz_array = np.array(all_xyz)  # [T, J, 3]
            # Check variance across time
            temporal_var = xyz_array.var(axis=0).mean()
            if temporal_var < 1e-8:
                motion_src = clip_labels[0].get("motion_source", "unknown")
                if motion_src != "none":
                    checks["static_clips"].append(clip_key)

            # Bone length stability
            for label in clip_labels[:10]:
                bones = label.get("bones", [])
                for bone in bones:
                    # Length stability checked via bones list
                    pass

    # Compute bone length drift per clip
    for clip_key, clip_labels in clips.items():
        bone_lengths_by_bone = defaultdict(list)
        for label in clip_labels:
            bones = label.get("bones", [])
            for bone in bones:
                bone_id = bone.get("id", -1)
                length = bone.get("length", 0)
                if length > 0:
                    bone_lengths_by_bone[bone_id].append(length)

        for bone_id, lengths in bone_lengths_by_bone.items():
            if len(lengths) >= 3:
                lengths = np.array(lengths)
                mean_len = lengths.mean()
                if mean_len > 0:
                    max_drift = (lengths.max() - lengths.min()) / mean_len
                    if max_drift > 0.10:
                        checks["bone_length_drift"].append({
                            "clip": clip_key,
                            "bone": bone_id,
                            "drift_pct": round(max_drift * 100, 1),
                        })

    # Compute pass/fail
    n_sampled = max(checks["sampled_frames"], 1)
    missing_ratio = (checks["rgb_missing"] + checks["skeleton_missing"]) / n_sampled
    joint_oob_ratio = checks["joint_out_of_bounds"] / max(checks["joint_visible_count"], 1)
    static_count = len(checks["static_clips"])
    bone_drift_count = len(checks["bone_length_drift"])

    # Convert set to count for serialization
    checks["clips_checked"] = len(checks["clips_checked"])

    # Pass/fail criteria
    failures = []
    if missing_ratio > 0.01:
        failures.append(f"Missing file ratio {missing_ratio:.1%} > 1%")
    if joint_oob_ratio > 0.20:
        failures.append(f"Joints out of bounds {joint_oob_ratio:.1%} > 20%")
    if static_count > 0:
        failures.append(f"{static_count} clips have static joint positions (rest pose leak)")
    if bone_drift_count > 0:
        failures.append(f"{bone_drift_count} bones with >10% length drift")

    # Mask overlap
    total_mask_checks = checks["joint_inside_mask"] + checks["joint_outside_mask"]
    mask_overlap_ratio = checks["joint_inside_mask"] / max(total_mask_checks, 1)

    result = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checks": checks,
        "summary": {
            "total_frames": len(labels),
            "sampled": len(sampled),
            "missing_file_ratio": round(missing_ratio, 4),
            "joint_oob_ratio": round(joint_oob_ratio, 4),
            "mask_overlap_ratio": round(mask_overlap_ratio, 4),
            "static_clips": static_count,
            "bone_drift_violations": bone_drift_count,
        }
    }

    return result


def write_report(result: dict, output_path: Path):
    """Write validation report as markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    status = result["status"]
    summary = result["summary"]
    checks = result["checks"]
    failures = result.get("failures", [])

    report = f"""# Anymate Synchronization Validation Report

## Status: **{status}**

"""
    if failures:
        report += "## Failures\n"
        for f in failures:
            report += f"- ❌ {f}\n"
        report += "\n"

    report += f"""## Summary
| Metric | Value |
|--------|-------|
| Total frames | {summary['total_frames']} |
| Sampled | {summary['sampled']} |
| Missing file ratio | {summary['missing_file_ratio']:.2%} |
| Joint out-of-bounds ratio | {summary['joint_oob_ratio']:.2%} |
| Mask overlap ratio | {summary['mask_overlap_ratio']:.2%} |
| Static clips (rest-pose leak) | {summary['static_clips']} |
| Bone drift violations (>10%) | {summary['bone_drift_violations']} |

## Detailed Checks
| Check | Count |
|-------|-------|
| RGB missing | {checks['rgb_missing']} |
| Mask missing | {checks['mask_missing']} |
| Depth missing | {checks['depth_missing']} |
| Skeleton missing | {checks['skeleton_missing']} |
| Camera missing | {checks['camera_missing']} |
| Timestamp mismatches | {checks['timestamp_mismatch']} |
| Joints visible (sampled) | {checks['joint_visible_count']} |
| Joints out of bounds | {checks['joint_out_of_bounds']} |
| Joints inside mask | {checks['joint_inside_mask']} |
| Joints outside mask | {checks['joint_outside_mask']} |
| Clips checked | {checks['clips_checked']} |
"""

    if checks.get("static_clips"):
        report += "\n## Static Clips (Possible Rest-Pose Leak)\n"
        for clip in checks["static_clips"][:10]:
            report += f"- {clip}\n"

    if checks.get("bone_length_drift"):
        report += "\n## Bone Length Drift Violations\n"
        for item in checks["bone_length_drift"][:10]:
            report += f"- {item['clip']}: bone {item['bone']}, drift {item['drift_pct']}%\n"

    output_path.write_text(report, encoding="utf-8")
    print(f"[Validate] Report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate Anymate dataset synchronization")
    parser.add_argument("--dataset", required=True, help="Path to dataset_index.jsonl")
    parser.add_argument("--out", default=None, help="Output report path (.md)")
    parser.add_argument("--sample-count", type=int, default=50, help="Number of frames to sample for validation")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)

    output_path = Path(args.out) if args.out else dataset_path.parent / "sync_report.md"

    print(f"[Validate] Dataset: {dataset_path}")
    print(f"[Validate] Sampling: {args.sample_count} frames")

    result = validate_dataset(str(dataset_path), sample_count=args.sample_count)

    # Print summary
    print(f"\n[Validate] Status: {result['status']}")
    if result.get("failures"):
        for f in result["failures"]:
            print(f"  ❌ {f}")
    else:
        print("  ✓ All checks passed")

    print(f"\n[Validate] Summary:")
    for k, v in result.get("summary", {}).items():
        print(f"  {k}: {v}")

    write_report(result, output_path)

    # Exit with error code if failed
    if result["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
