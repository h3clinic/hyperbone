"""
Audit Anymate dataset normalization consistency.

Reports:
- bbox scale distribution
- joint coordinate range (before/after normalization)
- point cloud range (before/after normalization)
- whether all samples are centered consistently
- whether GT joints fall inside normalized object bounds
- whether predicted scale matches GT scale

Usage:
    python scripts/audit_anymate_normalization.py
    python scripts/audit_anymate_normalization.py --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def audit_dataset(pt_path: str, split_path: str, max_samples: int = 0):
    """Audit normalization of Anymate dataset."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    # Load split indices
    indices = []
    with open(split_path) as f:
        for line in f:
            record = json.loads(line.strip())
            indices.append(record["idx"])

    if max_samples > 0:
        indices = indices[:max_samples]

    print(f"[Audit] Analyzing {len(indices)} samples from {split_path}")
    print("=" * 70)

    # Collect statistics
    stats = {
        "raw_pc_bbox_diag": [],
        "raw_pc_centroid_norm": [],
        "raw_joint_range_min": [],
        "raw_joint_range_max": [],
        "raw_joint_centroid_norm": [],
        "norm_pc_range_min": [],
        "norm_pc_range_max": [],
        "norm_pc_spread": [],
        "norm_joint_range_min": [],
        "norm_joint_range_max": [],
        "norm_joint_spread": [],
        "scale_values": [],
        "joints_inside_pc_bbox": [],
        "joints_outside_unit_sphere": [],
        "n_joints": [],
        "joint_centroid_dist_to_origin": [],
        "pc_centroid_dist_to_origin_normed": [],
    }

    for idx in indices:
        d = data[idx]
        pc = d["pc"]  # [N, 3+]
        joints = d["joints"]  # [J, 3]
        n_joints = int(d["joints_num"])

        if pc.dim() == 1:
            pc = pc.unsqueeze(0)
        pc_xyz = pc[:, :3].float()
        joints_xyz = joints[:n_joints, :3].float()

        # --- Raw stats ---
        pc_min = pc_xyz.min(dim=0)[0]
        pc_max = pc_xyz.max(dim=0)[0]
        pc_bbox_diag = (pc_max - pc_min).norm().item()
        pc_centroid = pc_xyz.mean(dim=0)

        stats["raw_pc_bbox_diag"].append(pc_bbox_diag)
        stats["raw_pc_centroid_norm"].append(pc_centroid.norm().item())

        if n_joints > 0:
            j_min = joints_xyz.min(dim=0)[0]
            j_max = joints_xyz.max(dim=0)[0]
            stats["raw_joint_range_min"].append(j_min.min().item())
            stats["raw_joint_range_max"].append(j_max.max().item())
            j_centroid = joints_xyz.mean(dim=0)
            stats["raw_joint_centroid_norm"].append(j_centroid.norm().item())

        # --- Normalized stats (replicate dataset normalization) ---
        centroid = pc_xyz.mean(dim=0)
        pc_centered = pc_xyz - centroid
        scale = pc_centered.norm(dim=-1).max().clamp(min=1e-6)
        pc_norm = pc_centered / scale

        stats["scale_values"].append(scale.item())

        pc_norm_min = pc_norm.min(dim=0)[0]
        pc_norm_max = pc_norm.max(dim=0)[0]
        stats["norm_pc_range_min"].append(pc_norm_min.min().item())
        stats["norm_pc_range_max"].append(pc_norm_max.max().item())
        stats["norm_pc_spread"].append((pc_norm_max - pc_norm_min).norm().item())
        stats["pc_centroid_dist_to_origin_normed"].append(
            pc_norm.mean(dim=0).norm().item()
        )

        if n_joints > 0:
            joints_norm = (joints_xyz - centroid[:3]) / scale
            j_norm_min = joints_norm.min(dim=0)[0]
            j_norm_max = joints_norm.max(dim=0)[0]
            stats["norm_joint_range_min"].append(j_norm_min.min().item())
            stats["norm_joint_range_max"].append(j_norm_max.max().item())
            stats["norm_joint_spread"].append(
                (j_norm_max - j_norm_min).norm().item()
            )
            stats["n_joints"].append(n_joints)

            # Check if joints inside PC bbox (normalized)
            inside = (
                (joints_norm >= pc_norm_min.unsqueeze(0) - 0.1)
                & (joints_norm <= pc_norm_max.unsqueeze(0) + 0.1)
            ).all(dim=-1)
            pct_inside = inside.float().mean().item()
            stats["joints_inside_pc_bbox"].append(pct_inside)

            # Check if joints outside unit sphere
            outside = joints_norm.norm(dim=-1) > 1.0
            pct_outside = outside.float().mean().item()
            stats["joints_outside_unit_sphere"].append(pct_outside)

            # Joint centroid distance to origin (after normalization)
            j_centroid_norm = joints_norm.mean(dim=0)
            stats["joint_centroid_dist_to_origin"].append(
                j_centroid_norm.norm().item()
            )

    # --- Print report ---
    def report(name, values, unit=""):
        arr = np.array(values)
        print(f"  {name}:")
        print(f"    mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}  "
              f"median={np.median(arr):.4f} {unit}")
        # Distribution buckets
        p5, p25, p75, p95 = np.percentile(arr, [5, 25, 75, 95])
        print(f"    p5={p5:.4f}  p25={p25:.4f}  p75={p75:.4f}  p95={p95:.4f}")

    print("\n--- RAW DATA (before normalization) ---")
    report("PC bbox diagonal", stats["raw_pc_bbox_diag"])
    report("PC centroid norm (distance from origin)", stats["raw_pc_centroid_norm"])
    if stats["raw_joint_range_min"]:
        report("Joint coordinate min", stats["raw_joint_range_min"])
        report("Joint coordinate max", stats["raw_joint_range_max"])
        report("Joint centroid norm", stats["raw_joint_centroid_norm"])

    print("\n--- NORMALIZED DATA (after dataset processing) ---")
    report("Scale values used", stats["scale_values"])
    report("PC range min (normalized)", stats["norm_pc_range_min"])
    report("PC range max (normalized)", stats["norm_pc_range_max"])
    report("PC spread (bbox diagonal, normalized)", stats["norm_pc_spread"])
    report("PC centroid dist to origin (should be ~0)", stats["pc_centroid_dist_to_origin_normed"])

    if stats["norm_joint_range_min"]:
        report("Joint range min (normalized)", stats["norm_joint_range_min"])
        report("Joint range max (normalized)", stats["norm_joint_range_max"])
        report("Joint spread (bbox diagonal, normalized)", stats["norm_joint_spread"])
        report("Joint centroid dist to origin (normalized)", stats["joint_centroid_dist_to_origin"])

    print("\n--- JOINT-PC ALIGNMENT ---")
    if stats["joints_inside_pc_bbox"]:
        report("Fraction of joints inside PC bbox (+0.1 margin)", stats["joints_inside_pc_bbox"])
        report("Fraction of joints outside unit sphere", stats["joints_outside_unit_sphere"])
        report("Number of joints per asset", stats["n_joints"])

    # Diagnosis
    print("\n--- DIAGNOSIS ---")
    arr_outside = np.array(stats["joints_outside_unit_sphere"])
    arr_jcent = np.array(stats["joint_centroid_dist_to_origin"])
    arr_jspread = np.array(stats["norm_joint_spread"])

    issues = []
    if arr_outside.mean() > 0.1:
        issues.append(f"WARNING: {arr_outside.mean()*100:.1f}% of joints fall OUTSIDE unit sphere on average")
    if arr_jcent.mean() > 0.15:
        issues.append(f"WARNING: Joint centroids are far from origin (mean={arr_jcent.mean():.4f}). "
                      "Normalization may not center joints properly.")
    if arr_jspread.mean() < 0.5:
        issues.append(f"WARNING: Joint spread is small (mean={arr_jspread.mean():.4f}). "
                      "Joints may be naturally clustered.")
    if arr_jspread.std() > 0.5:
        issues.append(f"WARNING: Joint spread has high variance (std={arr_jspread.std():.4f}). "
                      "Variable topology makes same-slot training hard.")

    if not issues:
        print("  No major normalization issues detected.")
    else:
        for iss in issues:
            print(f"  {iss}")

    # Save results
    out_path = Path(split_path).parent / "normalization_audit.json"
    summary = {}
    for k, v in stats.items():
        arr = np.array(v)
        summary[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "median": float(np.median(arr)),
        }
    summary["n_samples"] = len(indices)
    summary["issues"] = issues

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Audit] Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Audit Anymate normalization")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=0, help="0=all")
    args = parser.parse_args()

    split_path = f"{args.splits_dir}/{args.split}.jsonl"
    audit_dataset(args.pt, split_path, args.max_samples)


if __name__ == "__main__":
    main()
