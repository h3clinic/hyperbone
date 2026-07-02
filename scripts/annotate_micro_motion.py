"""HyperBone Track C2 — annotate a micro-motion dataset with mobility features.

Runs the mobility feature layer over every clip in a dataset dir and writes a
compact feature table plus a separability report (mean feature by class).

Usage:
    python scripts/annotate_micro_motion.py \
        --dataset outputs/track_c_micro_motion_1000 \
        --out outputs/track_c_micro_motion_1000
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.motion.mobility import annotate_clip, FEATURE_NAMES

CLASSES = ["valid", "bone_length_scale_error", "detached_child",
           "wrong_parent_motion", "temporal_jitter",
           "impossible_large_rotation", "swapped_limb_motion"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    clip_paths = sorted(glob.glob(str(Path(args.dataset) / "clips" / "*.npz")))
    print(f"Annotating {len(clip_paths)} clips...", flush=True)

    X = []
    y_valid = []
    y_class = []
    clip_ids = []
    asset_ids = []
    presets = []
    mob_counts = {t: 0 for t in
                  ["fixed", "hinge", "ball", "flexible_chain", "root"]}
    role_counts = {r: 0 for r in
                   ["rigid_bone", "flexible_branch", "soft_field_proxy"]}

    for i, cp in enumerate(clip_paths):
        z = np.load(cp, allow_pickle=True)
        world = z["joints_world"].astype(np.float64)
        edges = z["edges"].astype(np.int64)
        parents = z["parents"].astype(np.int64)
        rest = z["joints_rest"].astype(np.float64)
        fps = int(z["fps"])
        ann = annotate_clip(world, edges, parents, rest, fps)
        X.append([ann["features"][k] for k in FEATURE_NAMES])
        valid = bool(z["is_valid_motion"])
        ctype = str(z["corruption_type"]) if not valid else "valid"
        y_valid.append(valid)
        y_class.append(CLASSES.index(ctype) if ctype in CLASSES else 0)
        clip_ids.append(str(z["clip_id"]))
        asset_ids.append(int(z["asset_idx"]))
        presets.append(str(z["preset"]))
        for t in ann["mobility_type"]:
            if t in mob_counts:
                mob_counts[t] += 1
        for r in ann["motion_role"]:
            if r in role_counts:
                role_counts[r] += 1
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(clip_paths)}", flush=True)

    X = np.array(X, dtype=np.float32)
    y_valid = np.array(y_valid, dtype=bool)
    y_class = np.array(y_class, dtype=np.int64)
    asset_ids = np.array(asset_ids, dtype=np.int64)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "features.npz",
        X=X, y_valid=y_valid, y_class=y_class,
        asset_idx=asset_ids, clip_ids=np.array(clip_ids),
        presets=np.array(presets),
        feature_names=np.array(FEATURE_NAMES), classes=np.array(CLASSES),
    )

    # separability report: mean feature by class
    report = {"n_clips": int(len(X)), "n_valid": int(y_valid.sum()),
              "n_invalid": int((~y_valid).sum()),
              "feature_names": FEATURE_NAMES, "classes": CLASSES,
              "mobility_type_counts": mob_counts,
              "motion_role_counts": role_counts,
              "mean_feature_by_class": {}}
    print(f"\n{'class':<26}" + "".join(f"{n[:11]:>13}" for n in FEATURE_NAMES[:6]), flush=True)
    for ci, cls in enumerate(CLASSES):
        m = X[y_class == ci]
        if len(m) == 0:
            continue
        means = m.mean(axis=0)
        report["mean_feature_by_class"][cls] = {FEATURE_NAMES[k]: float(means[k])
                                                for k in range(len(FEATURE_NAMES))}
        print(f"  {cls:<24}" + "".join(f"{means[k]:13.4f}" for k in range(6)), flush=True)

    # quick rules-separability check: can simple thresholds flag the two
    # length-preserving corruptions?
    def mean_feat(cls, feat):
        idx = FEATURE_NAMES.index(feat)
        m = X[y_class == CLASSES.index(cls)]
        return float(m[:, idx].mean()) if len(m) else float("nan")

    checks = {
        "impossible_large_rotation: max_joint_angle_range >> valid":
            mean_feat("impossible_large_rotation", "max_joint_angle_range")
            > 2.0 * mean_feat("valid", "max_joint_angle_range"),
        "swapped_limb_motion: motion_symmetry < valid":
            mean_feat("swapped_limb_motion", "motion_symmetry")
            < mean_feat("valid", "motion_symmetry"),
        "temporal_jitter: temporal_smoothness < valid":
            mean_feat("temporal_jitter", "temporal_smoothness")
            < mean_feat("valid", "temporal_smoothness"),
        "length corruptions: bone_length_deviation >> valid":
            mean_feat("bone_length_scale_error", "bone_length_deviation")
            > 0.02,
    }
    report["rules_separability_checks"] = {k: bool(v) for k, v in checks.items()}
    print("\nRules separability checks:", flush=True)
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}", flush=True)

    with open(out_dir / "mobility_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved features -> {out_dir/'features.npz'}", flush=True)
    print(f"Report -> {out_dir/'mobility_report.json'}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
