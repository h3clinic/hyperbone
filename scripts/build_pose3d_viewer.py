"""
Build a self-contained Pose3D viewer directory.

Copies viewer_pose3d/ files and pose data, writes a manifest so the viewer
can load everything locally.

Usage:
  python scripts/build_pose3d_viewer.py \
    --gt outputs/pose3d/wolf_dataset/pose3d_gt.jsonl \
    --hyperbone outputs/pose3d/wolf_hyperbone/graphs3d/hyperbone_graph3d.jsonl \
    --out outputs/pose3d/wolf_pose3d_viewer
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build Pose3D viewer")
    parser.add_argument("--gt", required=True, help="GT armature JSONL")
    parser.add_argument("--hyperbone", help="HyperBone canonical 3D graph JSONL")
    parser.add_argument("--out", required=True, help="Output viewer directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    viewer_src = Path("viewer_pose3d")
    if not viewer_src.exists():
        print("ERROR: viewer_pose3d/ directory not found")
        return

    # Copy viewer files
    for f in viewer_src.glob("*"):
        if f.is_file():
            shutil.copy2(f, out_dir / f.name)
    src_dir = viewer_src / "src"
    if src_dir.exists():
        dst_src = out_dir / "src"
        dst_src.mkdir(exist_ok=True)
        for f in src_dir.glob("*"):
            if f.is_file():
                shutil.copy2(f, dst_src / f.name)

    # Copy data
    data_dir = out_dir / "data"
    data_dir.mkdir(exist_ok=True)

    gt_path = Path(args.gt)
    if gt_path.exists():
        shutil.copy2(gt_path, data_dir / "pose3d_gt.jsonl")
        print(f"  Copied GT: {gt_path}")
    else:
        print(f"  WARNING: GT file not found: {gt_path}")

    if args.hyperbone:
        hb_path = Path(args.hyperbone)
        if hb_path.exists():
            shutil.copy2(hb_path, data_dir / "hyperbone_graph3d.jsonl")
            print(f"  Copied HyperBone: {hb_path}")
        else:
            print(f"  WARNING: HyperBone file not found: {hb_path}")

    # Write manifest
    manifest = {
        "gt_file": "data/pose3d_gt.jsonl",
        "hyperbone_file": "data/hyperbone_graph3d.jsonl" if args.hyperbone else None,
        "gt_source": str(gt_path),
        "hyperbone_source": str(args.hyperbone) if args.hyperbone else None,
    }
    with open(data_dir / "data_manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nViewer built: {out_dir}")
    print(f"Open: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
