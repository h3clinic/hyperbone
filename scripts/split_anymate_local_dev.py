"""
Split Anymate_test.pt (5,598 samples) into local dev train/val/test.

Split is by ASSET (not by frame/sample) to prevent shape leakage.
80% train / 10% val / 10% test by unique asset name.

This is NOT an official Anymate benchmark split.
It is an internal HyperBone engineering dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import torch


def deterministic_asset_split(
    asset_names: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Dict[str, str]:
    """Assign each unique asset to train/val/test deterministically via hash."""
    assignments = {}
    for name in sorted(set(asset_names)):
        h = int(hashlib.sha256(name.encode()).hexdigest(), 16) % 1000
        if h < int(train_ratio * 1000):
            assignments[name] = "train"
        elif h < int((train_ratio + val_ratio) * 1000):
            assignments[name] = "val"
        else:
            assignments[name] = "test"
    return assignments


def main():
    parser = argparse.ArgumentParser(description="Split Anymate local dev dataset")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--out", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Split] Loading {args.pt}...")
    data = torch.load(args.pt, map_location="cpu", weights_only=False)
    print(f"[Split] Loaded {len(data)} samples")

    # Extract asset names
    asset_names = [d["name"] for d in data]
    unique_assets = sorted(set(asset_names))
    print(f"[Split] Unique assets: {len(unique_assets)}")

    # Deterministic split
    assignments = deterministic_asset_split(
        asset_names, args.train_ratio, args.val_ratio
    )

    # Partition samples
    splits = {"train": [], "val": [], "test": []}
    for idx, d in enumerate(data):
        name = d["name"]
        split = assignments[name]
        record = {
            "idx": idx,
            "name": name,
            "split": split,
            "joints_num": int(d["joints_num"]),
            "bones_num": int(d["bones_num"]),
            "mesh_verts": int(d["mesh_pc"].shape[0]) if "mesh_pc" in d else 0,
            "pc_points": int(d["pc"].shape[0]) if "pc" in d else 0,
        }
        splits[split].append(record)

    # Write split files
    for split_name, records in splits.items():
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"[Split] {split_name}: {len(records)} samples → {path}")

    # Summary
    split_assets = {}
    for split_name, records in splits.items():
        split_assets[split_name] = len(set(r["name"] for r in records))

    summary = {
        "total_samples": len(data),
        "total_assets": len(unique_assets),
        "train_samples": len(splits["train"]),
        "val_samples": len(splits["val"]),
        "test_samples": len(splits["test"]),
        "train_assets": split_assets["train"],
        "val_assets": split_assets["val"],
        "test_assets": split_assets["test"],
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": round(1.0 - args.train_ratio - args.val_ratio, 2),
        "split_method": "deterministic_sha256_by_asset",
        "source": args.pt,
        "note": "LOCAL DEV SPLIT - NOT official Anymate benchmark",
    }
    with open(out_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Split] Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
