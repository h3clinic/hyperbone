"""
Evaluate trained HyperBone3D v0 model.

Usage:
  python scripts/eval_hyperbone3d_v0.py \
    --model outputs/models/hyperbone3d_v0/model.pt \
    --config outputs/models/hyperbone3d_v0/config.json \
    --dataset outputs/pose3d/wolf_dataset \
    --out outputs/models/hyperbone3d_v0/eval

  # Or from video:
  python scripts/eval_hyperbone3d_v0.py \
    --model outputs/models/hyperbone3d_v0/model.pt \
    --config outputs/models/hyperbone3d_v0/config.json \
    --video output/fox3d/fox3d_walk.mp4 \
    --gt output/fox3d/fox_armature_gt.jsonl \
    --asset-id Fox.glb \
    --out outputs/models/hyperbone3d_v0/eval
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hyperbone.models.hyperbone3d import HyperBone3D
from hyperbone.pose3d.dataset import Pose3DDataset, Pose3DDatasetFromVideo
from hyperbone.pose3d.joint_map import QUADRUPED_JOINTS, QUADRUPED_BONES, NUM_JOINTS


def collate_fn(batch):
    result = {}
    tensor_keys = ["rgb", "mask", "depth", "bbox_xywh", "joint_xyz_canonical",
                   "joint_visible", "bone_edges", "scale", "root_xyz"]
    for key in tensor_keys:
        result[key] = torch.stack([b[key] for b in batch])
    result["asset_id"] = [b["asset_id"] for b in batch]
    result["frame_idx"] = [b["frame_idx"] for b in batch]
    return result


def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HyperBone3D v0 Evaluation")
    print("=" * 60)

    # Load config
    config_path = Path(args.config)
    with open(config_path) as f:
        config = json.load(f)

    # Load model
    model = HyperBone3D(
        num_joints=config.get("num_joints", NUM_JOINTS),
        input_channels=config.get("input_channels", 5),
        base_channels=config.get("base_channels", 64),
        hidden_dim=config.get("hidden_dim", 256),
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()
    print(f"  Model: {args.model} ({model.count_parameters():,} params)")

    # Load dataset
    if args.video and args.gt:
        dataset = Pose3DDatasetFromVideo(
            video_path=args.video,
            gt_jsonl_path=args.gt,
            resolution=(config.get("resolution", 256), config.get("resolution", 256)),
            asset_id=args.asset_id,
        )
    elif args.dataset:
        dataset = Pose3DDataset(
            root_dir=args.dataset,
            resolution=(config.get("resolution", 256), config.get("resolution", 256)),
            asset_id=args.asset_id,
        )
    else:
        raise ValueError("Provide --dataset or --video + --gt")

    # Apply split
    split = getattr(args, 'split', 'all')
    val_frac = getattr(args, 'val_split', 0.2)
    if split in ("train", "val") and val_frac > 0:
        from torch.utils.data import Subset
        n = len(dataset)
        n_val = int(n * val_frac)
        n_train = n - n_val
        if split == "train":
            dataset = Subset(dataset, list(range(n_train)))
            print(f"  Split: evaluating TRAIN ({n_train} frames, first {1-val_frac:.0%})")
        else:
            dataset = Subset(dataset, list(range(n_train, n)))
            print(f"  Split: evaluating VAL ({n_val} frames, last {val_frac:.0%})")
    else:
        print(f"  Split: all ({len(dataset)} frames)")

    print(f"  Eval samples: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    # Evaluate
    per_joint_errors = {name: [] for name in QUADRUPED_JOINTS}
    all_errors = []
    all_vis_preds = []
    all_vis_gt = []
    bone_errors = []
    predictions = []

    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            mask = batch["mask"].to(device)
            depth = batch["depth"].to(device)
            x = torch.cat([rgb, mask, depth], dim=1)

            gt_xyz = batch["joint_xyz_canonical"].to(device)
            gt_vis = batch["joint_visible"].to(device)

            pred = model(x)
            pred_xyz = pred["joint_xyz_canonical"]
            pred_vis = torch.sigmoid(pred["joint_visibility_logits"])

            B = pred_xyz.shape[0]
            for b in range(B):
                for j in range(NUM_JOINTS):
                    if gt_vis[b, j] > 0.5:
                        err = torch.norm(pred_xyz[b, j] - gt_xyz[b, j]).item()
                        per_joint_errors[QUADRUPED_JOINTS[j]].append(err)
                        all_errors.append(err)

                all_vis_preds.append(pred_vis[b].cpu().numpy())
                all_vis_gt.append(gt_vis[b].cpu().numpy())

                # Bone length errors
                for pi, ci in QUADRUPED_BONES:
                    if gt_vis[b, pi] > 0.5 and gt_vis[b, ci] > 0.5:
                        pred_len = torch.norm(pred_xyz[b, ci] - pred_xyz[b, pi]).item()
                        gt_len = torch.norm(gt_xyz[b, ci] - gt_xyz[b, pi]).item()
                        bone_errors.append(abs(pred_len - gt_len))

                # Save prediction
                predictions.append({
                    "frame_idx": batch["frame_idx"][b],
                    "pred_xyz": pred_xyz[b].cpu().tolist(),
                    "pred_vis": pred_vis[b].cpu().tolist(),
                    "gt_xyz": gt_xyz[b].cpu().tolist(),
                    "gt_vis": gt_vis[b].cpu().tolist(),
                })

    all_errors = np.array(all_errors)
    bone_errors = np.array(bone_errors) if bone_errors else np.array([0.0])

    # Visibility accuracy
    vis_preds_flat = np.concatenate(all_vis_preds)
    vis_gt_flat = np.concatenate(all_vis_gt)
    vis_acc = float(np.mean((vis_preds_flat > 0.5) == (vis_gt_flat > 0.5)))

    # Compute metrics
    metrics = {
        "model": args.model,
        "split": split,
        "n_samples": len(dataset),
        "n_joints_canonical": NUM_JOINTS,
        "mpjpe_canonical": round(float(all_errors.mean()), 6),
        "mpjpe_median": round(float(np.median(all_errors)), 6),
        "mpjpe_std": round(float(all_errors.std()), 6),
        "pck_005": round(float(np.mean(all_errors < 0.05)), 4),
        "pck_010": round(float(np.mean(all_errors < 0.10)), 4),
        "pck_020": round(float(np.mean(all_errors < 0.20)), 4),
        "bone_length_error_mean": round(float(bone_errors.mean()), 6),
        "visibility_accuracy": round(vis_acc, 4),
        "per_joint_mpjpe": {
            name: round(float(np.mean(errs)), 5) if errs else None
            for name, errs in per_joint_errors.items()
        },
    }

    # Save
    with open(out_dir / "eval_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "predictions.jsonl", 'w') as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    # Print report
    print(f"\n  Results:")
    print(f"  {'─'*50}")
    print(f"  MPJPE (canonical):    {metrics['mpjpe_canonical']:.5f}")
    print(f"  MPJPE (median):       {metrics['mpjpe_median']:.5f}")
    print(f"  PCK@0.05:             {metrics['pck_005']:.1%}")
    print(f"  PCK@0.10:             {metrics['pck_010']:.1%}")
    print(f"  PCK@0.20:             {metrics['pck_020']:.1%}")
    print(f"  Bone length err:      {metrics['bone_length_error_mean']:.5f}")
    print(f"  Visibility accuracy:  {metrics['visibility_accuracy']:.1%}")
    print(f"\n  Per-joint MPJPE:")
    for name, mpjpe in metrics["per_joint_mpjpe"].items():
        if mpjpe is not None:
            print(f"    {name:25s} {mpjpe:.5f}")
        else:
            print(f"    {name:25s} (not visible)")
    print(f"\n  Saved: {out_dir}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate HyperBone3D v0")
    parser.add_argument("--model", required=True, help="Path to model.pt")
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--dataset", help="Dataset directory")
    parser.add_argument("--video", help="Video path (alternative)")
    parser.add_argument("--gt", help="GT JSONL path (with --video)")
    parser.add_argument("--asset-id", default="")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=["all", "train", "val"], default="all",
                        help="Evaluate on 'all', 'train', or 'val' split")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Val fraction (used with --split train/val)")
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
