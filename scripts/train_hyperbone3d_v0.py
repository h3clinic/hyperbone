"""
Train HyperBone3D v0 — supervised internal 3D pose predictor.

Usage:
  python scripts/train_hyperbone3d_v0.py \
    --dataset outputs/pose3d/wolf_dataset \
    --out outputs/models/hyperbone3d_v0 \
    --epochs 20 --batch-size 8 --device cuda

  # Or train from video + GT JSONL directly:
  python scripts/train_hyperbone3d_v0.py \
    --video output/fox3d/fox3d_walk.mp4 \
    --gt output/fox3d/fox_armature_gt.jsonl \
    --asset-id Fox.glb \
    --out outputs/models/hyperbone3d_v0 \
    --epochs 20 --batch-size 8 --device cuda
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hyperbone.models.hyperbone3d import HyperBone3D, HyperBone3DLoss
from hyperbone.pose3d.dataset import Pose3DDataset, Pose3DDatasetFromVideo
from hyperbone.pose3d.joint_map import QUADRUPED_BONES, NUM_JOINTS


def collate_fn(batch):
    """Custom collate that handles string fields."""
    result = {}
    tensor_keys = ["rgb", "mask", "depth", "bbox_xywh", "joint_xyz_canonical",
                   "joint_visible", "bone_edges", "scale", "root_xyz"]
    for key in tensor_keys:
        result[key] = torch.stack([b[key] for b in batch])
    result["asset_id"] = [b["asset_id"] for b in batch]
    result["frame_idx"] = [b["frame_idx"] for b in batch]
    return result


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HyperBone3D v0 Training")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Output: {out_dir}")

    # Load dataset
    if args.video and args.gt:
        print(f"\n  Loading from video: {args.video}")
        print(f"  GT: {args.gt}")
        dataset = Pose3DDatasetFromVideo(
            video_path=args.video,
            gt_jsonl_path=args.gt,
            resolution=(args.resolution, args.resolution),
            asset_id=args.asset_id,
        )
    elif args.dataset:
        print(f"\n  Loading dataset: {args.dataset}")
        dataset = Pose3DDataset(
            root_dir=args.dataset,
            resolution=(args.resolution, args.resolution),
            asset_id=args.asset_id,
        )
    else:
        raise ValueError("Provide --dataset or --video + --gt")

    print(f"  Samples: {len(dataset)}")
    print(f"  Joints: {NUM_JOINTS} (canonical quadruped)")
    print(f"  Resolution: {args.resolution}x{args.resolution}")

    # Train/val split
    val_split = getattr(args, 'val_split', 0.0)
    val_dataset = None
    if val_split > 0:
        from torch.utils.data import Subset
        n = len(dataset)
        n_val = int(n * val_split)
        n_train = n - n_val
        # Use last N frames as val (temporal split, not random)
        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n))
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        print(f"  Split: train={n_train}, val={n_val} (last {val_split:.0%} held out)")
    else:
        train_dataset = dataset
        print(f"  Split: none (training on all {len(dataset)} frames)")

    # Verify first sample
    sample = dataset[0]
    print(f"  Sample RGB shape: {sample['rgb'].shape}")
    print(f"  Sample joints visible: {sample['joint_visible'].sum().item():.0f}/{NUM_JOINTS}")

    # DataLoader
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=False,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0, drop_last=False,
        )

    # Model
    model = HyperBone3D(
        num_joints=NUM_JOINTS,
        input_channels=5,  # RGB + mask + depth
        base_channels=args.base_channels,
        hidden_dim=args.hidden_dim,
    ).to(device)
    print(f"\n  Model parameters: {model.count_parameters():,}")

    # Loss
    bone_edges = torch.tensor(QUADRUPED_BONES, dtype=torch.long)
    criterion = HyperBone3DLoss(
        pos_weight=1.0,
        vis_weight=0.5,
        bone_weight=0.2,
        bone_edges=bone_edges,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f"\n  Training for {args.epochs} epochs...")
    train_log = []
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = {"total": 0, "pos": 0, "vis": 0, "bone": 0}
        n_batches = 0
        t0 = time.time()

        for batch in loader:
            # Build 5-channel input
            rgb = batch["rgb"].to(device)        # [B, 3, H, W]
            mask = batch["mask"].to(device)      # [B, 1, H, W]
            depth = batch["depth"].to(device)    # [B, 1, H, W]
            x = torch.cat([rgb, mask, depth], dim=1)  # [B, 5, H, W]

            gt_xyz = batch["joint_xyz_canonical"].to(device)   # [B, J, 3]
            gt_vis = batch["joint_visible"].to(device)         # [B, J]

            # Forward
            pred = model(x)
            pred_xyz = pred["joint_xyz_canonical"]
            pred_vis = pred["joint_visibility_logits"]

            # Loss
            losses = criterion(pred_xyz, pred_vis, gt_xyz, gt_vis)
            loss = losses["total"]

            # Backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses["total"] += loss.item()
            epoch_losses["pos"] += losses["pos_loss"].item()
            epoch_losses["vis"] += losses["vis_loss"].item()
            epoch_losses["bone"] += losses["bone_loss"].item()
            n_batches += 1

        scheduler.step()
        dt = time.time() - t0

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        # Validation loss
        val_loss_total = None
        if val_loader is not None:
            model.eval()
            val_losses_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    rgb = batch["rgb"].to(device)
                    mask = batch["mask"].to(device)
                    depth = batch["depth"].to(device)
                    x = torch.cat([rgb, mask, depth], dim=1)
                    gt_xyz = batch["joint_xyz_canonical"].to(device)
                    gt_vis = batch["joint_visible"].to(device)
                    pred = model(x)
                    losses = criterion(pred["joint_xyz_canonical"],
                                       pred["joint_visibility_logits"], gt_xyz, gt_vis)
                    val_losses_sum += losses["total"].item()
                    val_n += 1
            val_loss_total = val_losses_sum / max(val_n, 1)

        log_entry = {
            "epoch": epoch,
            "total_loss": round(epoch_losses["total"], 6),
            "pos_loss": round(epoch_losses["pos"], 6),
            "vis_loss": round(epoch_losses["vis"], 6),
            "bone_loss": round(epoch_losses["bone"], 6),
            "val_loss": round(val_loss_total, 6) if val_loss_total is not None else None,
            "lr": round(scheduler.get_last_lr()[0], 8),
            "time_sec": round(dt, 2),
        }
        train_log.append(log_entry)

        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            val_str = f"  val={val_loss_total:.5f}" if val_loss_total is not None else ""
            print(f"    Epoch {epoch:3d}/{args.epochs}  "
                  f"loss={epoch_losses['total']:.5f}  "
                  f"pos={epoch_losses['pos']:.5f}  "
                  f"vis={epoch_losses['vis']:.5f}  "
                  f"bone={epoch_losses['bone']:.5f}{val_str}  "
                  f"({dt:.1f}s)")

        # Save best
        if epoch_losses["total"] < best_loss:
            best_loss = epoch_losses["total"]
            torch.save(model.state_dict(), out_dir / "model.pt")

    # Save final
    torch.save(model.state_dict(), out_dir / "model_final.pt")

    # Save config
    config = {
        "model": "HyperBone3D-v0",
        "num_joints": NUM_JOINTS,
        "input_channels": 5,
        "base_channels": args.base_channels,
        "hidden_dim": args.hidden_dim,
        "resolution": args.resolution,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dataset": args.dataset or f"{args.video}+{args.gt}",
        "asset_id": args.asset_id,
        "parameters": model.count_parameters(),
        "best_loss": round(best_loss, 6),
        "val_split": val_split,
        "n_train": len(train_dataset),
        "n_val": len(val_dataset) if val_dataset else 0,
    }
    with open(out_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Save train log
    with open(out_dir / "train_log.jsonl", 'w') as f:
        for entry in train_log:
            f.write(json.dumps(entry) + "\n")

    # Generate sample predictions
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            x = torch.cat([sample["rgb"], sample["mask"], sample["depth"]], dim=0)
            x = x.unsqueeze(0).to(device)
            pred = model(x)
            predictions.append({
                "frame_idx": sample["frame_idx"],
                "pred_xyz": pred["joint_xyz_canonical"][0].cpu().tolist(),
                "pred_vis": torch.sigmoid(pred["joint_visibility_logits"][0]).cpu().tolist(),
                "gt_xyz": sample["joint_xyz_canonical"].tolist(),
                "gt_vis": sample["joint_visible"].tolist(),
            })

    with open(out_dir / "sample_predictions.jsonl", 'w') as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    # Final metrics on full dataset
    model.eval()
    all_errors = []
    all_val_errors = []
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

            # Per-joint error (visible only)
            err = torch.norm(pred_xyz - gt_xyz, dim=-1)  # [B, J]
            vis_mask = gt_vis > 0.5
            visible_err = err[vis_mask]
            if len(visible_err) > 0:
                all_errors.extend(visible_err.cpu().tolist())

        if val_loader is not None:
            for batch in val_loader:
                rgb = batch["rgb"].to(device)
                mask = batch["mask"].to(device)
                depth = batch["depth"].to(device)
                x = torch.cat([rgb, mask, depth], dim=1)
                gt_xyz = batch["joint_xyz_canonical"].to(device)
                gt_vis = batch["joint_visible"].to(device)
                pred = model(x)
                pred_xyz = pred["joint_xyz_canonical"]
                err = torch.norm(pred_xyz - gt_xyz, dim=-1)
                vis_mask = gt_vis > 0.5
                visible_err = err[vis_mask]
                if len(visible_err) > 0:
                    all_val_errors.extend(visible_err.cpu().tolist())

    all_errors = np.array(all_errors) if all_errors else np.array([float('inf')])
    all_val_errors = np.array(all_val_errors) if all_val_errors else None

    metrics = {
        "mpjpe_train": round(float(all_errors.mean()), 6),
        "mpjpe_train_median": round(float(np.median(all_errors)), 6),
        "pck_005_train": round(float(np.mean(all_errors < 0.05)), 4),
        "pck_010_train": round(float(np.mean(all_errors < 0.10)), 4),
        "pck_020_train": round(float(np.mean(all_errors < 0.20)), 4),
        "n_train_samples": len(train_dataset),
        "best_train_loss": round(best_loss, 6),
        "first_loss": train_log[0]["total_loss"] if train_log else None,
        "final_loss": train_log[-1]["total_loss"] if train_log else None,
    }

    if all_val_errors is not None:
        metrics["mpjpe_val"] = round(float(all_val_errors.mean()), 6)
        metrics["mpjpe_val_median"] = round(float(np.median(all_val_errors)), 6)
        metrics["pck_005_val"] = round(float(np.mean(all_val_errors < 0.05)), 4)
        metrics["pck_010_val"] = round(float(np.mean(all_val_errors < 0.10)), 4)
        metrics["pck_020_val"] = round(float(np.mean(all_val_errors < 0.20)), 4)
        metrics["n_val_samples"] = len(val_dataset)
        metrics["val_loss_final"] = train_log[-1].get("val_loss") if train_log else None

    with open(out_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"{'='*60}")
    print(f"  Model saved: {out_dir / 'model.pt'}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Best train loss: {best_loss:.6f}")
    print(f"  Loss start→end: {train_log[0]['total_loss']:.5f} → {train_log[-1]['total_loss']:.5f}")
    print(f"  Train MPJPE: {metrics['mpjpe_train']:.5f}")
    print(f"  Train PCK@0.05: {metrics['pck_005_train']:.1%}")
    print(f"  Train PCK@0.10: {metrics['pck_010_train']:.1%}")
    print(f"  Train PCK@0.20: {metrics['pck_020_train']:.1%}")
    if all_val_errors is not None:
        print(f"  --- Validation (held-out {val_split:.0%}) ---")
        print(f"  Val MPJPE: {metrics['mpjpe_val']:.5f}")
        print(f"  Val PCK@0.05: {metrics['pck_005_val']:.1%}")
        print(f"  Val PCK@0.10: {metrics['pck_010_val']:.1%}")
        print(f"  Val PCK@0.20: {metrics['pck_020_val']:.1%}")
        print(f"  Val loss (final): {metrics['val_loss_final']}")
    print(f"{'='*60}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train HyperBone3D v0")
    parser.add_argument("--dataset", help="Path to pose3d dataset directory")
    parser.add_argument("--video", help="Path to video (alternative to --dataset)")
    parser.add_argument("--gt", help="Path to GT JSONL (with --video)")
    parser.add_argument("--asset-id", default="", help="Asset identifier for joint mapping")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.0,
                        help="Fraction of data to hold out for validation (0=no split)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
