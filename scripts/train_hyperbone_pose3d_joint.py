"""
Train HyperBonePose3D-Joint: crop-based 3D canonical joint predictor.

Primary target: 3D canonical joints from cropped animal images.
Auxiliary: 2D reprojection loss + heatmap loss.

Usage:
  python scripts/train_hyperbone_pose3d_joint.py \
    --labels outputs/pose3d_joint/fox_walk/labels.jsonl \
    --out outputs/models/hyperbone_pose3d_joint_fox \
    --epochs 40 \
    --batch-size 8 \
    --resolution 192 \
    --device cuda
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
from torch.utils.data import DataLoader, Subset

from hyperbone.models.hyperbone_pose3d_joint import (
    HyperBonePose3DJoint, HyperBonePose3DJointLoss,
)
from hyperbone.pose2d.dataset_3djoint import Pose3DJointDataset
from hyperbone.pose2d.quadruped_schema import NUM_JOINTS_2D, QUADRUPED_BONES_2D


def compute_metrics(pred_xyz, gt_xyz, gt_vis):
    """Compute MPJPE and PCK on visible joints."""
    # pred_xyz, gt_xyz: [B, J, 3], gt_vis: [B, J]
    diff = pred_xyz - gt_xyz
    dist = torch.norm(diff, dim=-1)  # [B, J]
    mask = gt_vis > 0.5

    if mask.sum() == 0:
        return {"mpjpe": 0.0, "pck_005": 0.0, "pck_010": 0.0, "pck_020": 0.0}

    visible_dist = dist[mask]
    mpjpe = visible_dist.mean().item()
    pck_005 = (visible_dist < 0.05).float().mean().item()
    pck_010 = (visible_dist < 0.10).float().mean().item()
    pck_020 = (visible_dist < 0.20).float().mean().item()

    return {"mpjpe": mpjpe, "pck_005": pck_005, "pck_010": pck_010, "pck_020": pck_020}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=192)
    parser.add_argument("--heatmap-resolution", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--input-channels", type=int, default=3)
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    dataset = Pose3DJointDataset(
        labels_path=args.labels,
        resolution=args.resolution,
        heatmap_resolution=args.heatmap_resolution,
        augment=args.augment,
        input_channels=args.input_channels,
    )
    print(f"Dataset: {len(dataset)} samples")

    # Temporal split (last val_split fraction)
    n = len(dataset)
    n_val = int(n * args.val_split)
    n_train = n - n_val
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n))

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    print(f"Split: {n_train} train, {n_val} val")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = HyperBonePose3DJoint(
        num_joints=NUM_JOINTS_2D,
        input_channels=args.input_channels,
        base_channels=args.base_channels,
        hidden_dim=args.hidden_dim,
        use_aux_heatmaps=True,
        heatmap_resolution=args.heatmap_resolution,
    ).to(device)
    print(f"Model: {model.count_parameters():,} parameters")

    # Loss
    criterion = HyperBonePose3DJointLoss(
        pos_weight=1.0,
        heatmap_weight=1.0,
        coord2d_weight=0.5,
        vis_weight=0.2,
        bone_weight=0.2,
    ).to(device)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    train_log = []
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        t0 = time.time()

        # Train
        model.train()
        train_losses = []
        train_metrics_list = []

        for batch in train_dl:
            img = batch["image"].to(device)
            gt_xyz = batch["joint_xyz_canonical"].to(device)
            gt_vis = batch["joint_visible"].to(device)
            gt_2d = batch["joints_2d"].to(device)
            gt_hm = batch["heatmaps"].to(device)

            pred = model(img)
            losses = criterion(pred, gt_xyz, gt_vis, gt_2d, gt_hm)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(losses["total"].item())
            with torch.no_grad():
                m = compute_metrics(pred["joint_xyz_canonical"], gt_xyz, gt_vis)
                train_metrics_list.append(m)

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        val_metrics_list = []

        with torch.no_grad():
            for batch in val_dl:
                img = batch["image"].to(device)
                gt_xyz = batch["joint_xyz_canonical"].to(device)
                gt_vis = batch["joint_visible"].to(device)
                gt_2d = batch["joints_2d"].to(device)
                gt_hm = batch["heatmaps"].to(device)

                pred = model(img)
                losses = criterion(pred, gt_xyz, gt_vis, gt_2d, gt_hm)
                val_losses.append(losses["total"].item())

                m = compute_metrics(pred["joint_xyz_canonical"], gt_xyz, gt_vis)
                val_metrics_list.append(m)

        # Aggregate
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses) if val_losses else 0
        train_mpjpe = np.mean([m["mpjpe"] for m in train_metrics_list])
        val_mpjpe = np.mean([m["mpjpe"] for m in val_metrics_list]) if val_metrics_list else 0
        val_pck010 = np.mean([m["pck_010"] for m in val_metrics_list]) if val_metrics_list else 0

        elapsed = time.time() - t0
        print(f"E{epoch+1:02d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
              f"train_mpjpe={train_mpjpe:.4f} val_mpjpe={val_mpjpe:.4f} "
              f"val_pck@0.10={val_pck010:.1%} | {elapsed:.1f}s")

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_mpjpe": round(train_mpjpe, 6),
            "val_mpjpe": round(val_mpjpe, 6),
            "val_pck_010": round(val_pck010, 4),
            "lr": optimizer.param_groups[0]["lr"],
        }
        train_log.append(log_entry)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "model.pt")

    # Save config
    config = {
        "model": "HyperBonePose3D-Joint",
        "num_joints": NUM_JOINTS_2D,
        "input_channels": args.input_channels,
        "base_channels": args.base_channels,
        "hidden_dim": args.hidden_dim,
        "resolution": args.resolution,
        "heatmap_resolution": args.heatmap_resolution,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_split": args.val_split,
        "n_train": n_train,
        "n_val": n_val,
        "parameters": model.count_parameters(),
        "best_val_loss": round(best_val_loss, 6),
        "dataset": args.labels,
    }
    with open(out_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Save training log
    with open(out_dir / "train_log.jsonl", 'w') as f:
        for entry in train_log:
            f.write(json.dumps(entry) + "\n")

    # Final metrics
    final = train_log[-1] if train_log else {}
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Model saved: {out_dir / 'model.pt'}")
    print(f"Config saved: {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
