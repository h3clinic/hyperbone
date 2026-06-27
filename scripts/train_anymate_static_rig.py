"""
Train HyperBone static rig model on Anymate local dev split.

Trains: point cloud → joint positions + connectivity + skinning weights

Usage:
    python scripts/train_anymate_static_rig.py
    python scripts/train_anymate_static_rig.py --epochs 50 --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.datasets.anymate_static_dataset import AnymateStaticRigDataset
from hyperbone.models.hyperbone_rig_graph_static import HyperBoneStaticRigModel
from hyperbone.losses.rig_graph_losses import RigGraphLoss


def compute_metrics(pred, batch):
    """Compute evaluation metrics."""
    with torch.no_grad():
        gt_pos = batch["joint_pos"]
        gt_active = batch["joint_active"]
        pred_pos = pred["joint_pos"]
        pred_active = torch.sigmoid(pred["active_logits"])

        # MPJPE on active joints
        mask = gt_active.unsqueeze(-1)  # [B, J, 1]
        diff = (pred_pos - gt_pos) * mask
        per_joint_error = diff.norm(dim=-1)  # [B, J]
        n_active = gt_active.sum()
        mpjpe = (per_joint_error * gt_active).sum() / n_active.clamp(min=1)

        # Active accuracy
        pred_active_binary = (pred_active > 0.5).float()
        active_acc = (pred_active_binary == gt_active).float().mean()

        # Edge F1
        if "adj_logits" in pred:
            pred_adj = (torch.sigmoid(pred["adj_logits"]) > 0.5).float()
            gt_adj = batch["adj_matrix"]
            # Only consider upper triangle where both nodes active
            active_mask = gt_active.unsqueeze(-1) * gt_active.unsqueeze(-2)
            triu_mask = torch.triu(torch.ones_like(gt_adj[0]), diagonal=1).unsqueeze(0) * active_mask

            tp = (pred_adj * gt_adj * triu_mask).sum()
            fp = (pred_adj * (1 - gt_adj) * triu_mask).sum()
            fn = ((1 - pred_adj) * gt_adj * triu_mask).sum()

            precision = tp / (tp + fp).clamp(min=1)
            recall = tp / (tp + fn).clamp(min=1)
            edge_f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
        else:
            edge_f1 = torch.tensor(0.0)

        return {
            "mpjpe": mpjpe.item(),
            "active_acc": active_acc.item(),
            "edge_f1": edge_f1.item(),
        }


def train_epoch(model, loader, criterion, optimizer, device, amp_scaler):
    model.train()
    total_loss = 0.0
    total_metrics = {"mpjpe": 0.0, "active_acc": 0.0, "edge_f1": 0.0}
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=amp_scaler is not None):
            pred = model(batch)
            losses = criterion(pred, batch)
            loss = losses["total"]

        if amp_scaler:
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        metrics = compute_metrics(pred, batch)
        for k in total_metrics:
            total_metrics[k] += metrics[k]
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {k: v / max(n_batches, 1) for k, v in total_metrics.items()}
    return avg_loss, avg_metrics


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_metrics = {"mpjpe": 0.0, "active_acc": 0.0, "edge_f1": 0.0}
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        pred = model(batch)
        losses = criterion(pred, batch)

        total_loss += losses["total"].item()
        metrics = compute_metrics(pred, batch)
        for k in total_metrics:
            total_metrics[k] += metrics[k]
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {k: v / max(n_batches, 1) for k, v in total_metrics.items()}
    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train static rig model")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--out", default="outputs/models/hyperbone_anymate_local_dev")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-joints", type=int, default=64)
    parser.add_argument("--pc-points", type=int, default=2048)
    parser.add_argument("--feat-dim", type=int, default=512)
    parser.add_argument("--no-skinning", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    print(f"[Train] Output: {out_dir}")

    # Datasets
    print("[Train] Loading datasets...")
    train_ds = AnymateStaticRigDataset(
        args.pt, f"{args.splits_dir}/train.jsonl",
        max_joints=args.max_joints, pc_points=args.pc_points,
    )
    val_ds = AnymateStaticRigDataset(
        args.pt, f"{args.splits_dir}/val.jsonl",
        max_joints=args.max_joints, pc_points=args.pc_points,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    print(f"[Train] Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    print(f"[Train] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = HyperBoneStaticRigModel(
        in_channels=3,
        feat_dim=args.feat_dim,
        max_joints=args.max_joints,
        predict_skinning=not args.no_skinning,
    ).to(device)

    param_count = model.count_parameters()
    print(f"[Train] Model parameters: {param_count:,}")

    # Loss
    criterion = RigGraphLoss(
        w_joint_pos=1.0,
        w_edge=0.5,
        w_active=0.3,
        w_skinning=0.2 if not args.no_skinning else 0.0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # AMP
    amp_scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and not args.no_amp) else None

    # Training loop
    best_val_loss = float("inf")
    best_val_mpjpe = float("inf")
    training_log = []

    print(f"\n[Train] Starting training: {args.epochs} epochs")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, amp_scaler)
        val_loss, val_metrics = eval_epoch(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        # Log
        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_mpjpe": round(train_metrics["mpjpe"], 6),
            "val_mpjpe": round(val_metrics["mpjpe"], 6),
            "train_active_acc": round(train_metrics["active_acc"], 4),
            "val_active_acc": round(val_metrics["active_acc"], 4),
            "train_edge_f1": round(train_metrics["edge_f1"], 4),
            "val_edge_f1": round(val_metrics["edge_f1"], 4),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
            "time_s": round(elapsed, 1),
        }
        training_log.append(log_entry)

        # Print
        improved = ""
        if val_metrics["mpjpe"] < best_val_mpjpe:
            best_val_mpjpe = val_metrics["mpjpe"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            improved = " ★"

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(
            f"E{epoch:03d} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_mpjpe={train_metrics['mpjpe']:.4f} val_mpjpe={val_metrics['mpjpe']:.4f} | "
            f"edge_f1={val_metrics['edge_f1']:.3f} | "
            f"{elapsed:.1f}s{improved}"
        )

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), out_dir / f"checkpoint_e{epoch:03d}.pt")

    # Final save
    torch.save(model.state_dict(), out_dir / "model_final.pt")

    # Save training log
    with open(out_dir / "training_log.jsonl", "w") as f:
        for entry in training_log:
            f.write(json.dumps(entry) + "\n")

    # Save config and metrics
    config = {
        "model": "HyperBoneStaticRigModel",
        "feat_dim": args.feat_dim,
        "max_joints": args.max_joints,
        "pc_points": args.pc_points,
        "predict_skinning": not args.no_skinning,
        "param_count": param_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "best_val_loss": round(best_val_loss, 6),
        "best_val_mpjpe": round(best_val_mpjpe, 6),
        "final_train_loss": training_log[-1]["train_loss"],
        "final_val_loss": training_log[-1]["val_loss"],
        "final_train_mpjpe": training_log[-1]["train_mpjpe"],
        "final_val_mpjpe": training_log[-1]["val_mpjpe"],
        "final_edge_f1": training_log[-1]["val_edge_f1"],
        "dataset": "anymate_local_dev (NOT official benchmark)",
        "split_method": "deterministic_sha256_by_asset 80/10/10",
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[Train] DONE")
    print(f"  Best val MPJPE: {best_val_mpjpe:.6f}")
    print(f"  Best val loss:  {best_val_loss:.6f}")
    print(f"  Final edge F1:  {training_log[-1]['val_edge_f1']:.4f}")
    print(f"  Saved: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
