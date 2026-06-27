"""
Train HyperBone static rig model v2.

Key differences from v1:
- Hungarian matching loss (no same-slot position training)
- Chamfer + repulsion losses to prevent centroid collapse
- Matched edge and bone length losses
- Same PointNet backbone (fix loss before upgrading encoder)

Usage:
    python scripts/train_anymate_static_rig_v2.py --epochs 50 --batch-size 8
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
from hyperbone.losses.rig_graph_losses_v2 import RigGraphLossV2


def compute_metrics(pred, batch):
    """Quick training metrics (no Hungarian for speed — just Chamfer proxy)."""
    with torch.no_grad():
        gt_pos = batch["joint_pos"].float()
        gt_active = batch["joint_active"].float()
        pred_pos = pred["joint_pos"].float()
        pred_active = torch.sigmoid(pred["active_logits"].float())

        B = gt_pos.shape[0]
        chamfer_sum = 0.0
        spread_sum = 0.0
        active_acc_sum = 0.0
        count_error_sum = 0.0
        count_ratio_sum = 0.0
        overpred_sum = 0.0
        underpred_sum = 0.0
        count = 0

        for b in range(B):
            gt_mask = gt_active[b] > 0.5
            pred_mask = pred_active[b] > 0.5
            gt_pts = gt_pos[b, gt_mask]
            pred_pts = pred_pos[b, pred_mask]

            # Active accuracy
            pred_binary = (pred_active[b] > 0.5).float()
            active_acc_sum += (pred_binary == gt_active[b]).float().mean().item()

            # Count metrics
            n_gt = gt_mask.sum().item()
            n_pred = pred_mask.sum().item()
            count_ratio = n_pred / max(n_gt, 1)
            count_ratio_sum += count_ratio
            count_error_sum += abs(count_ratio - 1.0)
            overpred_sum += max(0.0, count_ratio - 1.0)
            underpred_sum += max(0.0, 1.0 - count_ratio)

            if gt_pts.shape[0] == 0 or pred_pts.shape[0] == 0:
                count += 1
                continue

            # Chamfer
            dist = torch.cdist(pred_pts, gt_pts, p=2)
            chamfer = dist.min(dim=1)[0].mean() + dist.min(dim=0)[0].mean()
            chamfer_sum += chamfer.item()

            # Spread collapse
            gt_spread = (gt_pts.max(dim=0)[0] - gt_pts.min(dim=0)[0]).norm()
            pred_spread = (pred_pts.max(dim=0)[0] - pred_pts.min(dim=0)[0]).norm()
            spread_sum += (pred_spread / gt_spread.clamp(min=1e-6)).item()
            count += 1

        n = max(count, 1)
        return {
            "chamfer": chamfer_sum / n,
            "spread_score": spread_sum / n,
            "active_acc": active_acc_sum / max(B, 1),
            "active_count_abs_error": count_error_sum / max(B, 1),
            "count_ratio": count_ratio_sum / max(B, 1),
            "overprediction_ratio": overpred_sum / max(B, 1),
            "underprediction_ratio": underpred_sum / max(B, 1),
        }


@torch.no_grad()
def compute_gate_metrics(model, loader, device, active_threshold=0.7, max_batches=16):
    """Compute lightweight matched MPJPE and node recall for preservation gates."""
    model.eval()
    mpjpe_sum = 0.0
    recall_sum = 0.0
    count_ratio_sum = 0.0
    n = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        pred = model(batch)

        gt_pos = batch["joint_pos"].float()
        gt_active = batch["joint_active"].float()
        pred_pos = pred["joint_pos"].float()
        pred_active = torch.sigmoid(pred["active_logits"].float())

        B = gt_pos.shape[0]
        for b in range(B):
            gt_mask = gt_active[b] > 0.5
            pred_mask = pred_active[b] > active_threshold

            gt_pts = gt_pos[b, gt_mask]
            pred_pts = pred_pos[b, pred_mask]
            n_gt = gt_pts.shape[0]
            n_pred = pred_pts.shape[0]
            if n_gt > 0:
                count_ratio_sum += n_pred / max(n_gt, 1)
            if n_gt == 0 or n_pred == 0:
                continue

            dist = torch.cdist(pred_pts, gt_pts, p=2)
            min_pred, _ = dist.min(dim=1)
            min_gt, _ = dist.min(dim=0)

            # Symmetric nearest-neighbor proxy for matched MPJPE.
            mpjpe_proxy = 0.5 * (min_pred.mean().item() + min_gt.mean().item())
            recall_proxy = (min_gt < 0.1).float().mean().item()

            mpjpe_sum += mpjpe_proxy
            recall_sum += recall_proxy
            n += 1

    if n == 0:
        return {
            "mpjpe_proxy": float("inf"),
            "node_recall_proxy": 0.0,
            "count_ratio_proxy": 0.0,
        }
    return {
        "mpjpe_proxy": mpjpe_sum / n,
        "node_recall_proxy": recall_sum / n,
        "count_ratio_proxy": count_ratio_sum / n,
    }


def train_epoch(model, loader, criterion, optimizer, device, amp_scaler, grad_accum=1, schedule=None):
    model.train()
    total_loss = 0.0
    total_metrics = {"chamfer": 0.0, "spread_score": 0.0, "active_acc": 0.0,
                     "active_count_abs_error": 0.0, "count_ratio": 0.0,
                     "overprediction_ratio": 0.0, "underprediction_ratio": 0.0}
    loss_components = {}
    n_batches = 0

    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.amp.autocast("cuda", enabled=amp_scaler is not None):
            pred = model(batch)
            losses = criterion(pred, batch, schedule=schedule)
            loss = losses["total"] / grad_accum

        if amp_scaler:
            amp_scaler.scale(loss).backward()
            if (i + 1) % grad_accum == 0:
                amp_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (i + 1) % grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += losses["total"].item()
        metrics = compute_metrics(pred, batch)
        for k in total_metrics:
            total_metrics[k] += metrics[k]

        # Track loss components
        for k, v in losses.items():
            if k != "total" and isinstance(v, torch.Tensor):
                loss_components[k] = loss_components.get(k, 0.0) + v.item()

        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {k: v / max(n_batches, 1) for k, v in total_metrics.items()}
    avg_components = {k: v / max(n_batches, 1) for k, v in loss_components.items()}
    return avg_loss, avg_metrics, avg_components


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_metrics = {"chamfer": 0.0, "spread_score": 0.0, "active_acc": 0.0,
                     "active_count_abs_error": 0.0, "count_ratio": 0.0,
                     "overprediction_ratio": 0.0, "underprediction_ratio": 0.0}
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
    parser = argparse.ArgumentParser(description="Train static rig model v2")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--out", default="outputs/models/hyperbone_anymate_static_v2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-joints", type=int, default=64)
    parser.add_argument("--pc-points", type=int, default=2048)
    parser.add_argument("--points-per-sample", type=int, default=None,
                        help="Alias for --pc-points")
    parser.add_argument("--feat-dim", type=int, default=512)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume model weights from checkpoint path")
    parser.add_argument("--amp", action="store_true",
                        help="Enable AMP explicitly (default behavior; accepted for convenience)")
    parser.add_argument("--backbone", choices=["pointnet", "dgcnn"], default="pointnet")
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--no-skinning", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--grad-accum", type=int, default=1)
    # Loss weights
    parser.add_argument("--w-hungarian", type=float, default=2.0)
    parser.add_argument("--w-chamfer", type=float, default=1.0)
    parser.add_argument("--w-repulsion", type=float, default=0.5)
    parser.add_argument("--w-edge", type=float, default=0.5)
    parser.add_argument("--w-bone", type=float, default=1.0)
    parser.add_argument("--w-active", type=float, default=0.3)
    parser.add_argument("--w-unmatched", type=float, default=0.5)
    parser.add_argument("--w-count", type=float, default=0.5)
    parser.add_argument("--min-repulsion-dist", type=float, default=0.05)
    parser.add_argument("--edge-pos-weight", type=float, default=3.0)
    parser.add_argument("--edge-fp-weight", type=float, default=2.0)
    parser.add_argument("--count-overpredict-scale", type=float, default=0.25)
    parser.add_argument("--count-underpredict-scale", type=float, default=1.0)
    parser.add_argument("--active-pos-weight", type=float, default=2.0)
    # Schedule: ramp epochs
    parser.add_argument("--ramp-start", type=int, default=10,
                        help="Epoch to start ramping count/bone/edge-fp losses")
    parser.add_argument("--ramp-end", type=int, default=30,
                        help="Epoch by which ramp reaches full strength")
    # Early stopping
    parser.add_argument("--kill-spread", type=float, default=0.45,
                        help="Kill training if val spread below this after ramp-start")
    parser.add_argument("--gate-start-epoch", type=int, default=5,
                        help="Epoch to start applying v2.6 preservation gates")
    parser.add_argument("--kill-count-min", type=float, default=None,
                        help="Kill if val count ratio drops below this after gate-start")
    parser.add_argument("--kill-count-max", type=float, default=None,
                        help="Kill if val count ratio rises above this after gate-start")
    parser.add_argument("--kill-mpjpe", type=float, default=None,
                        help="Kill if gate MPJPE proxy exceeds this after gate-start")
    parser.add_argument("--kill-recall-drop", type=float, default=None,
                        help="Kill if gate recall proxy drops by more than this from baseline")
    parser.add_argument("--gate-active-threshold", type=float, default=0.7,
                        help="Active threshold used by gate metric proxies")
    parser.add_argument("--gate-eval-batches", type=int, default=16,
                        help="Max val batches for gate metric proxies")
    args = parser.parse_args()
    if args.points_per_sample is not None:
        args.pc_points = args.points_per_sample

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train-v2] Device: {device}")
    print(f"[Train-v2] Output: {out_dir}")

    # Datasets
    print("[Train-v2] Loading datasets...")
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

    print(f"[Train-v2] Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    print(f"[Train-v2] Batches/epoch: {len(train_loader)} train, {len(val_loader)} val")

    # Model (same architecture as v1 — fix loss first, not model)
    model = HyperBoneStaticRigModel(
        in_channels=3,
        feat_dim=args.feat_dim,
        max_joints=args.max_joints,
        predict_skinning=not args.no_skinning,
        backbone=args.backbone,
        knn_k=args.knn_k,
    ).to(device)

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        state = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[Train-v2] Resumed model from: {resume_path}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Train-v2] Model: {param_count:,} parameters")

    # Loss v2
    criterion = RigGraphLossV2(
        w_hungarian_pos=args.w_hungarian,
        w_chamfer=args.w_chamfer,
        w_repulsion=args.w_repulsion,
        w_edge=args.w_edge,
        w_bone_length=args.w_bone,
        w_active=args.w_active,
        w_unmatched=args.w_unmatched,
        w_count=args.w_count,
        w_skinning=0.1 if not args.no_skinning else 0.0,
        min_repulsion_dist=args.min_repulsion_dist,
        edge_pos_weight=args.edge_pos_weight,
        edge_fp_weight=args.edge_fp_weight,
        count_overpredict_scale=args.count_overpredict_scale,
        count_underpredict_scale=args.count_underpredict_scale,
        active_pos_weight=args.active_pos_weight,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # AMP
    amp_scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and not args.no_amp) else None

    # Training loop
    best_val_chamfer = float("inf")
    best_val_loss = float("inf")
    training_log = []
    baseline_recall_proxy = None

    print(f"\n[Train-v2] Starting: {args.epochs} epochs, lr={args.lr}, bs={args.batch_size}")
    print(f"[Train-v2] Loss: hungarian={args.w_hungarian} chamfer={args.w_chamfer} "
          f"repulsion={args.w_repulsion} edge={args.w_edge} bone={args.w_bone} "
          f"count={args.w_count} unmatched={args.w_unmatched}")
    print(f"[Train-v2] Edge: pos_weight={args.edge_pos_weight} fp_weight={args.edge_fp_weight}")
    print(f"[Train-v2] Count: overpredict_scale={args.count_overpredict_scale} "
          f"underpredict_scale={args.count_underpredict_scale}")
    print(f"[Train-v2] Active BCE pos_weight={args.active_pos_weight}")
    print(f"[Train-v2] Schedule: ramp {args.ramp_start}-{args.ramp_end}, "
          f"kill_spread={args.kill_spread}")
    print(f"[Train-v2] Gates: start_epoch={args.gate_start_epoch} "
          f"count=[{args.kill_count_min},{args.kill_count_max}] "
          f"mpjpe<={args.kill_mpjpe} recall_drop<={args.kill_recall_drop}")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Compute schedule ramp factor
        if epoch < args.ramp_start:
            ramp = 0.0
        elif epoch >= args.ramp_end:
            ramp = 1.0
        else:
            ramp = (epoch - args.ramp_start) / (args.ramp_end - args.ramp_start)

        schedule = {
            "count": ramp,
            "bone": ramp,
            "edge_fp": ramp,
            "unmatched": 0.5 + 0.5 * ramp,  # start at 50%, ramp to 100%
        }

        train_loss, train_metrics, train_components = train_epoch(
            model, train_loader, criterion, optimizer, device, amp_scaler, args.grad_accum,
            schedule=schedule
        )
        val_loss, val_metrics = eval_epoch(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        # Log
        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_chamfer": round(train_metrics["chamfer"], 6),
            "val_chamfer": round(val_metrics["chamfer"], 6),
            "train_spread": round(train_metrics["spread_score"], 4),
            "val_spread": round(val_metrics["spread_score"], 4),
            "train_active_acc": round(train_metrics["active_acc"], 4),
            "val_active_acc": round(val_metrics["active_acc"], 4),
            "val_active_count_abs_error": round(val_metrics["active_count_abs_error"], 4),
            "val_count_ratio": round(val_metrics["count_ratio"], 4),
            "val_overprediction_ratio": round(val_metrics["overprediction_ratio"], 4),
            "val_underprediction_ratio": round(val_metrics["underprediction_ratio"], 4),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
            "time_s": round(elapsed, 1),
        }
        # Add loss components
        for k, v in train_components.items():
            log_entry[f"loss_{k}"] = round(v, 6)
        training_log.append(log_entry)

        # Checkpointing
        improved = ""
        if val_metrics["chamfer"] < best_val_chamfer:
            best_val_chamfer = val_metrics["chamfer"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            improved = " ★ (chamfer)"
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(
            f"E{epoch:03d} | "
            f"loss={train_loss:.4f}/{val_loss:.4f} | "
            f"chamfer={train_metrics['chamfer']:.4f}/{val_metrics['chamfer']:.4f} | "
            f"spread={train_metrics['spread_score']:.3f}/{val_metrics['spread_score']:.3f} | "
            f"acc={val_metrics['active_acc']:.3f} | "
            f"cnt={val_metrics['count_ratio']:.3f} "
            f"err={val_metrics['active_count_abs_error']:.3f} "
            f"ovp={val_metrics['overprediction_ratio']:.3f} "
            f"unp={val_metrics['underprediction_ratio']:.3f} | "
            f"ramp={ramp:.2f} {elapsed:.1f}s{improved}"
        )

        # Early stopping: kill if spread collapses after ramp starts
        if epoch >= args.ramp_start and val_metrics["spread_score"] < args.kill_spread:
            print(f"\n[KILL] Spread collapsed to {val_metrics['spread_score']:.3f} "
                  f"< {args.kill_spread} after epoch {epoch}. Stopping.")
            break

        # v2.6 preservation gates to keep coverage gains while improving structure.
        if epoch >= args.gate_start_epoch:
            if (
                args.kill_count_min is not None
                or args.kill_count_max is not None
                or args.kill_mpjpe is not None
                or args.kill_recall_drop is not None
            ):
                gate_metrics = compute_gate_metrics(
                    model, val_loader, device,
                    active_threshold=args.gate_active_threshold,
                    max_batches=args.gate_eval_batches,
                )
                log_entry["val_gate_mpjpe_proxy"] = round(gate_metrics["mpjpe_proxy"], 6)
                log_entry["val_gate_node_recall_proxy"] = round(gate_metrics["node_recall_proxy"], 6)
                log_entry["val_gate_count_ratio_proxy"] = round(gate_metrics["count_ratio_proxy"], 6)

                if baseline_recall_proxy is None and gate_metrics["node_recall_proxy"] > 0:
                    baseline_recall_proxy = gate_metrics["node_recall_proxy"]

                print(
                    f"      gate: mpjpe_proxy={gate_metrics['mpjpe_proxy']:.4f} "
                    f"recall_proxy={gate_metrics['node_recall_proxy']:.4f} "
                    f"count_ratio_proxy={gate_metrics['count_ratio_proxy']:.4f} "
                    f"baseline_recall={baseline_recall_proxy if baseline_recall_proxy is not None else -1:.4f}"
                )

                if args.kill_count_min is not None and gate_metrics["count_ratio_proxy"] < args.kill_count_min:
                    print(f"\n[KILL] Gate count ratio proxy {gate_metrics['count_ratio_proxy']:.3f} "
                          f"< {args.kill_count_min} at epoch {epoch}. Stopping.")
                    break

                if args.kill_count_max is not None and gate_metrics["count_ratio_proxy"] > args.kill_count_max:
                    print(f"\n[KILL] Gate count ratio proxy {gate_metrics['count_ratio_proxy']:.3f} "
                          f"> {args.kill_count_max} at epoch {epoch}. Stopping.")
                    break

                if args.kill_mpjpe is not None and gate_metrics["mpjpe_proxy"] > args.kill_mpjpe:
                    print(f"\n[KILL] Gate MPJPE proxy {gate_metrics['mpjpe_proxy']:.4f} "
                          f"> {args.kill_mpjpe} at epoch {epoch}. Stopping.")
                    break

                if (
                    args.kill_recall_drop is not None
                    and baseline_recall_proxy is not None
                    and gate_metrics["node_recall_proxy"] < (baseline_recall_proxy - args.kill_recall_drop)
                ):
                    print(f"\n[KILL] Gate recall proxy dropped from {baseline_recall_proxy:.4f} "
                          f"to {gate_metrics['node_recall_proxy']:.4f} (> {args.kill_recall_drop:.4f} drop) "
                          f"at epoch {epoch}. Stopping.")
                    break

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), out_dir / f"checkpoint_e{epoch:03d}.pt")

    # Final save
    torch.save(model.state_dict(), out_dir / "model_final.pt")

    # Save log
    with open(out_dir / "training_log.jsonl", "w") as f:
        for entry in training_log:
            f.write(json.dumps(entry) + "\n")

    # Save config
    config = {
        "model": "HyperBoneStaticRigModel",
        "version": "v2",
        "loss": "RigGraphLossV2 (hungarian+chamfer+repulsion)",
        "feat_dim": args.feat_dim,
        "max_joints": args.max_joints,
        "pc_points": args.pc_points,
        "predict_skinning": not args.no_skinning,
        "backbone": args.backbone,
        "knn_k": args.knn_k,
        "param_count": param_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "grad_accum": args.grad_accum,
        "loss_weights": {
            "hungarian_pos": args.w_hungarian,
            "chamfer": args.w_chamfer,
            "repulsion": args.w_repulsion,
            "edge": args.w_edge,
            "bone_length": args.w_bone,
            "active": args.w_active,
            "unmatched": args.w_unmatched,
        },
        "min_repulsion_dist": args.min_repulsion_dist,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "best_val_loss": round(best_val_loss, 6),
        "best_val_chamfer": round(best_val_chamfer, 6),
        "final_val_spread": training_log[-1]["val_spread"],
        "final_val_active_acc": training_log[-1]["val_active_acc"],
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[Train-v2] DONE")
    print(f"  Best val Chamfer: {best_val_chamfer:.6f}")
    print(f"  Best val loss:    {best_val_loss:.6f}")
    print(f"  Final spread:     {training_log[-1]['val_spread']:.4f}")
    print(f"  Saved: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
