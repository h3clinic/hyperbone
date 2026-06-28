"""Track B v1: Full geometry-only student training.

Trains on train split, selects best checkpoint on val, tests once.
No skinning features in student input.

Usage:
    python scripts/train_geometry_student_full.py \
        --train-cache outputs/models/hyperbone_track_b_student/cache_train.pt \
        --val-cache outputs/models/hyperbone_track_b_student/cache_val.pt \
        --epochs 50 \
        --out outputs/models/hyperbone_track_b_student
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.models.geometry_edge_student import GeometryEdgeStudent


class CachedEdgeDataset(torch.utils.data.Dataset):
    """Flat edge dataset from precomputed cache."""

    def __init__(self, cached_samples):
        self.edges = []
        for s in cached_samples:
            n = s["n_edges"]
            for e in range(n):
                self.edges.append({
                    "patch_i": s["patch_i"][e],
                    "patch_j": s["patch_j"][e],
                    "corridor": s["corridor"][e],
                    "geom_feats": s["geom_feats"][e],
                    "label": s["gt_labels"][e],
                })

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        e = self.edges[idx]
        return e["patch_i"], e["patch_j"], e["corridor"], e["geom_feats"], e["label"]


def train_epoch(model, loader, optimizer, device, pos_weight):
    model.train()
    total_loss = 0
    n = 0
    for patch_i, patch_j, corridor, geom, label in loader:
        patch_i = patch_i.to(device)
        patch_j = patch_j.to(device)
        corridor = corridor.to(device)
        geom = geom.to(device)
        label = label.to(device)

        logit = model(patch_i, patch_j, corridor, geom)
        loss = F.binary_cross_entropy_with_logits(logit, label, pos_weight=pos_weight)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * label.shape[0]
        n += label.shape[0]
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_edge_metrics(model, loader, device):
    """Compute edge-level classification metrics."""
    model.eval()
    all_logits, all_labels = [], []
    for patch_i, patch_j, corridor, geom, label in loader:
        logit = model(
            patch_i.to(device), patch_j.to(device),
            corridor.to(device), geom.to(device),
        )
        all_logits.append(logit.cpu())
        all_labels.append(label)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    if len(np.unique(labels)) < 2:
        return {"roc_auc": 0.0, "pr_auc": 0.0}

    return {
        "roc_auc": float(roc_auc_score(labels, logits)),
        "pr_auc": float(average_precision_score(labels, logits)),
    }


@torch.no_grad()
def score_all_edges(model, cached_samples, device, batch_size=512):
    """Run model on all edges in cached samples, return per-sample score matrices."""
    model.eval()
    results = []

    for s in cached_samples:
        n_edges = s["n_edges"]
        if n_edges == 0:
            results.append(None)
            continue

        scores = []
        for start in range(0, n_edges, batch_size):
            end = min(start + batch_size, n_edges)
            logit = model(
                s["patch_i"][start:end].to(device),
                s["patch_j"][start:end].to(device),
                s["corridor"][start:end].to(device),
                s["geom_feats"][start:end].to(device),
            )
            scores.append(logit.cpu())

        score_vec = torch.cat(scores)

        # Build score matrix
        max_nodes = s["joint_pos"].shape[0]
        score_mat = torch.zeros(max_nodes, max_nodes)
        for e_idx in range(n_edges):
            i, j = int(s["edge_pairs"][e_idx, 0]), int(s["edge_pairs"][e_idx, 1])
            score_mat[i, j] = score_vec[e_idx]
            score_mat[j, i] = score_vec[e_idx]

        results.append(score_mat)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Track B: Full geometry-only student training")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patch-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Track B v1: Full Geometry-Only Student Training", flush=True)
    print("NO skinning in student input.", flush=True)
    print(f"Device: {device}", flush=True)

    # Load caches
    print("\nLoading caches...", flush=True)
    train_cached = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    val_cached = torch.load(args.val_cache, map_location="cpu", weights_only=False)
    print(f"  Train: {len(train_cached)} samples", flush=True)
    print(f"  Val:   {len(val_cached)} samples", flush=True)

    train_ds = CachedEdgeDataset(train_cached)
    val_ds = CachedEdgeDataset(val_cached)
    print(f"  Train edges: {len(train_ds)}", flush=True)
    print(f"  Val edges:   {len(val_ds)}", flush=True)

    # Class balance
    train_labels = [train_ds[i][4].item() for i in range(len(train_ds))]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32).to(device)
    print(f"  Train pos: {n_pos}, neg: {n_neg}, pos_weight: {pos_weight.item():.2f}", flush=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Model
    model = GeometryEdgeStudent(
        patch_in_channels=6, patch_out_dim=args.patch_dim,
        geom_feat_dim=16, hidden_dim=args.hidden_dim, dropout=0.1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel params: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Baseline
    val_labels_np = np.array([val_ds[i][4].item() for i in range(len(val_ds))])
    val_euc = np.array([val_ds[i][3][0].item() for i in range(len(val_ds))])
    base_roc = float(roc_auc_score(val_labels_np, -val_euc))
    base_pr = float(average_precision_score(val_labels_np, -val_euc))
    print(f"\nBaseline (Euclidean): ROC={base_roc:.4f}  PR={base_pr:.4f}", flush=True)

    # Train
    print(f"\n{'Epoch':>6} {'Loss':>10} {'Val ROC':>10} {'Val PR':>10} "
          f"{'vs base':>10}", flush=True)
    print("-" * 55, flush=True)

    best_pr = 0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device, pos_weight)
        metrics = eval_edge_metrics(model, val_loader, device)
        scheduler.step()

        delta = metrics["roc_auc"] - base_roc
        marker = " *" if metrics["pr_auc"] > best_pr else ""
        print(f"  {epoch:4d}   {loss:10.4f} {metrics['roc_auc']:10.4f} "
              f"{metrics['pr_auc']:10.4f} {delta:+10.4f}{marker}", flush=True)

        history.append({
            "epoch": epoch, "loss": loss,
            "val_roc_auc": metrics["roc_auc"], "val_pr_auc": metrics["pr_auc"],
        })

        if metrics["pr_auc"] > best_pr:
            best_pr = metrics["pr_auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, str(out_dir / "best_student.pt"))

    # Load best and report
    model.load_state_dict(best_state)
    final = eval_edge_metrics(model, val_loader, device)
    best_epoch = max(history, key=lambda x: x["val_pr_auc"])

    print(f"\n{'='*60}", flush=True)
    print("Training Complete", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Baseline (Euclidean):    ROC={base_roc:.4f}  PR={base_pr:.4f}", flush=True)
    print(f"  Best student (ep {best_epoch['epoch']}):  "
          f"ROC={best_epoch['val_roc_auc']:.4f}  PR={best_epoch['val_pr_auc']:.4f}", flush=True)
    print(f"  Delta:                  ROC={best_epoch['val_roc_auc'] - base_roc:+.4f}  "
          f"PR={best_epoch['val_pr_auc'] - base_pr:+.4f}", flush=True)

    # Save report
    report = {
        "track": "B_geometry_student_full",
        "skinning_used_as_input": False,
        "n_train_samples": len(train_cached),
        "n_val_samples": len(val_cached),
        "n_train_edges": len(train_ds),
        "n_val_edges": len(val_ds),
        "model_params": n_params,
        "baseline_roc": base_roc,
        "baseline_pr": base_pr,
        "best_val_roc": best_epoch["val_roc_auc"],
        "best_val_pr": best_epoch["val_pr_auc"],
        "best_epoch": best_epoch["epoch"],
        "history": history,
    }
    with open(out_dir / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport -> {out_dir / 'training_report.json'}", flush=True)
    print(f"Model  -> {out_dir / 'best_student.pt'}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
