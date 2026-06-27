"""
HyperNodeNet v0 training script.

Usage:
  python scripts/train_hypernode_net_v0.py \
    --graphs outputs/labelforge_v05/graphs/trainable_graphs.jsonl \
    --out outputs/models/hypernode_net_v0 \
    --epochs 20 --batch-size 16 --resolution 192 --max-nodes 64 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hyperbone.labels.schema import load_graph_labels, save_graph_labels, GraphLabel
from hyperbone.hypernodes.dataset import (
    HyperNodeDataset,
    split_train_val,
    NUM_NODE_TYPES,
    NUM_EDGE_TYPES,
)
from hyperbone.hypernodes.losses import HyperNodeLoss, LossWeights
from hyperbone.hypernodes.decode import decode_predictions, save_predictions_jsonl, DecodeConfig
from hyperbone.hypernodes.eval import compute_metrics
from hyperbone.models.hypernode_net import HyperNodeNet


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate that handles string fields."""
    keys = batch[0].keys()
    result = {}
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            result[k] = torch.stack([b[k] for b in batch])
        elif isinstance(batch[0][k], str):
            result[k] = [b[k] for b in batch]
        elif isinstance(batch[0][k], (int, float)):
            result[k] = [b[k] for b in batch]
        else:
            result[k] = [b[k] for b in batch]
    return result


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: HyperNodeLoss,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> dict[str, float]:
    model.train()
    total_losses = {}
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = {
            "heatmaps": batch["heatmaps"].to(device),
            "radius_map": batch["radius_map"].to(device),
            "node_active": batch["node_active"].to(device),
            "node_xy": batch["node_xy"].to(device),
            "node_type": batch["node_type"].to(device),
            "edge_active": batch["edge_active"].to(device),
            "edge_type": batch["edge_type"].to(device),
        }

        pred = model(images)
        loss, loss_dict = loss_fn(pred, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0.0) + v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: HyperNodeLoss,
    device: str,
    val_graphs: list[GraphLabel],
    decode_config: DecodeConfig,
) -> tuple[dict[str, float], list[GraphLabel]]:
    model.eval()
    total_losses = {}
    n_batches = 0
    all_predictions = []

    for batch in loader:
        images = batch["image"].to(device)
        targets = {
            "heatmaps": batch["heatmaps"].to(device),
            "radius_map": batch["radius_map"].to(device),
            "node_active": batch["node_active"].to(device),
            "node_xy": batch["node_xy"].to(device),
            "node_type": batch["node_type"].to(device),
            "edge_active": batch["edge_active"].to(device),
            "edge_type": batch["edge_type"].to(device),
        }

        pred = model(images)
        loss, loss_dict = loss_fn(pred, targets)

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0.0) + v
        n_batches += 1

        # Decode predictions
        sample_ids = batch["sample_id"]
        decoded = decode_predictions(pred, decode_config, sample_ids)
        all_predictions.extend(decoded)

    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    return avg_losses, all_predictions


def main():
    parser = argparse.ArgumentParser(description="Train HyperNodeNet v0")
    parser.add_argument("--graphs", required=True, help="Path to trainable_graphs.jsonl")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=192)
    parser.add_argument("--max-nodes", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading graphs from {args.graphs}...")
    all_graphs = load_graph_labels(Path(args.graphs))
    print(f"  Loaded {len(all_graphs)} graphs")

    # Split
    train_graphs, val_graphs = split_train_val(all_graphs, args.val_fraction, args.seed)
    print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}")

    # Save splits
    save_graph_labels(train_graphs, out_dir / "train_split.jsonl")
    save_graph_labels(val_graphs, out_dir / "val_split.jsonl")

    # Count input modalities
    train_ds = HyperNodeDataset(
        graphs_path="",
        resolution=args.resolution,
        max_nodes=args.max_nodes,
        augment=True,
        graphs=train_graphs,
    )
    val_ds = HyperNodeDataset(
        graphs_path="",
        resolution=args.resolution,
        max_nodes=args.max_nodes,
        augment=False,
        graphs=val_graphs,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.workers, pin_memory=True,
    )

    # Model
    model = HyperNodeNet(
        in_channels=1,
        base_channels=args.base_channels,
        max_nodes=args.max_nodes,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
    ).to(args.device)

    print(f"  Model params: {model.param_count():,}")

    # Loss & optimizer
    loss_fn = HyperNodeLoss(LossWeights())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Config
    config = {
        "architecture": "HyperNodeNet-v0",
        "in_channels": 1,
        "base_channels": args.base_channels,
        "max_nodes": args.max_nodes,
        "num_node_types": NUM_NODE_TYPES,
        "num_edge_types": NUM_EDGE_TYPES,
        "resolution": args.resolution,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_graphs": len(train_graphs),
        "val_graphs": len(val_graphs),
        "total_params": model.param_count(),
        "device": args.device,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    train_log = []
    best_val_loss = float("inf")
    decode_config = DecodeConfig(
        max_nodes=args.max_nodes,
        active_threshold=0.3,
        confidence_threshold=0.2,
        edge_threshold=0.3,
        nms_radius=0.03,
    )

    print(f"\n{'='*60}")
    print(f"Training HyperNodeNet v0 — {args.epochs} epochs")
    print(f"{'='*60}")

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()

        # Train
        train_losses = train_epoch(model, train_loader, loss_fn, optimizer, args.device)

        # Validate
        val_losses, val_predictions = validate(
            model, val_loader, loss_fn, args.device, val_graphs, decode_config
        )

        scheduler.step()

        epoch_time = time.time() - epoch_t0

        # Log
        entry = {
            "epoch": epoch,
            "train": train_losses,
            "val": val_losses,
            "lr": scheduler.get_last_lr()[0],
            "time": epoch_time,
        }
        train_log.append(entry)

        # Print
        print(
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_losses['total']:.4f} | "
            f"val_loss={val_losses['total']:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        # Save best
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            torch.save(model.state_dict(), out_dir / "model.pt")

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.1f}s")

    # Save train log
    with open(out_dir / "train_log.jsonl", "w") as f:
        for entry in train_log:
            f.write(json.dumps(entry) + "\n")

    # Final validation with metrics
    print("\n=== Final Evaluation ===")
    model.load_state_dict(torch.load(out_dir / "model.pt", weights_only=True))
    val_losses, val_predictions = validate(
        model, val_loader, loss_fn, args.device, val_graphs, decode_config
    )

    # Save val predictions
    save_predictions_jsonl(val_predictions, out_dir / "val_predictions.jsonl")

    # Compute metrics (0.08 threshold is forgiving for normalized coords)
    metrics = compute_metrics(val_predictions, val_graphs, distance_threshold=0.08)

    # Count input modalities
    real_count = sum(1 for g in all_graphs if g.image_path and Path(g.image_path).exists())
    rasterized_count = len(all_graphs) - real_count

    # Final metrics
    final_metrics = {
        "train_graphs": len(train_graphs),
        "val_graphs": len(val_graphs),
        "input_modality": {
            "real_image": real_count,
            "graph_rasterized": rasterized_count,
        },
        "train_loss_start": train_log[0]["train"]["total"],
        "train_loss_end": train_log[-1]["train"]["total"],
        "val_loss_start": train_log[0]["val"]["total"],
        "val_loss_end": train_log[-1]["val"]["total"],
        "node_f1": metrics.node_f1,
        "edge_f1": metrics.edge_f1,
        "graph_chamfer": metrics.graph_chamfer,
        "node_type_accuracy": metrics.node_type_accuracy,
        "invalid_graph_rate": metrics.invalid_graph_rate,
        "per_node_type_f1": metrics.per_node_type_f1,
        "per_source_metrics": metrics.per_source_metrics,
        "avg_pred_node_count": metrics.avg_pred_node_count,
        "avg_target_node_count": metrics.avg_target_node_count,
        "total_time_seconds": total_time,
    }

    # Verdict
    loss_decreased = train_log[-1]["train"]["total"] < train_log[0]["train"]["total"] * 0.8
    node_f1_ok = metrics.node_f1 > 0.50
    edge_f1_ok = metrics.edge_f1 > 0.30
    invalid_ok = metrics.invalid_graph_rate < 0.20

    if loss_decreased and node_f1_ok and edge_f1_ok and invalid_ok:
        verdict = "ARCHITECTURE_WORKS"
    elif loss_decreased and (not node_f1_ok or not edge_f1_ok):
        verdict = "DATASET_TOO_SYNTHETIC"
    else:
        verdict = "FAIL"

    final_metrics["verdict"] = verdict

    # Convert numpy types for JSON serialization
    def _jsonable(obj):
        import numpy as np
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonable(v) for v in obj]
        return obj

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(_jsonable(final_metrics), f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"HyperNodeNet v0 — RESULTS")
    print(f"{'='*60}")
    print(f"  Train loss: {final_metrics['train_loss_start']:.4f} → {final_metrics['train_loss_end']:.4f}")
    print(f"  Val loss:   {final_metrics['val_loss_start']:.4f} → {final_metrics['val_loss_end']:.4f}")
    print(f"  Node F1:    {metrics.node_f1:.4f}")
    print(f"  Edge F1:    {metrics.edge_f1:.4f}")
    print(f"  Chamfer:    {metrics.graph_chamfer:.4f}")
    print(f"  Invalid:    {metrics.invalid_graph_rate:.2%}")
    print(f"  Verdict:    {verdict}")
    print(f"\n  Per-type F1:")
    for nt, f1 in sorted(metrics.per_node_type_f1.items()):
        print(f"    {nt}: {f1:.4f}")
    if metrics.per_source_metrics:
        print(f"\n  Per-source:")
        for src, sm in sorted(metrics.per_source_metrics.items()):
            print(f"    {src}: node_f1={sm['node_f1']:.4f}, edge_f1={sm['edge_f1']:.4f}, n={sm['count']}")
    print(f"\n  Output: {out_dir}")


if __name__ == "__main__":
    main()
