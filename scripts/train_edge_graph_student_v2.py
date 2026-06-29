"""Track B v2: Edge-graph message passing student — full training.

Trains per-sample with line graph context. Selects checkpoint by val
topology F1. Uses BCE + pairwise ranking loss.

Usage:
    python scripts/train_edge_graph_student_v2.py \
        --train-cache outputs/models/hyperbone_track_b_student/cache_train.pt \
        --val-cache outputs/models/hyperbone_track_b_student/cache_val.pt \
        --epochs 50 \
        --out outputs/models/hyperbone_track_b_student_v2
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.models.geometry_edge_graph_student import (
    GeometryEdgeGraphStudent,
    build_line_graph,
)
from hyperbone.rigs.undirected_topology import edge_prf


@dataclass
class EdgeCandidate:
    i: int
    j: int
    dist: float
    score: float


def kruskal_mst_from_scores(n_nodes, active_nodes, edges, max_edges=None):
    if max_edges is None:
        max_edges = max(len(active_nodes) - 1, 0)
    edge_mask = torch.zeros(n_nodes, n_nodes, dtype=torch.bool)
    if len(active_nodes) <= 1 or max_edges <= 0:
        return edge_mask
    parent = {n: n for n in active_nodes}
    rank = {n: 0 for n in active_nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    selected = 0
    for e in sorted(edges, key=lambda x: (x.score, -x.dist), reverse=True):
        if selected >= max_edges:
            break
        if find(e.i) == find(e.j):
            continue
        if union(e.i, e.j):
            edge_mask[e.i, e.j] = True
            edge_mask[e.j, e.i] = True
            selected += 1
    return edge_mask


def compute_sample_loss(model, sample, device, pos_weight, margin, hard_neg_k,
                        rank_weight):
    n_edges = sample["n_edges"]
    if n_edges < 2:
        return torch.tensor(0.0, device=device)

    patch_i = sample["patch_i"].to(device)
    patch_j = sample["patch_j"].to(device)
    corridor = sample["corridor"].to(device)
    geom_feats = sample["geom_feats"].to(device)
    gt_labels = sample["gt_labels"].to(device)
    edge_pairs = sample["edge_pairs"]
    lg_edges = build_line_graph(edge_pairs).to(device)

    logits = model(patch_i, patch_j, corridor, geom_feats, lg_edges)

    bce = F.binary_cross_entropy_with_logits(
        logits, gt_labels, pos_weight=pos_weight)

    rank_loss = torch.tensor(0.0, device=device)
    pos_mask = gt_labels > 0.5
    neg_mask = ~pos_mask
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos > 0 and n_neg > 0:
        pos_scores = logits[pos_mask]
        neg_scores = logits[neg_mask]
        k = min(hard_neg_k, n_neg)
        hard_neg_scores, _ = torch.topk(neg_scores, k)
        rank_loss = F.relu(
            margin - pos_scores.unsqueeze(1) + hard_neg_scores.unsqueeze(0)
        ).mean()

    return bce + rank_weight * rank_loss


@torch.no_grad()
def eval_val_topology_f1(model, val_cached, device, dist_weight=10.0):
    model.eval()
    f1s = []
    for sample in val_cached:
        n_edges = sample["n_edges"]
        if n_edges == 0:
            continue

        patch_i = sample["patch_i"].to(device)
        patch_j = sample["patch_j"].to(device)
        corridor = sample["corridor"].to(device)
        geom_feats = sample["geom_feats"].to(device)
        edge_pairs = sample["edge_pairs"]
        lg_edges = build_line_graph(edge_pairs).to(device)

        logits = model(patch_i, patch_j, corridor, geom_feats, lg_edges)
        scores = logits.cpu()

        active_mask = sample["active_mask"]
        active_nodes = torch.where(active_mask)[0].tolist()
        n_nodes = active_mask.shape[0]
        joint_pos = sample["joint_pos"]

        candidates = []
        for e_idx in range(n_edges):
            i, j = int(edge_pairs[e_idx, 0]), int(edge_pairs[e_idx, 1])
            dist = float(torch.norm(joint_pos[i] - joint_pos[j]))
            rel_dist = float(sample["geom_feats"][e_idx, 1])
            score = float(scores[e_idx]) - dist_weight * rel_dist
            candidates.append(EdgeCandidate(i=i, j=j, dist=dist, score=score))

        pred_mask = kruskal_mst_from_scores(n_nodes, active_nodes, candidates)
        prf = edge_prf(pred_mask, sample["gt_adj"], active_mask)
        f1s.append(prf["f1"])

    return float(np.mean(f1s)) if f1s else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Track B v2: Edge-graph message passing student")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--edge-dim", type=int, default=128)
    parser.add_argument("--n-mp-rounds", type=int, default=3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rank-weight", type=float, default=2.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--hard-neg-k", type=int, default=16)
    parser.add_argument("--dist-weight", type=float, default=10.0)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--accum-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--warmup-epochs", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Track B v2: Edge-Graph Message Passing Student", flush=True)
    print("NO skinning in student input.", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"edge_dim={args.edge_dim} n_mp_rounds={args.n_mp_rounds}", flush=True)
    print(f"rank_weight={args.rank_weight} margin={args.margin} "
          f"hard_neg_k={args.hard_neg_k}", flush=True)

    train_cached = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    val_cached = torch.load(args.val_cache, map_location="cpu", weights_only=False)
    print(f"\nTrain: {len(train_cached)} samples", flush=True)
    print(f"Val:   {len(val_cached)} samples", flush=True)

    total_pos = sum(float(s["gt_labels"].sum()) for s in train_cached)
    total_neg = sum(float((1 - s["gt_labels"]).sum()) for s in train_cached)
    pw = total_neg / max(total_pos, 1)
    pos_weight = torch.tensor([pw], device=device)
    print(f"pos_weight: {pw:.2f}", flush=True)

    model = GeometryEdgeGraphStudent(
        patch_in_channels=6, patch_out_dim=64, geom_feat_dim=16,
        edge_dim=args.edge_dim, n_mp_rounds=args.n_mp_rounds, dropout=0.1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_f1 = 0.0
    best_epoch = 0
    history = []
    accum = args.accum_steps

    print(f"accum_steps={accum} warmup_epochs={args.warmup_epochs}", flush=True)
    print(f"\n{'Epoch':>6} {'Loss':>10} {'Val F1':>10} {'Best':>6}", flush=True)
    print("-" * 40, flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_samples = 0

        indices = list(range(len(train_cached)))
        np.random.shuffle(indices)
        if args.samples_per_epoch > 0:
            indices = indices[:args.samples_per_epoch]

        optimizer.zero_grad()
        for step_i, si in enumerate(indices):
            sample = train_cached[si]
            loss = compute_sample_loss(
                model, sample, device, pos_weight,
                args.margin, args.hard_neg_k, args.rank_weight,
            )
            if loss.item() == 0:
                continue

            (loss / accum).backward()
            epoch_loss += loss.item()
            n_samples += 1

            if (step_i + 1) % accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        if n_samples % accum != 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        avg_loss = epoch_loss / max(n_samples, 1)

        val_f1 = eval_val_topology_f1(model, val_cached, device, args.dist_weight)

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best_student_v2.pt")

        marker = " *" if is_best else ""
        print(f"  {epoch:4d}   {avg_loss:10.4f}   {val_f1:10.4f}  {marker}", flush=True)

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "val_topology_f1": val_f1,
        })

    print(f"\n{'='*50}", flush=True)
    print(f"Training Complete", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"  Best epoch: {best_epoch}", flush=True)
    print(f"  Best val topology F1: {best_val_f1:.4f}", flush=True)

    report = {
        "track": "B_v2_edge_graph_student",
        "skinning_used_as_input": False,
        "n_train_samples": len(train_cached),
        "n_val_samples": len(val_cached),
        "model_params": n_params,
        "edge_dim": args.edge_dim,
        "n_mp_rounds": args.n_mp_rounds,
        "rank_weight": args.rank_weight,
        "margin": args.margin,
        "hard_neg_k": args.hard_neg_k,
        "dist_weight": args.dist_weight,
        "best_val_topology_f1": best_val_f1,
        "best_epoch": best_epoch,
        "history": history,
    }
    with open(out_dir / "training_report_v2.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport -> {out_dir / 'training_report_v2.json'}", flush=True)
    print(f"Model  -> {out_dir / 'best_student_v2.pt'}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
