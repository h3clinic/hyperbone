"""Track B v1.1: Topology-aware geometry student with teacher distillation.

Fixes v1's objective mismatch: trains per-sample edge ranking instead of
independent edge BCE. Uses v4.1 teacher-selected edges as ranking targets.

Loss = gt_bce + teacher_bce + rank_loss + listwise_loss

Select best checkpoint by val topology F1, not PR-AUC.

Usage:
    python scripts/train_geometry_student_v11.py \
        --train-cache outputs/models/hyperbone_track_b_student/cache_train_v11.pt \
        --val-cache outputs/models/hyperbone_track_b_student/cache_val_v11.pt \
        --epochs 50 \
        --out outputs/models/hyperbone_track_b_student_v11
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.models.geometry_edge_student import GeometryEdgeStudent
from hyperbone.rigs.undirected_topology import edge_prf, graph_stats


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


def compute_sample_loss(
    model, sample, device,
    gt_bce_weight, teacher_bce_weight, rank_loss_weight, listwise_loss_weight,
    margin, hard_neg_k, pos_weight_tensor,
):
    """Compute per-sample topology-aware loss."""
    n_edges = sample["n_edges"]
    if n_edges < 2:
        return torch.tensor(0.0, device=device)

    patch_i = sample["patch_i"].to(device)
    patch_j = sample["patch_j"].to(device)
    corridor = sample["corridor"].to(device)
    geom_feats = sample["geom_feats"].to(device)
    gt_labels = sample["gt_labels"].to(device)
    teacher_selected = sample["teacher_selected"].to(device)

    logits = model(patch_i, patch_j, corridor, geom_feats)

    loss = torch.tensor(0.0, device=device)

    # A: GT BCE
    if gt_bce_weight > 0:
        gt_bce = F.binary_cross_entropy_with_logits(
            logits, gt_labels, pos_weight=pos_weight_tensor,
        )
        loss = loss + gt_bce_weight * gt_bce

    # B: Teacher BCE
    if teacher_bce_weight > 0:
        teacher_bce = F.binary_cross_entropy_with_logits(
            logits, teacher_selected, pos_weight=pos_weight_tensor,
        )
        loss = loss + teacher_bce_weight * teacher_bce

    # C: Pairwise ranking loss
    # For each positive, enforce score(pos) > score(hard_neg) + margin
    if rank_loss_weight > 0:
        pos_mask = (gt_labels > 0.5) | (teacher_selected > 0.5)
        neg_mask = ~pos_mask
        n_pos = int(pos_mask.sum().item())
        n_neg = int(neg_mask.sum().item())

        if n_pos > 0 and n_neg > 0:
            pos_scores = logits[pos_mask]
            neg_scores = logits[neg_mask]

            # Hard negatives: top-k highest-scoring negatives
            k = min(hard_neg_k, n_neg)
            hard_neg_scores, _ = torch.topk(neg_scores, k)

            # All-pairs margin loss: each pos vs each hard neg
            rank_loss = F.relu(
                margin - pos_scores.unsqueeze(1) + hard_neg_scores.unsqueeze(0)
            ).mean()
            loss = loss + rank_loss_weight * rank_loss

    # D: Listwise loss (approx NDCG / softmax cross-entropy over edges)
    # Encourage top-E edges by score to match the GT/teacher positive set
    if listwise_loss_weight > 0:
        target = torch.clamp(gt_labels + teacher_selected, 0.0, 1.0)
        n_target = int(target.sum().item())
        if n_target > 0 and n_target < n_edges:
            # Softmax cross-entropy: treat edge selection as classification
            # log_softmax over all edges, supervised by target distribution
            log_probs = F.log_softmax(logits, dim=0)
            target_dist = target / target.sum()
            listwise = -(target_dist * log_probs).sum()
            loss = loss + listwise_loss_weight * listwise

    return loss


@torch.no_grad()
def eval_val_topology_f1(model, val_cached, device, dist_weight=10.0):
    """Evaluate topology F1 on val via MST decoder. Returns mean F1."""
    model.eval()
    f1s = []
    for sample in val_cached:
        n_edges = sample["n_edges"]
        if n_edges == 0:
            continue

        # Score edges in batches
        all_scores = []
        for start in range(0, n_edges, 512):
            end = min(start + 512, n_edges)
            logit = model(
                sample["patch_i"][start:end].to(device),
                sample["patch_j"][start:end].to(device),
                sample["corridor"][start:end].to(device),
                sample["geom_feats"][start:end].to(device),
            )
            all_scores.append(logit.cpu())
        edge_scores = torch.cat(all_scores)

        # Decode via MST
        active_mask = sample["active_mask"]
        active_nodes = torch.where(active_mask)[0].tolist()
        n_nodes = active_mask.shape[0]
        pairs = sample["edge_pairs"]
        joint_pos = sample["joint_pos"]

        candidates = []
        for e_idx in range(n_edges):
            i, j = int(pairs[e_idx, 0]), int(pairs[e_idx, 1])
            dist = float(torch.norm(joint_pos[i] - joint_pos[j]))
            rel_dist = float(sample["geom_feats"][e_idx, 1])
            score = float(edge_scores[e_idx]) - dist_weight * rel_dist
            candidates.append(EdgeCandidate(i=i, j=j, dist=dist, score=score))

        pred_mask = kruskal_mst_from_scores(n_nodes, active_nodes, candidates)
        prf = edge_prf(pred_mask, sample["gt_adj"], active_mask)
        f1s.append(prf["f1"])

    return float(np.mean(f1s)) if f1s else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Track B v1.1: Topology-aware geometry student")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patch-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--out", required=True)
    # Loss weights
    parser.add_argument("--gt-bce-weight", type=float, default=1.0)
    parser.add_argument("--teacher-bce-weight", type=float, default=0.5)
    parser.add_argument("--rank-loss-weight", type=float, default=2.0)
    parser.add_argument("--listwise-loss-weight", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--hard-neg-k", type=int, default=16)
    # Eval
    parser.add_argument("--dist-weight", type=float, default=10.0)
    parser.add_argument("--samples-per-epoch", type=int, default=0,
                        help="Subsample train samples per epoch (0=all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Track B v1.1: Topology-Aware Geometry Student", flush=True)
    print("NO skinning in student input.", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Loss weights: gt_bce={args.gt_bce_weight} teacher_bce={args.teacher_bce_weight} "
          f"rank={args.rank_loss_weight} listwise={args.listwise_loss_weight}", flush=True)
    print(f"Margin={args.margin} hard_neg_k={args.hard_neg_k}", flush=True)

    train_cached = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    val_cached = torch.load(args.val_cache, map_location="cpu", weights_only=False)
    print(f"\nLoading caches...", flush=True)
    print(f"  Train: {len(train_cached)} samples", flush=True)
    print(f"  Val:   {len(val_cached)} samples", flush=True)

    # Verify teacher labels present
    assert "teacher_selected" in train_cached[0], "Train cache missing teacher_selected"
    assert "teacher_selected" in val_cached[0], "Val cache missing teacher_selected"

    # Compute pos_weight from train
    total_pos = sum(float(s["gt_labels"].sum()) for s in train_cached)
    total_neg = sum(float((1 - s["gt_labels"]).sum()) for s in train_cached)
    pw = total_neg / max(total_pos, 1)
    pos_weight = torch.tensor([pw], device=device)
    print(f"  Train pos: {total_pos}, neg: {total_neg}, pos_weight: {pw:.2f}", flush=True)

    model = GeometryEdgeStudent(
        patch_in_channels=6, patch_out_dim=args.patch_dim,
        geom_feat_dim=16, hidden_dim=args.hidden_dim, dropout=0.1,
    ).to(device)
    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = 0.0
    best_epoch = 0
    history = []

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

        for si in indices:
            sample = train_cached[si]
            loss = compute_sample_loss(
                model, sample, device,
                args.gt_bce_weight, args.teacher_bce_weight,
                args.rank_loss_weight, args.listwise_loss_weight,
                args.margin, args.hard_neg_k, pos_weight,
            )
            if loss.item() == 0:
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_samples += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_samples, 1)

        # Val topology F1
        val_f1 = eval_val_topology_f1(model, val_cached, device, args.dist_weight)

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best_student_v11.pt")

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
        "track": "B_v1.1_topology_aware_student",
        "skinning_used_as_input": False,
        "n_train_samples": len(train_cached),
        "n_val_samples": len(val_cached),
        "loss_weights": {
            "gt_bce": args.gt_bce_weight,
            "teacher_bce": args.teacher_bce_weight,
            "rank_loss": args.rank_loss_weight,
            "listwise_loss": args.listwise_loss_weight,
        },
        "margin": args.margin,
        "hard_neg_k": args.hard_neg_k,
        "dist_weight": args.dist_weight,
        "best_val_topology_f1": best_val_f1,
        "best_epoch": best_epoch,
        "history": history,
    }
    with open(out_dir / "training_report_v11.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport -> {out_dir / 'training_report_v11.json'}", flush=True)
    print(f"Model  -> {out_dir / 'best_student_v11.pt'}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
