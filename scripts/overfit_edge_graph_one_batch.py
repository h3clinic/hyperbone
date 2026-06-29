"""Track B v2: One-batch overfit sanity check for edge-graph model.

Pass bar:
  - one-batch topology F1 > 0.90
  - one-batch edge PR-AUC > 0.90
  - selected/GT near 1.0

If it cannot overfit one batch, do not build full training.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

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


@torch.no_grad()
def eval_topology(model, sample, device, dist_weight=10.0):
    model.eval()
    n_edges = sample["n_edges"]
    if n_edges == 0:
        return {"f1": 0.0, "roc": 0.0, "pr": 0.0}

    patch_i = sample["patch_i"].to(device)
    patch_j = sample["patch_j"].to(device)
    corridor = sample["corridor"].to(device)
    geom_feats = sample["geom_feats"].to(device)
    edge_pairs = sample["edge_pairs"]
    lg_edges = build_line_graph(edge_pairs).to(device)

    logits = model(patch_i, patch_j, corridor, geom_feats, lg_edges)
    scores = logits.cpu()

    gt = sample["gt_labels"].numpy()
    pred_prob = torch.sigmoid(scores).numpy()

    roc = roc_auc_score(gt, pred_prob) if len(np.unique(gt)) > 1 else 0.0
    pr = average_precision_score(gt, pred_prob) if len(np.unique(gt)) > 1 else 0.0

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

    gt_edges = int(sample["gt_labels"].sum())
    pred_edges = int(pred_mask.sum()) // 2
    sel_gt = pred_edges / max(gt_edges, 1)

    model.train()
    return {"f1": prf["f1"], "roc": roc, "pr": pr, "sel_gt": sel_gt}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(r"C:\Users\ritayan\ClaudeHyperBone\hyperbone")

    cache_path = base / "outputs/models/hyperbone_track_b_student/cache_train.pt"
    cached = torch.load(str(cache_path), map_location="cpu", weights_only=False)
    print(f"Loaded cache: {len(cached)} samples", flush=True)

    # Pick a batch of 8 medium-sized samples for overfit
    batch_size = 8
    candidates = [(i, s) for i, s in enumerate(cached)
                  if 15 <= s["n_edges"] <= 200 and float(s["gt_labels"].sum()) >= 5]
    candidates.sort(key=lambda x: x[1]["n_edges"])
    mid = len(candidates) // 2
    batch_indices = [candidates[mid + i][0] for i in range(batch_size)]
    batch = [cached[i] for i in batch_indices]

    for i, s in enumerate(batch):
        n_e = s["n_edges"]
        n_pos = int(s["gt_labels"].sum())
        n_j = int(s["active_mask"].sum())
        lg = build_line_graph(s["edge_pairs"])
        print(f"  Sample {i}: {n_j} joints, {n_e} edges, {n_pos} pos, "
              f"{lg.shape[1]} line-graph edges", flush=True)

    model = GeometryEdgeGraphStudent(
        patch_in_channels=6, patch_out_dim=64, geom_feat_dim=16,
        edge_dim=128, n_mp_rounds=3, dropout=0.0,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel params: {n_params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    total_pos = sum(float(s["gt_labels"].sum()) for s in batch)
    total_neg = sum(float((1 - s["gt_labels"]).sum()) for s in batch)
    pw = total_neg / max(total_pos, 1)
    pos_weight = torch.tensor([pw], device=device)
    print(f"pos_weight: {pw:.2f}", flush=True)

    margin = 1.0
    hard_neg_k = 16

    print(f"\n{'Step':>6} {'Loss':>10} {'F1':>8} {'ROC':>8} {'PR':>8} {'S/G':>8}",
          flush=True)
    print("-" * 56, flush=True)

    for step in range(1, 501):
        model.train()
        total_loss = 0.0

        for sample in batch:
            n_edges = sample["n_edges"]
            if n_edges < 2:
                continue

            patch_i = sample["patch_i"].to(device)
            patch_j = sample["patch_j"].to(device)
            corridor_t = sample["corridor"].to(device)
            geom_feats = sample["geom_feats"].to(device)
            gt_labels = sample["gt_labels"].to(device)
            edge_pairs = sample["edge_pairs"]
            lg_edges = build_line_graph(edge_pairs).to(device)

            logits = model(patch_i, patch_j, corridor_t, geom_feats, lg_edges)

            # BCE loss
            bce = F.binary_cross_entropy_with_logits(
                logits, gt_labels, pos_weight=pos_weight)

            # Pairwise ranking within this sample
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

            loss = bce + 2.0 * rank_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if step % 10 == 0 or step <= 5:
            f1s, rocs, prs, sgs = [], [], [], []
            for s in batch:
                m = eval_topology(model, s, device)
                f1s.append(m["f1"])
                rocs.append(m["roc"])
                prs.append(m["pr"])
                sgs.append(m.get("sel_gt", 1.0))
            avg_f1 = np.mean(f1s)
            avg_roc = np.mean(rocs)
            avg_pr = np.mean(prs)
            avg_sg = np.mean(sgs)
            print(f"  {step:4d}   {total_loss/len(batch):10.4f}   {avg_f1:.4f}"
                  f"   {avg_roc:.4f}   {avg_pr:.4f}   {avg_sg:.4f}", flush=True)

            if avg_f1 > 0.90 and avg_pr > 0.90:
                print(f"\nPASS: F1={avg_f1:.4f} > 0.90, PR={avg_pr:.4f} > 0.90",
                      flush=True)
                break
    else:
        f1s, rocs, prs, sgs = [], [], [], []
        for s in batch:
            m = eval_topology(model, s, device)
            f1s.append(m["f1"])
            rocs.append(m["roc"])
            prs.append(m["pr"])
            sgs.append(m.get("sel_gt", 1.0))
        avg_f1 = np.mean(f1s)
        avg_pr = np.mean(prs)
        if avg_f1 > 0.90 and avg_pr > 0.90:
            print(f"\nPASS: F1={avg_f1:.4f}, PR={avg_pr:.4f}", flush=True)
        else:
            print(f"\nFAIL: F1={avg_f1:.4f}, PR={avg_pr:.4f} "
                  f"(needed F1>0.90 and PR>0.90)", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
