"""Track B: Error analysis for edge-graph student topology.

Given a checkpoint, MST-decodes each sample (student_only) and
characterizes where the model fails:
  - F1 by joint-count bucket
  - F1 vs GT max-degree (hub-heavy skeletons)
  - False-positive vs false-negative edge geometry (edge-length percentile)
  - Worst-N samples

Read-only diagnostics; informs next-architecture decisions and reports.

Usage:
    python scripts/analyze_edge_graph_errors.py \
        --cache outputs/models/hyperbone_track_b_student/cache_test_v11.pt \
        --ckpt  outputs/models/hyperbone_track_b_student_v21/best_student_v21.pt \
        --edge-dim 128 --n-mp-rounds 3 \
        --out outputs/models/hyperbone_track_b_student_v21/error_analysis
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.models.geometry_edge_graph_student import (
    GeometryEdgeGraphStudent,
    build_line_graph,
)
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


@torch.no_grad()
def decode_sample(model, sample, device):
    n_edges = sample["n_edges"]
    pi = sample["patch_i"].to(device)
    pj = sample["patch_j"].to(device)
    co = sample["corridor"].to(device)
    gf = sample["geom_feats"].to(device)
    lg = build_line_graph(sample["edge_pairs"]).to(device)
    scores = model(pi, pj, co, gf, lg).cpu()

    active_mask = sample["active_mask"]
    active_nodes = torch.where(active_mask)[0].tolist()
    n_nodes = active_mask.shape[0]
    joint_pos = sample["joint_pos"]
    pairs = sample["edge_pairs"]

    candidates = []
    dists = []
    for e in range(n_edges):
        i, j = int(pairs[e, 0]), int(pairs[e, 1])
        d = float(torch.norm(joint_pos[i] - joint_pos[j]))
        dists.append(d)
        candidates.append(EdgeCandidate(i=i, j=j, dist=d, score=float(scores[e])))
    pred_mask = kruskal_mst_from_scores(n_nodes, active_nodes, candidates)
    return pred_mask, np.array(dists, dtype=np.float32)


def gt_max_degree(sample):
    adj = sample["gt_adj"].bool()
    am = sample["active_mask"]
    deg = adj[am][:, am].sum(dim=1)
    return int(deg.max()) if deg.numel() else 0


def main():
    parser = argparse.ArgumentParser(description="Edge-graph error analysis")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--edge-dim", type=int, default=128)
    parser.add_argument("--n-mp-rounds", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Error analysis: {args.ckpt}", flush=True)

    model = GeometryEdgeGraphStudent(
        patch_in_channels=6, patch_out_dim=64, geom_feat_dim=16,
        edge_dim=args.edge_dim, n_mp_rounds=args.n_mp_rounds, dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()

    cached = torch.load(args.cache, map_location="cpu", weights_only=False)
    print(f"Samples: {len(cached)}", flush=True)

    per_sample = []
    # FP / FN edge-length percentile accumulation (relative to sample's edge lengths)
    fp_len_pct, fn_len_pct = [], []

    for si, sample in enumerate(cached):
        if sample["n_edges"] == 0:
            continue
        pred_mask, dists = decode_sample(model, sample, device)
        prf = edge_prf(pred_mask, sample["gt_adj"], sample["active_mask"])
        gs = graph_stats(pred_mask, sample["active_mask"])
        n_j = int(sample["active_mask"].sum())
        maxdeg = gt_max_degree(sample)

        # Per-edge FP/FN characterization.
        am = sample["active_mask"]
        gt = sample["gt_adj"].bool()
        pred = pred_mask.bool()
        # edge-length percentile lookup per candidate
        order = np.argsort(dists)
        pct = np.empty_like(dists)
        pct[order] = np.linspace(0, 1, len(dists)) if len(dists) > 1 else np.array([0.5])
        pairs = sample["edge_pairs"]
        for e in range(sample["n_edges"]):
            i, j = int(pairs[e, 0]), int(pairs[e, 1])
            if not (am[i] and am[j]):
                continue
            is_gt = bool(gt[i, j])
            is_pred = bool(pred[i, j])
            if is_pred and not is_gt:
                fp_len_pct.append(float(pct[e]))
            elif is_gt and not is_pred:
                fn_len_pct.append(float(pct[e]))

        per_sample.append({
            "idx": si, "n_joints": n_j, "gt_max_degree": maxdeg,
            "f1": prf["f1"], "precision": prf["precision"], "recall": prf["recall"],
            "cycles": gs["cycle_count"],
        })

    f1s = np.array([r["f1"] for r in per_sample])
    print(f"\nOverall test F1: {f1s.mean():.4f}", flush=True)

    # By joint-count bucket
    print("\n== F1 by joint-count bucket ==", flush=True)
    jbuckets = [("tiny(<=10)", 0, 10), ("small(11-25)", 11, 25),
                ("medium(26-50)", 26, 50), ("large(>50)", 51, 10**9)]
    for name, lo, hi in jbuckets:
        rows = [r for r in per_sample if lo <= r["n_joints"] <= hi]
        if rows:
            print(f"  {name:<16} N={len(rows):4d}  F1={np.mean([r['f1'] for r in rows]):.4f}",
                  flush=True)

    # By GT max-degree bucket (hub-heaviness)
    print("\n== F1 by GT max-degree (hub-heaviness) ==", flush=True)
    dbuckets = [("deg<=3", 0, 3), ("deg4-5", 4, 5), ("deg6-8", 6, 8),
                ("deg>=9", 9, 10**9)]
    for name, lo, hi in dbuckets:
        rows = [r for r in per_sample if lo <= r["gt_max_degree"] <= hi]
        if rows:
            print(f"  {name:<10} N={len(rows):4d}  F1={np.mean([r['f1'] for r in rows]):.4f}",
                  flush=True)

    # FP/FN edge-length distribution
    print("\n== Error edge-length percentile (0=shortest, 1=longest cand.) ==", flush=True)
    if fp_len_pct:
        print(f"  false positives: N={len(fp_len_pct):5d}  "
              f"mean_len_pct={np.mean(fp_len_pct):.3f}", flush=True)
    if fn_len_pct:
        print(f"  false negatives: N={len(fn_len_pct):5d}  "
              f"mean_len_pct={np.mean(fn_len_pct):.3f}", flush=True)
    print("  (FN skewed high => model misses long true edges; "
          "FP skewed low => model over-connects near neighbors)", flush=True)

    # Worst samples
    print("\n== Worst 10 samples ==", flush=True)
    worst = sorted(per_sample, key=lambda r: r["f1"])[:10]
    print(f"  {'idx':>5} {'joints':>7} {'maxdeg':>7} {'F1':>7}", flush=True)
    for r in worst:
        print(f"  {r['idx']:5d} {r['n_joints']:7d} {r['gt_max_degree']:7d} "
              f"{r['f1']:7.4f}", flush=True)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "ckpt": args.ckpt,
            "overall_f1": float(f1s.mean()),
            "fp_mean_len_pct": float(np.mean(fp_len_pct)) if fp_len_pct else None,
            "fn_mean_len_pct": float(np.mean(fn_len_pct)) if fn_len_pct else None,
            "n_fp": len(fp_len_pct), "n_fn": len(fn_len_pct),
            "worst": worst,
            "per_sample": per_sample,
        }
        with open(out_dir / "error_analysis.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport -> {out_dir / 'error_analysis.json'}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
