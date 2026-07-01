"""Track B: N-way score ensemble of edge-graph students.

Generalizes the 2-way v2.2 ensemble to any number of checkpoints.
Ensemble logits = sum_k w_k * logits_k, weights on the simplex.
Grid-search weights on VALIDATION topology F1 only, test once.

Decode: student_only MST.

Usage:
    python scripts/eval_edge_graph_ensemble_nway.py \
        --val-cache  outputs/models/hyperbone_track_b_student/cache_val_v11.pt \
        --test-cache outputs/models/hyperbone_track_b_student/cache_test_v11.pt \
        --ckpts  ".../best_student_v2.pt,.../best_student_v21.pt,.../best_student_v23.pt" \
        --labels "v2,v21,v23" \
        --grid-step 0.1 \
        --out outputs/models/hyperbone_track_b_student_ens
"""
from __future__ import annotations

import argparse
import itertools
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
def compute_logits(model, sample, device):
    n_edges = sample["n_edges"]
    if n_edges == 0:
        return np.zeros(0, dtype=np.float32)
    pi = sample["patch_i"].to(device)
    pj = sample["patch_j"].to(device)
    co = sample["corridor"].to(device)
    gf = sample["geom_feats"].to(device)
    lg = build_line_graph(sample["edge_pairs"]).to(device)
    return model(pi, pj, co, gf, lg).cpu().numpy().astype(np.float32)


def precompute(models, cached, device, tag):
    entries = []
    for si, sample in enumerate(cached):
        if sample["n_edges"] == 0:
            entries.append(None)
            continue
        pairs = sample["edge_pairs"]
        joint_pos = sample["joint_pos"]
        n_edges = sample["n_edges"]
        dists = np.array([
            float(torch.norm(joint_pos[int(pairs[e, 0])] - joint_pos[int(pairs[e, 1])]))
            for e in range(n_edges)
        ], dtype=np.float32)
        entries.append({
            "logits": [compute_logits(m, sample, device) for m in models],
            "dists": dists,
            "ij": [(int(pairs[e, 0]), int(pairs[e, 1])) for e in range(n_edges)],
        })
        if (si + 1) % 100 == 0:
            print(f"  precompute {tag}: {si+1}/{len(cached)}", flush=True)
    return entries


def decode_weighted(sample, entry, weights):
    active_mask = sample["active_mask"]
    active_nodes = torch.where(active_mask)[0].tolist()
    n_nodes = active_mask.shape[0]
    ens = np.zeros_like(entry["logits"][0])
    for w, lg in zip(weights, entry["logits"]):
        ens = ens + w * lg
    candidates = [
        EdgeCandidate(i=entry["ij"][e][0], j=entry["ij"][e][1],
                      dist=float(entry["dists"][e]), score=float(ens[e]))
        for e in range(len(ens))
    ]
    return kruskal_mst_from_scores(n_nodes, active_nodes, candidates)


def eval_weights(cached, entries, weights, full=False):
    rows = []
    for sample, entry in zip(cached, entries):
        if entry is None:
            continue
        pred_mask = decode_weighted(sample, entry, weights)
        prf = edge_prf(pred_mask, sample["gt_adj"], sample["active_mask"])
        if not full:
            rows.append(prf["f1"])
        else:
            gs = graph_stats(pred_mask, sample["active_mask"])
            gt_edges = int(sample["gt_adj"].sum()) // 2
            pred_edges = int(pred_mask.sum()) // 2
            rows.append({
                "f1": prf["f1"], "precision": prf["precision"],
                "recall": prf["recall"],
                "sel_gt": pred_edges / max(gt_edges, 1),
                "cycles": gs["cycle_count"],
                "components": gs["component_count"],
                "n_joints": int(sample["active_mask"].sum()),
            })
    if not full:
        return float(np.mean(rows)) if rows else 0.0
    return rows


def simplex_grid(n, step):
    """All weight vectors of length n on the simplex, grid resolution `step`."""
    m = int(round(1.0 / step))
    pts = []
    for combo in itertools.product(range(m + 1), repeat=n - 1):
        if sum(combo) <= m:
            last = m - sum(combo)
            w = [c / m for c in combo] + [last / m]
            pts.append(w)
    return pts


def main():
    parser = argparse.ArgumentParser(description="Track B N-way ensemble eval")
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--ckpts", required=True, help="comma-separated checkpoint paths")
    parser.add_argument("--labels", default=None, help="comma-separated labels")
    parser.add_argument("--edge-dim", type=int, default=128)
    parser.add_argument("--n-mp-rounds", type=int, default=3)
    parser.add_argument("--grid-step", type=float, default=0.1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = [c.strip() for c in args.ckpts.split(",")]
    labels = ([l.strip() for l in args.labels.split(",")] if args.labels
              else [f"m{i}" for i in range(len(ckpts))])
    assert len(labels) == len(ckpts)

    print("Track B: N-way Score Ensemble", flush=True)
    print(f"Models: {labels}", flush=True)
    print("Select weights by VAL only. Test once.", flush=True)
    print(f"Device: {device}", flush=True)

    def load(ckpt):
        m = GeometryEdgeGraphStudent(
            patch_in_channels=6, patch_out_dim=64, geom_feat_dim=16,
            edge_dim=args.edge_dim, n_mp_rounds=args.n_mp_rounds, dropout=0.0,
        ).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        m.load_state_dict(state)
        m.eval()
        return m

    models = [load(c) for c in ckpts]
    for lbl, c in zip(labels, ckpts):
        print(f"  loaded {lbl}: {c}", flush=True)

    val_cached = torch.load(args.val_cache, map_location="cpu", weights_only=False)
    test_cached = torch.load(args.test_cache, map_location="cpu", weights_only=False)
    print(f"Val: {len(val_cached)}  Test: {len(test_cached)}", flush=True)

    print("\nPrecomputing logits...", flush=True)
    val_entries = precompute(models, val_cached, device, "val")
    test_entries = precompute(models, test_cached, device, "test")

    # Pure-model references (one-hot weights).
    n = len(models)
    print(f"\n{'='*50}", flush=True)
    print("Pure-model reference F1 (val / test)", flush=True)
    print(f"{'='*50}", flush=True)
    for k, lbl in enumerate(labels):
        w = [1.0 if j == k else 0.0 for j in range(n)]
        vf = eval_weights(val_cached, val_entries, w)
        tf = eval_weights(test_cached, test_entries, w)
        print(f"  {lbl:<8} val={vf:.4f}  test={tf:.4f}", flush=True)

    # Simplex grid search on VAL.
    grid = simplex_grid(n, args.grid_step)
    print(f"\n{'='*50}", flush=True)
    print(f"Simplex grid search on VAL ({len(grid)} weight vectors, "
          f"step={args.grid_step})", flush=True)
    print(f"{'='*50}", flush=True)

    best_w = None
    best_vf = -1.0
    for w in grid:
        vf = eval_weights(val_cached, val_entries, w)
        if vf > best_vf:
            best_vf = vf
            best_w = w
            print(f"  new best: w={[round(x,2) for x in w]}  val_F1={vf:.4f}",
                  flush=True)

    print(f"\nSelected weights (by val F1): "
          f"{dict(zip(labels, [round(x,3) for x in best_w]))}", flush=True)
    print(f"  val F1 @ best: {best_vf:.4f}", flush=True)

    # TEST once.
    print(f"\n{'='*50}", flush=True)
    print("TEST evaluation @ best weights (once)", flush=True)
    print(f"{'='*50}", flush=True)
    test_rows = eval_weights(test_cached, test_entries, best_w, full=True)

    def agg(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    test_f1 = agg(test_rows, "f1")
    print(f"  test F1:         {test_f1:.4f}", flush=True)
    print(f"  test precision:  {agg(test_rows,'precision'):.4f}", flush=True)
    print(f"  test recall:     {agg(test_rows,'recall'):.4f}", flush=True)
    print(f"  test sel/GT:     {agg(test_rows,'sel_gt'):.4f}", flush=True)
    print(f"  test cycles:     {agg(test_rows,'cycles'):.4f}", flush=True)
    print(f"  test components: {agg(test_rows,'components'):.4f}", flush=True)

    if test_f1 >= 0.85:
        verdict = "RESEARCH PASS"
    elif test_f1 >= 0.82:
        verdict = "STRONG PASS"
    elif test_f1 > 0.77:
        verdict = "MINIMUM PASS"
    else:
        verdict = "BELOW THRESHOLD"
    print(f"\n  Verdict: {verdict} (test F1={test_f1:.4f})", flush=True)

    buckets = {"tiny(<=10)": [], "small(11-25)": [], "medium(26-50)": [],
               "large(>50)": []}
    for r in test_rows:
        n_j = r["n_joints"]
        if n_j <= 10:
            buckets["tiny(<=10)"].append(r)
        elif n_j <= 25:
            buckets["small(11-25)"].append(r)
        elif n_j <= 50:
            buckets["medium(26-50)"].append(r)
        else:
            buckets["large(>50)"].append(r)
    print(f"\nDegree buckets (test):", flush=True)
    bucket_summary = {}
    for bname, brows in buckets.items():
        if brows:
            bf1 = agg(brows, "f1")
            bucket_summary[bname] = {"n": len(brows), "f1": bf1}
            print(f"  {bname:<18} {len(brows):6d} {bf1:8.4f}", flush=True)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "track": "B_nway_ensemble",
            "labels": labels,
            "ckpts": ckpts,
            "selection": "val topology F1 only",
            "best_weights": dict(zip(labels, best_w)),
            "val_f1_best": best_vf,
            "test_f1": test_f1,
            "verdict": verdict,
            "degree_buckets": bucket_summary,
        }
        with open(out_dir / "ensemble_nway_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport -> {out_dir / 'ensemble_nway_report.json'}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
