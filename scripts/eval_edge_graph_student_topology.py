"""Track B v2: Topology evaluation for edge-graph message passing student.

Usage:
    python scripts/eval_edge_graph_student_topology.py \
        --test-cache outputs/models/hyperbone_track_b_student/cache_test.pt \
        --ckpt outputs/models/hyperbone_track_b_student_v2/best_student_v2.pt \
        --dist-weight 10.0
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


def decode_topology(sample, edge_scores, method, dist_weight):
    active_mask = sample["active_mask"]
    active_nodes = torch.where(active_mask)[0].tolist()
    n_nodes = active_mask.shape[0]
    pairs = sample["edge_pairs"]
    joint_pos = sample["joint_pos"]
    n_edges = sample["n_edges"]

    candidates = []
    for e_idx in range(n_edges):
        i, j = int(pairs[e_idx, 0]), int(pairs[e_idx, 1])
        dist = float(torch.norm(joint_pos[i] - joint_pos[j]))

        if method == "distance_only":
            score = -dist
        elif method == "density":
            density = float(sample["geom_feats"][e_idx, 2])
            score = density - dist
        elif method == "student_only":
            score = float(edge_scores[e_idx])
        elif method == "student_dist_hybrid":
            rel_dist = float(sample["geom_feats"][e_idx, 1])
            score = float(edge_scores[e_idx]) - dist_weight * rel_dist
        else:
            score = -dist

        candidates.append(EdgeCandidate(i=i, j=j, dist=dist, score=score))

    return kruskal_mst_from_scores(n_nodes, active_nodes, candidates)


def main():
    parser = argparse.ArgumentParser(
        description="Track B v2: Edge-graph student topology eval")
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dist-weight", type=float, default=10.0)
    parser.add_argument("--edge-dim", type=int, default=128)
    parser.add_argument("--n-mp-rounds", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Track B v2: Topology Evaluation", flush=True)
    print("NO skinning in student input.", flush=True)
    print(f"Device: {device}", flush=True)

    model = GeometryEdgeGraphStudent(
        patch_in_channels=6, patch_out_dim=64, geom_feat_dim=16,
        edge_dim=args.edge_dim, n_mp_rounds=args.n_mp_rounds, dropout=0.0,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model: {args.ckpt}", flush=True)

    test_cached = torch.load(args.test_cache, map_location="cpu", weights_only=False)
    print(f"Test samples: {len(test_cached)}", flush=True)

    methods = ["distance_only", "density", "student_only", "student_dist_hybrid"]
    all_results = {m: [] for m in methods}

    for si, sample in enumerate(test_cached):
        n_edges = sample["n_edges"]
        if n_edges == 0:
            continue

        with torch.no_grad():
            pi = sample["patch_i"].to(device)
            pj = sample["patch_j"].to(device)
            co = sample["corridor"].to(device)
            gf = sample["geom_feats"].to(device)
            lg = build_line_graph(sample["edge_pairs"]).to(device)
            logits = model(pi, pj, co, gf, lg)
            edge_scores = logits.cpu()

        for method in methods:
            pred_mask = decode_topology(sample, edge_scores, method, args.dist_weight)
            prf = edge_prf(pred_mask, sample["gt_adj"], sample["active_mask"])
            gs = graph_stats(pred_mask, sample["active_mask"])
            gt_edges = int(sample["gt_adj"].sum()) // 2
            pred_edges = int(pred_mask.sum()) // 2
            all_results[method].append({
                "f1": prf["f1"],
                "precision": prf["precision"],
                "recall": prf["recall"],
                "sel_gt": pred_edges / max(gt_edges, 1),
                "cycles": gs["cycle_count"],
                "components": gs["component_count"],
            })

        if (si + 1) % 50 == 0:
            stud_f1 = np.mean([r["f1"] for r in all_results["student_only"]])
            print(f"  {si+1}/{len(test_cached)}: student_only F1={stud_f1:.4f}",
                  flush=True)

    print(f"\n{'='*70}", flush=True)
    print("Track B v2 Topology Results (Test Set)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\n{'Method':<25} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Sel/GT':>8} "
          f"{'Cyc':>6} {'Comp':>6}", flush=True)
    print("-" * 70, flush=True)

    best_method = None
    best_f1 = 0.0
    summary = {}
    for method in methods:
        results = all_results[method]
        f1 = np.mean([r["f1"] for r in results])
        prec = np.mean([r["precision"] for r in results])
        rec = np.mean([r["recall"] for r in results])
        sel_gt = np.mean([r["sel_gt"] for r in results])
        cyc = np.mean([r["cycles"] for r in results])
        comp = np.mean([r["components"] for r in results])
        print(f"  {method:<23} {f1:8.4f} {prec:8.4f} {rec:8.4f} {sel_gt:8.4f} "
              f"{cyc:6.2f} {comp:6.2f}", flush=True)
        summary[method] = {"f1": f1, "precision": prec, "recall": rec}
        if f1 > best_f1:
            best_f1 = f1
            best_method = method

    print(f"\n{'='*70}", flush=True)
    print(f"  Best method: {best_method}", flush=True)
    print(f"  Best F1: {best_f1:.4f}", flush=True)
    dist_f1 = np.mean([r["f1"] for r in all_results["distance_only"]])
    print(f"  Distance baseline F1: {dist_f1:.4f} (delta +{best_f1-dist_f1:.4f})",
          flush=True)

    if best_f1 > 0.85:
        verdict = "RESEARCH PASS"
    elif best_f1 >= 0.82:
        verdict = "STRONG PASS"
    elif best_f1 > 0.77:
        verdict = "MINIMUM PASS"
    else:
        verdict = "BELOW THRESHOLD"
    print(f"\n  Verdict: {verdict} (F1={best_f1:.4f})", flush=True)

    # Degree-bucketed F1
    print(f"\n{'='*70}", flush=True)
    print(f"Degree-Bucketed F1 ({best_method})", flush=True)
    print(f"{'='*70}", flush=True)

    buckets = {"tiny(<=10)": [], "small(11-25)": [], "medium(26-50)": [],
               "large(>50)": []}
    for si, sample in enumerate(test_cached):
        n_j = int(sample["active_mask"].sum())
        r = all_results[best_method][si]
        if n_j <= 10:
            buckets["tiny(<=10)"].append(r)
        elif n_j <= 25:
            buckets["small(11-25)"].append(r)
        elif n_j <= 50:
            buckets["medium(26-50)"].append(r)
        else:
            buckets["large(>50)"].append(r)

    print(f"{'Bucket':<20} {'N':>6} {'F1':>8} {'Prec':>8} {'Rec':>8}", flush=True)
    print("-" * 50, flush=True)
    for bname, bresults in buckets.items():
        if bresults:
            f1 = np.mean([r["f1"] for r in bresults])
            prec = np.mean([r["precision"] for r in bresults])
            rec = np.mean([r["recall"] for r in bresults])
            print(f"  {bname:<18} {len(bresults):6d} {f1:8.4f} {prec:8.4f} {rec:8.4f}",
                  flush=True)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "track": "B_v2_edge_graph_student",
            "verdict": verdict,
            "best_method": best_method,
            "best_f1": best_f1,
            "summary": summary,
        }
        with open(out_dir / "eval_report_v2.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport -> {out_dir / 'eval_report_v2.json'}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
