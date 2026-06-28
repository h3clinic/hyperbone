"""Track B: Evaluate geometry student topology through MST optimizer.

Loads the trained student model and cached test data, scores all candidate
edges, decodes through constrained MST, and reports topology metrics.

Baselines compared:
  - density MST (~0.632)
  - v3.1 neural hybrid MST (~0.756)
  - student-cost MST (geometry only)
  - student + distance hybrid MST

Usage:
    python scripts/eval_geometry_student_topology.py \
        --test-cache outputs/models/hyperbone_track_b_student/cache_test.pt \
        --ckpt outputs/models/hyperbone_track_b_student/best_student.pt \
        --out outputs/models/hyperbone_track_b_student
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch

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


def kruskal_mst_from_scores(
    n_nodes: int,
    active_nodes: List[int],
    edges: List[EdgeCandidate],
    max_edges: int = None,
) -> torch.Tensor:
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
def score_sample_edges(model, sample, device, batch_size=512):
    """Score all candidate edges for one sample."""
    model.eval()
    n_edges = sample["n_edges"]
    if n_edges == 0:
        return torch.zeros(0)

    all_scores = []
    for start in range(0, n_edges, batch_size):
        end = min(start + batch_size, n_edges)
        logit = model(
            sample["patch_i"][start:end].to(device),
            sample["patch_j"][start:end].to(device),
            sample["corridor"][start:end].to(device),
            sample["geom_feats"][start:end].to(device),
        )
        all_scores.append(logit.cpu())

    return torch.cat(all_scores)


def decode_topology(sample, edge_scores, method="student_only", dist_weight=0.5):
    """Decode topology from edge scores using Kruskal MST."""
    active_mask = sample["active_mask"]
    active_nodes = torch.where(active_mask)[0].tolist()
    n_j = len(active_nodes)
    n_nodes = active_mask.shape[0]
    pairs = sample["edge_pairs"]
    joint_pos = sample["joint_pos"]

    candidates = []
    for e_idx in range(sample["n_edges"]):
        i, j = int(pairs[e_idx, 0]), int(pairs[e_idx, 1])
        dist = float(torch.norm(joint_pos[i] - joint_pos[j]))

        if method == "student_only":
            score = float(edge_scores[e_idx])
        elif method == "student_dist_hybrid":
            rel_dist = float(sample["geom_feats"][e_idx, 1])
            score = float(edge_scores[e_idx]) - dist_weight * rel_dist
        elif method == "distance_only":
            score = -dist
        elif method == "density":
            geom = sample["geom_feats"][e_idx]
            mid_density = float(geom[4])
            score = mid_density - dist
        else:
            score = float(edge_scores[e_idx])

        candidates.append(EdgeCandidate(i=i, j=j, dist=dist, score=score))

    pred_mask = kruskal_mst_from_scores(n_nodes, active_nodes, candidates)
    return pred_mask


def eval_sample(sample, pred_mask):
    """Compute topology metrics for one sample."""
    active_mask = sample["active_mask"]
    gt_adj = sample["gt_adj"]

    prf = edge_prf(pred_mask, gt_adj, active_mask)
    stats = graph_stats(pred_mask, active_mask)

    n_j = int(active_mask.sum())
    gt_edges = int((gt_adj[active_mask][:, active_mask] > 0.5).sum()) // 2
    pred_edges = int(pred_mask[active_mask][:, active_mask].sum()) // 2
    sel_gt = pred_edges / max(gt_edges, 1)

    return {
        "f1": prf["f1"],
        "precision": prf["precision"],
        "recall": prf["recall"],
        "tp": prf["tp"],
        "fp": prf["fp"],
        "fn": prf["fn"],
        "n_joints": n_j,
        "gt_edges": gt_edges,
        "pred_edges": pred_edges,
        "sel_gt": sel_gt,
        "components": stats["component_count"],
        "cycles": stats["cycle_count"],
    }


def degree_bucket(n_j):
    if n_j <= 10:
        return "tiny(<=10)"
    elif n_j <= 25:
        return "small(11-25)"
    elif n_j <= 50:
        return "medium(26-50)"
    else:
        return "large(>50)"


def main():
    parser = argparse.ArgumentParser(
        description="Track B: Evaluate student topology through MST")
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--patch-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dist-weight", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Track B: Topology Evaluation", flush=True)
    print("NO skinning in student input.", flush=True)
    print(f"Device: {device}", flush=True)

    # Load model
    model = GeometryEdgeStudent(
        patch_in_channels=6, patch_out_dim=args.patch_dim,
        geom_feat_dim=16, hidden_dim=args.hidden_dim, dropout=0.0,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model: {args.ckpt}", flush=True)

    # Load test data
    test_cached = torch.load(args.test_cache, map_location="cpu", weights_only=False)
    print(f"Test samples: {len(test_cached)}", flush=True)

    methods = ["distance_only", "density", "student_only", "student_dist_hybrid"]
    all_results = {m: [] for m in methods}

    for si, sample in enumerate(test_cached):
        edge_scores = score_sample_edges(model, sample, device)

        for method in methods:
            pred_mask = decode_topology(
                sample, edge_scores, method=method, dist_weight=args.dist_weight)
            metrics = eval_sample(sample, pred_mask)
            metrics["method"] = method
            all_results[method].append(metrics)

        if (si + 1) % 50 == 0:
            stud_f1 = np.mean([r["f1"] for r in all_results["student_only"]])
            print(f"  {si+1}/{len(test_cached)}: student F1={stud_f1:.4f}", flush=True)

    # Aggregate
    print(f"\n{'='*70}", flush=True)
    print("Track B Topology Results (Test Set)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\n{'Method':<25} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Sel/GT':>8} "
          f"{'Cyc':>6} {'Comp':>6}", flush=True)
    print("-" * 70, flush=True)

    summary = {}
    for method in methods:
        results = all_results[method]
        agg = {
            "mean_f1": float(np.mean([r["f1"] for r in results])),
            "mean_precision": float(np.mean([r["precision"] for r in results])),
            "mean_recall": float(np.mean([r["recall"] for r in results])),
            "mean_sel_gt": float(np.mean([r["sel_gt"] for r in results])),
            "mean_cycles": float(np.mean([r["cycles"] for r in results])),
            "mean_components": float(np.mean([r["components"] for r in results])),
            "n_samples": len(results),
        }
        summary[method] = agg

        print(f"  {method:<23} {agg['mean_f1']:8.4f} {agg['mean_precision']:8.4f} "
              f"{agg['mean_recall']:8.4f} {agg['mean_sel_gt']:8.4f} "
              f"{agg['mean_cycles']:6.2f} {agg['mean_components']:6.2f}", flush=True)

    # Best student method
    best_method = max(["student_only", "student_dist_hybrid"],
                      key=lambda m: summary[m]["mean_f1"])
    best_f1 = summary[best_method]["mean_f1"]
    dist_f1 = summary["distance_only"]["mean_f1"]
    density_f1 = summary["density"]["mean_f1"]

    print(f"\n{'='*70}", flush=True)
    print(f"  Best student method: {best_method}", flush=True)
    print(f"  Student F1:  {best_f1:.4f}", flush=True)
    print(f"  Distance F1: {dist_f1:.4f}  (delta {best_f1 - dist_f1:+.4f})", flush=True)
    print(f"  Density F1:  {density_f1:.4f}  (delta {best_f1 - density_f1:+.4f})", flush=True)

    # Pass criteria
    if best_f1 >= 0.85:
        verdict = "RESEARCH PASS (F1 >= 0.85)"
    elif best_f1 >= 0.82:
        verdict = "STRONG (F1 >= 0.82)"
    elif best_f1 > 0.77:
        verdict = "MINIMUM PASS (F1 > 0.77)"
    else:
        verdict = "BELOW THRESHOLD (F1 <= 0.77)"
    print(f"\n  Verdict: {verdict}", flush=True)

    # Degree-bucketed results
    print(f"\n{'='*70}", flush=True)
    print(f"Degree-Bucketed F1 ({best_method})", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Bucket':<20} {'N':>5} {'F1':>8} {'Prec':>8} {'Rec':>8}", flush=True)
    print("-" * 50, flush=True)

    buckets = {}
    for r in all_results[best_method]:
        b = degree_bucket(r["n_joints"])
        buckets.setdefault(b, []).append(r)

    for b in ["tiny(<=10)", "small(11-25)", "medium(26-50)", "large(>50)"]:
        if b not in buckets:
            continue
        rs = buckets[b]
        bf1 = np.mean([r["f1"] for r in rs])
        bp = np.mean([r["precision"] for r in rs])
        br = np.mean([r["recall"] for r in rs])
        print(f"  {b:<18} {len(rs):5d} {bf1:8.4f} {bp:8.4f} {br:8.4f}", flush=True)

    # Save report
    report = {
        "track": "B_geometry_student_topology_eval",
        "skinning_used_as_input": False,
        "n_test_samples": len(test_cached),
        "dist_weight": args.dist_weight,
        "methods": summary,
        "best_method": best_method,
        "best_f1": best_f1,
        "verdict": verdict,
        "reference": {
            "density_mst": "~0.632",
            "v31_neural": "~0.756",
            "v41_skinned": "0.888 (teacher only)",
        },
    }
    with open(out_dir / "topology_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport -> {out_dir / 'topology_eval_report.json'}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
