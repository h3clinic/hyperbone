"""Track B v2.2: Score ensemble of v2 + v2.1 edge-graph students.

Ensembles per-edge logits:
    ensemble_logits = alpha * logits_v2 + (1 - alpha) * logits_v21

Sweeps alpha on VALIDATION topology F1 only, then evaluates the
selected alpha on TEST exactly once. Decode is student_only MST
(distance hybrid hurt v2/v2.1).

Logits are cached per sample (they don't depend on alpha), so the
alpha sweep is a cheap MST re-decode over cached scores.

Usage:
    python scripts/eval_edge_graph_ensemble.py \
        --val-cache   outputs/models/hyperbone_track_b_student/cache_val_v11.pt \
        --test-cache  outputs/models/hyperbone_track_b_student/cache_test_v11.pt \
        --ckpt-v2     outputs/models/hyperbone_track_b_student_v2/best_student_v2.pt \
        --ckpt-v21    outputs/models/hyperbone_track_b_student_v21/best_student_v21.pt \
        --out         outputs/models/hyperbone_track_b_student_v22
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
def compute_logits(model, sample, device):
    """Per-edge logits for one sample. Returns np.ndarray [n_edges]."""
    n_edges = sample["n_edges"]
    if n_edges == 0:
        return np.zeros(0, dtype=np.float32)
    pi = sample["patch_i"].to(device)
    pj = sample["patch_j"].to(device)
    co = sample["corridor"].to(device)
    gf = sample["geom_feats"].to(device)
    lg = build_line_graph(sample["edge_pairs"]).to(device)
    logits = model(pi, pj, co, gf, lg)
    return logits.cpu().numpy().astype(np.float32)


def precompute(model_v2, model_v21, cached, device, tag):
    """Cache per-sample logits from both models. Returns list of dicts."""
    entries = []
    for si, sample in enumerate(cached):
        n_edges = sample["n_edges"]
        if n_edges == 0:
            entries.append(None)
            continue
        lv2 = compute_logits(model_v2, sample, device)
        lv21 = compute_logits(model_v21, sample, device)
        # Precompute per-edge geometry the MST decode needs.
        pairs = sample["edge_pairs"]
        joint_pos = sample["joint_pos"]
        dists = np.array([
            float(torch.norm(joint_pos[int(pairs[e, 0])] - joint_pos[int(pairs[e, 1])]))
            for e in range(n_edges)
        ], dtype=np.float32)
        entries.append({
            "logits_v2": lv2,
            "logits_v21": lv21,
            "dists": dists,
            "ij": [(int(pairs[e, 0]), int(pairs[e, 1])) for e in range(n_edges)],
        })
        if (si + 1) % 100 == 0:
            print(f"  precompute {tag}: {si+1}/{len(cached)}", flush=True)
    return entries


def decode_ensemble(sample, entry, alpha):
    """student_only MST decode of ensemble logits for one sample."""
    active_mask = sample["active_mask"]
    active_nodes = torch.where(active_mask)[0].tolist()
    n_nodes = active_mask.shape[0]
    ens = alpha * entry["logits_v2"] + (1.0 - alpha) * entry["logits_v21"]
    candidates = [
        EdgeCandidate(i=entry["ij"][e][0], j=entry["ij"][e][1],
                      dist=float(entry["dists"][e]), score=float(ens[e]))
        for e in range(len(ens))
    ]
    return kruskal_mst_from_scores(n_nodes, active_nodes, candidates)


def eval_alpha(cached, entries, alpha):
    """Mean topology F1 over a split for a given alpha."""
    f1s = []
    for sample, entry in zip(cached, entries):
        if entry is None:
            continue
        pred_mask = decode_ensemble(sample, entry, alpha)
        prf = edge_prf(pred_mask, sample["gt_adj"], sample["active_mask"])
        f1s.append(prf["f1"])
    return float(np.mean(f1s)) if f1s else 0.0


def eval_alpha_full(cached, entries, alpha):
    """Full metrics for one alpha (used on test after selection)."""
    rows = []
    for sample, entry in zip(cached, entries):
        if entry is None:
            continue
        pred_mask = decode_ensemble(sample, entry, alpha)
        prf = edge_prf(pred_mask, sample["gt_adj"], sample["active_mask"])
        gs = graph_stats(pred_mask, sample["active_mask"])
        gt_edges = int(sample["gt_adj"].sum()) // 2
        pred_edges = int(pred_mask.sum()) // 2
        rows.append({
            "f1": prf["f1"],
            "precision": prf["precision"],
            "recall": prf["recall"],
            "sel_gt": pred_edges / max(gt_edges, 1),
            "cycles": gs["cycle_count"],
            "components": gs["component_count"],
            "n_joints": int(sample["active_mask"].sum()),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Track B v2.2 ensemble eval")
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--ckpt-v2", required=True)
    parser.add_argument("--ckpt-v21", required=True)
    parser.add_argument("--edge-dim", type=int, default=128)
    parser.add_argument("--n-mp-rounds", type=int, default=3)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Track B v2.2: Score Ensemble (v2 + v2.1)", flush=True)
    print("NO skinning in student input. Select alpha by VAL only.", flush=True)
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

    model_v2 = load(args.ckpt_v2)
    model_v21 = load(args.ckpt_v21)
    print(f"Loaded v2:  {args.ckpt_v2}", flush=True)
    print(f"Loaded v21: {args.ckpt_v21}", flush=True)

    val_cached = torch.load(args.val_cache, map_location="cpu", weights_only=False)
    test_cached = torch.load(args.test_cache, map_location="cpu", weights_only=False)
    print(f"Val: {len(val_cached)}  Test: {len(test_cached)}", flush=True)

    print("\nPrecomputing logits...", flush=True)
    val_entries = precompute(model_v2, model_v21, val_cached, device, "val")
    test_entries = precompute(model_v2, model_v21, test_cached, device, "test")

    # Alpha sweep on VALIDATION only.
    alphas = [round(a, 4) for a in np.arange(0.0, 1.0 + 1e-9, args.alpha_step)]
    print(f"\n{'='*50}", flush=True)
    print(f"Alpha sweep on VALIDATION ({len(alphas)} values)", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"{'alpha':>8} {'val_F1':>10}", flush=True)
    print("-" * 20, flush=True)

    sweep = []
    best_alpha = 0.0
    best_val_f1 = -1.0
    for a in alphas:
        f1 = eval_alpha(val_cached, val_entries, a)
        sweep.append({"alpha": a, "val_f1": f1})
        marker = ""
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_alpha = a
            marker = " *"
        print(f"{a:8.2f} {f1:10.4f}{marker}", flush=True)

    # Endpoint references: alpha=1.0 is pure v2, alpha=0.0 is pure v21.
    val_v2_only = next(s["val_f1"] for s in sweep if s["alpha"] == 1.0)
    val_v21_only = next(s["val_f1"] for s in sweep if s["alpha"] == 0.0)

    print(f"\n{'='*50}", flush=True)
    print(f"Selected alpha (by val F1): {best_alpha:.2f}", flush=True)
    print(f"  val F1 @ best alpha: {best_val_f1:.4f}", flush=True)
    print(f"  val F1 pure v2  (alpha=1.0): {val_v2_only:.4f}", flush=True)
    print(f"  val F1 pure v21 (alpha=0.0): {val_v21_only:.4f}", flush=True)

    # Evaluate on TEST once with the selected alpha.
    print(f"\n{'='*50}", flush=True)
    print(f"TEST evaluation @ alpha={best_alpha:.2f} (once)", flush=True)
    print(f"{'='*50}", flush=True)
    test_rows = eval_alpha_full(test_cached, test_entries, best_alpha)

    def agg(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    test_f1 = agg(test_rows, "f1")
    test_prec = agg(test_rows, "precision")
    test_rec = agg(test_rows, "recall")
    test_selgt = agg(test_rows, "sel_gt")
    test_cyc = agg(test_rows, "cycles")
    test_comp = agg(test_rows, "components")

    print(f"  test F1:         {test_f1:.4f}", flush=True)
    print(f"  test precision:  {test_prec:.4f}", flush=True)
    print(f"  test recall:     {test_rec:.4f}", flush=True)
    print(f"  test sel/GT:     {test_selgt:.4f}", flush=True)
    print(f"  test cycles:     {test_cyc:.4f}", flush=True)
    print(f"  test components: {test_comp:.4f}", flush=True)

    # Reference: pure-model test F1 for context (NOT used for selection).
    test_v2_rows = eval_alpha_full(test_cached, test_entries, 1.0)
    test_v21_rows = eval_alpha_full(test_cached, test_entries, 0.0)
    test_v2_f1 = agg(test_v2_rows, "f1")
    test_v21_f1 = agg(test_v21_rows, "f1")
    print(f"\n  (ref) test F1 pure v2:  {test_v2_f1:.4f}", flush=True)
    print(f"  (ref) test F1 pure v21: {test_v21_f1:.4f}", flush=True)
    print(f"  ensemble delta vs best single: "
          f"{test_f1 - max(test_v2_f1, test_v21_f1):+.4f}", flush=True)

    if test_f1 >= 0.85:
        verdict = "RESEARCH PASS"
    elif test_f1 >= 0.82:
        verdict = "STRONG PASS"
    elif test_f1 > 0.77:
        verdict = "MINIMUM PASS"
    else:
        verdict = "BELOW THRESHOLD"
    print(f"\n  Verdict: {verdict} (test F1={test_f1:.4f})", flush=True)

    # Degree-bucketed F1 on test.
    print(f"\n{'='*50}", flush=True)
    print("Degree-Bucketed F1 (test, best alpha)", flush=True)
    print(f"{'='*50}", flush=True)
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
    print(f"{'Bucket':<20} {'N':>6} {'F1':>8}", flush=True)
    print("-" * 36, flush=True)
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
            "track": "B_v2.2_ensemble",
            "method": "student_only MST on alpha*v2 + (1-alpha)*v21 logits",
            "selection": "val topology F1 only",
            "best_alpha": best_alpha,
            "val_f1_best_alpha": best_val_f1,
            "val_f1_pure_v2": val_v2_only,
            "val_f1_pure_v21": val_v21_only,
            "test_f1": test_f1,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_sel_gt": test_selgt,
            "test_cycles": test_cyc,
            "test_components": test_comp,
            "test_f1_pure_v2": test_v2_f1,
            "test_f1_pure_v21": test_v21_f1,
            "verdict": verdict,
            "degree_buckets": bucket_summary,
            "alpha_sweep": sweep,
        }
        with open(out_dir / "ensemble_report_v22.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport -> {out_dir / 'ensemble_report_v22.json'}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
