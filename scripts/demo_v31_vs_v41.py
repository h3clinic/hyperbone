"""
Demo: v3.1 vs v4.1 side-by-side comparison on 5 test samples.

Produces per-sample overlays showing GT, v3.1 prediction, and v4.1 prediction,
plus a summary table. Selects 5 samples spanning the F1 range:
  1 best, 1 above-median, 1 median, 1 below-median, 1 challenging.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.rigs.topology_optimizers import (
    hybrid_neural_cost_optimize,
    hybrid_skinning_cost_optimize,
)
from hyperbone.rigs.undirected_topology import edge_prf, graph_stats

CACHE_PATH = Path("outputs/models/hyperbone_v4_1_skinning_topology/cache_test_v41.pt")
OUT_DIR = Path("outputs/demo_v31_vs_v41")

V31_CONFIG = {
    "mode": "density_normalized_mst",
    "k": 12,
    "neural_weight": 2.0,
    "degree_penalty": 0.05,
    "long_edge_penalty": 0.25,
    "mutual_bonus": 0.2,
    "max_degree": 4,
}

V41_CONFIG = {
    "mode": "density_normalized_mst",
    "k": 16,
    "neural_weight": 1.0,
    "skin_cosine_weight": 4.0,
    "shared_weight": 4.0,
    "distance_weight": 1.0,
    "degree_penalty": 0.05,
    "long_edge_penalty": 0.25,
    "mutual_bonus": 0.2,
    "max_degree": 4,
}


def edge_count(adj, active):
    tri = torch.triu(torch.ones_like(adj, dtype=torch.bool), diagonal=1)
    return int((adj & tri & active.unsqueeze(0) & active.unsqueeze(1)).sum().item())


def run_v31(s):
    pred = hybrid_neural_cost_optimize(
        joint_pos=s["joint_pos"], active_mask=s["active_mask"],
        neural_scores=s["neural_scores"],
        mode=V31_CONFIG["mode"], k=V31_CONFIG["k"],
        neural_weight=V31_CONFIG["neural_weight"],
        degree_penalty=V31_CONFIG["degree_penalty"],
        long_edge_penalty=V31_CONFIG["long_edge_penalty"],
        mutual_bonus=V31_CONFIG["mutual_bonus"],
        max_degree=V31_CONFIG["max_degree"],
        candidate_mask=s["candidate_mask"],
    )
    return pred


def run_v41(s):
    pred = hybrid_skinning_cost_optimize(
        joint_pos=s["joint_pos"], active_mask=s["active_mask"],
        neural_scores=s["neural_scores"],
        skinning_cosine=s.get("skinning_cosine"),
        max_shared_weight=s.get("max_shared_weight"),
        mode=V41_CONFIG["mode"], k=V41_CONFIG["k"],
        distance_weight=V41_CONFIG["distance_weight"],
        neural_weight=V41_CONFIG["neural_weight"],
        skin_cosine_weight=V41_CONFIG["skin_cosine_weight"],
        shared_weight=V41_CONFIG["shared_weight"],
        degree_penalty=V41_CONFIG["degree_penalty"],
        long_edge_penalty=V41_CONFIG["long_edge_penalty"],
        mutual_bonus=V41_CONFIG["mutual_bonus"],
        max_degree=V41_CONFIG["max_degree"],
        candidate_mask=s["candidate_mask"],
    )
    return pred


def select_demo_samples(cached):
    """Pick 5 samples: best, above-median, median, below-median, challenging."""
    f1s = []
    for i, s in enumerate(cached):
        pred41 = run_v41(s)
        prf = edge_prf(pred41, s["gt_adj"], s["active_mask"])
        f1s.append((i, prf["f1"]))

    sorted_f1 = sorted(f1s, key=lambda x: x[1])
    n = len(sorted_f1)

    picks = [
        sorted_f1[-1],                  # best
        sorted_f1[int(n * 0.75)],       # above median
        sorted_f1[n // 2],              # median
        sorted_f1[int(n * 0.25)],       # below median
        sorted_f1[max(4, int(n * 0.05))],  # challenging (5th percentile, not worst)
    ]
    return picks


def draw_overlay(ax, pos_2d, active_idx, adj, color, title, linestyle="-"):
    for i in active_idx:
        for j in active_idx:
            if j > i and adj[i, j] > 0.5:
                ax.plot([pos_2d[i, 0], pos_2d[j, 0]], [pos_2d[i, 1], pos_2d[j, 1]],
                        color=color, alpha=0.6, linewidth=1.5, linestyle=linestyle)
    ax.scatter(pos_2d[active_idx, 0], pos_2d[active_idx, 1],
               c=color, s=25, zorder=5, edgecolors="black", linewidths=0.3)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])


def draw_diff(ax, pos_2d, active_idx, pred_adj, gt_adj, title):
    for i in active_idx:
        for j in active_idx:
            if j <= i:
                continue
            gt_e = gt_adj[i, j] > 0.5
            pred_e = pred_adj[i, j] > 0.5
            if gt_e and pred_e:
                ax.plot([pos_2d[i, 0], pos_2d[j, 0]], [pos_2d[i, 1], pos_2d[j, 1]],
                        color="blue", alpha=0.5, linewidth=1.5)
            elif gt_e and not pred_e:
                ax.plot([pos_2d[i, 0], pos_2d[j, 0]], [pos_2d[i, 1], pos_2d[j, 1]],
                        color="green", alpha=0.7, linewidth=2.5, linestyle="--")
            elif pred_e and not gt_e:
                ax.plot([pos_2d[i, 0], pos_2d[j, 0]], [pos_2d[i, 1], pos_2d[j, 1]],
                        color="red", alpha=0.7, linewidth=2.5, linestyle=":")
    ax.scatter(pos_2d[active_idx, 0], pos_2d[active_idx, 1],
               c="black", s=25, zorder=5, edgecolors="white", linewidths=0.3)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Loading cache...", flush=True)
    cached = torch.load(str(CACHE_PATH), map_location="cpu", weights_only=False)
    print(f"Loaded {len(cached)} samples.", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Selecting 5 demo samples...", flush=True)
    picks = select_demo_samples(cached)

    results = []
    for rank, (sample_idx, v41_f1) in enumerate(picks):
        s = cached[sample_idx]
        am = s["active_mask"]
        gt = s["gt_adj"]
        jp = s["joint_pos"].numpy()

        pred31 = run_v31(s)
        pred41 = run_v41(s)

        prf31 = edge_prf(pred31, gt, am)
        prf41 = edge_prf(pred41, gt, am)
        stats31 = graph_stats(pred31, am)
        stats41 = graph_stats(pred41, am)

        n_joints = int(am.sum().item())
        gt_edges = edge_count(gt, am)

        row = {
            "rank": rank + 1,
            "sample_idx": sample_idx,
            "n_joints": n_joints,
            "gt_edges": gt_edges,
            "v31_f1": round(prf31["f1"], 4),
            "v41_f1": round(prf41["f1"], 4),
            "delta_f1": round(prf41["f1"] - prf31["f1"], 4),
            "v31_cycles": stats31["cycle_count"],
            "v41_cycles": stats41["cycle_count"],
        }
        results.append(row)
        print(f"  Sample {sample_idx}: v3.1 F1={prf31['f1']:.3f} -> v4.1 F1={prf41['f1']:.3f} "
              f"(delta={prf41['f1'] - prf31['f1']:+.3f})", flush=True)

        # Generate 4-panel figure: GT | v3.1 diff | v4.1 diff | v3.1 vs v4.1 edges
        pos_2d = jp[:, :2]
        active_idx = np.where(am.numpy() > 0.5)[0]

        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

        draw_overlay(axes[0], pos_2d, active_idx, gt.numpy(), "green",
                     f"GT ({gt_edges} edges, {n_joints} joints)")

        draw_diff(axes[1], pos_2d, active_idx, pred31.numpy(), gt.numpy(),
                  f"v3.1 F1={prf31['f1']:.3f}  TP=blue FN=green-- FP=red:")

        draw_diff(axes[2], pos_2d, active_idx, pred41.numpy(), gt.numpy(),
                  f"v4.1 F1={prf41['f1']:.3f}  TP=blue FN=green-- FP=red:")

        # Panel 4: direct comparison — v3.1 edges in orange, v4.1 edges in purple, both in gray
        ax = axes[3]
        for i in active_idx:
            for j in active_idx:
                if j <= i:
                    continue
                e31 = pred31[i, j].item() > 0.5
                e41 = pred41[i, j].item() > 0.5
                if e31 and e41:
                    ax.plot([pos_2d[i, 0], pos_2d[j, 0]], [pos_2d[i, 1], pos_2d[j, 1]],
                            color="gray", alpha=0.4, linewidth=1.5)
                elif e31 and not e41:
                    ax.plot([pos_2d[i, 0], pos_2d[j, 0]], [pos_2d[i, 1], pos_2d[j, 1]],
                            color="orange", alpha=0.7, linewidth=2.5, linestyle="--")
                elif e41 and not e31:
                    ax.plot([pos_2d[i, 0], pos_2d[j, 0]], [pos_2d[i, 1], pos_2d[j, 1]],
                            color="purple", alpha=0.7, linewidth=2.5, linestyle=":")
        ax.scatter(pos_2d[active_idx, 0], pos_2d[active_idx, 1],
                   c="black", s=25, zorder=5, edgecolors="white", linewidths=0.3)
        ax.set_title("v3.1 vs v4.1: shared=gray  v3.1-only=orange--  v4.1-only=purple:", fontsize=9)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])

        labels = ["best", "above_median", "median", "below_median", "challenging"]
        fig.suptitle(
            f"Demo sample #{rank+1} ({labels[rank]}): idx={sample_idx}, "
            f"{n_joints} joints — v3.1 F1={prf31['f1']:.3f} -> v4.1 F1={prf41['f1']:.3f}",
            fontsize=11)
        fig.tight_layout()
        out_path = OUT_DIR / f"demo_{rank+1}_{labels[rank]}_sample{sample_idx}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"    Saved {out_path}", flush=True)

    # Save summary
    summary = {
        "version_comparison": "v3.1 vs v4.1",
        "track": "rigged_skinned_asset_topology",
        "v31_config": V31_CONFIG,
        "v41_config": V41_CONFIG,
        "samples": results,
    }
    summary_path = OUT_DIR / "demo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print table
    print("\n" + "=" * 70)
    print("v3.1 vs v4.1 Demo Summary")
    print("=" * 70)
    print(f"{'#':<4} {'Sample':<8} {'Joints':<8} {'v3.1 F1':<10} {'v4.1 F1':<10} {'Delta':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['rank']:<4} {r['sample_idx']:<8} {r['n_joints']:<8} "
              f"{r['v31_f1']:<10.4f} {r['v41_f1']:<10.4f} {r['delta_f1']:+.4f}")
    print("-" * 50)
    mean_31 = np.mean([r["v31_f1"] for r in results])
    mean_41 = np.mean([r["v41_f1"] for r in results])
    print(f"{'Mean':<12} {'':8} {mean_31:<10.4f} {mean_41:<10.4f} {mean_41 - mean_31:+.4f}")

    print(f"\nSaved: {summary_path}")
    print(f"Overlays: {OUT_DIR}")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
