"""
Visualize rig graph predictions v2 — failure mode analysis.

Adds over v1:
- GT point cloud background
- Centroid markers
- Spread ratio text
- Worst/best samples by MPJPE
- Worst samples by bone length error
- Per-sample bbox visualization

Usage:
    python scripts/vis_anymate_rig_graph_v2.py
    python scripts/vis_anymate_rig_graph_v2.py --checkpoint outputs/models/hyperbone_anymate_static_v2/best_model.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.datasets.anymate_static_dataset import AnymateStaticRigDataset
from hyperbone.models.hyperbone_rig_graph_static import HyperBoneStaticRigModel
from hyperbone.rigs.structured_decoder import StructuredDecodeConfig, decode_structured_graph

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def compute_spread(pts):
    if pts.shape[0] < 2:
        return 0.0
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return np.linalg.norm(mx - mn)


def draw_skeleton(ax, pos_2d, active_mask, adj, color, label, alpha=0.8, linewidth=1.5, markersize=25):
    """Draw skeleton graph on axis."""
    active_idx = np.where(active_mask > 0.5)[0]
    # Edges
    for i in active_idx:
        for j in active_idx:
            if j > i and adj[i, j] > 0.5:
                ax.plot([pos_2d[i, 0], pos_2d[j, 0]],
                        [pos_2d[i, 1], pos_2d[j, 1]],
                        color=color, alpha=alpha * 0.6, linewidth=linewidth)
    # Nodes
    if len(active_idx) > 0:
        ax.scatter(pos_2d[active_idx, 0], pos_2d[active_idx, 1],
                   c=color, s=markersize, zorder=5, label=label, alpha=alpha,
                   edgecolors="black", linewidths=0.3)


@torch.no_grad()
def collect_per_sample_data(model, dataset, device, max_samples=None, decode_config=None):
    """Run inference and collect per-sample metrics for sorting."""
    model.eval()
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    results = []
    for idx in range(n):
        sample = dataset[idx]
        batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in sample.items()}
        pred = model(batch)

        gt_pos = sample["joint_pos"].numpy()
        gt_active = sample["joint_active"].numpy()
        gt_adj = sample["adj_matrix"].numpy()
        pc = sample["pc"].numpy()[:, :3]

        pred_pos = pred["joint_pos"][0].cpu().numpy()
        pred_active_prob = torch.sigmoid(pred["active_logits"][0]).cpu().numpy()
        pred_adj_prob = torch.sigmoid(pred["adj_logits"][0]).cpu().numpy()

        threshold_decoded = decode_structured_graph(
            pred_pos, pred_active_prob, pred_adj_prob, point_cloud=pc,
            node_confidence=pred_active_prob,
            config=StructuredDecodeConfig(active_threshold=decode_config.active_threshold if decode_config else 0.5,
                                          edge_decode_mode="threshold"),
        )
        structured_decoded = decode_structured_graph(
            pred_pos, pred_active_prob, pred_adj_prob, point_cloud=pc,
            node_confidence=pred_active_prob,
            config=decode_config or StructuredDecodeConfig(),
        )

        gt_mask = gt_active > 0.5
        pred_mask = structured_decoded["active_mask"]

        gt_pts = gt_pos[gt_mask]
        pred_pts = pred_pos[pred_mask]

        # Matched MPJPE
        if pred_pts.shape[0] > 0 and gt_pts.shape[0] > 0:
            cost = np.linalg.norm(pred_pts[:, None] - gt_pts[None, :], axis=-1)
            row_ind, col_ind = linear_sum_assignment(cost)
            matched_dists = np.linalg.norm(pred_pts[row_ind] - gt_pts[col_ind], axis=-1)
            mpjpe = matched_dists.mean()
        else:
            mpjpe = float("inf")

        # Spread
        gt_spread = compute_spread(gt_pts)
        pred_spread = compute_spread(pred_pts) if pred_pts.shape[0] >= 2 else 0.0
        spread_score = pred_spread / max(gt_spread, 1e-6)

        # Bone length + edge quality (after matching)
        bone_ratios = []
        edge_f1 = 0.0
        if pred_pts.shape[0] > 0 and gt_pts.shape[0] > 0 and len(row_ind) >= 2:
            gt_active_idx = np.where(gt_mask)[0]

            pred_active_idx = np.where(pred_mask)[0]
            matched_pred_global = pred_active_idx[row_ind]
            matched_gt_global = gt_active_idx[col_ind]

            pred_sub = structured_decoded["adjacency"][np.ix_(matched_pred_global, matched_pred_global)]
            gt_sub = gt_adj[np.ix_(matched_gt_global, matched_gt_global)]
            M = len(row_ind)
            triu = np.triu(np.ones((M, M), dtype=bool), k=1)
            pred_edges = (pred_sub > 0.5) & triu
            gt_edges = (gt_sub > 0.5) & triu

            tp = np.logical_and(pred_edges, gt_edges).sum()
            fp = np.logical_and(pred_edges, ~gt_edges).sum()
            fn = np.logical_and(~pred_edges, gt_edges).sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            edge_f1 = 2 * prec * rec / max(prec + rec, 1e-8)

            for i in range(len(row_ind)):
                for j in range(i + 1, len(row_ind)):
                    gi, gj = col_ind[i], col_ind[j]
                    # Check if there's an edge in GT between these matched GT nodes
                    gi_global = gt_active_idx[gi] if gi < len(gt_active_idx) else -1
                    gj_global = gt_active_idx[gj] if gj < len(gt_active_idx) else -1
                    if gi_global >= 0 and gj_global >= 0 and gi_global < gt_adj.shape[0] and gj_global < gt_adj.shape[1]:
                        if gt_adj[gi_global, gj_global] > 0.5:
                            gt_len = np.linalg.norm(gt_pts[gi] - gt_pts[gj])
                            pred_len = np.linalg.norm(pred_pts[row_ind[i]] - pred_pts[row_ind[j]])
                            if gt_len > 1e-4:
                                bone_ratios.append(pred_len / gt_len)

        bone_ratio = np.mean(bone_ratios) if bone_ratios else float("inf")
        bone_error = np.mean([abs(r - 1.0) for r in bone_ratios]) if bone_ratios else float("inf")
        count_ratio = int(pred_mask.sum()) / max(int(gt_mask.sum()), 1)
        count_error = abs(count_ratio - 1.0)

        results.append({
            "idx": idx,
            "mpjpe": mpjpe,
            "spread_score": spread_score,
            "gt_spread": gt_spread,
            "pred_spread": pred_spread,
            "bone_error": bone_error,
            "bone_ratio": bone_ratio,
            "edge_f1": edge_f1,
            "count_ratio": count_ratio,
            "count_error": count_error,
            "n_gt": int(gt_mask.sum()),
            "n_pred": int(pred_mask.sum()),
            "gt_pos": gt_pos,
            "gt_active": gt_active,
            "gt_adj": gt_adj,
            "pred_pos": pred_pos,
            "pred_active": pred_active_prob,
            "pred_adj": structured_decoded["adjacency"],
            "pred_adj_threshold": threshold_decoded["adjacency"],
            "pred_adj_structured": structured_decoded["adjacency"],
            "pred_active_threshold": threshold_decoded["active_mask"].astype(float),
            "pred_active_structured": structured_decoded["active_mask"].astype(float),
            "pred_component_count": structured_decoded["metadata"]["component_count"],
            "pred_edge_count": structured_decoded["metadata"]["pred_edge_count"],
            "pred_max_degree": structured_decoded["metadata"]["max_degree"],
            "pc": pc,
        })

    return results


def make_grid_plot(samples, out_path, title, n_cols=4, active_threshold=0.5):
    """Generate grid comparison plot."""
    n = len(samples)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, s in enumerate(samples):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        pc_2d = s["pc"][:, :2]
        gt_2d = s["gt_pos"][:, :2]
        pred_2d = s["pred_pos"][:, :2]

        # Point cloud background
        ax.scatter(pc_2d[:, 0], pc_2d[:, 1], c="lightgray", s=1, alpha=0.3, zorder=1)

        # GT skeleton
        draw_skeleton(ax, gt_2d, s["gt_active"], s["gt_adj"], "green", "GT")

        # Predicted skeleton
        pred_mask_binary = s["pred_active_structured"]
        pred_adj_binary = s["pred_adj_structured"]
        draw_skeleton(ax, pred_2d, pred_mask_binary, pred_adj_binary, "red", "Pred", alpha=0.6)

        # Centroid markers
        gt_pts = s["gt_pos"][s["gt_active"] > 0.5]
        pred_pts = s["pred_pos"][s["pred_active"] > 0.5]
        if gt_pts.shape[0] > 0:
            gc = gt_pts.mean(axis=0)[:2]
            ax.plot(gc[0], gc[1], "g+", markersize=12, markeredgewidth=2, zorder=10)
        if pred_pts.shape[0] > 0:
            pc_c = pred_pts.mean(axis=0)[:2]
            ax.plot(pc_c[0], pc_c[1], "rx", markersize=12, markeredgewidth=2, zorder=10)

        # Info text
        info = (f"MPJPE={s['mpjpe']:.3f}  spread={s['spread_score']:.2f}  edgeF1={s['edge_f1']:.2f}\n"
            f"GT:{s['n_gt']} Pred:{s['n_pred']} cnt={s['count_ratio']:.2f} bone={s['bone_ratio']:.2f} cc={s['pred_component_count']} md={s['pred_max_degree']}")
        ax.set_title(info, fontsize=7)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty
    for i in range(n, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Vis-v2] Saved: {out_path}")


def make_compare_plot(samples, out_path, title):
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(15, max(4, n * 4)))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, s in enumerate(samples):
        pc_2d = s["pc"][:, :2]
        gt_2d = s["gt_pos"][:, :2]
        pred_2d = s["pred_pos"][:, :2]
        panels = [
            ("GT", s["gt_active"], s["gt_adj"], "green"),
            ("Threshold", s["pred_active_threshold"], s["pred_adj_threshold"], "orange"),
            ("Structured", s["pred_active_structured"], s["pred_adj_structured"], "red"),
        ]
        for col, (label, active_mask, adj, color) in enumerate(panels):
            ax = axes[row, col]
            ax.scatter(pc_2d[:, 0], pc_2d[:, 1], c="lightgray", s=1, alpha=0.25, zorder=1)
            pos_2d = gt_2d if label == "GT" else pred_2d
            draw_skeleton(ax, pos_2d, active_mask, adj, color, label, alpha=0.8)
            if row == 0:
                ax.set_title(label)
            if col == 2:
                ax.text(
                    0.02, 0.02,
                    f"edgeF1={s['edge_f1']:.2f}\nbone={s['bone_ratio']:.2f}\ncc={s['pred_component_count']}\nedges={s['pred_edge_count']}\nmaxdeg={s['pred_max_degree']}",
                    transform=ax.transAxes, fontsize=7,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                )
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.15)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Vis-v2] Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize rig predictions v2")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--checkpoint", default="outputs/models/hyperbone_anymate_static_v2/best_model.pt")
    parser.add_argument("--out", default="outputs/models/hyperbone_anymate_static_v2/visualizations")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=200, help="Max samples to evaluate for sorting")
    parser.add_argument("--n-grid", type=int, default=16, help="Samples per grid plot")
    parser.add_argument("--max-joints", type=int, default=64)
    parser.add_argument("--pc-points", type=int, default=2048)
    parser.add_argument("--points-per-sample", type=int, default=None,
                        help="Alias for --pc-points")
    parser.add_argument("--feat-dim", type=int, default=512)
    parser.add_argument("--backbone", choices=["pointnet", "dgcnn"], default="pointnet")
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--active-threshold", type=float, default=0.5)
    parser.add_argument("--edge-decode-mode",
                        choices=["threshold", "mst", "knn_mst", "forest_mst", "degree_limited"],
                        default="threshold")
    parser.add_argument("--decoder-k", type=int, default=6)
    parser.add_argument("--max-degree", type=int, default=4)
    parser.add_argument("--max-components", type=int, default=1)
    parser.add_argument("--edge-alpha", type=float, default=1.0)
    parser.add_argument("--edge-beta", type=float, default=0.5)
    parser.add_argument("--edge-gamma", type=float, default=1.0)
    parser.add_argument("--edge-eta", type=float, default=0.25)
    args = parser.parse_args()
    if args.points_per_sample is not None:
        args.pc_points = args.points_per_sample

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Vis-v2] Device: {device}")

    # Dataset
    split_path = f"{args.splits_dir}/{args.split}.jsonl"
    ds = AnymateStaticRigDataset(
        args.pt, split_path,
        max_joints=args.max_joints, pc_points=args.pc_points,
    )
    print(f"[Vis-v2] {args.split}: {len(ds)} samples")

    # Model
    model = HyperBoneStaticRigModel(
        in_channels=3, feat_dim=args.feat_dim,
        max_joints=args.max_joints, predict_skinning=True,
        backbone=args.backbone, knn_k=args.knn_k,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"[Vis-v2] Model loaded")

    decode_config = StructuredDecodeConfig(
        active_threshold=args.active_threshold,
        edge_decode_mode=args.edge_decode_mode,
        decoder_k=args.decoder_k,
        max_degree=args.max_degree,
        max_components=args.max_components,
        edge_alpha=args.edge_alpha,
        edge_beta=args.edge_beta,
        edge_gamma=args.edge_gamma,
        edge_eta=args.edge_eta,
    )

    # Collect data
    print(f"[Vis-v2] Running inference on {min(args.max_samples, len(ds))} samples...")
    all_data = collect_per_sample_data(model, ds, device, args.max_samples, decode_config=decode_config)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort and generate plots
    valid = [s for s in all_data if s["mpjpe"] != float("inf")]

    # Random 32 samples
    rng = np.random.default_rng(42)
    random_samples = list(rng.choice(valid, size=min(32, len(valid)), replace=False))
    make_grid_plot(random_samples, out_dir / "random_32.png",
                   f"Random {len(random_samples)} samples", active_threshold=args.active_threshold)

    # Worst by MPJPE
    worst_mpjpe = sorted(valid, key=lambda x: -x["mpjpe"])[:args.n_grid]
    make_grid_plot(worst_mpjpe, out_dir / "worst_by_mpjpe.png",
                   f"WORST {args.n_grid} by MPJPE (higher=worse)", active_threshold=args.active_threshold)

    # Best by MPJPE
    best_mpjpe = sorted(valid, key=lambda x: x["mpjpe"])[:args.n_grid]
    make_grid_plot(best_mpjpe, out_dir / "best_by_mpjpe.png",
                   f"BEST {args.n_grid} by MPJPE (lower=better)", active_threshold=args.active_threshold)

    # Worst by bone length
    valid_bone = [s for s in valid if s["bone_error"] != float("inf")]
    if valid_bone:
        worst_bone = sorted(valid_bone, key=lambda x: -x["bone_error"])[:args.n_grid]
        make_grid_plot(worst_bone, out_dir / "worst_by_bone_error.png",
                       f"WORST {args.n_grid} by Bone Length Error", active_threshold=args.active_threshold)

    # Worst by edge F1
    worst_edge = sorted(valid, key=lambda x: x["edge_f1"])[:args.n_grid]
    make_grid_plot(worst_edge, out_dir / "worst_by_edge_f1.png",
                   f"WORST {args.n_grid} by Edge F1", active_threshold=args.active_threshold)

    # Worst by count error
    worst_count = sorted(valid, key=lambda x: -x["count_error"])[:args.n_grid]
    make_grid_plot(worst_count, out_dir / "worst_by_count_error.png",
                   f"WORST {args.n_grid} by Count Error", active_threshold=args.active_threshold)

    # Most collapsed (lowest spread score)
    worst_spread = sorted(valid, key=lambda x: x["spread_score"])[:args.n_grid]
    make_grid_plot(worst_spread, out_dir / "worst_by_spread_collapse.png",
                   f"WORST {args.n_grid} by Spread Collapse (score→0 = total collapse)", active_threshold=args.active_threshold)

    if args.edge_decode_mode != "threshold":
        compare_random = list(rng.choice(valid, size=min(8, len(valid)), replace=False))
        make_compare_plot(compare_random, out_dir / "compare_random_8.png",
                          "GT vs Threshold vs Structured Decode")
        compare_worst_edge = sorted(valid, key=lambda x: x["edge_f1"])[:min(8, len(valid))]
        make_compare_plot(compare_worst_edge, out_dir / "compare_worst_edge_8.png",
                          "Worst Edge-F1: Threshold vs Structured Decode")

    # Summary stats
    summary = {
        "n_evaluated": len(valid),
        "mpjpe_mean": float(np.mean([s["mpjpe"] for s in valid])),
        "mpjpe_median": float(np.median([s["mpjpe"] for s in valid])),
        "spread_score_mean": float(np.mean([s["spread_score"] for s in valid])),
        "spread_score_median": float(np.median([s["spread_score"] for s in valid])),
        "edge_f1_mean": float(np.mean([s["edge_f1"] for s in valid])),
        "count_ratio_mean": float(np.mean([s["count_ratio"] for s in valid])),
        "count_error_mean": float(np.mean([s["count_error"] for s in valid])),
    }
    if valid_bone:
        summary["bone_error_mean"] = float(np.mean([s["bone_error"] for s in valid_bone]))

    with open(out_dir / "vis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Vis-v2] Summary: MPJPE={summary['mpjpe_mean']:.4f}, "
          f"spread_score={summary['spread_score_mean']:.3f}")
    print(f"[Vis-v2] All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
