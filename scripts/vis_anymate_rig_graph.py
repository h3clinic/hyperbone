"""
Visualize predicted vs GT joint graphs for Anymate static rig model.

Generates side-by-side 2D projections of:
- GT joints + edges (green)
- Predicted joints + edges (red)
- Matched pairs (blue lines)

Usage:
    python scripts/vis_anymate_rig_graph.py
    python scripts/vis_anymate_rig_graph.py --n-samples 20 --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.datasets.anymate_static_dataset import AnymateStaticRigDataset
from hyperbone.models.hyperbone_rig_graph_static import HyperBoneStaticRigModel

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def project_3d_to_2d(points: np.ndarray) -> np.ndarray:
    """Simple orthographic projection: use XY plane."""
    return points[:, :2]


def draw_graph(ax, pos_2d, active, adj, color, label, alpha=0.8):
    """Draw a skeleton graph on a matplotlib axis."""
    n = pos_2d.shape[0]
    active_idx = np.where(active > 0.5)[0]

    # Draw edges
    for i in active_idx:
        for j in active_idx:
            if j > i and adj[i, j] > 0.5:
                ax.plot(
                    [pos_2d[i, 0], pos_2d[j, 0]],
                    [pos_2d[i, 1], pos_2d[j, 1]],
                    color=color, alpha=alpha * 0.7, linewidth=1.5,
                )

    # Draw nodes
    if len(active_idx) > 0:
        ax.scatter(
            pos_2d[active_idx, 0], pos_2d[active_idx, 1],
            c=color, s=30, zorder=5, label=label, alpha=alpha,
            edgecolors="black", linewidths=0.5,
        )


@torch.no_grad()
def generate_visualizations(model, dataset, device, out_dir, n_samples=16, max_joints=64):
    """Generate grid of predicted vs GT skeleton graphs."""
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_samples = min(n_samples, len(dataset))
    # Pick evenly spaced samples
    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)

    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for plot_idx, data_idx in enumerate(indices):
        row, col = plot_idx // cols, plot_idx % cols
        ax = axes[row, col]

        # Get sample
        sample = dataset[data_idx]
        batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in sample.items()}

        # Predict
        pred = model(batch)

        # Extract data
        gt_pos = sample["joint_pos"].numpy()  # [J, 3]
        gt_active = sample["joint_active"].numpy()  # [J]
        gt_adj = sample["adj_matrix"].numpy()  # [J, J]

        pred_pos = pred["joint_pos"][0].cpu().numpy()  # [J, 3]
        pred_active_logits = pred["active_logits"][0].cpu().numpy()
        pred_active = (1 / (1 + np.exp(-pred_active_logits)))  # sigmoid
        pred_adj_logits = pred["adj_logits"][0].cpu().numpy()
        pred_adj = (1 / (1 + np.exp(-pred_adj_logits)))  # sigmoid

        # Project to 2D
        gt_2d = project_3d_to_2d(gt_pos)
        pred_2d = project_3d_to_2d(pred_pos)

        # Draw
        ax.set_aspect("equal")
        draw_graph(ax, gt_2d, gt_active, gt_adj, "green", "GT")
        draw_graph(ax, pred_2d, (pred_active > 0.5).astype(float), (pred_adj > 0.5).astype(float), "red", "Pred", alpha=0.6)

        n_gt = int(gt_active.sum())
        n_pred = int((pred_active > 0.5).sum())
        gt_edges = int((gt_adj * np.triu(np.ones_like(gt_adj), k=1)).sum())
        pred_edges = int(((pred_adj > 0.5) * np.triu(np.ones_like(pred_adj), k=1)).sum())

        ax.set_title(f"#{data_idx} | GT:{n_gt}j/{gt_edges}e  Pred:{n_pred}j/{pred_edges}e", fontsize=8)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.grid(True, alpha=0.2)
        ax.set_xticks([])
        ax.set_yticks([])

    # Remove empty axes
    for plot_idx in range(n_samples, rows * cols):
        row, col = plot_idx // cols, plot_idx % cols
        axes[row, col].set_visible(False)

    fig.suptitle("Predicted (red) vs GT (green) Rig Graphs — XY Projection", fontsize=12)
    fig.tight_layout()

    save_path = out_dir / "rig_graph_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Vis] Saved: {save_path}")

    # Also save individual samples for closer inspection
    for i, data_idx in enumerate(indices[:4]):
        sample = dataset[data_idx]
        batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in sample.items()}
        pred = model(batch)

        gt_pos = sample["joint_pos"].numpy()
        gt_active = sample["joint_active"].numpy()
        gt_adj = sample["adj_matrix"].numpy()
        pred_pos = pred["joint_pos"][0].cpu().numpy()
        pred_active = (1 / (1 + np.exp(-pred["active_logits"][0].cpu().numpy())))
        pred_adj = (1 / (1 + np.exp(-pred["adj_logits"][0].cpu().numpy())))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        gt_2d = project_3d_to_2d(gt_pos)
        pred_2d = project_3d_to_2d(pred_pos)

        ax1.set_title(f"Ground Truth ({int(gt_active.sum())} joints)")
        draw_graph(ax1, gt_2d, gt_active, gt_adj, "green", "GT")
        ax1.set_aspect("equal")
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.2, 1.2)
        ax1.grid(True, alpha=0.3)

        ax2.set_title(f"Predicted ({int((pred_active > 0.5).sum())} joints)")
        draw_graph(ax2, pred_2d, (pred_active > 0.5).astype(float), (pred_adj > 0.5).astype(float), "red", "Pred")
        ax2.set_aspect("equal")
        ax2.set_xlim(-1.2, 1.2)
        ax2.set_ylim(-1.2, 1.2)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{i:02d}_idx{data_idx}.png", dpi=150)
        plt.close(fig)

    print(f"[Vis] Saved {min(4, n_samples)} individual comparison plots")


def main():
    parser = argparse.ArgumentParser(description="Visualize rig graph predictions")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--checkpoint", default="outputs/models/hyperbone_anymate_local_dev/best_model.pt")
    parser.add_argument("--out", default="outputs/models/hyperbone_anymate_local_dev/visualizations")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--max-joints", type=int, default=64)
    parser.add_argument("--pc-points", type=int, default=2048)
    parser.add_argument("--feat-dim", type=int, default=512)
    args = parser.parse_args()

    if not HAS_MPL:
        print("[Vis] ERROR: matplotlib not available. Install with: pip install matplotlib")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Vis] Device: {device}")

    # Dataset
    split_path = f"{args.splits_dir}/{args.split}.jsonl"
    ds = AnymateStaticRigDataset(
        args.pt, split_path,
        max_joints=args.max_joints, pc_points=args.pc_points,
    )
    print(f"[Vis] {args.split} split: {len(ds)} samples")

    # Model
    model = HyperBoneStaticRigModel(
        in_channels=3, feat_dim=args.feat_dim,
        max_joints=args.max_joints, predict_skinning=True,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"[Vis] Model loaded")

    # Generate
    out_dir = Path(args.out)
    generate_visualizations(model, ds, device, out_dir, args.n_samples, args.max_joints)


if __name__ == "__main__":
    main()
