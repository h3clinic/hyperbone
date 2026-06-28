"""
HyperBone v4.1 — Production topology prediction CLI.

Track A: Rigged/skinned asset topology extraction.
Skinning features are NOT available for unrigged video or raw object meshes.

Input: asset/sample with known joints + mesh skinning weights.
Output: adjacency JSON + optional overlay PNG.

Usage:
    # Single sample by index
    python scripts/predict_topology_v41.py --sample-idx 42

    # Batch over a split
    python scripts/predict_topology_v41.py --split test --max-samples 10

    # With overlay images
    python scripts/predict_topology_v41.py --sample-idx 42 --overlay
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.datasets.anymate_static_dataset import AnymateStaticRigDataset
from hyperbone.models.hyperbone_static_parent import HyperBoneStaticParentModel
from hyperbone.models.topology_edge_scorer import TopologyEdgeScorer
from hyperbone.rigs.skinning_topology_features import compute_skinning_score_matrices
from hyperbone.rigs.topology_optimizers import hybrid_skinning_cost_optimize
from hyperbone.rigs.undirected_topology import (
    build_undirected_adjacency,
    build_undirected_knn_candidates,
    compute_topology_edge_inputs,
    edge_prf,
    graph_stats,
    symmetrize_pair_scores,
)

# v4.1 locked config
V41_CONFIG = {
    "method": "hybrid_skinning_cost_mst",
    "optimizer": "density_normalized_mst",
    "candidate_k": 16,
    "neural_weight": 1.0,
    "skin_cosine_weight": 4.0,
    "shared_weight": 4.0,
    "distance_weight": 1.0,
    "degree_penalty": 0.05,
    "long_edge_penalty": 0.25,
    "mutual_bonus": 0.2,
    "max_degree": 4,
}


def _edge_count(edge_mask):
    tri = torch.triu(torch.ones_like(edge_mask, dtype=torch.bool), diagonal=1)
    return int((edge_mask & tri).sum().item())


def _load_models(ckpt_path, device, max_nodes=128, feat_dim=512):
    model = HyperBoneStaticParentModel(
        in_channels=3, feat_dim=feat_dim, max_joints=max_nodes,
        predict_skinning=False, backbone="dgcnn", knn_k=16,
        parent_head="pairwise", root_feature_mode="structural",
    ).to(device)
    scorer = TopologyEdgeScorer(
        node_feature_dim=256, edge_feature_dim=23, node_local_dim=8,
        global_context_dim=4, hidden_dim=256, dropout=0.1, num_blocks=3,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
        scorer_state = state.get("topology_scorer_state_dict", state.get("edge_refiner_state_dict"))
        if scorer_state is not None:
            scorer.load_state_dict(scorer_state, strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()
    scorer.eval()
    return model, scorer


def extract_edge_list(adj, active_mask):
    active_idx = torch.where(active_mask)[0].tolist()
    edges = []
    for i in active_idx:
        for j in active_idx:
            if j > i and adj[i, j] > 0.5:
                edges.append([i, j])
    return edges


@torch.no_grad()
def predict_single(
    dataset, sample_idx, model, scorer, device, config=None, compute_gt=True,
):
    """Run v4.1 topology prediction on a single sample.

    Returns dict with prediction results.
    """
    cfg = config or V41_CONFIG
    batch = dataset[sample_idx]
    raw_sample = dataset.data[dataset.indices[sample_idx]]

    joint_pos = batch["joint_pos"].unsqueeze(0).to(device)
    active_mask = (batch["joint_active"] > 0.5).unsqueeze(0).to(device)

    pred = model.forward_parent_from_joints(
        joint_pos, active_mask=active_mask, no_backbone_for_gt_nodes=True,
    )
    base_scores = symmetrize_pair_scores(pred["parent_pair_logits"], mode="average")
    node_tokens = pred["node_tokens"]

    k = cfg["candidate_k"]

    if compute_gt:
        gt_adj = build_undirected_adjacency(batch["adj_matrix"].unsqueeze(0), active_mask)
    else:
        gt_adj = None

    cand = build_undirected_knn_candidates(
        joint_pos, active_mask, k=k, gt_adj=gt_adj, force_include_gt=False,
    )
    candidate_mask = cand["candidate_mask"]
    topo_inputs = compute_topology_edge_inputs(joint_pos, active_mask, candidate_mask, k=k)
    neural_scores = scorer(
        base_scores, topo_inputs["pair_features"], node_tokens,
        topo_inputs["node_local_features"], topo_inputs["global_context"],
    )

    # Skinning features
    jp_cpu = joint_pos[0].cpu()
    am_cpu = active_mask[0].cpu()
    cm_cpu = candidate_mask[0].cpu()
    skinning_mats = compute_skinning_score_matrices(jp_cpu, am_cpu, cm_cpu, raw_sample)

    # Run v4.1 optimizer
    pred_adj = hybrid_skinning_cost_optimize(
        joint_pos=jp_cpu,
        active_mask=am_cpu,
        neural_scores=neural_scores[0].cpu(),
        skinning_cosine=skinning_mats["skinning_cosine"],
        max_shared_weight=skinning_mats["max_shared_weight"],
        mode=cfg["optimizer"],
        k=k,
        distance_weight=cfg["distance_weight"],
        neural_weight=cfg["neural_weight"],
        skin_cosine_weight=cfg["skin_cosine_weight"],
        shared_weight=cfg["shared_weight"],
        degree_penalty=cfg["degree_penalty"],
        long_edge_penalty=cfg["long_edge_penalty"],
        mutual_bonus=cfg["mutual_bonus"],
        max_degree=cfg["max_degree"],
        candidate_mask=cm_cpu,
    )

    stats = graph_stats(pred_adj, am_cpu)
    pred_edges = extract_edge_list(pred_adj, am_cpu)

    result = {
        "sample_idx": sample_idx,
        "dataset_idx": dataset.indices[sample_idx],
        "n_joints": int(am_cpu.sum().item()),
        "n_edges": len(pred_edges),
        "edges": pred_edges,
        "cycle_count": stats["cycle_count"],
        "max_degree": stats["max_degree"],
        "average_degree": stats["average_degree"],
        "component_count": stats["component_count"],
        "config": cfg,
    }

    if compute_gt and gt_adj is not None:
        prf = edge_prf(pred_adj, gt_adj[0].cpu(), am_cpu)
        gt_edges = extract_edge_list(gt_adj[0].cpu(), am_cpu)
        result["gt_edges"] = gt_edges
        result["edge_f1"] = prf["f1"]
        result["edge_precision"] = prf["precision"]
        result["edge_recall"] = prf["recall"]
        result["selected_to_gt_ratio"] = len(pred_edges) / max(len(gt_edges), 1)

    # Store numpy arrays for overlay generation
    result["_joint_pos"] = jp_cpu.numpy()
    result["_active_mask"] = am_cpu.numpy()
    result["_pred_adj"] = pred_adj.numpy()
    if compute_gt and gt_adj is not None:
        result["_gt_adj"] = gt_adj[0].cpu().numpy()

    return result


def generate_overlay(result, out_path):
    """Generate a 3-panel overlay image for a prediction result."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos_2d = result["_joint_pos"][:, :2]
    active = result["_active_mask"]
    pred_adj = result["_pred_adj"]
    gt_adj = result.get("_gt_adj")
    active_idx = np.where(active > 0.5)[0]

    n_panels = 3 if gt_adj is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    def draw(ax, adj, color, title):
        for i in active_idx:
            for j in active_idx:
                if j > i and adj[i, j] > 0.5:
                    ax.plot([pos_2d[i, 0], pos_2d[j, 0]], [pos_2d[i, 1], pos_2d[j, 1]],
                            color=color, alpha=0.6, linewidth=1.5)
        ax.scatter(pos_2d[active_idx, 0], pos_2d[active_idx, 1],
                   c=color, s=30, zorder=5, edgecolors="black", linewidths=0.3)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.invert_yaxis()

    if gt_adj is not None:
        draw(axes[0], gt_adj, "green", f"GT ({len(result.get('gt_edges', []))} edges)")
        draw(axes[1], pred_adj, "red", f"Pred ({result['n_edges']} edges)")

        ax = axes[2]
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
                   c="black", s=30, zorder=5, edgecolors="white", linewidths=0.3)
        f1_str = f"F1={result.get('edge_f1', 0):.3f}" if "edge_f1" in result else ""
        ax.set_title(f"Overlay: TP=blue, FN=green--, FP=red: {f1_str}")
        ax.set_aspect("equal")
        ax.invert_yaxis()
    else:
        draw(axes[0], pred_adj, "steelblue", f"v4.1 Pred ({result['n_edges']} edges)")

    fig.suptitle(
        f"v4.1 Sample {result['sample_idx']} — "
        f"{result['n_joints']} joints, {result['n_edges']} edges, "
        f"cycles={result['cycle_count']:.0f}",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="HyperBone v4.1 Topology Prediction (Track A: Rigged/Skinned)")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--split", default="test")
    parser.add_argument("--ckpt", default="outputs/models/hyperbone_anymate_static_v2.16_topology_full/best_model.pt")
    parser.add_argument("--out-dir", default="outputs/predictions/v41_topology")
    parser.add_argument("--sample-idx", type=int, default=-1, help="Predict a single sample by split index")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples to predict (0=all)")
    parser.add_argument("--overlay", action="store_true", help="Generate overlay images")
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument("--feat-dim", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("HyperBone v4.1 Topology Prediction", flush=True)
    print("Track A: Rigged/skinned asset topology.", flush=True)
    print("Caveat: Skinning features are NOT available for unrigged objects.", flush=True)
    print(f"Config: {json.dumps(V41_CONFIG, indent=2)}", flush=True)

    dataset = AnymateStaticRigDataset(
        args.pt, f"{args.splits_dir}/{args.split}.jsonl",
        max_joints=args.max_nodes, pc_points=1024,
    )
    model, scorer = _load_models(args.ckpt, device, args.max_nodes, args.feat_dim)

    if args.sample_idx >= 0:
        indices = [args.sample_idx]
    else:
        n = len(dataset)
        if args.max_samples > 0:
            n = min(n, args.max_samples)
        indices = list(range(n))

    results = []
    for i, idx in enumerate(indices):
        result = predict_single(dataset, idx, model, scorer, device)

        # Remove numpy arrays from JSON output
        json_result = {k: v for k, v in result.items() if not k.startswith("_")}
        results.append(json_result)

        f1_str = f"F1={result.get('edge_f1', 0):.3f}" if "edge_f1" in result else ""
        print(f"  [{i+1}/{len(indices)}] sample={idx} edges={result['n_edges']} "
              f"cycles={result['cycle_count']:.0f} {f1_str}", flush=True)

        if args.overlay:
            overlay_dir = out_dir / "overlays"
            overlay_dir.mkdir(exist_ok=True)
            generate_overlay(result, overlay_dir / f"sample_{idx:04d}.png")

        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(indices)} done", flush=True)

    # Save predictions
    pred_path = out_dir / "predictions.json"
    with open(pred_path, "w") as f:
        json.dump({
            "version": "v4.1",
            "track": "rigged_skinned_asset_topology",
            "config": V41_CONFIG,
            "caveat": "Skinning features are NOT available for unrigged video or raw object meshes.",
            "predictions": results,
        }, f, indent=2)

    # Summary
    if any("edge_f1" in r for r in results):
        f1s = [r["edge_f1"] for r in results if "edge_f1" in r]
        print(f"\nMean F1: {np.mean(f1s):.4f} (n={len(f1s)})", flush=True)

    print(f"Saved {pred_path}", flush=True)
    if args.overlay:
        print(f"Overlays: {out_dir / 'overlays'}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
