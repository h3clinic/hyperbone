"""
Evaluate all checkpoints for a model run + select best by composite score.

Evaluates every checkpoint_e*.pt and best_model.pt on val split,
sweeps active threshold, selects best composite, then evaluates on test.

Usage:
    python scripts/eval_checkpoints_composite.py \
        --model-dir outputs/models/hyperbone_anymate_static_v2.4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.datasets.anymate_static_dataset import AnymateStaticRigDataset
from hyperbone.models.hyperbone_rig_graph_static import HyperBoneStaticRigModel


def compute_spread(positions: torch.Tensor) -> float:
    if positions.shape[0] < 2:
        return 0.0
    mn = positions.min(dim=0)[0]
    mx = positions.max(dim=0)[0]
    return (mx - mn).norm().item()


def hungarian_match(pred_pts, gt_pts):
    if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        return np.array([]), np.array([])
    cost = torch.cdist(pred_pts, gt_pts, p=2).cpu().numpy()
    return linear_sum_assignment(cost)


@torch.no_grad()
def fast_eval(model, loader, device, active_threshold=0.5, match_threshold=0.1):
    """Fast evaluation returning aggregate metrics."""
    model.eval()
    metrics_accum = {
        "mpjpe": [], "chamfer": [], "spread_score": [],
        "node_f1": [], "node_precision": [], "node_recall": [],
        "edge_f1": [], "edge_precision": [], "edge_recall": [],
        "bone_ratio": [], "count_ratio": [], "underprediction_ratio": [],
        "overprediction_ratio": [], "active_count_abs_error": [],
        "n_pred": [], "n_gt": [],
    }

    for batch in loader:
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
        pred = model(batch_dev)

        B = batch_dev["joint_pos"].shape[0]
        gt_pos = batch_dev["joint_pos"]
        gt_active = batch_dev["joint_active"]
        pred_pos = pred["joint_pos"].float()
        pred_active_prob = torch.sigmoid(pred["active_logits"].float())
        pred_adj = torch.sigmoid(pred["adj_logits"].float())
        gt_adj = batch_dev["adj_matrix"]

        for b in range(B):
            gt_mask = gt_active[b] > 0.5
            pred_mask = pred_active_prob[b] > active_threshold
            gt_pts = gt_pos[b, gt_mask]
            pred_pts = pred_pos[b, pred_mask]

            n_gt = gt_mask.sum().item()
            n_pred = pred_mask.sum().item()

            if n_gt == 0:
                continue

            count_ratio = n_pred / max(n_gt, 1)
            metrics_accum["n_gt"].append(n_gt)
            metrics_accum["n_pred"].append(n_pred)
            metrics_accum["count_ratio"].append(count_ratio)
            metrics_accum["underprediction_ratio"].append(max(0.0, 1.0 - count_ratio))
            metrics_accum["overprediction_ratio"].append(max(0.0, count_ratio - 1.0))
            metrics_accum["active_count_abs_error"].append(abs(count_ratio - 1.0))

            # Spread
            gt_spread = compute_spread(gt_pts)
            pred_spread = compute_spread(pred_pts) if n_pred >= 2 else 0.0
            metrics_accum["spread_score"].append(
                pred_spread / max(gt_spread, 1e-6))

            if n_pred == 0:
                metrics_accum["mpjpe"].append(float("inf"))
                metrics_accum["chamfer"].append(float("inf"))
                metrics_accum["node_f1"].append(0.0)
                metrics_accum["node_precision"].append(0.0)
                metrics_accum["node_recall"].append(0.0)
                metrics_accum["edge_f1"].append(0.0)
                metrics_accum["edge_precision"].append(0.0)
                metrics_accum["edge_recall"].append(0.0)
                metrics_accum["bone_ratio"].append(0.0)
                continue

            # Chamfer
            dist_p2g = torch.cdist(pred_pts, gt_pts, p=2)
            chamfer = (dist_p2g.min(dim=1)[0].mean() + dist_p2g.min(dim=0)[0].mean()).item()
            metrics_accum["chamfer"].append(chamfer)

            # Hungarian matching
            row_ind, col_ind = hungarian_match(pred_pts, gt_pts)
            n_matched = len(row_ind)

            if n_matched > 0:
                matched_pred = pred_pts[row_ind]
                matched_gt = gt_pts[col_ind]
                matched_dists = (matched_pred - matched_gt).norm(dim=-1)
                metrics_accum["mpjpe"].append(matched_dists.mean().item())

                # Node P/R/F1
                within = (matched_dists < match_threshold).sum().item()
                tp = within
                fp = n_pred - tp
                fn = n_gt - tp
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-8)
                metrics_accum["node_precision"].append(prec)
                metrics_accum["node_recall"].append(rec)
                metrics_accum["node_f1"].append(f1)

                # Edge F1
                if n_matched >= 2:
                    pred_active_idx = pred_mask.nonzero(as_tuple=True)[0]
                    gt_active_idx = gt_mask.nonzero(as_tuple=True)[0]
                    m_pred_g = pred_active_idx[row_ind]
                    m_gt_g = gt_active_idx[col_ind]

                    pred_sub = pred_adj[b][m_pred_g][:, m_pred_g]
                    gt_sub = gt_adj[b][m_gt_g][:, m_gt_g]
                    M = len(row_ind)
                    triu = torch.triu(torch.ones(M, M, device=device), diagonal=1).bool()
                    pred_edges = (pred_sub > 0.5) & triu
                    gt_edges = (gt_sub > 0.5) & triu

                    e_tp = (pred_edges & gt_edges).sum().item()
                    e_fp = (pred_edges & ~gt_edges).sum().item()
                    e_fn = (~pred_edges & gt_edges).sum().item()

                    e_prec = e_tp / max(e_tp + e_fp, 1)
                    e_rec = e_tp / max(e_tp + e_fn, 1)
                    e_f1 = 2 * e_prec * e_rec / max(e_prec + e_rec, 1e-8)
                    metrics_accum["edge_precision"].append(e_prec)
                    metrics_accum["edge_recall"].append(e_rec)
                    metrics_accum["edge_f1"].append(e_f1)
                else:
                    metrics_accum["edge_f1"].append(0.0)
                    metrics_accum["edge_precision"].append(0.0)
                    metrics_accum["edge_recall"].append(0.0)

                # Bone ratio
                if n_matched >= 2:
                    gt_sub_adj = gt_adj[b][m_gt_g][:, m_gt_g]
                    edges_in_matched = (gt_sub_adj.triu(diagonal=1) > 0.5).nonzero(as_tuple=False)
                    bone_ratios = []
                    for e in edges_in_matched:
                        i, j = e[0].item(), e[1].item()
                        gt_len = (gt_pts[col_ind[i]] - gt_pts[col_ind[j]]).norm().item()
                        pred_len = (pred_pts[row_ind[i]] - pred_pts[row_ind[j]]).norm().item()
                        if gt_len > 1e-4:
                            bone_ratios.append(pred_len / gt_len)
                    if bone_ratios:
                        metrics_accum["bone_ratio"].append(float(np.mean(bone_ratios)))
                    else:
                        metrics_accum["bone_ratio"].append(1.0)
                else:
                    metrics_accum["bone_ratio"].append(1.0)
            else:
                metrics_accum["mpjpe"].append(float("inf"))
                metrics_accum["node_f1"].append(0.0)
                metrics_accum["node_precision"].append(0.0)
                metrics_accum["node_recall"].append(0.0)
                metrics_accum["edge_f1"].append(0.0)
                metrics_accum["edge_precision"].append(0.0)
                metrics_accum["edge_recall"].append(0.0)
                metrics_accum["bone_ratio"].append(0.0)

    # Aggregate
    result = {}
    for k, vals in metrics_accum.items():
        valid = [v for v in vals if v != float("inf")]
        result[k] = float(np.mean(valid)) if valid else 0.0
    return result


def composite_score(m: dict) -> float:
    """Composite model selection score. Higher = better."""
    return (
        2.0 * m.get("edge_f1", 0)
        + 1.0 * m.get("node_f1", 0)
        - 1.0 * m.get("mpjpe", 0.5)
        - 0.5 * m.get("chamfer", 0.5)
        - 0.5 * abs(m.get("spread_score", 0) - 1.0)
        - 0.5 * max(0, m.get("bone_ratio", 2.0) - 1.25)
        - 0.75 * abs(m.get("count_ratio", 0.0) - 1.0)
    )


def main():
    parser = argparse.ArgumentParser(description="Composite checkpoint evaluation")
    parser.add_argument("--model-dir", required=True,
                        help="Directory with checkpoints")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--max-joints", type=int, default=64)
    parser.add_argument("--pc-points", type=int, default=2048)
    parser.add_argument("--points-per-sample", type=int, default=None,
                        help="Alias for --pc-points")
    parser.add_argument("--feat-dim", type=int, default=512)
    parser.add_argument("--backbone", choices=["pointnet", "dgcnn"], default="pointnet")
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--thresholds", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                        help="Comma-separated active thresholds to sweep")
    args = parser.parse_args()
    if args.points_per_sample is not None:
        args.pc_points = args.points_per_sample

    model_dir = Path(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = [float(t) for t in args.thresholds.split(",")]

    print(f"[Composite] Model dir: {model_dir}")
    print(f"[Composite] Thresholds: {thresholds}")
    print(f"[Composite] Device: {device}")

    # Find checkpoints
    checkpoints = sorted(model_dir.glob("checkpoint_e*.pt"))
    best_model = model_dir / "best_model.pt"
    final_model = model_dir / "model_final.pt"
    if best_model.exists():
        checkpoints.append(best_model)
    if final_model.exists():
        checkpoints.append(final_model)

    if not checkpoints:
        print("[ERROR] No checkpoints found!")
        return

    print(f"[Composite] Found {len(checkpoints)} checkpoints")

    # Load val dataset
    val_ds = AnymateStaticRigDataset(
        args.pt, f"{args.splits_dir}/val.jsonl",
        max_joints=args.max_joints, pc_points=args.pc_points,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[Composite] Val samples: {len(val_ds)}")

    # Model
    model = HyperBoneStaticRigModel(
        in_channels=3, feat_dim=args.feat_dim,
        max_joints=args.max_joints, predict_skinning=True,
        backbone=args.backbone, knn_k=args.knn_k,
    ).to(device)

    out_dir = model_dir / "composite_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each checkpoint at each threshold
    all_results = []

    for ckpt_path in checkpoints:
        ckpt_name = ckpt_path.stem
        print(f"\n--- {ckpt_name} ---")

        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)

        best_score = -999
        best_thresh = 0.5
        best_metrics = None

        for thresh in thresholds:
            metrics = fast_eval(model, val_loader, device, active_threshold=thresh)
            score = composite_score(metrics)

            if score > best_score:
                best_score = score
                best_thresh = thresh
                best_metrics = metrics

        best_metrics["composite_score"] = best_score
        best_metrics["best_threshold"] = best_thresh
        best_metrics["checkpoint"] = ckpt_name

        print(f"  best_thresh={best_thresh:.1f} score={best_score:.4f} | "
              f"mpjpe={best_metrics['mpjpe']:.4f} edge_f1={best_metrics['edge_f1']:.4f} "
              f"bone={best_metrics['bone_ratio']:.3f} spread={best_metrics['spread_score']:.3f} "
              f"cnt={best_metrics['count_ratio']:.3f} node_f1={best_metrics['node_f1']:.4f}")

        # Save per-checkpoint metric file
        if ckpt_name.startswith("checkpoint_e"):
            suffix = ckpt_name.split("checkpoint_e", 1)[1]
            per_ckpt_name = f"metrics_epoch_{suffix}.json"
        else:
            per_ckpt_name = f"metrics_{ckpt_name}.json"
        with open(out_dir / per_ckpt_name, "w") as f:
            json.dump(best_metrics, f, indent=2)

        all_results.append(best_metrics)

    # Select overall best
    best_idx = max(range(len(all_results)), key=lambda i: all_results[i]["composite_score"])
    selected = all_results[best_idx]

    print(f"\n{'='*70}")
    print(f"[Composite] SELECTED: {selected['checkpoint']} "
          f"(threshold={selected['best_threshold']}, score={selected['composite_score']:.4f})")
    print(f"{'='*70}")

    # Now evaluate selected checkpoint on TEST
    print(f"\n[Composite] Evaluating selected on TEST split...")
    test_ds = AnymateStaticRigDataset(
        args.pt, f"{args.splits_dir}/test.jsonl",
        max_joints=args.max_joints, pc_points=args.pc_points,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    selected_ckpt_path = model_dir / f"{selected['checkpoint']}.pt"
    state = torch.load(selected_ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    test_metrics = fast_eval(model, test_loader, device,
                             active_threshold=selected["best_threshold"])
    test_metrics["composite_score"] = composite_score(test_metrics)
    test_metrics["checkpoint"] = selected["checkpoint"]
    test_metrics["active_threshold"] = selected["best_threshold"]

    print(f"\n{'='*70}")
    print(f"[Composite] TEST RESULTS ({selected['checkpoint']}, "
          f"threshold={selected['best_threshold']})")
    print(f"{'='*70}")
    print(f"  MPJPE:             {test_metrics['mpjpe']:.4f}")
    print(f"  Chamfer:           {test_metrics['chamfer']:.4f}")
    print(f"  Spread:            {test_metrics['spread_score']:.3f}")
    print(f"  Node F1:           {test_metrics['node_f1']:.4f}")
    print(f"  Node Precision:    {test_metrics['node_precision']:.4f}")
    print(f"  Node Recall:       {test_metrics['node_recall']:.4f}")
    print(f"  Edge F1:           {test_metrics['edge_f1']:.4f}")
    print(f"  Edge Precision:    {test_metrics['edge_precision']:.4f}")
    print(f"  Edge Recall:       {test_metrics['edge_recall']:.4f}")
    print(f"  Bone Ratio:        {test_metrics['bone_ratio']:.3f}")
    print(f"  Count Ratio:       {test_metrics['count_ratio']:.3f}")
    print(f"  Underpred Ratio:   {test_metrics['underprediction_ratio']:.3f}")
    print(f"  Overpred Ratio:    {test_metrics['overprediction_ratio']:.3f}")
    print(f"  Count Abs Error:   {test_metrics['active_count_abs_error']:.3f}")
    print(f"  Pred Count:        {test_metrics['n_pred']:.1f}")
    print(f"  GT Count:          {test_metrics['n_gt']:.1f}")
    print(f"  Composite Score:   {test_metrics['composite_score']:.4f}")
    print(f"{'='*70}")

    # Save all results
    with open(out_dir / "val_checkpoints.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(out_dir / "test_selected.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    with open(out_dir / "selection.json", "w") as f:
        json.dump({
            "selected_checkpoint": selected["checkpoint"],
            "selected_threshold": selected["best_threshold"],
            "val_composite_score": selected["composite_score"],
            "test_composite_score": test_metrics["composite_score"],
        }, f, indent=2)

    print(f"\n  Saved: {out_dir}")

    # Pass/fail assessment
    print(f"\n{'='*70}")
    print("[Composite] PASS/FAIL ASSESSMENT")
    print(f"{'='*70}")

    checks = {
        "Edge F1 >= 0.50": test_metrics["edge_f1"] >= 0.50,
        "Edge recall >= 0.45": test_metrics["edge_recall"] >= 0.45,
        "Edge precision >= 0.40": test_metrics["edge_precision"] >= 0.40,
        "MPJPE <= 0.14": test_metrics["mpjpe"] <= 0.14,
        "Spread >= 0.85": test_metrics["spread_score"] >= 0.85,
        "Bone ratio <= 1.50": test_metrics["bone_ratio"] <= 1.50,
        "Count ratio in [0.8, 1.2]": 0.8 <= test_metrics["count_ratio"] <= 1.2,
    }

    passed = sum(checks.values())
    total = len(checks)
    for name, ok in checks.items():
        status = "✓" if ok else "✗"
        print(f"  [{status}] {name}")

    if passed == total:
        print(f"\n  VERDICT: PASS ({passed}/{total})")
    elif passed >= 4:
        print(f"\n  VERDICT: PARTIAL ({passed}/{total})")
    else:
        print(f"\n  VERDICT: FAIL ({passed}/{total})")


if __name__ == "__main__":
    main()
