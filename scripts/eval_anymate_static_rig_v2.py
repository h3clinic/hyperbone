"""
Evaluate HyperBone static rig model v2 — includes spread-collapse metrics.

New metrics over v1:
- Chamfer distance (bidirectional)
- Node precision / recall / F1 after Hungarian matching
- Predicted spread ratio vs GT spread ratio
- Spread collapse score = pred_spread / gt_spread
- Coverage: % of GT extremity joints matched within threshold
- Bone length error after matching
- Active node overprediction ratio
- Edge F1 after matching

Usage:
    python scripts/eval_anymate_static_rig_v2.py --checkpoint outputs/models/hyperbone_anymate_static_v2/best_model.pt
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
from hyperbone.rigs.structured_decoder import StructuredDecodeConfig, decode_structured_graph


def compute_spread(positions: torch.Tensor) -> float:
    """Compute spread = bbox diagonal of a set of 3D points."""
    if positions.shape[0] < 2:
        return 0.0
    mn = positions.min(dim=0)[0]
    mx = positions.max(dim=0)[0]
    return (mx - mn).norm().item()


def hungarian_match(pred_pts, gt_pts):
    """Match pred to gt by L2 cost. Returns (row_ind, col_ind)."""
    if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        return np.array([]), np.array([])
    cost = torch.cdist(pred_pts, gt_pts, p=2).cpu().numpy()
    return linear_sum_assignment(cost)


def count_gt_edges(gt_adj: torch.Tensor, gt_mask: torch.Tensor) -> int:
    gt_active_idx = gt_mask.nonzero(as_tuple=True)[0]
    if gt_active_idx.numel() < 2:
        return 0
    gt_sub = gt_adj[gt_active_idx][:, gt_active_idx]
    return int((gt_sub.triu(diagonal=1) > 0.5).sum().item())


@torch.no_grad()
def evaluate_v2(
    model,
    loader,
    device,
    max_joints,
    match_threshold=0.1,
    active_threshold=0.5,
    decode_config: StructuredDecodeConfig | None = None,
):
    """Evaluate with full v2 metrics."""
    model.eval()

    metrics_list = []

    for batch in loader:
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        pred = model(batch_dev)

        B = batch_dev["joint_pos"].shape[0]
        gt_pos = batch_dev["joint_pos"]
        gt_active = batch_dev["joint_active"]
        pred_pos = pred["joint_pos"]
        pred_active_prob = torch.sigmoid(pred["active_logits"])
        pred_adj = torch.sigmoid(pred["adj_logits"])
        gt_adj = batch_dev["adj_matrix"]

        for b in range(B):
            gt_mask = gt_active[b] > 0.5
            pred_mask = pred_active_prob[b] > active_threshold
            decoded = decode_structured_graph(
                pred_pos[b].detach().cpu().numpy(),
                pred_active_prob[b].detach().cpu().numpy(),
                pred_adj[b].detach().cpu().numpy(),
                point_cloud=batch_dev["pc"][b].detach().cpu().numpy()[:, :3] if "pc" in batch_dev else None,
                node_confidence=pred_active_prob[b].detach().cpu().numpy(),
                config=decode_config or StructuredDecodeConfig(active_threshold=active_threshold),
            )
            pred_mask_np = decoded["active_mask"]
            pred_adj_np = decoded["adjacency"]
            pred_mask = torch.from_numpy(pred_mask_np).to(device=device)
            pred_adj_decoded = torch.from_numpy(pred_adj_np).to(device=device)

            gt_pts = gt_pos[b, gt_mask]
            pred_pts = pred_pos[b, pred_mask]

            n_gt = gt_mask.sum().item()
            n_pred = pred_mask.sum().item()

            if n_gt == 0:
                continue

            m = {}
            count_ratio = n_pred / max(n_gt, 1)
            m["gt_count"] = n_gt
            m["pred_count"] = n_pred
            m["count_ratio"] = count_ratio
            m["underprediction_ratio"] = max(0.0, 1.0 - count_ratio)
            m["overprediction_ratio"] = max(0.0, count_ratio - 1.0)
            m["active_count_abs_error"] = abs(count_ratio - 1.0)
            m["pred_edge_count"] = decoded["metadata"]["pred_edge_count"]
            m["gt_edge_count"] = count_gt_edges(gt_adj[b], gt_mask)
            m["component_count"] = decoded["metadata"]["component_count"]
            m["connected_ratio"] = decoded["metadata"]["connected_ratio"]
            m["average_degree"] = decoded["metadata"]["average_degree"]
            m["max_degree"] = decoded["metadata"]["max_degree"]

            # --- Spread ---
            gt_spread = compute_spread(gt_pts)
            pred_spread = compute_spread(pred_pts) if n_pred >= 2 else 0.0
            m["gt_spread"] = gt_spread
            m["pred_spread"] = pred_spread
            m["spread_collapse_score"] = pred_spread / max(gt_spread, 1e-6)

            # --- Chamfer ---
            if n_pred > 0:
                dist_p2g = torch.cdist(pred_pts, gt_pts, p=2)
                min_p2g = dist_p2g.min(dim=1)[0]
                min_g2p = dist_p2g.min(dim=0)[0]
                m["chamfer_p2g"] = min_p2g.mean().item()
                m["chamfer_g2p"] = min_g2p.mean().item()
                m["chamfer"] = (min_p2g.mean() + min_g2p.mean()).item()
            else:
                m["chamfer_p2g"] = float("inf")
                m["chamfer_g2p"] = float("inf")
                m["chamfer"] = float("inf")

            # --- Hungarian matching ---
            if n_pred > 0:
                row_ind, col_ind = hungarian_match(pred_pts, gt_pts)
                n_matched = len(row_ind)

                if n_matched > 0:
                    matched_pred_pts = pred_pts[row_ind]
                    matched_gt_pts = gt_pts[col_ind]
                    matched_dists = (matched_pred_pts - matched_gt_pts).norm(dim=-1)

                    m["mpjpe_matched"] = matched_dists.mean().item()
                    m["mpjpe_matched_median"] = matched_dists.median().item()

                    # Coverage: matched within threshold
                    within_thresh = (matched_dists < match_threshold).sum().item()
                    m["coverage"] = within_thresh / n_gt

                    # Node precision/recall/F1 (using threshold)
                    node_tp = within_thresh
                    node_fp = n_pred - node_tp
                    node_fn = n_gt - node_tp
                    node_prec = node_tp / max(node_tp + node_fp, 1)
                    node_rec = node_tp / max(node_tp + node_fn, 1)
                    node_f1 = 2 * node_prec * node_rec / max(node_prec + node_rec, 1e-8)
                    m["node_precision"] = node_prec
                    m["node_recall"] = node_rec
                    m["node_f1"] = node_f1
                else:
                    m["mpjpe_matched"] = float("inf")
                    m["mpjpe_matched_median"] = float("inf")
                    m["coverage"] = 0.0
                    m["node_precision"] = 0.0
                    m["node_recall"] = 0.0
                    m["node_f1"] = 0.0
            else:
                m["mpjpe_matched"] = float("inf")
                m["mpjpe_matched_median"] = float("inf")
                m["coverage"] = 0.0
                m["node_precision"] = 0.0
                m["node_recall"] = 0.0
                m["node_f1"] = 0.0
                row_ind, col_ind = np.array([]), np.array([])

            # --- Edge F1 after matching ---
            if n_pred > 0 and len(row_ind) >= 2:
                # Get pred adj for active nodes
                pred_active_idx = pred_mask.nonzero(as_tuple=True)[0]
                gt_active_idx = gt_mask.nonzero(as_tuple=True)[0]

                # Submatrix for matched nodes
                matched_pred_global = pred_active_idx[row_ind]
                matched_gt_global = gt_active_idx[col_ind]

                pred_sub = pred_adj_decoded[matched_pred_global][:, matched_pred_global]
                gt_sub = gt_adj[b][matched_gt_global][:, matched_gt_global]

                M = len(row_ind)
                triu = torch.triu(torch.ones(M, M, device=device), diagonal=1).bool()
                pred_edges_matched = (pred_sub > 0.5) & triu
                gt_edges_matched = (gt_sub > 0.5) & triu

                tp = (pred_edges_matched & gt_edges_matched).sum().item()
                fp = (pred_edges_matched & ~gt_edges_matched).sum().item()
                fn = (~pred_edges_matched & gt_edges_matched).sum().item()

                edge_prec = tp / max(tp + fp, 1)
                edge_rec = tp / max(tp + fn, 1)
                edge_f1 = 2 * edge_prec * edge_rec / max(edge_prec + edge_rec, 1e-8)
                m["edge_f1_matched"] = edge_f1
                m["edge_precision_matched"] = edge_prec
                m["edge_recall_matched"] = edge_rec
            else:
                m["edge_f1_matched"] = 0.0
                m["edge_precision_matched"] = 0.0
                m["edge_recall_matched"] = 0.0

            # --- Bone length error after matching ---
            if n_pred > 0 and len(row_ind) >= 2:
                gt_active_idx = gt_mask.nonzero(as_tuple=True)[0]
                matched_gt_global = gt_active_idx[col_ind]
                gt_sub_adj = gt_adj[b][matched_gt_global][:, matched_gt_global]
                edges_in_matched = (gt_sub_adj.triu(diagonal=1) > 0.5).nonzero(as_tuple=False)

                bone_errors = []
                bone_errors_tp = []
                bone_errors_fp = []
                for e in edges_in_matched:
                    i, j = e[0].item(), e[1].item()
                    gt_len = (gt_pts[col_ind[i]] - gt_pts[col_ind[j]]).norm().item()
                    pred_len = (pred_pts[row_ind[i]] - pred_pts[row_ind[j]]).norm().item()
                    if gt_len > 1e-4:
                        bone_errors.append(pred_len / gt_len)

                # Split diagnostics: matched true-positive edges vs. false-positive edges.
                pred_active_idx = pred_mask.nonzero(as_tuple=True)[0]
                m_pred_global = pred_active_idx[row_ind]
                pred_sub_all = pred_adj_decoded[m_pred_global][:, m_pred_global]
                triu = torch.triu(torch.ones(len(row_ind), len(row_ind), device=device), diagonal=1).bool()
                pred_edges_matched = (pred_sub_all > 0.5) & triu
                gt_edges_matched = (gt_sub_adj > 0.5) & triu

                tp_edge_mask = pred_edges_matched & gt_edges_matched
                fp_edge_mask = pred_edges_matched & ~gt_edges_matched

                tp_edges = tp_edge_mask.nonzero(as_tuple=False)
                fp_edges = fp_edge_mask.nonzero(as_tuple=False)

                for e in tp_edges:
                    i, j = e[0].item(), e[1].item()
                    gt_len = (gt_pts[col_ind[i]] - gt_pts[col_ind[j]]).norm().item()
                    pred_len = (pred_pts[row_ind[i]] - pred_pts[row_ind[j]]).norm().item()
                    if gt_len > 1e-4:
                        bone_errors_tp.append(pred_len / gt_len)

                for e in fp_edges:
                    i, j = e[0].item(), e[1].item()
                    gt_len = (gt_pts[col_ind[i]] - gt_pts[col_ind[j]]).norm().item()
                    pred_len = (pred_pts[row_ind[i]] - pred_pts[row_ind[j]]).norm().item()
                    if gt_len > 1e-4:
                        bone_errors_fp.append(pred_len / gt_len)

                if bone_errors:
                    m["bone_length_ratio_mean"] = float(np.mean(bone_errors))
                    m["bone_length_ratio_median"] = float(np.median(bone_errors))
                    m["bone_length_error_abs"] = float(np.mean([abs(r - 1.0) for r in bone_errors]))
                else:
                    m["bone_length_ratio_mean"] = 0.0
                    m["bone_length_ratio_median"] = 0.0
                    m["bone_length_error_abs"] = 0.0

                pred_edge_bone_ratios = []
                pred_edges = pred_edges_matched.nonzero(as_tuple=False)
                for e in pred_edges:
                    i, j = e[0].item(), e[1].item()
                    ref_len = (gt_pts[col_ind[i]] - gt_pts[col_ind[j]]).norm().item()
                    pred_len = (pred_pts[row_ind[i]] - pred_pts[row_ind[j]]).norm().item()
                    if ref_len > 1e-4:
                        pred_edge_bone_ratios.append(pred_len / ref_len)

                m["bone_length_ratio_all_pred_edges"] = float(np.mean(pred_edge_bone_ratios)) if pred_edge_bone_ratios else 0.0
                m["bone_length_ratio_true_positive_edges"] = float(np.mean(bone_errors_tp)) if bone_errors_tp else 0.0
                m["bone_length_ratio_false_positive_edges"] = float(np.mean(bone_errors_fp)) if bone_errors_fp else 0.0
            else:
                m["bone_length_ratio_mean"] = 0.0
                m["bone_length_ratio_median"] = 0.0
                m["bone_length_error_abs"] = 0.0
                m["bone_length_ratio_all_pred_edges"] = 0.0
                m["bone_length_ratio_true_positive_edges"] = 0.0
                m["bone_length_ratio_false_positive_edges"] = 0.0

            # --- Same-slot MPJPE (for comparison with v1) ---
            active_gt = gt_pos[b, gt_mask]
            active_pred_ss = pred_pos[b, gt_mask]  # same-slot
            if n_gt > 0:
                m["mpjpe_same_slot"] = (active_gt - active_pred_ss).norm(dim=-1).mean().item()
            else:
                m["mpjpe_same_slot"] = 0.0

            metrics_list.append(m)

    # --- Aggregate ---
    results = {"n_samples": len(metrics_list)}
    keys = [
        "mpjpe_matched", "mpjpe_matched_median", "mpjpe_same_slot",
        "chamfer", "chamfer_p2g", "chamfer_g2p",
        "coverage", "node_precision", "node_recall", "node_f1",
        "edge_f1_matched", "edge_precision_matched", "edge_recall_matched",
        "bone_length_ratio_mean", "bone_length_ratio_median", "bone_length_error_abs",
        "bone_length_ratio_all_pred_edges", "bone_length_ratio_true_positive_edges",
        "bone_length_ratio_false_positive_edges",
        "pred_edge_count", "gt_edge_count", "component_count", "connected_ratio",
        "average_degree", "max_degree",
        "gt_spread", "pred_spread", "spread_collapse_score",
        "gt_count", "pred_count", "count_ratio", "underprediction_ratio",
        "overprediction_ratio", "active_count_abs_error",
    ]

    for k in keys:
        vals = [m[k] for m in metrics_list if m[k] != float("inf")]
        if vals:
            results[k + "_mean"] = round(float(np.mean(vals)), 6)
            results[k + "_median"] = round(float(np.median(vals)), 6)
            results[k + "_std"] = round(float(np.std(vals)), 6)
        else:
            results[k + "_mean"] = None
            results[k + "_median"] = None
            results[k + "_std"] = None

    # Per-sample breakdown (for vis sorting)
    results["per_sample"] = metrics_list

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate static rig v2")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--checkpoint", default="outputs/models/hyperbone_anymate_static_v2/best_model.pt")
    parser.add_argument("--out", default="outputs/models/hyperbone_anymate_static_v2/eval")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-joints", type=int, default=64)
    parser.add_argument("--pc-points", type=int, default=2048)
    parser.add_argument("--points-per-sample", type=int, default=None,
                        help="Alias for --pc-points")
    parser.add_argument("--feat-dim", type=int, default=512)
    parser.add_argument("--backbone", choices=["pointnet", "dgcnn"], default="pointnet")
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--match-threshold", type=float, default=0.1,
                        help="Distance threshold for counting a match as 'correct'")
    parser.add_argument("--active-threshold", type=float, default=0.5,
                        help="Fixed active threshold for node activation")
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
    parser.add_argument("--sweep-active-threshold", action="store_true",
                        help="Sweep active threshold from 0.3 to 0.9 and report best by node F1")
    args = parser.parse_args()
    if args.points_per_sample is not None:
        args.pc_points = args.points_per_sample

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval-v2] Device: {device}")
    print(f"[Eval-v2] Checkpoint: {args.checkpoint}")
    print(f"[Eval-v2] Split: {args.split}")

    # Dataset
    split_path = f"{args.splits_dir}/{args.split}.jsonl"
    ds = AnymateStaticRigDataset(
        args.pt, split_path,
        max_joints=args.max_joints, pc_points=args.pc_points,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[Eval-v2] Samples: {len(ds)}")

    # Model
    model = HyperBoneStaticRigModel(
        in_channels=3, feat_dim=args.feat_dim,
        max_joints=args.max_joints, predict_skinning=True,
        backbone=args.backbone, knn_k=args.knn_k,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Eval-v2] Model loaded ({n_params:,} params)")

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

    # Evaluate
    results = evaluate_v2(
        model, loader, device, args.max_joints,
        args.match_threshold, active_threshold=args.active_threshold,
        decode_config=decode_config,
    )
    results["checkpoint"] = args.checkpoint
    results["split"] = args.split
    results["match_threshold"] = args.match_threshold
    results["active_threshold"] = args.active_threshold

    # --- Active Threshold Sweep ---
    if args.sweep_active_threshold:
        print("\n[Eval-v2] Sweeping active threshold...")
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        sweep_results = []
        best_f1 = -1.0
        best_thresh = 0.5
        for thresh in thresholds:
            r = evaluate_v2(model, loader, device, args.max_joints, args.match_threshold, active_threshold=thresh)
            f1 = r.get("node_f1_mean", 0.0) or 0.0
            prec = r.get("node_precision_mean", 0.0) or 0.0
            rec = r.get("node_recall_mean", 0.0) or 0.0
            n_pred = r.get("pred_count_mean", 0.0) or 0.0
            n_gt = r.get("gt_count_mean", 0.0) or 0.0
            cnt_ratio = r.get("count_ratio_mean", 0.0) or 0.0
            unp = r.get("underprediction_ratio_mean", 0.0) or 0.0
            ovp = r.get("overprediction_ratio_mean", 0.0) or 0.0
            edge_f1 = r.get("edge_f1_matched_mean", 0.0) or 0.0
            bone_err = r.get("bone_length_ratio_mean_mean", 0.0) or 0.0
            sweep_results.append({
                "threshold": thresh, "node_f1": round(f1, 4),
                "node_precision": round(prec, 4), "node_recall": round(rec, 4),
                "pred_count": round(n_pred, 1), "gt_count": round(n_gt, 1),
                "count_ratio": round(cnt_ratio, 3),
                "underprediction_ratio": round(unp, 3),
                "overprediction_ratio": round(ovp, 3),
                "edge_f1": round(edge_f1, 4), "bone_length_ratio": round(bone_err, 3),
            })
            print(f"  thresh={thresh:.1f}: F1={f1:.4f} prec={prec:.4f} rec={rec:.4f} "
                  f"pred={n_pred:.1f} gt={n_gt:.1f} cnt={cnt_ratio:.3f} "
                  f"unp={unp:.3f} ovp={ovp:.3f} edge_f1={edge_f1:.4f} bone={bone_err:.3f}")
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        results["threshold_sweep"] = sweep_results
        results["best_active_threshold"] = best_thresh
        results["best_node_f1"] = round(best_f1, 4)
        print(f"\n  Best threshold: {best_thresh} (F1={best_f1:.4f})")

        # Re-run eval at best threshold if different from 0.5
        if best_thresh != 0.5:
            print(f"  Re-evaluating at threshold={best_thresh}...")
            decode_config.active_threshold = best_thresh
            results = evaluate_v2(
                model, loader, device, args.max_joints, args.match_threshold,
                active_threshold=best_thresh, decode_config=decode_config
            )
            results["checkpoint"] = args.checkpoint
            results["split"] = args.split
            results["match_threshold"] = args.match_threshold
            results["active_threshold"] = best_thresh
            results["threshold_sweep"] = sweep_results
            results["best_active_threshold"] = best_thresh
            results["best_node_f1"] = round(best_f1, 4)

    # Save (without per-sample for the summary)
    summary = {k: v for k, v in results.items() if k != "per_sample"}
    with open(out_dir / "eval_metrics_v2.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print
    print(f"\n{'='*70}")
    print(f"[Eval-v2] Results on {args.split} split ({results['n_samples']} assets)")
    print(f"{'='*70}")
    print(f"  --- Position ---")
    print(f"  MPJPE (matched, mean):      {results.get('mpjpe_matched_mean', 'N/A')}")
    print(f"  MPJPE (matched, median):    {results.get('mpjpe_matched_median_median', 'N/A')}")
    print(f"  MPJPE (same-slot, mean):    {results.get('mpjpe_same_slot_mean', 'N/A')}")
    print(f"  Chamfer (mean):             {results.get('chamfer_mean', 'N/A')}")
    print(f"  --- Spread ---")
    print(f"  GT spread (mean):           {results.get('gt_spread_mean', 'N/A')}")
    print(f"  Pred spread (mean):         {results.get('pred_spread_mean', 'N/A')}")
    print(f"  Spread collapse score:      {results.get('spread_collapse_score_mean', 'N/A')}")
    print(f"  --- Node Detection ---")
    print(f"  Node precision:             {results.get('node_precision_mean', 'N/A')}")
    print(f"  Node recall:                {results.get('node_recall_mean', 'N/A')}")
    print(f"  Node F1:                    {results.get('node_f1_mean', 'N/A')}")
    print(f"  Coverage (within {args.match_threshold}):   {results.get('coverage_mean', 'N/A')}")
    print(f"  Pred count:                 {results.get('pred_count_mean', 'N/A')}")
    print(f"  GT count:                   {results.get('gt_count_mean', 'N/A')}")
    print(f"  Count ratio:                {results.get('count_ratio_mean', 'N/A')}")
    print(f"  Underprediction ratio:      {results.get('underprediction_ratio_mean', 'N/A')}")
    print(f"  Overprediction ratio:       {results.get('overprediction_ratio_mean', 'N/A')}")
    print(f"  Active count abs error:     {results.get('active_count_abs_error_mean', 'N/A')}")
    print(f"  --- Edges (after matching) ---")
    print(f"  Edge F1 (matched):          {results.get('edge_f1_matched_mean', 'N/A')}")
    print(f"  Edge precision (matched):   {results.get('edge_precision_matched_mean', 'N/A')}")
    print(f"  Edge recall (matched):      {results.get('edge_recall_matched_mean', 'N/A')}")
    print(f"  Pred edge count:            {results.get('pred_edge_count_mean', 'N/A')}")
    print(f"  GT edge count:              {results.get('gt_edge_count_mean', 'N/A')}")
    print(f"  Component count:            {results.get('component_count_mean', 'N/A')}")
    print(f"  Connected ratio:            {results.get('connected_ratio_mean', 'N/A')}")
    print(f"  Average degree:             {results.get('average_degree_mean', 'N/A')}")
    print(f"  Max degree:                 {results.get('max_degree_mean', 'N/A')}")
    print(f"  --- Bone Geometry ---")
    print(f"  Bone length ratio (mean):   {results.get('bone_length_ratio_mean_mean', 'N/A')}")
    print(f"  Bone length error (abs):    {results.get('bone_length_error_abs_mean', 'N/A')}")
    print(f"  Bone ratio all pred edges:   {results.get('bone_length_ratio_all_pred_edges_mean', 'N/A')}")
    print(f"  Bone ratio TP edges:         {results.get('bone_length_ratio_true_positive_edges_mean', 'N/A')}")
    print(f"  Bone ratio FP edges:         {results.get('bone_length_ratio_false_positive_edges_mean', 'N/A')}")
    print(f"{'='*70}")
    print(f"  Saved: {out_dir / 'eval_metrics_v2.json'}")


if __name__ == "__main__":
    main()
