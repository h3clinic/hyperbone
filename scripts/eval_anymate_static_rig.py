"""
Evaluate HyperBone static rig model on Anymate local dev test split.

Reports:
- MPJPE (normalized)
- Joint active accuracy
- Edge F1 / precision / recall
- Bone length error
- Per-asset breakdown
- Comparison to pilot model (if available)

Usage:
    python scripts/eval_anymate_static_rig.py
    python scripts/eval_anymate_static_rig.py --checkpoint outputs/models/hyperbone_anymate_local_dev/best_model.pt
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
from hyperbone.models.hyperbone_rig_graph_static import HyperBoneStaticRigModel


@torch.no_grad()
def evaluate(model, loader, device, max_joints):
    model.eval()

    all_mpjpe = []
    all_active_acc = []
    all_edge_tp = 0
    all_edge_fp = 0
    all_edge_fn = 0
    all_bone_len_errors = []
    per_asset_mpjpe = []
    n_joints_hist = []

    for batch in loader:
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        pred = model(batch_dev)

        B = batch_dev["joint_pos"].shape[0]
        gt_pos = batch_dev["joint_pos"]
        gt_active = batch_dev["joint_active"]
        pred_pos = pred["joint_pos"]
        pred_active = torch.sigmoid(pred["active_logits"])
        pred_adj = torch.sigmoid(pred["adj_logits"])
        gt_adj = batch_dev["adj_matrix"]

        for b in range(B):
            active_mask = gt_active[b] > 0.5
            n_active = active_mask.sum().item()
            n_joints_hist.append(n_active)

            if n_active == 0:
                continue

            # MPJPE per asset
            diff = (pred_pos[b, active_mask] - gt_pos[b, active_mask]).norm(dim=-1)
            mpjpe = diff.mean().item()
            all_mpjpe.append(mpjpe)
            per_asset_mpjpe.append({"n_joints": n_active, "mpjpe": mpjpe})

            # Active accuracy
            pred_binary = (pred_active[b] > 0.5).float()
            acc = (pred_binary == gt_active[b]).float().mean().item()
            all_active_acc.append(acc)

            # Edge metrics (upper triangle, active nodes only)
            active_2d = active_mask.unsqueeze(-1) & active_mask.unsqueeze(0)
            triu = torch.triu(torch.ones(max_joints, max_joints, device=device), diagonal=1).bool()
            mask = active_2d & triu

            pred_edges = (pred_adj[b] > 0.5) & mask
            gt_edges = (gt_adj[b] > 0.5) & mask

            tp = (pred_edges & gt_edges).sum().item()
            fp = (pred_edges & ~gt_edges).sum().item()
            fn = (~pred_edges & gt_edges).sum().item()
            all_edge_tp += tp
            all_edge_fp += fp
            all_edge_fn += fn

            # Bone length error from adjacency
            gt_edge_pairs = gt_edges.nonzero(as_tuple=False)
            if gt_edge_pairs.shape[0] > 0:
                for e in gt_edge_pairs:
                    i, j = e[0].item(), e[1].item()
                    gt_len = (gt_pos[b, i] - gt_pos[b, j]).norm().item()
                    pred_len = (pred_pos[b, i] - pred_pos[b, j]).norm().item()
                    if gt_len > 1e-4:
                        rel_err = abs(pred_len - gt_len) / gt_len
                        all_bone_len_errors.append(rel_err)

    # Aggregate
    edge_precision = all_edge_tp / max(all_edge_tp + all_edge_fp, 1)
    edge_recall = all_edge_tp / max(all_edge_tp + all_edge_fn, 1)
    edge_f1 = 2 * edge_precision * edge_recall / max(edge_precision + edge_recall, 1e-8)

    results = {
        "n_samples": len(all_mpjpe),
        "mpjpe_mean": round(float(np.mean(all_mpjpe)), 6) if all_mpjpe else 0,
        "mpjpe_median": round(float(np.median(all_mpjpe)), 6) if all_mpjpe else 0,
        "mpjpe_std": round(float(np.std(all_mpjpe)), 6) if all_mpjpe else 0,
        "active_acc_mean": round(float(np.mean(all_active_acc)), 4) if all_active_acc else 0,
        "edge_precision": round(edge_precision, 4),
        "edge_recall": round(edge_recall, 4),
        "edge_f1": round(edge_f1, 4),
        "edge_tp": all_edge_tp,
        "edge_fp": all_edge_fp,
        "edge_fn": all_edge_fn,
        "bone_length_error_mean": round(float(np.mean(all_bone_len_errors)), 4) if all_bone_len_errors else 0,
        "bone_length_error_median": round(float(np.median(all_bone_len_errors)), 4) if all_bone_len_errors else 0,
        "n_joints_mean": round(float(np.mean(n_joints_hist)), 1),
        "n_joints_min": int(np.min(n_joints_hist)) if n_joints_hist else 0,
        "n_joints_max": int(np.max(n_joints_hist)) if n_joints_hist else 0,
    }

    # MPJPE by joint count bucket
    buckets = {"1-10": [], "11-20": [], "21-40": [], "41-64": []}
    for item in per_asset_mpjpe:
        n = item["n_joints"]
        if n <= 10:
            buckets["1-10"].append(item["mpjpe"])
        elif n <= 20:
            buckets["11-20"].append(item["mpjpe"])
        elif n <= 40:
            buckets["21-40"].append(item["mpjpe"])
        else:
            buckets["41-64"].append(item["mpjpe"])

    results["mpjpe_by_complexity"] = {
        k: {"count": len(v), "mean": round(float(np.mean(v)), 4) if v else 0}
        for k, v in buckets.items()
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate static rig model")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--checkpoint", default="outputs/models/hyperbone_anymate_local_dev/best_model.pt")
    parser.add_argument("--out", default="outputs/models/hyperbone_anymate_local_dev/eval_test")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-joints", type=int, default=64)
    parser.add_argument("--pc-points", type=int, default=2048)
    parser.add_argument("--feat-dim", type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")
    print(f"[Eval] Checkpoint: {args.checkpoint}")
    print(f"[Eval] Split: {args.split}")

    # Dataset
    split_path = f"{args.splits_dir}/{args.split}.jsonl"
    ds = AnymateStaticRigDataset(
        args.pt, split_path,
        max_joints=args.max_joints, pc_points=args.pc_points,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[Eval] Samples: {len(ds)}")

    # Model
    model = HyperBoneStaticRigModel(
        in_channels=3, feat_dim=args.feat_dim,
        max_joints=args.max_joints, predict_skinning=True,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"[Eval] Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # Evaluate
    results = evaluate(model, loader, device, args.max_joints)
    results["checkpoint"] = args.checkpoint
    results["split"] = args.split
    results["dataset"] = "anymate_local_dev (NOT official benchmark)"

    # Save
    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print
    print(f"\n{'='*60}")
    print(f"[Eval] Results on {args.split} split ({results['n_samples']} assets)")
    print(f"{'='*60}")
    print(f"  MPJPE (mean):       {results['mpjpe_mean']:.4f}")
    print(f"  MPJPE (median):     {results['mpjpe_median']:.4f}")
    print(f"  Active accuracy:    {results['active_acc_mean']:.4f}")
    print(f"  Edge F1:            {results['edge_f1']:.4f}")
    print(f"  Edge precision:     {results['edge_precision']:.4f}")
    print(f"  Edge recall:        {results['edge_recall']:.4f}")
    print(f"  Bone length error:  {results['bone_length_error_mean']:.4f} (mean rel)")
    print(f"  Joint count:        {results['n_joints_min']}–{results['n_joints_max']} (avg {results['n_joints_mean']})")
    print(f"\n  MPJPE by complexity:")
    for bucket, stats in results["mpjpe_by_complexity"].items():
        print(f"    {bucket} joints: {stats['mean']:.4f} ({stats['count']} assets)")
    print(f"{'='*60}")
    print(f"  Saved: {out_dir / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()
