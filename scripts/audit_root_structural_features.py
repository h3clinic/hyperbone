from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.datasets.anymate_static_dataset import AnymateStaticRigDataset
from hyperbone.rigs.parent_candidates import build_parent_candidates
from hyperbone.rigs.root_features import (
    ROOT_STRUCTURAL_FEATURE_NAMES,
    compute_root_structural_features,
)


def _tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    if x.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
        "p10": float(torch.quantile(x, 0.10).item()),
        "p50": float(torch.quantile(x, 0.50).item()),
        "p90": float(torch.quantile(x, 0.90).item()),
    }


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2 or y.numel() < 2:
        return 0.0
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.pow(2).mean()) * (y.pow(2).mean())).clamp(min=1e-8)
    return float((x * y).mean().item() / denom.item())


def _auc_like(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    """Rank-based AUC estimate in [0,1]."""
    n_pos = int(pos_scores.numel())
    n_neg = int(neg_scores.numel())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    scores = torch.cat([pos_scores.float(), neg_scores.float()], dim=0)
    labels = torch.cat([
        torch.ones(n_pos, dtype=torch.float32),
        torch.zeros(n_neg, dtype=torch.float32),
    ], dim=0)

    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    sum_ranks_pos = float(ranks[labels > 0.5].sum().item())
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))


def audit_split(loader: DataLoader, k: int) -> Dict:
    all_feat = []
    all_root = []

    for batch in loader:
        joint_pos = batch["joint_pos"]
        active = batch["joint_active"] > 0.5
        root = batch["root_mask"] > 0.5

        cand = build_parent_candidates(
            joint_pos,
            active,
            batch["parent_index"].long(),
            k=k,
            include_root=False,
            force_gt_parent=False,
        )

        feat = compute_root_structural_features(
            joint_pos,
            active,
            candidate_indices=cand["candidate_indices"],
            candidate_mask=cand["candidate_mask"],
            pair_logits=None,
            k=k,
        )

        valid = active
        if valid.any():
            all_feat.append(feat[valid])
            all_root.append(root[valid].float())

    if not all_feat:
        return {
            "n_active_nodes": 0,
            "feature_names": ROOT_STRUCTURAL_FEATURE_NAMES,
            "root_mean": {},
            "nonroot_mean": {},
            "root_std": {},
            "nonroot_std": {},
            "auc_like": {},
            "corr_with_root": {},
            "top_abs_corr": [],
            "candidate_indegree_distribution": {},
            "local_density_distribution": {},
        }

    feat_all = torch.cat(all_feat, dim=0)
    root_all = torch.cat(all_root, dim=0)

    root_mask = root_all > 0.5
    nonroot_mask = ~root_mask

    root_mean = {}
    nonroot_mean = {}
    root_std = {}
    nonroot_std = {}
    auc_like = {}
    corr = {}

    for i, name in enumerate(ROOT_STRUCTURAL_FEATURE_NAMES):
        vals = feat_all[:, i]
        vals_root = vals[root_mask]
        vals_nonroot = vals[nonroot_mask]

        root_mean[name] = float(vals_root.mean().item()) if vals_root.numel() > 0 else 0.0
        nonroot_mean[name] = float(vals_nonroot.mean().item()) if vals_nonroot.numel() > 0 else 0.0
        root_std[name] = float(vals_root.std(unbiased=False).item()) if vals_root.numel() > 1 else 0.0
        nonroot_std[name] = float(vals_nonroot.std(unbiased=False).item()) if vals_nonroot.numel() > 1 else 0.0
        auc_like[name] = _auc_like(vals_root, vals_nonroot)
        corr[name] = _pearson_corr(vals, root_all)

    top_abs_corr = sorted(
        [{"feature": n, "corr": c, "abs_corr": abs(c)} for n, c in corr.items()],
        key=lambda x: x["abs_corr"],
        reverse=True,
    )[:10]

    indegree_idx = ROOT_STRUCTURAL_FEATURE_NAMES.index("knn_indegree")
    density_idx = ROOT_STRUCTURAL_FEATURE_NAMES.index("local_density_mean_knn_dist")

    indegree_root = feat_all[root_mask, indegree_idx]
    indegree_nonroot = feat_all[nonroot_mask, indegree_idx]
    density_root = feat_all[root_mask, density_idx]
    density_nonroot = feat_all[nonroot_mask, density_idx]

    return {
        "n_active_nodes": int(feat_all.shape[0]),
        "feature_names": ROOT_STRUCTURAL_FEATURE_NAMES,
        "root_mean": root_mean,
        "nonroot_mean": nonroot_mean,
        "root_std": root_std,
        "nonroot_std": nonroot_std,
        "auc_like": auc_like,
        "corr_with_root": corr,
        "top_abs_corr": top_abs_corr,
        "candidate_indegree_distribution": {
            "root": _tensor_stats(indegree_root),
            "nonroot": _tensor_stats(indegree_nonroot),
        },
        "local_density_distribution": {
            "root": _tensor_stats(density_root),
            "nonroot": _tensor_stats(density_nonroot),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit structural root features on train/val/test splits")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument("--points-per-sample", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument(
        "--out",
        default="outputs/models/hyperbone_anymate_static_v2.14b_root_feature_audit",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in ["train", "val", "test"]:
        ds = AnymateStaticRigDataset(
            args.pt,
            f"{args.splits_dir}/{split}.jsonl",
            max_joints=args.max_nodes,
            pc_points=args.points_per_sample,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
        report = audit_split(loader, k=args.k)
        summary[split] = report
        (out_dir / f"audit_{split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[audit] {split}: active_nodes={report['n_active_nodes']} top_abs_corr={report['top_abs_corr'][:3]}")

    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[audit] wrote reports to {out_dir}")


if __name__ == "__main__":
    main()
