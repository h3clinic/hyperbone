"""Root ranking diagnostics for v2.14b.1 collapse audit.

Computes:
- ROC-AUC and PR-AUC over all test nodes
- Precision/recall/F1 at thresholds 0.05–0.95
- Budgeted top-k root accuracy (select exactly GT root count per sample)
- Proportional top-k (select round(gt_ratio * active_count) per sample)
- Logit distribution stats for GT roots vs non-roots
- Overlap rate: % non-root logits above median GT-root logit
- Ambiguous coincident nodes excluded and reported
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.datasets.anymate_static_dataset import AnymateStaticRigDataset
from hyperbone.models.hyperbone_static_parent import HyperBoneStaticParentModel
from hyperbone.rigs.parent_candidates import build_parent_candidates
from hyperbone.rigs.root_features import compute_root_structural_features


# ---------------------------------------------------------------------------
# AUC helpers (no sklearn dependency)
# ---------------------------------------------------------------------------

def _roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    n_pos = int(labels.sum().item())
    n_neg = int((1 - labels).sum().item())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = torch.argsort(scores, descending=True)
    labels_s = labels[order].float()
    tp = fp = 0.0
    fpr_prev = tpr_prev = 0.0
    auc = 0.0
    for lab in labels_s.tolist():
        if lab > 0.5:
            tp += 1.0
        else:
            fp += 1.0
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
        fpr_prev = fpr
        tpr_prev = tpr
    return float(auc)


def _pr_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    n_pos = int(labels.sum().item())
    if n_pos == 0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    labels_s = labels[order].float()
    tp = 0.0
    rec_prev = 0.0
    prec_prev = 1.0
    auc = 0.0
    for i, lab in enumerate(labels_s.tolist()):
        if lab > 0.5:
            tp += 1.0
        prec = tp / (i + 1)
        rec = tp / n_pos
        auc += (rec - rec_prev) * (prec + prec_prev) / 2.0
        rec_prev = rec
        prec_prev = prec
    return float(auc)


def _threshold_metrics(scores: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict:
    pred = (scores > threshold).float()
    tp = float((pred * labels).sum().item())
    fp = float((pred * (1 - labels)).sum().item())
    fn = float(((1 - pred) * labels).sum().item())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"threshold": threshold, "precision": prec, "recall": rec, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "pred_root_ratio": float(pred.mean().item())}


# ---------------------------------------------------------------------------
# Per-sample top-k budgeted decode
# ---------------------------------------------------------------------------

def _budgeted_topk_sample(logits_active: torch.Tensor, gt_root_active: torch.Tensor, k: int) -> dict:
    """Select exactly k nodes by logit rank, compute TP against GT roots."""
    n = logits_active.numel()
    if n == 0 or k == 0:
        return {"k": k, "tp": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    k = min(k, n)
    topk_idx = torch.argsort(logits_active, descending=True)[:k]
    pred_set = set(topk_idx.tolist())
    gt_set = set(torch.where(gt_root_active)[0].tolist())
    tp = len(pred_set & gt_set)
    prec = tp / max(k, 1)
    rec = tp / max(len(gt_set), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"k": k, "tp": tp, "precision": prec, "recall": rec, "f1": f1}


# ---------------------------------------------------------------------------
# Ambiguity exclusion (mirrors overfit script)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _ambiguous_mask_single(pos: torch.Tensor, active: torch.Tensor, gt_root: torch.Tensor, eps: float) -> torch.Tensor:
    """Return [N] bool mask of ambiguous nodes for one sample."""
    N = pos.shape[0]
    ambiguous = torch.zeros(N, dtype=torch.bool)
    valid = torch.where(active)[0]
    if valid.numel() < 2 or eps <= 0.0:
        return ambiguous
    pos_v = pos[valid]
    dmat = torch.cdist(pos_v.float(), pos_v.float())
    root_v = gt_root[valid]
    conflict = root_v.unsqueeze(0) != root_v.unsqueeze(1)
    coincident = (dmat < eps) & conflict
    coincident.fill_diagonal_(False)
    ambig_local = coincident.any(dim=1)
    ambiguous[valid[ambig_local]] = True
    return ambiguous


# ---------------------------------------------------------------------------
# Main diagnostic loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_diagnostics(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = AnymateStaticRigDataset(
        args.pt,
        f"{args.splits_dir}/{args.split}.jsonl",
        max_joints=args.max_nodes,
        pc_points=args.pc_points,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = HyperBoneStaticParentModel(
        in_channels=3,
        feat_dim=512,
        max_joints=args.max_nodes,
        predict_skinning=False,
        backbone=args.backbone,
        knn_k=args.knn_k,
        parent_head=args.parent_head,
        root_bias_init=getattr(args, "root_bias_init", None),
        root_feature_mode=args.root_feature_mode,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Accumulators over the full split (CPU tensors).
    all_logits = []     # [N_total]
    all_gt_root = []    # [N_total]
    all_active = []     # bool [N_total]
    all_ambiguous = []  # bool [N_total]

    # Per-sample budgeted results
    budgeted_results_gt = []     # select k = GT root count
    budgeted_results_prop = []   # select k = round(gt_ratio * active_count)

    n_samples = 0
    total_ambiguous_nodes = 0

    for batch in loader:
        batch_cpu = batch
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        if args.teacher_force_gt_nodes:
            pred = model.forward_parent_from_joints(
                batch["joint_pos"],
                active_mask=batch["joint_active"] > 0.5,
                no_backbone_for_gt_nodes=args.no_backbone_for_gt_nodes,
            )
        else:
            pred = model(batch)

        root_logits = pred.get("root_logits")
        if root_logits is None:
            raise ValueError("Checkpoint has no root_logits — ensure --parent-head pairwise")

        root_logits_cpu = root_logits.detach().cpu()
        active_cpu = (batch_cpu["joint_active"] > 0.5)
        gt_root_cpu = (batch_cpu["root_mask"] > 0.5)

        B = root_logits_cpu.shape[0]
        for b in range(B):
            n_samples += 1
            act = active_cpu[b]
            gt = gt_root_cpu[b]
            rl = root_logits_cpu[b]

            ambig = _ambiguous_mask_single(
                batch_cpu["joint_pos"][b], act, gt, eps=args.root_ambiguity_eps
            )

            eval_mask = act & ~ambig
            total_ambiguous_nodes += int(ambig.sum().item())

            active_idx = torch.where(eval_mask)[0]
            if active_idx.numel() == 0:
                continue

            all_logits.append(rl[active_idx])
            all_gt_root.append(gt[active_idx].float())
            all_active.append(torch.ones(active_idx.numel(), dtype=torch.bool))
            all_ambiguous.append(torch.zeros(active_idx.numel(), dtype=torch.bool))

            # Budgeted top-k with GT count.
            gt_count = int(gt[eval_mask].sum().item())
            budgeted_results_gt.append(
                _budgeted_topk_sample(rl[active_idx], gt[active_idx], k=gt_count)
            )

            # Proportional top-k.
            n_active = int(eval_mask.sum().item())
            gt_ratio = gt_count / max(n_active, 1)
            k_prop = max(1, round(gt_ratio * n_active))
            budgeted_results_prop.append(
                _budgeted_topk_sample(rl[active_idx], gt[active_idx], k=k_prop)
            )

    if not all_logits:
        return {"error": "no_data"}

    logits_all = torch.cat(all_logits, dim=0)
    gt_all = torch.cat(all_gt_root, dim=0)

    roc = _roc_auc(logits_all, gt_all)
    pr = _pr_auc(logits_all, gt_all)

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                  0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    prob_all = torch.sigmoid(logits_all)
    threshold_metrics = [_threshold_metrics(prob_all, gt_all, t) for t in thresholds]

    # Logit distribution.
    root_logits_vals = logits_all[gt_all > 0.5]
    nonroot_logits_vals = logits_all[gt_all < 0.5]
    root_stats = {
        "count": int(root_logits_vals.numel()),
        "mean": float(root_logits_vals.mean().item()) if root_logits_vals.numel() > 0 else 0.0,
        "std": float(root_logits_vals.std(unbiased=False).item()) if root_logits_vals.numel() > 1 else 0.0,
        "min": float(root_logits_vals.min().item()) if root_logits_vals.numel() > 0 else 0.0,
        "p10": float(torch.quantile(root_logits_vals, 0.10).item()) if root_logits_vals.numel() > 0 else 0.0,
        "p50": float(torch.quantile(root_logits_vals, 0.50).item()) if root_logits_vals.numel() > 0 else 0.0,
        "p90": float(torch.quantile(root_logits_vals, 0.90).item()) if root_logits_vals.numel() > 0 else 0.0,
        "max": float(root_logits_vals.max().item()) if root_logits_vals.numel() > 0 else 0.0,
    }
    nonroot_stats = {
        "count": int(nonroot_logits_vals.numel()),
        "mean": float(nonroot_logits_vals.mean().item()) if nonroot_logits_vals.numel() > 0 else 0.0,
        "std": float(nonroot_logits_vals.std(unbiased=False).item()) if nonroot_logits_vals.numel() > 1 else 0.0,
        "min": float(nonroot_logits_vals.min().item()) if nonroot_logits_vals.numel() > 0 else 0.0,
        "p10": float(torch.quantile(nonroot_logits_vals, 0.10).item()) if nonroot_logits_vals.numel() > 0 else 0.0,
        "p50": float(torch.quantile(nonroot_logits_vals, 0.50).item()) if nonroot_logits_vals.numel() > 0 else 0.0,
        "p90": float(torch.quantile(nonroot_logits_vals, 0.90).item()) if nonroot_logits_vals.numel() > 0 else 0.0,
        "max": float(nonroot_logits_vals.max().item()) if nonroot_logits_vals.numel() > 0 else 0.0,
    }

    # Logit margin: mean(root logit) - mean(nonroot logit)
    logit_margin = root_stats["mean"] - nonroot_stats["mean"]

    # Overlap: fraction of non-root logits above median GT-root logit
    overlap_rate = 0.0
    if root_logits_vals.numel() > 0 and nonroot_logits_vals.numel() > 0:
        root_median = float(torch.quantile(root_logits_vals, 0.50).item())
        overlap_rate = float((nonroot_logits_vals > root_median).float().mean().item())

    # Aggregate budgeted top-k stats
    def _agg_budgeted(results):
        if not results:
            return {}
        prec = sum(r["precision"] for r in results) / len(results)
        rec = sum(r["recall"] for r in results) / len(results)
        f1 = sum(r["f1"] for r in results) / len(results)
        tp = sum(r["tp"] for r in results)
        return {"precision": prec, "recall": rec, "f1": f1, "total_tp": tp, "n_samples": len(results)}

    return {
        "n_samples": n_samples,
        "total_active_eval_nodes": int(logits_all.numel()),
        "total_ambiguous_excluded": total_ambiguous_nodes,
        "roc_auc": roc,
        "pr_auc": pr,
        "logit_margin": logit_margin,
        "overlap_rate": overlap_rate,
        "root_logit_stats": root_stats,
        "nonroot_logit_stats": nonroot_stats,
        "threshold_metrics": threshold_metrics,
        "budgeted_topk_gt_count": _agg_budgeted(budgeted_results_gt),
        "budgeted_topk_proportional": _agg_budgeted(budgeted_results_prop),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Root ranking diagnostics for v2.14b.1")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument("--points-per-sample", "--pc-points", type=int, default=1024, dest="pc_points")
    parser.add_argument("--backbone", choices=["pointnet", "dgcnn"], default="dgcnn")
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--parent-head", choices=["slot", "pairwise"], default="pairwise")
    parser.add_argument("--root-bias-init", type=float, default=None)
    parser.add_argument("--teacher-force-gt-nodes", action="store_true")
    parser.add_argument("--no-backbone-for-gt-nodes", action="store_true")
    parser.add_argument("--root-feature-mode", choices=["none", "structural"], default="none")
    parser.add_argument("--root-ambiguity-eps", type=float, default=0.0)
    parser.add_argument("--out", default="outputs/models/root_diagnostics")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[root-diag] Running on split={args.split} checkpoint={args.checkpoint}")
    results = run_diagnostics(args)

    out_path = out_dir / f"root_diagnostics_{args.split}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({
        "roc_auc": results.get("roc_auc"),
        "pr_auc": results.get("pr_auc"),
        "logit_margin": results.get("logit_margin"),
        "overlap_rate": results.get("overlap_rate"),
        "budgeted_topk_gt_count": results.get("budgeted_topk_gt_count"),
        "root_logit_stats": {k: round(v, 3) for k, v in results.get("root_logit_stats", {}).items() if isinstance(v, float)},
        "nonroot_logit_stats": {k: round(v, 3) for k, v in results.get("nonroot_logit_stats", {}).items() if isinstance(v, float)},
        "total_ambiguous_excluded": results.get("total_ambiguous_excluded"),
    }, indent=2))
    print(f"[root-diag] Full results: {out_path}")


if __name__ == "__main__":
    main()
