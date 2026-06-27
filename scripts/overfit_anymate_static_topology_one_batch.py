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
from hyperbone.models.topology_edge_scorer import TopologyEdgeScorer
from hyperbone.rigs.undirected_topology import EdgeScoreRefiner
from hyperbone.rigs.undirected_topology import decode_undirected_edges, edge_prf
from scripts.train_anymate_static_rig_topology import compute_topology_loss


@torch.no_grad()
def compute_score_diagnostics(
    sym_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    gt_adj: torch.Tensor,
    active_mask: torch.Tensor,
    hard_neg_k: int,
    thr_min: float,
    thr_max: float,
    thr_step: float,
) -> dict:
    tri = torch.triu(torch.ones_like(candidate_mask, dtype=torch.bool), diagonal=1)
    valid = candidate_mask & tri
    pos_mask = valid & gt_adj
    neg_mask = valid & (~gt_adj)

    pos_scores = sym_scores[pos_mask]
    neg_scores = sym_scores[neg_mask]

    hard_neg_parts = []
    for b in range(sym_scores.shape[0]):
        neg_b = sym_scores[b][neg_mask[b]]
        if neg_b.numel() == 0:
            continue
        k = min(int(hard_neg_k), int(neg_b.numel()))
        if k <= 0:
            continue
        hard_neg_parts.append(torch.topk(neg_b, k=k, largest=True).values)
    hard_neg_scores = torch.cat(hard_neg_parts, dim=0) if hard_neg_parts else torch.empty(0, device=sym_scores.device)

    pos_mean = float(pos_scores.mean().item()) if pos_scores.numel() > 0 else 0.0
    pos_std = float(pos_scores.std(unbiased=False).item()) if pos_scores.numel() > 1 else 0.0
    neg_mean = float(neg_scores.mean().item()) if neg_scores.numel() > 0 else 0.0
    neg_std = float(neg_scores.std(unbiased=False).item()) if neg_scores.numel() > 1 else 0.0
    hard_neg_mean = float(hard_neg_scores.mean().item()) if hard_neg_scores.numel() > 0 else 0.0
    hard_neg_std = float(hard_neg_scores.std(unbiased=False).item()) if hard_neg_scores.numel() > 1 else 0.0

    if pos_scores.numel() > 0 and neg_scores.numel() > 0:
        pos_median = float(pos_scores.median().item())
        overlap_rate = float((neg_scores > pos_median).float().mean().item())
    else:
        overlap_rate = 1.0

    thresholds = []
    t = thr_min
    while t <= thr_max + 1e-8:
        thresholds.append(round(float(t), 6))
        t += thr_step

    best_thr = thresholds[0]
    best_thr_f1 = -1.0
    best_sel = 0.0
    best_gt = 0.0
    for thr in thresholds:
        f1_sum = 0.0
        sel_sum = 0.0
        gt_sum = 0.0
        n = 0
        for b in range(sym_scores.shape[0]):
            pred_thr = decode_undirected_edges(
                sym_scores[b],
                candidate_mask[b],
                active_mask[b],
                mode="threshold",
                threshold=float(thr),
            )
            prf = edge_prf(pred_thr, gt_adj[b], active_mask[b])
            f1_sum += prf["f1"]
            sel_sum += float((pred_thr & torch.triu(torch.ones_like(pred_thr, dtype=torch.bool), diagonal=1)).sum().item())
            gt_sum += float((gt_adj[b] & torch.triu(torch.ones_like(gt_adj[b], dtype=torch.bool), diagonal=1)).sum().item())
            n += 1

        f1_avg = f1_sum / max(n, 1)
        if f1_avg > best_thr_f1:
            best_thr_f1 = f1_avg
            best_thr = float(thr)
            best_sel = sel_sum / max(n, 1)
            best_gt = gt_sum / max(n, 1)

    return {
        "positive_score_mean": pos_mean,
        "positive_score_std": pos_std,
        "negative_score_mean": neg_mean,
        "negative_score_std": neg_std,
        "hard_negative_score_mean": hard_neg_mean,
        "hard_negative_score_std": hard_neg_std,
        "overlap_rate": overlap_rate,
        "best_threshold": best_thr,
        "best_threshold_f1": float(best_thr_f1),
        "best_threshold_selected_edge_count": float(best_sel),
        "best_threshold_gt_edge_count": float(best_gt),
    }


def main():
    parser = argparse.ArgumentParser(description="Overfit one batch for v2.15b undirected topology")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--out", default="outputs/models/hyperbone_anymate_static_v2.15b_topology_overfit")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument("--points-per-sample", type=int, default=1024)
    parser.add_argument("--feat-dim", type=int, default=512)
    parser.add_argument("--backbone", choices=["pointnet", "dgcnn"], default="dgcnn")
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--parent-head", choices=["pairwise"], default="pairwise")
    parser.add_argument("--no-backbone-for-gt-nodes", action="store_true")
    parser.add_argument("--root-feature-mode", choices=["none", "structural"], default="structural")

    parser.add_argument("--candidate-parent-k", type=int, default=12)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--force-gt-edge-candidate", action="store_true")
    parser.add_argument("--edge-symmetrize", choices=["average", "max"], default="average")
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--edge-pos-weight-cap", type=float, default=8.0)
    parser.add_argument("--edge-rank-loss-weight", type=float, default=2.0)
    parser.add_argument("--edge-rank-margin", type=float, default=1.0)
    parser.add_argument("--edge-hard-neg-k", type=int, default=8)
    parser.add_argument("--hard-neg-k", type=int, default=None)
    parser.add_argument("--edge-hard-neg-weight", type=float, default=1.0)
    parser.add_argument("--topology-scorer", choices=["refiner", "edge_mlp"], default="edge_mlp")
    parser.add_argument("--degree-loss-weight", type=float, default=0.0)
    parser.add_argument("--sweep-threshold-min", type=float, default=0.05)
    parser.add_argument("--sweep-threshold-max", type=float, default=0.95)
    parser.add_argument("--sweep-threshold-step", type=float, default=0.05)
    args = parser.parse_args()

    if args.candidate_k is not None:
        args.candidate_parent_k = args.candidate_k
    if args.hard_neg_k is not None:
        args.edge_hard_neg_k = args.hard_neg_k

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = AnymateStaticRigDataset(
        args.pt,
        f"{args.splits_dir}/train.jsonl",
        max_joints=args.max_nodes,
        pc_points=args.points_per_sample,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    fixed_batch = next(iter(loader))
    fixed_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in fixed_batch.items()}

    model = HyperBoneStaticParentModel(
        in_channels=3,
        feat_dim=args.feat_dim,
        max_joints=args.max_nodes,
        predict_skinning=False,
        backbone=args.backbone,
        knn_k=args.knn_k,
        parent_head=args.parent_head,
        root_feature_mode=args.root_feature_mode,
    ).to(device)
    if args.topology_scorer == "edge_mlp":
        topology_scorer = TopologyEdgeScorer(
            node_feature_dim=256,
            edge_feature_dim=23,
            node_local_dim=8,
            global_context_dim=4,
            hidden_dim=256,
            dropout=0.1,
            num_blocks=3,
        ).to(device)
    else:
        topology_scorer = EdgeScoreRefiner(feature_dim=12, hidden_dim=64).to(device)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(topology_scorer.parameters()), lr=args.lr, weight_decay=1e-4)

    best = {
        "step": 0,
        "loss": float("inf"),
        "edge_f1_top_e": 0.0,
        "edge_f1_threshold": 0.0,
        "candidate_coverage": 0.0,
        "positive_score_mean": 0.0,
        "positive_score_std": 0.0,
        "negative_score_mean": 0.0,
        "negative_score_std": 0.0,
        "hard_negative_score_mean": 0.0,
        "hard_negative_score_std": 0.0,
        "overlap_rate": 1.0,
        "best_threshold": 0.5,
        "best_threshold_f1": 0.0,
        "best_threshold_selected_edge_count": 0.0,
        "best_threshold_gt_edge_count": 0.0,
    }

    print(f"[Overfit-topology] Device: {device}")
    print(f"[Overfit-topology] Steps: {args.steps} batch_size={args.batch_size}")

    for step in range(1, args.steps + 1):
        model.train()
        topology_scorer.train()
        pred = model.forward_parent_from_joints(
            fixed_batch["joint_pos"],
            active_mask=fixed_batch["joint_active"] > 0.5,
            no_backbone_for_gt_nodes=args.no_backbone_for_gt_nodes,
        )
        if pred.get("parent_pair_logits") is None:
            raise ValueError("v2.15b overfit requires --parent-head pairwise")

        loss, metrics, sym_scores, candidate_mask, gt_adj = compute_topology_loss(
            pred["parent_pair_logits"],
            topology_scorer,
            fixed_batch,
            args,
            force_include_gt=args.force_gt_edge_candidate,
            node_tokens=pred.get("node_tokens"),
        )
        score_diag = compute_score_diagnostics(
            sym_scores,
            candidate_mask,
            gt_adj,
            fixed_batch["joint_active"] > 0.5,
            hard_neg_k=args.edge_hard_neg_k,
            thr_min=args.sweep_threshold_min,
            thr_max=args.sweep_threshold_max,
            thr_step=args.sweep_threshold_step,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(topology_scorer.parameters()), 1.0)
        optimizer.step()

        if metrics["edge_f1_top_e"] > best["edge_f1_top_e"]:
            best = {
                "step": step,
                "loss": float(loss.detach().item()),
                "edge_f1_top_e": metrics["edge_f1_top_e"],
                "edge_f1_threshold": metrics["edge_f1_threshold"],
                "candidate_coverage": metrics["candidate_coverage"],
                **score_diag,
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "topology_scorer_type": args.topology_scorer,
                    "topology_scorer_state_dict": topology_scorer.state_dict(),
                    "best": best,
                },
                out_dir / "best_model.pt",
            )

        if step % 100 == 0 or step == 1:
            print(
                f"S{step:04d} | loss={loss.item():.4f} | "
                f"thr_f1={metrics['edge_f1_threshold']:.3f} topE_f1={metrics['edge_f1_top_e']:.3f} "
                f"deg={metrics['degree_loss']:.3f} "
                f"best_thr={score_diag['best_threshold']:.2f} best_thr_f1={score_diag['best_threshold_f1']:.3f} "
                f"ovlp={score_diag['overlap_rate']:.3f} cov={metrics['candidate_coverage']:.3f}"
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "topology_scorer_type": args.topology_scorer,
            "topology_scorer_state_dict": topology_scorer.state_dict(),
            "best": best,
        },
        out_dir / "model_final.pt",
    )

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "best": best,
            "pass_top_e": best["edge_f1_top_e"] >= 0.95,
            "pass_threshold": best["edge_f1_threshold"] >= 0.85,
            "pass_best_threshold": best["best_threshold_f1"] >= 0.90,
            "pass_coverage": best["candidate_coverage"] >= 0.95,
        }, f, indent=2)

    print("[Overfit-topology] DONE")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
