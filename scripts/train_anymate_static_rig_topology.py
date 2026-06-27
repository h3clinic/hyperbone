from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.datasets.anymate_static_dataset import AnymateStaticRigDataset
from hyperbone.models.hyperbone_static_parent import HyperBoneStaticParentModel
from hyperbone.models.topology_edge_scorer import TopologyEdgeScorer
from hyperbone.rigs.undirected_topology import (
    EdgeScoreRefiner,
    build_undirected_adjacency,
    build_undirected_knn_candidates,
    compute_edge_pair_features,
    compute_topology_edge_inputs,
    decode_undirected_edges,
    edge_prf,
    symmetrize_pair_scores,
)


def compute_topology_loss(
    pair_logits: torch.Tensor,
    topology_scorer: nn.Module,
    batch: Dict[str, torch.Tensor],
    args,
    force_include_gt: bool,
    node_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    active_mask = batch["joint_active"] > 0.5
    gt_adj = build_undirected_adjacency(batch["adj_matrix"], active_mask)
    cand_info = build_undirected_knn_candidates(
        batch["joint_pos"],
        active_mask,
        k=args.candidate_parent_k,
        gt_adj=gt_adj,
        force_include_gt=force_include_gt,
    )
    candidate_mask = cand_info["candidate_mask"]
    base_sym_scores = symmetrize_pair_scores(pair_logits, mode=args.edge_symmetrize)
    if args.topology_scorer == "edge_mlp":
        if node_tokens is None:
            raise ValueError("v2.16 edge_mlp scorer requires node tokens from model forward.")
        topo_inputs = compute_topology_edge_inputs(
            batch["joint_pos"],
            active_mask,
            candidate_mask,
            k=args.candidate_parent_k,
        )
        sym_scores = topology_scorer(
            base_sym_scores,
            topo_inputs["pair_features"],
            node_tokens,
            topo_inputs["node_local_features"],
            topo_inputs["global_context"],
        )
    else:
        edge_features = compute_edge_pair_features(
            batch["joint_pos"],
            active_mask,
            candidate_mask,
            k=args.candidate_parent_k,
        )
        sym_scores = topology_scorer(base_sym_scores, edge_features)

    tri = torch.triu(torch.ones_like(candidate_mask, dtype=torch.bool), diagonal=1)
    valid = candidate_mask & tri
    labels = gt_adj.float()

    pos_count = (labels * valid.float()).sum().clamp(min=1.0)
    neg_count = (((1.0 - labels) * valid.float()).sum()).clamp(min=1.0)
    pos_weight = (neg_count / pos_count).detach()
    if args.edge_pos_weight_cap > 0.0:
        pos_weight = torch.clamp(pos_weight, max=args.edge_pos_weight_cap)

    bce_raw = F.binary_cross_entropy_with_logits(
        sym_scores,
        labels,
        reduction="none",
        pos_weight=pos_weight,
    )
    valid_float = valid.float()
    bce_loss = (bce_raw * valid_float).sum() / valid_float.sum().clamp(min=1.0)

    # Edge-set hard-negative BCE weighting: emphasize top-k false candidate edges.
    hard_neg_bce_loss = torch.tensor(0.0, device=sym_scores.device)
    if args.edge_hard_neg_weight > 0.0:
        hard_neg_terms = []
        for b in range(sym_scores.shape[0]):
            neg_mask = valid[b] & (~gt_adj[b])
            if neg_mask.sum() == 0:
                continue
            neg_scores = sym_scores[b][neg_mask]
            hard_k = min(int(args.edge_hard_neg_k), int(neg_scores.numel()))
            if hard_k <= 0:
                continue
            top_ids = torch.topk(neg_scores, k=hard_k, largest=True).indices
            hard_neg = neg_scores[top_ids]
            hard_neg_terms.append(F.binary_cross_entropy_with_logits(hard_neg, torch.zeros_like(hard_neg)))
        if hard_neg_terms:
            hard_neg_bce_loss = torch.stack(hard_neg_terms).mean()

    rank_losses = []
    for b in range(sym_scores.shape[0]):
        valid_b = valid[b]
        pos_mask = valid_b & gt_adj[b]
        neg_mask = valid_b & (~gt_adj[b])
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue
        score_b = sym_scores[b]
        pos_idx = torch.where(pos_mask)
        pos_scores = score_b[pos_idx]

        # Endpoint hard negatives: for each positive edge (i,j), search false candidates touching i or j.
        endpoint_losses = []
        for i, j, pos_score in zip(pos_idx[0].tolist(), pos_idx[1].tolist(), pos_scores):
            touch = torch.zeros_like(valid_b)
            touch[i, :] = True
            touch[:, i] = True
            touch[j, :] = True
            touch[:, j] = True
            endpoint_mask = valid_b & (~gt_adj[b]) & touch
            endpoint_mask.fill_diagonal_(False)
            endpoint_candidates = score_b[endpoint_mask]
            if endpoint_candidates.numel() == 0:
                continue
            hard_k = min(int(args.edge_hard_neg_k), int(endpoint_candidates.numel()))
            hard_neg = torch.topk(endpoint_candidates, k=hard_k, largest=True).values
            margin_violation = args.edge_rank_margin - pos_score + hard_neg
            endpoint_losses.append(F.relu(margin_violation).pow(2).mean())

        if endpoint_losses:
            rank_losses.append(torch.stack(endpoint_losses).mean())

    edge_rank_loss = torch.stack(rank_losses).mean() if rank_losses else torch.tensor(0.0, device=sym_scores.device)
    edge_count_loss = torch.tensor(0.0, device=sym_scores.device)
    pred_edge_count_mean = 0.0
    gt_edge_count_mean = 0.0
    selected_to_gt_ratio = 0.0

    count_terms = []
    pred_counts = []
    gt_counts = []
    for b in range(sym_scores.shape[0]):
        valid_b = valid[b]
        pred_count_b = (torch.sigmoid(sym_scores[b]) * valid_b.float()).sum()
        gt_count_b = (gt_adj[b] & torch.triu(torch.ones_like(gt_adj[b], dtype=torch.bool), diagonal=1)).float().sum()
        pred_counts.append(float(pred_count_b.detach().item()))
        gt_counts.append(float(gt_count_b.detach().item()))
        if getattr(args, "edge_count_loss_weight", 0.0) > 0.0:
            count_terms.append(F.smooth_l1_loss(pred_count_b, gt_count_b))

    if count_terms:
        edge_count_loss = torch.stack(count_terms).mean()
    if pred_counts:
        pred_edge_count_mean = float(sum(pred_counts) / len(pred_counts))
    if gt_counts:
        gt_edge_count_mean = float(sum(gt_counts) / len(gt_counts))
        selected_to_gt_ratio = pred_edge_count_mean / max(gt_edge_count_mean, 1e-6)

    degree_loss = torch.tensor(0.0, device=sym_scores.device)
    if getattr(args, "degree_loss_weight", 0.0) > 0.0:
        valid_deg = candidate_mask & active_mask.unsqueeze(1) & active_mask.unsqueeze(2)
        valid_deg = valid_deg & (~torch.eye(valid_deg.shape[1], device=valid_deg.device, dtype=torch.bool).unsqueeze(0))
        soft_prob = torch.sigmoid(sym_scores) * valid_deg.float()
        pred_degree = soft_prob.sum(dim=-1)
        gt_degree = gt_adj.float().sum(dim=-1)
        active_float = active_mask.float()
        degree_raw = F.smooth_l1_loss(pred_degree, gt_degree, reduction="none")
        degree_loss = (degree_raw * active_float).sum() / active_float.sum().clamp(min=1.0)

    total = (
        bce_loss
        + args.edge_rank_loss_weight * edge_rank_loss
        + args.edge_hard_neg_weight * hard_neg_bce_loss
        + getattr(args, "edge_count_loss_weight", 0.0) * edge_count_loss
        + getattr(args, "degree_loss_weight", 0.0) * degree_loss
    )

    with torch.no_grad():
        thr_f1 = 0.0
        top_e_f1 = 0.0
        n_samples = 0
        for b in range(sym_scores.shape[0]):
            active_b = active_mask[b]
            gt_b = gt_adj[b]
            gt_edge_count = int((gt_b & torch.triu(torch.ones_like(gt_b, dtype=torch.bool), diagonal=1)).sum().item())

            pred_thr = decode_undirected_edges(
                sym_scores[b],
                candidate_mask[b],
                active_b,
                mode="threshold",
                threshold=args.edge_threshold,
            )
            pred_top = decode_undirected_edges(
                sym_scores[b],
                candidate_mask[b],
                active_b,
                mode="top_e_budget",
                budget_count=gt_edge_count,
            )
            thr_f1 += edge_prf(pred_thr, gt_b, active_b)["f1"]
            top_e_f1 += edge_prf(pred_top, gt_b, active_b)["f1"]
            n_samples += 1

    metrics = {
        "edge_bce_loss": float(bce_loss.detach().item()),
        "edge_rank_loss": float(edge_rank_loss.detach().item()),
        "edge_hard_neg_bce_loss": float(hard_neg_bce_loss.detach().item()),
        "edge_count_loss": float(edge_count_loss.detach().item()),
        "pred_edge_count": float(pred_edge_count_mean),
        "gt_edge_count": float(gt_edge_count_mean),
        "selected_to_gt_ratio": float(selected_to_gt_ratio),
        "degree_loss": float(degree_loss.detach().item()),
        "candidate_coverage": float(cand_info["candidate_coverage"].detach().item()),
        "edge_f1_threshold": float(thr_f1 / max(n_samples, 1)),
        "edge_f1_top_e": float(top_e_f1 / max(n_samples, 1)),
        "effective_pos_weight": float(pos_weight.item()),
    }
    return total, metrics, sym_scores, candidate_mask, gt_adj


def run_epoch(model, topology_scorer, loader, optimizer, device, args, train: bool) -> tuple[float, Dict[str, float]]:
    if train:
        model.train()
        topology_scorer.train()
    else:
        model.eval()
        topology_scorer.eval()

    total_loss = 0.0
    agg = {
        "edge_bce_loss": 0.0,
        "edge_rank_loss": 0.0,
        "edge_hard_neg_bce_loss": 0.0,
        "edge_count_loss": 0.0,
        "pred_edge_count": 0.0,
        "gt_edge_count": 0.0,
        "selected_to_gt_ratio": 0.0,
        "degree_loss": 0.0,
        "candidate_coverage": 0.0,
        "edge_f1_threshold": 0.0,
        "edge_f1_top_e": 0.0,
        "effective_pos_weight": 0.0,
    }
    n_batches = 0

    max_batches = args.max_batches if train else args.max_val_batches
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.set_grad_enabled(train):
            pred = model.forward_parent_from_joints(
                batch["joint_pos"],
                active_mask=batch["joint_active"] > 0.5,
                no_backbone_for_gt_nodes=args.no_backbone_for_gt_nodes,
            )
            if pred.get("parent_pair_logits") is None:
                raise ValueError("v2.15 topology training requires --parent-head pairwise")

            loss, metrics, _, _, _ = compute_topology_loss(
                pred["parent_pair_logits"],
                topology_scorer,
                batch,
                args,
                force_include_gt=args.force_gt_edge_candidate,
                node_tokens=pred.get("node_tokens"),
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(topology_scorer.parameters()), 1.0)
                optimizer.step()

        total_loss += float(loss.detach().item())
        for key in agg:
            agg[key] += metrics[key]
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {k: v / max(n_batches, 1) for k, v in agg.items()}
    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train HyperBone v2.15 root-free undirected topology model")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--out", default="outputs/models/hyperbone_anymate_static_v2.15_topology")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument("--points-per-sample", type=int, default=1024)
    parser.add_argument("--feat-dim", type=int, default=512)
    parser.add_argument("--backbone", choices=["pointnet", "dgcnn"], default="dgcnn")
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--parent-head", choices=["pairwise"], default="pairwise")
    parser.add_argument("--no-backbone-for-gt-nodes", action="store_true")
    parser.add_argument("--root-feature-mode", choices=["none", "structural"], default="structural")

    parser.add_argument("--candidate-parent-k", type=int, default=12)
    parser.add_argument("--candidate-k", type=int, default=None,
                        help="Alias for --candidate-parent-k.")
    parser.add_argument("--force-gt-edge-candidate", action="store_true")
    parser.add_argument("--edge-symmetrize", choices=["average", "max"], default="average")
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--edge-pos-weight-cap", type=float, default=8.0)
    parser.add_argument("--edge-rank-loss-weight", type=float, default=2.0)
    parser.add_argument("--edge-rank-margin", type=float, default=1.0)
    parser.add_argument("--edge-hard-neg-k", type=int, default=8)
    parser.add_argument("--hard-neg-k", type=int, default=None,
                        help="Alias for --edge-hard-neg-k.")
    parser.add_argument("--edge-hard-neg-weight", type=float, default=1.0)
    parser.add_argument("--edge-count-loss-weight", type=float, default=0.0)
    parser.add_argument("--topology-scorer", choices=["refiner", "edge_mlp"], default="edge_mlp")
    parser.add_argument("--degree-loss-weight", type=float, default=0.0)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Optional cap on train batches per epoch for smoke/debug runs.")
    parser.add_argument("--max-val-batches", type=int, default=None,
                        help="Optional cap on val batches per epoch for smoke/debug runs.")
    args = parser.parse_args()
    if args.candidate_k is not None:
        args.candidate_parent_k = args.candidate_k
    if args.hard_neg_k is not None:
        args.edge_hard_neg_k = args.hard_neg_k

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = AnymateStaticRigDataset(
        args.pt,
        f"{args.splits_dir}/train.jsonl",
        max_joints=args.max_nodes,
        pc_points=args.points_per_sample,
    )
    val_ds = AnymateStaticRigDataset(
        args.pt,
        f"{args.splits_dir}/val.jsonl",
        max_joints=args.max_nodes,
        pc_points=args.points_per_sample,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

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

    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
            scorer_state = state.get("topology_scorer_state_dict", state.get("edge_refiner_state_dict"))
            if scorer_state is not None:
                topology_scorer.load_state_dict(scorer_state, strict=False)
        else:
            model.load_state_dict(state, strict=False)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(topology_scorer.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val = float("inf")
    training_log = []

    print(f"[Train-topology] Device: {device}")
    print(f"[Train-topology] Train={len(train_ds)} Val={len(val_ds)}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_metrics = run_epoch(model, topology_scorer, train_loader, optimizer, device, args, train=True)
        val_loss, val_metrics = run_epoch(model, topology_scorer, val_loader, optimizer, device, args, train=False)
        scheduler.step()
        elapsed = time.time() - t0

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "topology_scorer_type": args.topology_scorer,
                    "topology_scorer_state_dict": topology_scorer.state_dict(),
                },
                out_dir / "best_model.pt",
            )

        if epoch % 5 == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "topology_scorer_type": args.topology_scorer,
                    "topology_scorer_state_dict": topology_scorer.state_dict(),
                },
                out_dir / f"checkpoint_e{epoch:03d}.pt",
            )

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_edge_bce_loss": round(train_metrics["edge_bce_loss"], 6),
            "val_edge_bce_loss": round(val_metrics["edge_bce_loss"], 6),
            "train_edge_rank_loss": round(train_metrics["edge_rank_loss"], 6),
            "val_edge_rank_loss": round(val_metrics["edge_rank_loss"], 6),
            "train_edge_hard_neg_bce_loss": round(train_metrics["edge_hard_neg_bce_loss"], 6),
            "val_edge_hard_neg_bce_loss": round(val_metrics["edge_hard_neg_bce_loss"], 6),
            "train_edge_count_loss": round(train_metrics["edge_count_loss"], 6),
            "val_edge_count_loss": round(val_metrics["edge_count_loss"], 6),
            "train_pred_edge_count": round(train_metrics["pred_edge_count"], 6),
            "val_pred_edge_count": round(val_metrics["pred_edge_count"], 6),
            "train_gt_edge_count": round(train_metrics["gt_edge_count"], 6),
            "val_gt_edge_count": round(val_metrics["gt_edge_count"], 6),
            "train_selected_to_gt_ratio": round(train_metrics["selected_to_gt_ratio"], 6),
            "val_selected_to_gt_ratio": round(val_metrics["selected_to_gt_ratio"], 6),
            "train_degree_loss": round(train_metrics["degree_loss"], 6),
            "val_degree_loss": round(val_metrics["degree_loss"], 6),
            "train_candidate_coverage": round(train_metrics["candidate_coverage"], 6),
            "val_candidate_coverage": round(val_metrics["candidate_coverage"], 6),
            "train_edge_f1_threshold": round(train_metrics["edge_f1_threshold"], 6),
            "val_edge_f1_threshold": round(val_metrics["edge_f1_threshold"], 6),
            "train_edge_f1_top_e": round(train_metrics["edge_f1_top_e"], 6),
            "val_edge_f1_top_e": round(val_metrics["edge_f1_top_e"], 6),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
            "time_s": round(elapsed, 1),
        }
        training_log.append(log_entry)

        print(
            f"E{epoch:03d} | loss={train_loss:.4f}/{val_loss:.4f} | "
            f"thr_f1={val_metrics['edge_f1_threshold']:.3f} topE_f1={val_metrics['edge_f1_top_e']:.3f} | "
            f"cov={val_metrics['candidate_coverage']:.3f} | {elapsed:.1f}s"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "topology_scorer_type": args.topology_scorer,
            "topology_scorer_state_dict": topology_scorer.state_dict(),
        },
        out_dir / "model_final.pt",
    )
    with open(out_dir / "training_log.jsonl", "w") as f:
        for row in training_log:
            f.write(json.dumps(row) + "\n")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "model": "HyperBoneStaticParentModel",
                "task": "v2.15_root_free_undirected_topology",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "topology_scorer": args.topology_scorer,
                "edge_count_loss_weight": args.edge_count_loss_weight,
                "degree_loss_weight": args.degree_loss_weight,
                "candidate_parent_k": args.candidate_parent_k,
                "best_val_loss": round(best_val, 6),
            },
            f,
            indent=2,
        )

    print("[Train-topology] DONE")
    print(f"  Best val loss: {best_val:.6f}")
    print(f"  Saved: {out_dir}")


if __name__ == "__main__":
    main()
