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
from hyperbone.rigs.undirected_topology import (
    EdgeScoreRefiner,
    build_undirected_adjacency,
    build_undirected_knn_candidates,
    compute_edge_pair_features,
    compute_topology_edge_inputs,
    decode_undirected_edges,
    edge_prf,
    graph_stats,
    symmetrize_pair_scores,
)


def bone_ratio(pred_pos: torch.Tensor, gt_pos: torch.Tensor, pred_edges: torch.Tensor) -> float:
    idx = torch.where(torch.triu(pred_edges, diagonal=1))
    if idx[0].numel() == 0:
        return 0.0
    ratios = []
    for i, j in zip(idx[0].tolist(), idx[1].tolist()):
        ref = float((gt_pos[i] - gt_pos[j]).norm().item())
        pred = float((pred_pos[i] - pred_pos[j]).norm().item())
        if ref > 1e-6:
            ratios.append(pred / ref)
    if not ratios:
        return 0.0
    return float(sum(ratios) / len(ratios))


def evaluate(model, topology_scorer, loader, device, args):
    model.eval()
    topology_scorer.eval()

    metrics = {
        "edge_precision": 0.0,
        "edge_recall": 0.0,
        "edge_f1": 0.0,
        "threshold_edge_f1": 0.0,
        "top_e_edge_f1": 0.0,
        "candidate_coverage": 0.0,
        "component_count": 0.0,
        "average_degree": 0.0,
        "max_degree": 0.0,
        "cycle_count": 0.0,
        "bone_length_ratio": 0.0,
        "selected_edge_count": 0.0,
        "gt_edge_count": 0.0,
        "selected_to_gt_ratio": 0.0,
    }
    n = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            if args.teacher_force_gt_nodes:
                pred = model.forward_parent_from_joints(
                    batch["joint_pos"],
                    active_mask=batch["joint_active"] > 0.5,
                    no_backbone_for_gt_nodes=args.no_backbone_for_gt_nodes,
                )
                pred_pos = batch["joint_pos"]
                active_mask = batch["joint_active"] > 0.5
            else:
                pred = model(batch)
                pred_pos = pred["joint_pos"]
                active_mask = torch.sigmoid(pred["active_logits"]) > args.active_threshold

            pair_logits = pred.get("parent_pair_logits")
            if pair_logits is None:
                raise ValueError("v2.15 topology eval requires --parent-head pairwise")

            gt_adj = build_undirected_adjacency(batch["adj_matrix"], active_mask)
            cand_info = build_undirected_knn_candidates(
                batch["joint_pos"],
                active_mask,
                k=args.candidate_parent_k,
                gt_adj=gt_adj,
                force_include_gt=args.force_gt_edge_candidate,
            )
            candidate_mask = cand_info["candidate_mask"]
            base_sym_scores = symmetrize_pair_scores(pair_logits, mode=args.edge_symmetrize)
            if args.topology_scorer == "edge_mlp":
                node_tokens = pred.get("node_tokens")
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

            calibrated_scores = (sym_scores + float(args.edge_logit_bias)) / max(float(args.edge_temperature), 1e-6)

            for b in range(batch["joint_pos"].shape[0]):
                active_b = active_mask[b]
                gt_b = gt_adj[b]
                cand_b = candidate_mask[b]
                scores_b = calibrated_scores[b]

                gt_edge_count = int((gt_b & torch.triu(torch.ones_like(gt_b, dtype=torch.bool), diagonal=1)).sum().item())

                pred_threshold = decode_undirected_edges(
                    scores_b,
                    cand_b,
                    active_b,
                    mode="threshold",
                    threshold=args.edge_threshold,
                )
                pred_top_e = decode_undirected_edges(
                    scores_b,
                    cand_b,
                    active_b,
                    mode="top_e_budget",
                    budget_count=gt_edge_count,
                )

                if args.edge_decode_mode == "threshold":
                    pred_mode = pred_threshold
                elif args.edge_decode_mode == "top_e_budget":
                    pred_mode = pred_top_e
                elif args.edge_decode_mode == "ratio_budget":
                    ratio = args.edge_budget_ratio
                    if ratio is None:
                        ratio = gt_edge_count / max(int(active_b.sum().item()), 1)
                    pred_mode = decode_undirected_edges(
                        scores_b,
                        cand_b,
                        active_b,
                        mode="ratio_budget",
                        budget_ratio=ratio,
                    )
                elif args.edge_decode_mode == "mst_forest":
                    pred_mode = decode_undirected_edges(
                        scores_b,
                        cand_b,
                        active_b,
                        mode="mst_forest",
                    )
                else:
                    raise ValueError(f"Unknown edge decode mode: {args.edge_decode_mode}")

                prf_mode = edge_prf(pred_mode, gt_b, active_b)
                prf_thr = edge_prf(pred_threshold, gt_b, active_b)
                prf_top = edge_prf(pred_top_e, gt_b, active_b)
                stats = graph_stats(pred_mode, active_b)

                metrics["edge_precision"] += prf_mode["precision"]
                metrics["edge_recall"] += prf_mode["recall"]
                metrics["edge_f1"] += prf_mode["f1"]
                metrics["threshold_edge_f1"] += prf_thr["f1"]
                metrics["top_e_edge_f1"] += prf_top["f1"]
                metrics["candidate_coverage"] += float(cand_info["candidate_coverage"].item())
                metrics["component_count"] += stats["component_count"]
                metrics["average_degree"] += stats["average_degree"]
                metrics["max_degree"] += stats["max_degree"]
                metrics["cycle_count"] += stats["cycle_count"]
                metrics["bone_length_ratio"] += bone_ratio(pred_pos[b], batch["joint_pos"][b], pred_mode)
                selected_count = float((pred_mode & torch.triu(torch.ones_like(pred_mode, dtype=torch.bool), diagonal=1)).sum().item())
                metrics["selected_edge_count"] += selected_count
                metrics["gt_edge_count"] += float(gt_edge_count)
                metrics["selected_to_gt_ratio"] += selected_count / max(float(gt_edge_count), 1.0)
                n += 1

    return {k: v / max(n, 1) for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate HyperBone v2.15 root-free undirected topology")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--checkpoint", "--ckpt", dest="ckpt", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument("--points-per-sample", type=int, default=1024)
    parser.add_argument("--feat-dim", type=int, default=512)
    parser.add_argument("--backbone", choices=["pointnet", "dgcnn"], default="dgcnn")
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--active-threshold", type=float, default=0.7)
    parser.add_argument("--parent-head", choices=["pairwise"], default="pairwise")
    parser.add_argument("--teacher-force-gt-nodes", action="store_true")
    parser.add_argument("--no-backbone-for-gt-nodes", action="store_true")
    parser.add_argument("--root-feature-mode", choices=["none", "structural"], default="structural")

    parser.add_argument("--candidate-parent-k", type=int, default=12)
    parser.add_argument("--candidate-k", type=int, default=None,
                        help="Alias for --candidate-parent-k.")
    parser.add_argument("--force-gt-edge-candidate", action="store_true")
    parser.add_argument("--edge-symmetrize", choices=["average", "max"], default="average")
    parser.add_argument("--decode-mode", type=str, default=None,
                        help="Alias for --edge-decode-mode.")
    parser.add_argument("--edge-decode-mode", choices=["threshold", "top_e_budget", "ratio_budget", "mst_forest"], default="threshold")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Alias for --edge-threshold.")
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--edge-logit-bias", type=float, default=0.0)
    parser.add_argument("--edge-temperature", type=float, default=1.0)
    parser.add_argument("--edge-budget-ratio", type=float, default=None)
    parser.add_argument("--topology-scorer", choices=["refiner", "edge_mlp"], default="edge_mlp")
    parser.add_argument("--out", "--output", dest="output", default=None)
    args = parser.parse_args()
    if args.candidate_k is not None:
        args.candidate_parent_k = args.candidate_k
    if args.decode_mode is not None:
        args.edge_decode_mode = args.decode_mode
    if args.threshold is not None:
        args.edge_threshold = args.threshold
    if args.edge_temperature <= 0.0:
        raise ValueError("--edge-temperature must be > 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = AnymateStaticRigDataset(
        args.pt,
        f"{args.splits_dir}/{args.split}.jsonl",
        max_joints=args.max_nodes,
        pc_points=args.points_per_sample,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

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

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
        scorer_state = state.get("topology_scorer_state_dict", state.get("edge_refiner_state_dict"))
        if scorer_state is not None:
            topology_scorer.load_state_dict(scorer_state, strict=False)
    else:
        model.load_state_dict(state, strict=False)

    metrics = evaluate(model, topology_scorer, loader, device, args)
    print(json.dumps(metrics, indent=2))

    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
