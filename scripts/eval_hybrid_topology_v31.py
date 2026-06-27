from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.datasets.anymate_static_dataset import AnymateStaticRigDataset
from hyperbone.models.hyperbone_static_parent import HyperBoneStaticParentModel
from hyperbone.models.topology_edge_scorer import TopologyEdgeScorer
from hyperbone.rigs.topology_optimizers import degree_capped_mst, density_normalized_mst, hybrid_mst_plus_branches, hybrid_neural_cost_optimize
from hyperbone.rigs.undirected_topology import (
    build_undirected_adjacency,
    build_undirected_knn_candidates,
    compute_topology_edge_inputs,
    decode_undirected_edges,
    edge_prf,
    graph_stats,
    symmetrize_pair_scores,
)


def _edge_count(edge_mask: torch.Tensor) -> int:
    tri = torch.triu(torch.ones_like(edge_mask, dtype=torch.bool), diagonal=1)
    return int((edge_mask & tri).sum().item())


def _load_v216_models(ckpt_path: str, device: torch.device, max_nodes: int, feat_dim: int, points_per_sample: int) -> tuple[HyperBoneStaticParentModel, TopologyEdgeScorer]:
    _ = points_per_sample
    model = HyperBoneStaticParentModel(
        in_channels=3,
        feat_dim=feat_dim,
        max_joints=max_nodes,
        predict_skinning=False,
        backbone="dgcnn",
        knn_k=16,
        parent_head="pairwise",
        root_feature_mode="structural",
    ).to(device)
    scorer = TopologyEdgeScorer(
        node_feature_dim=256,
        edge_feature_dim=23,
        node_local_dim=8,
        global_context_dim=4,
        hidden_dim=256,
        dropout=0.1,
        num_blocks=3,
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


def _avg(metrics: Dict[str, float], n: int) -> Dict[str, float]:
    if n <= 0:
        return metrics
    return {k: float(v) / float(n) for k, v in metrics.items()}


@torch.no_grad()
def eval_v216_decode(
    loader: DataLoader,
    model: HyperBoneStaticParentModel,
    scorer: TopologyEdgeScorer,
    device: torch.device,
    candidate_k: int,
    mode: str,
    threshold: float,
) -> Dict[str, float]:
    metrics = {
        "edge_precision": 0.0,
        "edge_recall": 0.0,
        "edge_f1": 0.0,
        "selected_to_gt_ratio": 0.0,
        "component_count": 0.0,
        "average_degree": 0.0,
        "max_degree": 0.0,
        "cycle_count": 0.0,
    }
    n = 0

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        active_mask = batch["joint_active"] > 0.5
        pred = model.forward_parent_from_joints(
            batch["joint_pos"],
            active_mask=active_mask,
            no_backbone_for_gt_nodes=True,
        )
        pair_logits = pred["parent_pair_logits"]
        node_tokens = pred["node_tokens"]
        gt_adj = build_undirected_adjacency(batch["adj_matrix"], active_mask)

        cand = build_undirected_knn_candidates(
            batch["joint_pos"],
            active_mask,
            k=candidate_k,
            gt_adj=gt_adj,
            force_include_gt=False,
        )
        candidate_mask = cand["candidate_mask"]
        base_scores = symmetrize_pair_scores(pair_logits, mode="average")
        topo_inputs = compute_topology_edge_inputs(batch["joint_pos"], active_mask, candidate_mask, k=candidate_k)
        scores = scorer(
            base_scores,
            topo_inputs["pair_features"],
            node_tokens,
            topo_inputs["node_local_features"],
            topo_inputs["global_context"],
        )

        for b in range(batch["joint_pos"].shape[0]):
            gt_b = gt_adj[b]
            active_b = active_mask[b]
            cand_b = candidate_mask[b]
            score_b = scores[b]
            gt_e = _edge_count(gt_b)

            if mode == "threshold":
                pred_b = decode_undirected_edges(
                    score_b,
                    cand_b,
                    active_b,
                    mode="threshold",
                    threshold=threshold,
                )
            elif mode == "top_e_budget":
                pred_b = decode_undirected_edges(
                    score_b,
                    cand_b,
                    active_b,
                    mode="top_e_budget",
                    budget_count=gt_e,
                )
            else:
                raise ValueError(mode)

            prf = edge_prf(pred_b, gt_b, active_b)
            stats = graph_stats(pred_b, active_b)
            sel_e = float(_edge_count(pred_b))

            metrics["edge_precision"] += prf["precision"]
            metrics["edge_recall"] += prf["recall"]
            metrics["edge_f1"] += prf["f1"]
            metrics["selected_to_gt_ratio"] += sel_e / max(float(gt_e), 1.0)
            metrics["component_count"] += stats["component_count"]
            metrics["average_degree"] += stats["average_degree"]
            metrics["max_degree"] += stats["max_degree"]
            metrics["cycle_count"] += stats["cycle_count"]
            n += 1

    return _avg(metrics, n)


def eval_deterministic_family(loader: DataLoader, method: str, k: int, max_degree: int) -> Dict[str, float]:
    metrics = {
        "edge_precision": 0.0,
        "edge_recall": 0.0,
        "edge_f1": 0.0,
        "selected_to_gt_ratio": 0.0,
        "component_count": 0.0,
        "average_degree": 0.0,
        "max_degree": 0.0,
        "cycle_count": 0.0,
    }
    n = 0

    for batch in loader:
        joint_pos = batch["joint_pos"]
        joint_active = batch["joint_active"] > 0.5
        gt_adj_raw = batch["adj_matrix"]

        for b in range(joint_pos.shape[0]):
            pos_b = joint_pos[b]
            active_b = joint_active[b]
            gt_b = build_undirected_adjacency(gt_adj_raw[b].unsqueeze(0), active_b.unsqueeze(0))[0]

            if method == "density_normalized_mst":
                pred_b = density_normalized_mst(pos_b, active_b, k=k)
            elif method == "degree_capped_mst":
                pred_b = degree_capped_mst(pos_b, active_b, k=k, degree_cap=max_degree)
            elif method == "hybrid_mst_plus_branches":
                pred_b = hybrid_mst_plus_branches(pos_b, active_b, k=k, extra_ratio=0.25)
            else:
                raise ValueError(method)

            prf = edge_prf(pred_b, gt_b, active_b)
            stats = graph_stats(pred_b, active_b)
            sel_e = float(_edge_count(pred_b))
            gt_e = float(_edge_count(gt_b))

            metrics["edge_precision"] += prf["precision"]
            metrics["edge_recall"] += prf["recall"]
            metrics["edge_f1"] += prf["f1"]
            metrics["selected_to_gt_ratio"] += sel_e / max(gt_e, 1.0)
            metrics["component_count"] += stats["component_count"]
            metrics["average_degree"] += stats["average_degree"]
            metrics["max_degree"] += stats["max_degree"]
            metrics["cycle_count"] += stats["cycle_count"]
            n += 1

    return _avg(metrics, n)


@torch.no_grad()
def eval_hybrid_v31(
    loader: DataLoader,
    model: HyperBoneStaticParentModel,
    scorer: TopologyEdgeScorer,
    device: torch.device,
    k: int,
    mode: str,
    neural_weight: float,
    distance_weight: float,
    max_degree: int,
    degree_penalty: float,
    long_edge_penalty: float,
) -> Dict[str, float]:
    metrics = {
        "edge_precision": 0.0,
        "edge_recall": 0.0,
        "edge_f1": 0.0,
        "selected_to_gt_ratio": 0.0,
        "component_count": 0.0,
        "average_degree": 0.0,
        "max_degree": 0.0,
        "cycle_count": 0.0,
    }
    n = 0

    for batch in loader:
        batch = {k0: v.to(device) if isinstance(v, torch.Tensor) else v for k0, v in batch.items()}
        joint_pos = batch["joint_pos"]
        active_mask = batch["joint_active"] > 0.5
        gt_adj = build_undirected_adjacency(batch["adj_matrix"], active_mask)

        pred = model.forward_parent_from_joints(
            joint_pos,
            active_mask=active_mask,
            no_backbone_for_gt_nodes=True,
        )
        pair_logits = pred["parent_pair_logits"]
        node_tokens = pred["node_tokens"]
        base_scores = symmetrize_pair_scores(pair_logits, mode="average")

        cand = build_undirected_knn_candidates(
            joint_pos,
            active_mask,
            k=k,
            gt_adj=gt_adj,
            force_include_gt=False,
        )
        candidate_mask = cand["candidate_mask"]
        topo_inputs = compute_topology_edge_inputs(joint_pos, active_mask, candidate_mask, k=k)
        neural_scores = scorer(
            base_scores,
            topo_inputs["pair_features"],
            node_tokens,
            topo_inputs["node_local_features"],
            topo_inputs["global_context"],
        )

        for b in range(joint_pos.shape[0]):
            pred_b = hybrid_neural_cost_optimize(
                joint_pos=joint_pos[b],
                active_mask=active_mask[b],
                neural_scores=neural_scores[b],
                mode=mode,
                k=k,
                distance_weight=distance_weight,
                neural_weight=neural_weight,
                degree_penalty=degree_penalty,
                long_edge_penalty=long_edge_penalty,
                mutual_bonus=0.2,
                max_degree=max_degree,
                branch_extra_ratio=0.25,
                branch_bonus=0.15,
                candidate_mask=candidate_mask[b],
            )
            gt_b = gt_adj[b]
            prf = edge_prf(pred_b, gt_b, active_mask[b])
            stats = graph_stats(pred_b, active_mask[b])

            sel_e = float(_edge_count(pred_b))
            gt_e = float(_edge_count(gt_b))
            metrics["edge_precision"] += prf["precision"]
            metrics["edge_recall"] += prf["recall"]
            metrics["edge_f1"] += prf["f1"]
            metrics["selected_to_gt_ratio"] += sel_e / max(gt_e, 1.0)
            metrics["component_count"] += stats["component_count"]
            metrics["average_degree"] += stats["average_degree"]
            metrics["max_degree"] += stats["max_degree"]
            metrics["cycle_count"] += stats["cycle_count"]
            n += 1

    return _avg(metrics, n)


def _table(rows: List[Dict[str, object]]) -> str:
    hdr = "| group | split | method | k | neural_w | dist_w | max_deg | edge_f1 | precision | recall | sel/gt | comp | avg_deg | max_deg_obs | cycles |"
    sep = "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [hdr, sep]
    for r in rows:
        lines.append(
            "| {group} | {split} | {method} | {k} | {neural_w:.2f} | {dist_w:.2f} | {max_degree} | {edge_f1:.4f} | {edge_precision:.4f} | {edge_recall:.4f} | {selected_to_gt_ratio:.3f} | {component_count:.2f} | {average_degree:.2f} | {max_degree_obs:.2f} | {cycle_count:.2f} |".format(
                **r
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="HyperBone v3.1 hybrid neural-cost topology optimizer sweep")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--ckpt", default="outputs/models/hyperbone_anymate_static_v2.16_topology_full/best_model.pt")
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument("--points-per-sample", type=int, default=1024)
    parser.add_argument("--feat-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--candidate-k-list", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--neural-weight-list", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--distance-weight-list", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--max-degree-list", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--hybrid-modes", nargs="+", default=["density_normalized_mst", "degree_capped_mst", "hybrid_mst_plus_branches"])
    parser.add_argument("--train-subset", type=int, default=200)
    parser.add_argument("--degree-penalty", type=float, default=0.05)
    parser.add_argument("--long-edge-penalty", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--out-dir", default="outputs/models/hyperbone_v3_1_hybrid_topology")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scorer = _load_v216_models(args.ckpt, device, args.max_nodes, args.feat_dim, args.points_per_sample)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_names = args.splits
    datasets = {
        split: AnymateStaticRigDataset(
            args.pt,
            f"{args.splits_dir}/{split}.jsonl",
            max_joints=args.max_nodes,
            pc_points=args.points_per_sample,
        )
        for split in split_names
    }

    loaders: Dict[str, DataLoader] = {}
    for split in split_names:
        ds = datasets[split]
        if split == "train" and args.train_subset > 0 and len(ds) > args.train_subset:
            ds = Subset(ds, list(range(args.train_subset)))
        loaders[split] = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    side_by_side_rows: List[Dict[str, object]] = []
    for split in split_names:
        loader = loaders[split]
        print(f"[side_by_side] split={split} v2.16_threshold", flush=True)

        r_thr = eval_v216_decode(loader, model, scorer, device, candidate_k=12, mode="threshold", threshold=args.threshold)
        side_by_side_rows.append({
            "group": "side_by_side",
            "split": split,
            "method": "v2.16_threshold",
            "k": 12,
            "neural_w": 1.0,
            "dist_w": 1.0,
            "max_degree": 0,
            "max_degree_obs": r_thr["max_degree"],
            **r_thr,
        })

        print(f"[side_by_side] split={split} v2.16_top_e", flush=True)
        r_top = eval_v216_decode(loader, model, scorer, device, candidate_k=12, mode="top_e_budget", threshold=args.threshold)
        side_by_side_rows.append({
            "group": "side_by_side",
            "split": split,
            "method": "v2.16_top_e",
            "k": 12,
            "neural_w": 1.0,
            "dist_w": 1.0,
            "max_degree": 0,
            "max_degree_obs": r_top["max_degree"],
            **r_top,
        })

        for method in ["density_normalized_mst", "degree_capped_mst", "hybrid_mst_plus_branches"]:
            print(f"[side_by_side] split={split} {method}", flush=True)
            r_det = eval_deterministic_family(loader, method=method, k=8, max_degree=3)
            side_by_side_rows.append({
                "group": "side_by_side",
                "split": split,
                "method": f"v3_{method}",
                "k": 8,
                "neural_w": 0.0,
                "dist_w": 1.0,
                "max_degree": 3,
                "max_degree_obs": r_det["max_degree"],
                **r_det,
            })

    side_path = out_dir / "side_by_side.json"
    with open(side_path, "w", encoding="utf-8") as f:
        json.dump({"rows": side_by_side_rows}, f, indent=2)

    side_md_path = out_dir / "side_by_side.md"
    side_md_path.write_text(
        "# HyperBone v3.1 Side-by-Side\n\n" + _table(side_by_side_rows) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {side_path}", flush=True)
    print(f"Saved {side_md_path}", flush=True)

    if args.skip_sweep:
        report_json_path = out_dir / "report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump({"side_by_side": side_by_side_rows, "hybrid_sweep": [], "best_by_split": []}, f, indent=2)
        report_md_path = out_dir / "report.md"
        report_md_path.write_text(
            "# HyperBone v3.1 Hybrid Neural-Cost Optimizer\n\n## Side-by-Side (Same Splits, Same Metrics)\n"
            + _table(side_by_side_rows)
            + "\n\n## Notes\n- Sweep skipped by --skip-sweep.\n",
            encoding="utf-8",
        )
        print(f"Saved {report_json_path}", flush=True)
        print(f"Saved {report_md_path}", flush=True)
        return

    sweep_rows: List[Dict[str, object]] = []
    iter_count = 0
    partial_path = out_dir / "hybrid_sweep_partial.json"
    for split in split_names:
        loader = loaders[split]
        for mode in args.hybrid_modes:
            for k in args.candidate_k_list:
                for nw in args.neural_weight_list:
                    for dw in args.distance_weight_list:
                        for md in args.max_degree_list:
                            r = eval_hybrid_v31(
                                loader=loader,
                                model=model,
                                scorer=scorer,
                                device=device,
                                k=int(k),
                                mode=mode,
                                neural_weight=float(nw),
                                distance_weight=float(dw),
                                max_degree=int(md),
                                degree_penalty=float(args.degree_penalty),
                                long_edge_penalty=float(args.long_edge_penalty),
                            )
                            row = {
                                "group": "v3.1_sweep",
                                "split": split,
                                "method": mode,
                                "k": int(k),
                                "neural_w": float(nw),
                                "dist_w": float(dw),
                                "max_degree": int(md),
                                "max_degree_obs": r["max_degree"],
                                **r,
                            }
                            sweep_rows.append(row)
                            iter_count += 1
                            print(
                                "[{split}] {mode} k={k} nw={nw} dw={dw} md={md}: f1={f1:.4f} sel/gt={sg:.3f} cycles={cy:.2f}".format(
                                    split=split,
                                    mode=mode,
                                    k=int(k),
                                    nw=float(nw),
                                    dw=float(dw),
                                    md=int(md),
                                    f1=r["edge_f1"],
                                    sg=r["selected_to_gt_ratio"],
                                    cy=r["cycle_count"],
                                )
                                ,
                                flush=True,
                            )
                            if iter_count % 20 == 0:
                                with open(partial_path, "w", encoding="utf-8") as f:
                                    json.dump({"rows": sweep_rows}, f, indent=2)
                                print(f"Saved partial {partial_path} (rows={iter_count})", flush=True)

    best_by_split = []
    for split in split_names:
        cands = [r for r in sweep_rows if r["split"] == split]
        if cands:
            best_by_split.append(max(cands, key=lambda x: float(x["edge_f1"])))

    sweep_path = out_dir / "hybrid_sweep.json"
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump({"rows": sweep_rows, "best_by_split": best_by_split}, f, indent=2)

    report_json_path = out_dir / "report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "candidate_k_list": args.candidate_k_list,
                    "neural_weight_list": args.neural_weight_list,
                    "distance_weight_list": args.distance_weight_list,
                    "max_degree_list": args.max_degree_list,
                    "hybrid_modes": args.hybrid_modes,
                    "degree_penalty": args.degree_penalty,
                    "long_edge_penalty": args.long_edge_penalty,
                },
                "side_by_side": side_by_side_rows,
                "hybrid_sweep": sweep_rows,
                "best_by_split": best_by_split,
            },
            f,
            indent=2,
        )

    md_parts = [
        "# HyperBone v3.1 Hybrid Neural-Cost Optimizer",
        "",
        "## Side-by-Side (Same Splits, Same Metrics)",
        _table(side_by_side_rows),
        "",
        "## Best v3.1 by Split",
        _table(best_by_split),
        "",
        "## Notes",
        "- v2.16 top-E is diagnostic (uses GT edge count budget).",
        "- Deterministic baselines are deployable constrained decoders.",
        "- v3.1 combines deterministic structural costs with neural edge logits.",
    ]
    report_md_path = out_dir / "report.md"
    report_md_path.write_text("\n".join(md_parts), encoding="utf-8")

    print(f"Saved {side_path}")
    print(f"Saved {sweep_path}")
    print(f"Saved {report_json_path}")
    print(f"Saved {report_md_path}")


if __name__ == "__main__":
    main()
