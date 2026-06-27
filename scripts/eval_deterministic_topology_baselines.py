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
from hyperbone.rigs.topology_optimizers import OPTIMIZER_REGISTRY
from hyperbone.rigs.undirected_topology import build_undirected_adjacency, edge_prf, graph_stats


def _edge_count(edge_mask: torch.Tensor) -> int:
    tri = torch.triu(torch.ones_like(edge_mask, dtype=torch.bool), diagonal=1)
    return int((edge_mask & tri).sum().item())


def evaluate_split(
    ds: AnymateStaticRigDataset,
    method: str,
    candidate_k: int,
    batch_size: int,
    train_subset: int | None,
    degree_cap: int,
    budget_ratio: float,
    branch_extra_ratio: float,
) -> Dict[str, float]:
    if train_subset is not None and train_subset > 0 and len(ds) > train_subset:
        ds_eval = Subset(ds, list(range(train_subset)))
    else:
        ds_eval = ds

    loader = DataLoader(ds_eval, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    optimize_fn = OPTIMIZER_REGISTRY[method]

    metrics = {
        "edge_precision": 0.0,
        "edge_recall": 0.0,
        "edge_f1": 0.0,
        "component_count": 0.0,
        "average_degree": 0.0,
        "max_degree": 0.0,
        "cycle_count": 0.0,
        "selected_edge_count": 0.0,
        "gt_edge_count": 0.0,
        "selected_to_gt_ratio": 0.0,
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

            if method == "degree_capped_mst":
                pred_b = optimize_fn(pos_b, active_b, k=candidate_k, degree_cap=degree_cap)
            elif method == "budgeted_knn_forest":
                pred_b = optimize_fn(pos_b, active_b, k=candidate_k, budget_ratio=budget_ratio)
            elif method == "hybrid_mst_plus_branches":
                pred_b = optimize_fn(pos_b, active_b, k=candidate_k, extra_ratio=branch_extra_ratio)
            else:
                pred_b = optimize_fn(pos_b, active_b, k=candidate_k)

            prf = edge_prf(pred_b, gt_b, active_b)
            stats = graph_stats(pred_b, active_b)

            gt_edges = float(_edge_count(gt_b))
            sel_edges = float(_edge_count(pred_b))

            metrics["edge_precision"] += prf["precision"]
            metrics["edge_recall"] += prf["recall"]
            metrics["edge_f1"] += prf["f1"]
            metrics["component_count"] += stats["component_count"]
            metrics["average_degree"] += stats["average_degree"]
            metrics["max_degree"] += stats["max_degree"]
            metrics["cycle_count"] += stats["cycle_count"]
            metrics["selected_edge_count"] += sel_edges
            metrics["gt_edge_count"] += gt_edges
            metrics["selected_to_gt_ratio"] += sel_edges / max(gt_edges, 1.0)
            n += 1

    if n == 0:
        return metrics
    return {k: v / n for k, v in metrics.items()}


def _format_md_table(rows: List[Dict[str, object]]) -> str:
    header = "| split | method | k | edge_f1 | precision | recall | sel/gt | comp | avg_deg | max_deg | cycles |"
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            "| {split} | {method} | {k} | {edge_f1:.4f} | {edge_precision:.4f} | {edge_recall:.4f} | {selected_to_gt_ratio:.3f} | {component_count:.2f} | {average_degree:.2f} | {max_degree:.2f} | {cycle_count:.2f} |".format(
                **r
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="HyperBone v3.0 deterministic constrained topology baseline eval")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument("--points-per-sample", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--candidate-k-list", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "knn_mst",
            "degree_capped_mst",
            "budgeted_knn_forest",
            "density_normalized_mst",
            "mutual_knn_sparse",
            "hybrid_mst_plus_branches",
        ],
    )
    parser.add_argument("--train-subset", type=int, default=200)
    parser.add_argument("--degree-cap", type=int, default=3)
    parser.add_argument("--budget-ratio", type=float, default=1.0)
    parser.add_argument("--branch-extra-ratio", type=float, default=0.25)
    parser.add_argument(
        "--out-dir",
        default="outputs/models/hyperbone_v3_deterministic_topology_baselines",
    )
    args = parser.parse_args()

    unknown = [m for m in args.methods if m not in OPTIMIZER_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_names = ["train", "val", "test"]
    datasets = {
        split: AnymateStaticRigDataset(
            args.pt,
            f"{args.splits_dir}/{split}.jsonl",
            max_joints=args.max_nodes,
            pc_points=args.points_per_sample,
        )
        for split in split_names
    }

    rows: List[Dict[str, object]] = []
    raw_report = {
        "config": {
            "candidate_k_list": args.candidate_k_list,
            "methods": args.methods,
            "train_subset": args.train_subset,
            "degree_cap": args.degree_cap,
            "budget_ratio": args.budget_ratio,
            "branch_extra_ratio": args.branch_extra_ratio,
        },
        "results": {},
    }

    for split in split_names:
        raw_report["results"][split] = {}
        for method in args.methods:
            raw_report["results"][split][method] = {}
            for k in args.candidate_k_list:
                train_subset = args.train_subset if split == "train" else None
                metrics = evaluate_split(
                    ds=datasets[split],
                    method=method,
                    candidate_k=int(k),
                    batch_size=args.batch_size,
                    train_subset=train_subset,
                    degree_cap=args.degree_cap,
                    budget_ratio=args.budget_ratio,
                    branch_extra_ratio=args.branch_extra_ratio,
                )
                raw_report["results"][split][method][str(k)] = metrics
                row = {
                    "split": split,
                    "method": method,
                    "k": int(k),
                    **metrics,
                }
                rows.append(row)
                print(
                    "[{split}] {method} k={k}: f1={f1:.4f} p={p:.4f} r={r:.4f} sel/gt={sg:.3f}".format(
                        split=split,
                        method=method,
                        k=int(k),
                        f1=metrics["edge_f1"],
                        p=metrics["edge_precision"],
                        r=metrics["edge_recall"],
                        sg=metrics["selected_to_gt_ratio"],
                    )
                )

    best_rows = []
    for split in split_names:
        candidates = [r for r in rows if r["split"] == split]
        best = max(candidates, key=lambda x: float(x["edge_f1"])) if candidates else None
        if best is not None:
            best_rows.append(best)

    report_json_path = out_dir / "report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "best_by_split": best_rows, **raw_report}, f, indent=2)

    md_parts = [
        "# HyperBone v3.0 Deterministic Topology Baselines",
        "",
        "## Best by Split (edge_f1)",
        _format_md_table(best_rows),
        "",
        "## Full Results",
        _format_md_table(rows),
        "",
        "## Notes",
        "- Train split uses the first N samples controlled by --train-subset (default 200).",
        "- All methods are deterministic and run from GT joints/active masks only.",
    ]
    report_md_path = out_dir / "report.md"
    report_md_path.write_text("\n".join(md_parts), encoding="utf-8")

    print(f"Saved {report_json_path}")
    print(f"Saved {report_md_path}")


if __name__ == "__main__":
    main()
