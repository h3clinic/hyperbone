from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.datasets.anymate_static_dataset import AnymateStaticRigDataset
from hyperbone.rigs.parent_decoder import ParentDecodeConfig, decode_parent_graph
from scripts.eval_anymate_static_rig_parent import edge_metrics, parent_edge_sets


def build_perfect_parent_logits(
    parent_index: np.ndarray,
    root_mask: np.ndarray,
    active_mask: np.ndarray,
    max_nodes: int,
) -> np.ndarray:
    root_class = max_nodes
    logits = np.full((max_nodes, max_nodes + 1), -20.0, dtype=np.float32)

    for child in range(max_nodes):
        if not active_mask[child]:
            logits[child, root_class] = 20.0
            continue

        p = int(parent_index[child])
        # root_mask is authoritative in identity convention checks.
        is_root = bool(root_mask[child]) or p < 0 or p >= max_nodes or p == child or (not active_mask[p] if 0 <= p < max_nodes else False)
        target = root_class if is_root else p
        logits[child, target] = 20.0

    return logits


def expected_parent_ptr(
    parent_index: np.ndarray,
    root_mask: np.ndarray,
    active_mask: np.ndarray,
    max_nodes: int,
) -> np.ndarray:
    out = np.full((max_nodes,), -1, dtype=np.int64)
    for child in range(max_nodes):
        if not active_mask[child]:
            continue
        p = int(parent_index[child])
        # root_mask is authoritative in identity convention checks.
        if bool(root_mask[child]) or p < 0 or p >= max_nodes or p == child or not active_mask[p]:
            out[child] = -1
        else:
            out[child] = p
    return out


def undirected_edge_set_from_parent(parent_ptr: np.ndarray, active_mask: np.ndarray) -> set[Tuple[int, int]]:
    edges: set[Tuple[int, int]] = set()
    for child, parent in enumerate(parent_ptr.tolist()):
        if parent >= 0 and active_mask[child] and active_mask[parent]:
            a, b = sorted((int(parent), int(child)))
            edges.add((a, b))
    return edges


def component_count(active_mask: np.ndarray, edges: set[Tuple[int, int]]) -> int:
    nodes = [i for i, a in enumerate(active_mask.tolist()) if a]
    if not nodes:
        return 0

    adj = {n: set() for n in nodes}
    for i, j in edges:
        if i in adj and j in adj:
            adj[i].add(j)
            adj[j].add(i)

    seen = set()
    comps = 0
    for n in nodes:
        if n in seen:
            continue
        comps += 1
        stack = [n]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(adj[cur] - seen)
    return comps


def evaluate_sample(sample: Dict[str, torch.Tensor], decode_cfg: ParentDecodeConfig, sample_key: str) -> Dict:
    max_nodes = int(sample["parent_index"].shape[0])
    root_class = max_nodes

    joint_pos = sample["joint_pos"].detach().cpu().numpy().astype(np.float32)
    active_mask = (sample["joint_active"].detach().cpu().numpy() > 0.5)
    parent_index = sample["parent_index"].detach().cpu().numpy().astype(np.int64)
    root_mask = (sample["root_mask"].detach().cpu().numpy() > 0.5)

    logits = build_perfect_parent_logits(parent_index, root_mask, active_mask, max_nodes)
    decoded = decode_parent_graph(
        positions=joint_pos,
        active_prob=active_mask.astype(np.float32),
        parent_logits=logits,
        parent_offset=np.zeros_like(joint_pos, dtype=np.float32),
        edge_confidence=np.ones((max_nodes,), dtype=np.float32),
        config=decode_cfg,
    )

    exp_parent = expected_parent_ptr(parent_index, root_mask, active_mask, max_nodes)
    dec_parent = decoded["parent_ptr"].astype(np.int64)
    argmax_class = logits.argmax(axis=-1).astype(np.int64)

    parent_total = 0
    parent_correct = 0
    nonroot_total = 0
    nonroot_correct = 0
    root_total = 0
    root_correct = 0

    mismatches: List[Dict] = []

    for child in range(max_nodes):
        gt_active = bool(active_mask[child])
        gt_parent = int(exp_parent[child])
        pred_arg = int(argmax_class[child])
        mapped_arg = -1 if (pred_arg == root_class or pred_arg == child) else pred_arg
        dec = int(dec_parent[child])

        if gt_active:
            parent_total += 1
            if gt_parent < 0:
                root_total += 1
                if dec < 0:
                    root_correct += 1
                    parent_correct += 1
            else:
                nonroot_total += 1
                if dec == gt_parent:
                    nonroot_correct += 1
                    parent_correct += 1

        mismatch = gt_active and (dec != gt_parent)
        if mismatch:
            active_parent = bool(active_mask[dec]) if dec >= 0 and dec < max_nodes else False
            padded_involved = (not gt_active) or (dec >= 0 and not active_parent)
            mismatches.append(
                {
                    "sample": sample_key,
                    "child": child,
                    "gt_parent": gt_parent,
                    "predicted_parent_argmax": pred_arg,
                    "decoded_parent": dec,
                    "active_child": gt_active,
                    "active_parent": active_parent,
                    "root_selected": pred_arg == root_class,
                    "self_parent_rejected": (pred_arg == child and dec < 0),
                    "cycle_repair_changed": mapped_arg != dec,
                    "padded_involved": padded_involved,
                }
            )

    pred_edges = parent_edge_sets(dec_parent, decoded["active_mask"])
    gt_edges = undirected_edge_set_from_parent(exp_parent, active_mask)
    f1 = edge_metrics(pred_edges, gt_edges)

    gt_components = component_count(active_mask, gt_edges)
    decoded_components = int(decoded["metadata"].get("component_count", 0))

    padded_decoded = 0
    for child, parent in enumerate(dec_parent.tolist()):
        if parent >= 0 and ((not active_mask[child]) or (not active_mask[parent])):
            padded_decoded += 1

    return {
        "sample": sample_key,
        "parent_acc": float(parent_correct / max(parent_total, 1)),
        "nonroot_parent_acc": float(nonroot_correct / max(nonroot_total, 1)),
        "root_acc": float(root_correct / max(root_total, 1)),
        "edge_precision": float(f1["precision"]),
        "edge_recall": float(f1["recall"]),
        "edge_f1": float(f1["f1"]),
        "cycle_rate": float(decoded["metadata"].get("cycle_rate", 0.0)),
        "component_count": float(decoded_components),
        "gt_component_count": float(gt_components),
        "padded_decoded_edges": int(padded_decoded),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def collect_indices(split_size: int, sample_count: int) -> List[int]:
    return list(range(min(split_size, sample_count)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic parent convention identity check")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--sample-count", type=int, default=40)
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument("--points-per-sample", type=int, default=1024)
    parser.add_argument("--decode-mode", choices=["parent_argmax", "parent_argmax_acyclic", "parent_mst_hybrid"], default="parent_argmax_acyclic")
    parser.add_argument("--out", default="outputs/models/hyperbone_anymate_static_v2.8d_parent_convention")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    decode_cfg = ParentDecodeConfig(
        active_threshold=0.5,
        decode_mode=args.decode_mode,
        max_degree=args.max_nodes,
    )

    if args.split == "both":
        half = max(args.sample_count // 2, 1)
        split_plan = [("train", half), ("test", args.sample_count - half)]
    else:
        split_plan = [(args.split, args.sample_count)]

    all_rows: List[Dict] = []
    all_mismatches: List[Dict] = []

    for split_name, n_samples in split_plan:
        ds = AnymateStaticRigDataset(
            args.pt,
            f"{args.splits_dir}/{split_name}.jsonl",
            max_joints=args.max_nodes,
            pc_points=args.points_per_sample,
        )
        indices = collect_indices(len(ds), n_samples)

        for idx in indices:
            sample = ds[idx]
            row = evaluate_sample(sample, decode_cfg, f"{split_name}:{idx}")
            all_rows.append(row)
            all_mismatches.extend(row["mismatches"])

    def avg(key: str) -> float:
        if not all_rows:
            return 0.0
        return float(sum(float(r[key]) for r in all_rows) / len(all_rows))

    max_mismatch = max((int(r["mismatch_count"]) for r in all_rows), default=0)
    any_padded = sum(int(r["padded_decoded_edges"]) for r in all_rows)

    summary = {
        "sample_count": len(all_rows),
        "split": args.split,
        "decode_mode": args.decode_mode,
        "parent_acc": avg("parent_acc"),
        "nonroot_parent_acc": avg("nonroot_parent_acc"),
        "root_acc": avg("root_acc"),
        "edge_precision": avg("edge_precision"),
        "edge_recall": avg("edge_recall"),
        "edge_f1": avg("edge_f1"),
        "cycle_rate": avg("cycle_rate"),
        "component_count": avg("component_count"),
        "gt_component_count": avg("gt_component_count"),
        "total_mismatches": len(all_mismatches),
        "max_sample_mismatch_count": max_mismatch,
        "padded_decoded_edges_total": any_padded,
        "identity_pass": bool(
            avg("parent_acc") > 0.999
            and avg("nonroot_parent_acc") > 0.999
            and avg("root_acc") > 0.999
            and avg("edge_f1") > 0.999
            and avg("cycle_rate") == 0.0
            and any_padded == 0
        ),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "per_sample.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    (out_dir / "mismatches.json").write_text(json.dumps(all_mismatches, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if all_mismatches:
        print("[Convention-check] mismatch rows:")
        for row in all_mismatches:
            print(json.dumps(row))


if __name__ == "__main__":
    main()
