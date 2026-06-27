from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.datasets.anymate_static_dataset import AnymateStaticRigDataset
from hyperbone.models.hyperbone_static_parent import HyperBoneStaticParentModel
from hyperbone.rigs.parent_targets import normalize_parent_index


def hungarian_match(pred_pts: torch.Tensor, gt_pts: torch.Tensor):
    if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    cost = torch.cdist(pred_pts, gt_pts, p=2).detach().cpu().numpy()
    return linear_sum_assignment(cost)


def sample_to_device(sample: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}


def canonical_parent(parent_index: np.ndarray, active: np.ndarray) -> np.ndarray:
    n = int(parent_index.shape[0])
    out = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if not active[i]:
            continue
        p = int(parent_index[i])
        if p < 0 or p >= n or p == i or (not active[p]):
            out[i] = -1
        else:
            out[i] = p
    return out


def count_cycles(parent_index: np.ndarray, active: np.ndarray) -> int:
    n = int(parent_index.shape[0])
    seen_cycles = set()
    for start in range(n):
        if not active[start]:
            continue
        trail = {}
        order = []
        cur = start
        step = 0
        while cur >= 0 and cur < n and active[cur]:
            if cur in trail:
                cyc_nodes = tuple(sorted(order[trail[cur]:]))
                seen_cycles.add(cyc_nodes)
                break
            trail[cur] = step
            order.append(cur)
            step += 1
            nxt = int(parent_index[cur])
            if nxt < 0 or nxt >= n or (not active[nxt]):
                break
            cur = nxt
    return int(len(seen_cycles))


def edge_set(parent_index: np.ndarray, active: np.ndarray) -> set[tuple[int, int]]:
    edges = set()
    for child, parent in enumerate(parent_index.tolist()):
        if parent >= 0 and active[child] and active[parent]:
            edges.add((int(parent), int(child)))
    return edges


def build_direct_stats(sample: Dict[str, torch.Tensor]) -> Dict:
    active = sample["joint_active"] > 0.5
    active_np = active.detach().cpu().numpy().astype(bool)
    n_gt = int(active.sum().item())
    root_class = int(sample["parent_index"].shape[0])
    parent_index = sample["parent_index"].long()
    parent_index_np = parent_index.detach().cpu().numpy().astype(np.int64)
    raw_conns_np = sample["conns"].long().detach().cpu().numpy().astype(np.int64)

    raw_canon = canonical_parent(raw_conns_np, active_np)
    norm_ref, _, _, norm_meta = normalize_parent_index(raw_conns_np, active_np)
    raw_edges = edge_set(raw_canon, active_np)
    norm_edges = edge_set(parent_index_np, active_np)

    normalization_changed = bool(np.any(raw_canon[active_np] != parent_index_np[active_np]))
    raw_cycles = count_cycles(raw_canon, active_np)
    normalized_cycles = count_cycles(parent_index_np, active_np)
    raw_root_count = int(((raw_canon < 0) & active_np).sum())
    normalized_root_count = int(((parent_index_np < 0) & active_np).sum())
    edges_preserved_ratio = float(len(raw_edges & norm_edges) / max(len(raw_edges), 1))

    root_mask = sample["root_mask"] > 0.5
    valid_parent_mask = sample["valid_parent_mask"] > 0.5
    valid = active.clone()
    root_count = int((root_mask & active).sum().item())
    nonroot_count = int((valid_parent_mask & active).sum().item())
    ignored_count = int((~valid).sum().item())
    parent_class = parent_index.clone()
    parent_class[parent_class < 0] = root_class
    parent_class[~active] = root_class
    target_hist = Counter(parent_class[active].tolist())
    target_min = int(parent_class[active].min().item()) if active.any() else -1
    target_max = int(parent_class[active].max().item()) if active.any() else -1
    return {
        "gt_joint_count": n_gt,
        "valid_parent_target_count": n_gt,
        "root_target_count": root_count,
        "nonroot_parent_target_count": nonroot_count,
        "percent_parent_targets_root": root_count / max(n_gt, 1),
        "percent_parent_targets_ignored": ignored_count / max(sample["joint_active"].numel(), 1),
        "parent_ce_class_distribution": {str(k): int(v) for k, v in target_hist.items()},
        "parent_target_min": target_min,
        "parent_target_max": target_max,
        "raw_parent_cycles": raw_cycles,
        "normalized_parent_cycles": normalized_cycles,
        "raw_root_count": raw_root_count,
        "normalized_root_count": normalized_root_count,
        "edges_preserved_ratio": edges_preserved_ratio,
        "samples_requiring_normalization": 1.0 if normalization_changed else 0.0,
        "normalization_cycles_detected": float(norm_meta.get("cycles_detected", 0)),
        "normalization_cycles_broken": float(norm_meta.get("cycles_broken", 0)),
        "normalization_invalid_parent_count": float(norm_meta.get("invalid_parent_count", 0)),
        "normalization_self_parent_count": float(norm_meta.get("self_parent_count", 0)),
        "normalization_no_root_component_count": float(norm_meta.get("no_root_component_count", 0)),
        "normalization_roots_added": float(norm_meta.get("roots_added", 0)),
        "normalization_edges_preserved": float(norm_meta.get("edges_preserved", 0)),
        "normalization_edges_removed": float(norm_meta.get("edges_removed", 0)),
    }


def build_hungarian_stats(sample: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor], active_threshold: float) -> Dict:
    gt_active = sample["joint_active"] > 0.5
    pred_active_prob = torch.sigmoid(pred["active_logits"])
    pred_active = pred_active_prob > active_threshold
    pred_pos = pred["joint_pos"]
    gt_pos = sample["joint_pos"]

    gt_idx = gt_active.nonzero(as_tuple=True)[0]
    pred_idx = pred_active.nonzero(as_tuple=True)[0]
    gt_pts = gt_pos[gt_idx]
    pred_pts = pred_pos[pred_idx]

    row_ind, col_ind = hungarian_match(pred_pts, gt_pts)
    matched_pred_count = int(len(row_ind))
    gt_to_pred = {int(col_ind[i]): int(row_ind[i]) for i in range(matched_pred_count)}
    pred_to_gt = {int(row_ind[i]): int(col_ind[i]) for i in range(matched_pred_count)}

    gt_parent_index = sample["parent_index"].long().cpu().numpy()
    gt_root_mask = sample["root_mask"].cpu().numpy() > 0.5
    active_count = int(gt_active.sum().item())
    root_class = int(sample["parent_index"].shape[0])

    parent_targets = []
    root_targets = 0
    nonroot_targets = 0
    valid_targets = 0
    valid_pred_parent_targets = 0
    recoverable_gt_edges = 0
    class_hist = Counter()

    matched_pred_set = set(pred_idx[row_ind].tolist()) if matched_pred_count > 0 else set()

    for gt_slot, pred_slot in gt_to_pred.items():
        valid_targets += 1
        parent_gt = int(gt_parent_index[gt_idx[gt_slot]].item())
        if gt_root_mask[gt_idx[gt_slot]]:
            root_targets += 1
            parent_targets.append(root_class)
            class_hist[root_class] += 1
            continue
        nonroot_targets += 1
        parent_targets.append(pred_slot)
        class_hist[pred_slot] += 1
        if parent_gt >= 0 and parent_gt in gt_to_pred:
            recoverable_gt_edges += 1
            parent_pred = gt_to_pred[parent_gt]
            if parent_pred in matched_pred_set:
                valid_pred_parent_targets += 1

    parent_targets_np = np.asarray(parent_targets, dtype=np.int64) if parent_targets else np.asarray([], dtype=np.int64)
    parent_target_min = int(parent_targets_np.min()) if parent_targets_np.size > 0 else -1
    parent_target_max = int(parent_targets_np.max()) if parent_targets_np.size > 0 else -1

    return {
        "gt_joint_count": active_count,
        "matched_pred_count": matched_pred_count,
        "valid_parent_target_count": valid_targets,
        "root_target_count": root_targets,
        "nonroot_parent_target_count": nonroot_targets,
        "percent_parent_targets_root": root_targets / max(valid_targets, 1),
        "percent_parent_targets_ignored": max(active_count - valid_targets, 0) / max(active_count, 1),
        "parent_ce_class_distribution": {str(k): int(v) for k, v in class_hist.items()},
        "parent_target_min": parent_target_min,
        "parent_target_max": parent_target_max,
        "valid_parent_targets_point_to_valid_predicted_matched_indices": valid_pred_parent_targets == nonroot_targets if nonroot_targets > 0 else True,
        "gt_edges_recoverable_after_matching": recoverable_gt_edges,
    }


def summarize(records: List[Dict]) -> Dict:
    summary = defaultdict(float)
    hist = Counter()
    if not records:
        return {}
    for record in records:
        for key in [
            "gt_joint_count",
            "matched_pred_count",
            "valid_parent_target_count",
            "root_target_count",
            "nonroot_parent_target_count",
            "percent_parent_targets_root",
            "percent_parent_targets_ignored",
            "parent_target_min",
            "parent_target_max",
            "gt_edges_recoverable_after_matching",
            "raw_parent_cycles",
            "normalized_parent_cycles",
            "raw_root_count",
            "normalized_root_count",
            "edges_preserved_ratio",
            "samples_requiring_normalization",
            "normalization_cycles_detected",
            "normalization_cycles_broken",
            "normalization_invalid_parent_count",
            "normalization_self_parent_count",
            "normalization_no_root_component_count",
            "normalization_roots_added",
            "normalization_edges_preserved",
            "normalization_edges_removed",
        ]:
            if key in record:
                summary[key] += float(record[key])
        for k, v in record.get("parent_ce_class_distribution", {}).items():
            hist[k] += int(v)
    count = float(len(records))
    out = {k: v / count for k, v in summary.items()}
    out["parent_ce_class_distribution"] = {k: int(v) for k, v in hist.items()}
    out["sample_count"] = len(records)
    return out


def main():
    parser = argparse.ArgumentParser(description="Debug parent supervision transfer")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--backbone", choices=["pointnet", "dgcnn"], default="dgcnn")
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument("--points-per-sample", type=int, default=1024)
    parser.add_argument("--active-threshold", type=float, default=0.70)
    parser.add_argument("--out-dir", default="outputs/models/hyperbone_anymate_static_v2.8_parent_debug")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = HyperBoneStaticParentModel(
        in_channels=3,
        feat_dim=512,
        max_joints=args.max_nodes,
        predict_skinning=False,
        backbone=args.backbone,
        knn_k=args.knn_k,
    ).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
    model.eval()

    report = {
        "args": vars(args),
        "splits": {},
    }

    for split in ["train", "test"]:
        ds = AnymateStaticRigDataset(args.pt, f"{args.splits_dir}/{split}.jsonl", max_joints=args.max_nodes, pc_points=args.points_per_sample)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        records = []
        for sample_idx, batch in enumerate(loader):
            if sample_idx >= 20:
                break
            batch = sample_to_device(batch, device)
            with torch.no_grad():
                pred = model(batch)
            sample = {k: v[0] if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == 1 else v for k, v in batch.items()}
            pred_sample = {k: v[0] if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == 1 else v for k, v in pred.items()}
            direct = build_direct_stats(sample)
            hungarian = build_hungarian_stats(sample, pred_sample, args.active_threshold)
            records.append({
                "sample_index": sample_idx,
                "direct": direct,
                "hungarian": hungarian,
            })

        summary_direct = summarize([r["direct"] for r in records])
        summary_hungarian = summarize([r["hungarian"] for r in records])
        report["splits"][split] = {
            "samples": records,
            "summary": {
                "direct": summary_direct,
                "hungarian": summary_hungarian,
            },
        }

    md_lines = [
        "# Parent Supervision Debug",
        "",
        f"Checkpoint: {args.checkpoint or 'none'}",
        f"Backbone: {args.backbone}",
        f"Threshold: {args.active_threshold}",
        "",
    ]
    for split, split_data in report["splits"].items():
        md_lines.append(f"## {split.title()}")
        for mode in ["direct", "hungarian"]:
            s = split_data["summary"][mode]
            md_lines.append(f"### {mode}")
            md_lines.append(f"- GT joint count: {s.get('gt_joint_count', 0):.2f}")
            md_lines.append(f"- Matched predicted count: {s.get('matched_pred_count', 0):.2f}")
            md_lines.append(f"- Valid parent target count: {s.get('valid_parent_target_count', 0):.2f}")
            md_lines.append(f"- Root target count: {s.get('root_target_count', 0):.2f}")
            md_lines.append(f"- Non-root parent target count: {s.get('nonroot_parent_target_count', 0):.2f}")
            md_lines.append(f"- Percent parent targets ROOT: {s.get('percent_parent_targets_root', 0.0):.3f}")
            md_lines.append(f"- Percent parent targets ignored: {s.get('percent_parent_targets_ignored', 0.0):.3f}")
            md_lines.append(f"- GT edges recoverable after matching: {s.get('gt_edges_recoverable_after_matching', 0.0):.2f}")
            md_lines.append(f"- Parent target min/max: {s.get('parent_target_min', -1):.2f} / {s.get('parent_target_max', -1):.2f}")
            md_lines.append(f"- Raw parent cycles: {s.get('raw_parent_cycles', 0.0):.2f}")
            md_lines.append(f"- Normalized parent cycles: {s.get('normalized_parent_cycles', 0.0):.2f}")
            md_lines.append(f"- Raw root count: {s.get('raw_root_count', 0.0):.2f}")
            md_lines.append(f"- Normalized root count: {s.get('normalized_root_count', 0.0):.2f}")
            md_lines.append(f"- Edges preserved ratio: {s.get('edges_preserved_ratio', 0.0):.3f}")
            md_lines.append(f"- Samples requiring normalization: {s.get('samples_requiring_normalization', 0.0):.2f}")
            md_lines.append(f"- Normalization cycles detected/broken: {s.get('normalization_cycles_detected', 0.0):.2f} / {s.get('normalization_cycles_broken', 0.0):.2f}")
            md_lines.append(f"- Normalization invalid/self parents: {s.get('normalization_invalid_parent_count', 0.0):.2f} / {s.get('normalization_self_parent_count', 0.0):.2f}")
            md_lines.append(f"- Normalization no-root components / roots added: {s.get('normalization_no_root_component_count', 0.0):.2f} / {s.get('normalization_roots_added', 0.0):.2f}")
            md_lines.append(f"- Normalization edges preserved/removed: {s.get('normalization_edges_preserved', 0.0):.2f} / {s.get('normalization_edges_removed', 0.0):.2f}")
            md_lines.append(f"- Parent class distribution: {json.dumps(s.get('parent_ce_class_distribution', {}), sort_keys=True)}")
            md_lines.append("")

    md_path = out_dir / "supervision_debug.md"
    json_path = out_dir / "supervision_debug.json"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[Debug-parent] Wrote {md_path}")
    print(f"[Debug-parent] Wrote {json_path}")


if __name__ == "__main__":
    main()
