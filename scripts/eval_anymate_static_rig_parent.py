from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.datasets.anymate_static_dataset import AnymateStaticRigDataset
from hyperbone.models.hyperbone_static_parent import HyperBoneStaticParentModel
from hyperbone.rigs.parent_candidates import build_parent_candidates, mask_parent_logits_with_candidates, gather_nonroot_candidate_logits
from hyperbone.rigs.parent_decoder import ParentDecodeConfig, decode_parent_graph, decode_two_stage_root_parent_graph


def compute_spread(positions: torch.Tensor) -> float:
    if positions.shape[0] < 2:
        return 0.0
    mn = positions.min(dim=0)[0]
    mx = positions.max(dim=0)[0]
    return float((mx - mn).norm().item())


def hungarian_match(pred_pts: torch.Tensor, gt_pts: torch.Tensor):
    if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    cost = torch.cdist(pred_pts, gt_pts, p=2).detach().cpu().numpy()
    return linear_sum_assignment(cost)


def count_gt_edges(gt_adj: torch.Tensor, gt_mask: torch.Tensor) -> int:
    gt_active_idx = gt_mask.nonzero(as_tuple=True)[0]
    if gt_active_idx.numel() < 2:
        return 0
    gt_sub = gt_adj[gt_active_idx][:, gt_active_idx]
    return int((gt_sub.triu(diagonal=1) > 0.5).sum().item())


def edge_metrics(pred_edges: set[tuple[int, int]], gt_edges: set[tuple[int, int]]) -> dict:
    tp = len(pred_edges & gt_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1}


def parent_edge_sets(parent_ptr: np.ndarray, active_mask: np.ndarray) -> set[tuple[int, int]]:
    edges = set()
    for child, parent in enumerate(parent_ptr):
        if parent >= 0 and active_mask[child] and active_mask[parent]:
            a, b = sorted((int(parent), int(child)))
            edges.add((a, b))
    return edges


def bone_ratio_metrics(pred_pos: torch.Tensor, gt_pos: torch.Tensor, pred_edges: set[tuple[int, int]], gt_edges: set[tuple[int, int]]) -> dict:
    all_pred = []
    tp_edges = []
    fp_edges = []
    for edge in pred_edges:
        i, j = edge
        ref_len = float((gt_pos[i] - gt_pos[j]).norm().item())
        pred_len = float((pred_pos[i] - pred_pos[j]).norm().item())
        if ref_len > 1e-4:
            ratio = pred_len / ref_len
            all_pred.append(ratio)
            if edge in gt_edges:
                tp_edges.append(ratio)
            else:
                fp_edges.append(ratio)
    return {
        "all_pred_edges": float(np.mean(all_pred)) if all_pred else 0.0,
        "true_positive_edges": float(np.mean(tp_edges)) if tp_edges else 0.0,
        "false_positive_edges": float(np.mean(fp_edges)) if fp_edges else 0.0,
    }


def evaluate(model, loader, device, active_threshold: float, decode_cfg: ParentDecodeConfig,
             teacher_force_gt_nodes: bool = False, no_backbone_for_gt_nodes: bool = False,
             root_decode_bias: float = 0.0, candidate_parent_k: int | None = None,
             force_gt_parent_candidate: bool = False,
             two_stage_root_parent: bool = False,
             root_ambiguity_eps: float = 0.0,
             root_decode_mode: str = "threshold",
             root_budget_ratio: float | None = None):
    model.eval()
    metrics = {
        "mpjpe": 0.0,
        "count_ratio": 0.0,
        "parent_acc": 0.0,
        "nonroot_parent_acc": 0.0,
        "root_acc": 0.0,
        "root_precision": 0.0,
        "root_recall": 0.0,
        "root_f1": 0.0,
        "pred_root_ratio": 0.0,
        "gt_root_ratio": 0.0,
        "selected_root_count": 0.0,
        "gt_root_count": 0.0,
        "edge_precision": 0.0,
        "edge_recall": 0.0,
        "edge_f1": 0.0,
        "bone_ratio_all": 0.0,
        "bone_ratio_tp": 0.0,
        "component_count": 0.0,
        "cycle_rate": 0.0,
        "average_degree": 0.0,
        "max_degree": 0.0,
        "ambiguous_root_nodes": 0.0,
    }
    n = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if teacher_force_gt_nodes:
                pred = model.forward_parent_from_joints(
                    batch["joint_pos"],
                    active_mask=batch["joint_active"] > 0.5,
                    no_backbone_for_gt_nodes=no_backbone_for_gt_nodes,
                )
                active_prob = batch["joint_active"].float()
                active_mask = active_prob > 0.5
                pred_pos = batch["joint_pos"].detach().cpu()
            else:
                pred = model(batch)
                active_prob = torch.sigmoid(pred["active_logits"])
                active_mask = active_prob > active_threshold
                pred_pos = pred["joint_pos"].detach().cpu()
            parent_logits = pred["parent_logits"]
            # Build candidate info for two-stage or slot-candidate modes.
            candidate_info = None
            if two_stage_root_parent and candidate_parent_k is not None:
                candidate_info = build_parent_candidates(
                    batch["joint_pos"],
                    batch["joint_active"] > 0.5,
                    batch["parent_index"].long(),
                    k=int(candidate_parent_k),
                    include_root=False,
                    force_gt_parent=force_gt_parent_candidate,
                )
                candidate_pair_logits = gather_nonroot_candidate_logits(
                    pred["parent_pair_logits"],
                    candidate_info["candidate_indices"],
                    candidate_info["candidate_mask"],
                )
            elif candidate_parent_k is not None:
                candidate_info = build_parent_candidates(
                    batch["joint_pos"],
                    batch["joint_active"] > 0.5,
                    batch["parent_index"].long(),
                    k=int(candidate_parent_k),
                    include_root=True,
                    force_gt_parent=force_gt_parent_candidate,
                )
                parent_logits = mask_parent_logits_with_candidates(
                    parent_logits,
                    candidate_info["candidate_indices"],
                    candidate_info["candidate_mask"],
                )
            gt_pos = batch["joint_pos"].detach().cpu()
            gt_adj = batch["adj_matrix"].detach().cpu()

            m_batch = batch["joint_pos"].shape[0]
            for b in range(m_batch):
                gt_mask = batch["joint_active"][b].detach().cpu() > 0.5
                pred_mask = active_mask[b].detach().cpu()

                gt_pts = gt_pos[b, gt_mask]
                pred_pts = pred_pos[b, pred_mask]
                n_gt = int(gt_mask.sum().item())
                n_pred = int(pred_mask.sum().item())
                gt_parent_index = batch["parent_index"][b].detach().cpu().numpy()
                gt_root_mask = batch["root_mask"][b].detach().cpu().numpy() > 0.5
                gt_root_count = int((gt_root_mask & gt_mask.numpy()).sum())

                if n_gt == 0:
                    continue

                if two_stage_root_parent and candidate_info is not None and pred.get("root_logits") is not None:
                    decoded = decode_two_stage_root_parent_graph(
                        positions=pred_pos[b].numpy(),
                        active_prob=active_prob[b].detach().cpu().numpy(),
                        root_logits=pred["root_logits"][b].detach().cpu().numpy(),
                        parent_candidate_logits=candidate_pair_logits[b].detach().cpu().numpy(),
                        candidate_indices=candidate_info["candidate_indices"][b].detach().cpu().numpy(),
                        candidate_mask=candidate_info["candidate_mask"][b].detach().cpu().numpy(),
                        config=decode_cfg,
                        root_threshold=decode_cfg.root_threshold,
                        root_decode_mode=root_decode_mode,
                        root_budget_ratio=root_budget_ratio,
                        root_count_budget=gt_root_count if root_decode_mode == "topk_budget" else None,
                    )
                else:
                    decoded = decode_parent_graph(
                        pred_pos[b].numpy(),
                        active_prob[b].detach().cpu().numpy(),
                        parent_logits[b].detach().cpu().numpy(),
                        parent_offset=pred["parent_offset"][b].detach().cpu().numpy(),
                        edge_confidence=pred["edge_confidence"].detach().cpu().numpy()[b],
                        config=decode_cfg,
                        root_decode_bias=root_decode_bias,
                        root_logits=pred.get("root_logits")[b].detach().cpu().numpy() if pred.get("root_logits") is not None else None,
                        root_decode_mode=root_decode_mode,
                        root_budget_ratio=root_budget_ratio,
                        root_count_budget=gt_root_count if root_decode_mode == "topk_budget" else None,
                    )

                root_class = pred["parent_logits"].shape[-1] - 1

                if teacher_force_gt_nodes:
                    # Direct GT-node path: slot correspondence is identity.
                    metrics["mpjpe"] += float((pred_pos[b][gt_mask] - gt_pos[b][gt_mask]).norm(dim=-1).mean().item()) if gt_mask.any() else 0.0
                    metrics["count_ratio"] += float(gt_mask.float().sum().item() / max(gt_mask.float().sum().item(), 1.0))
                    metrics["spread"] = metrics.get("spread", 0.0) + float(compute_spread(pred_pos[b, gt_mask]) / max(compute_spread(gt_pos[b, gt_mask]), 1e-6))

                    parent_ptr = decoded["parent_ptr"]
                    parent_correct = 0
                    root_correct = 0
                    parent_total = 0
                    nonroot_total = 0
                    nonroot_correct = 0
                    root_total = 0
                    pred_edges = parent_edge_sets(parent_ptr, decoded["active_mask"])
                    gt_edges = set()
                    for gt_child in range(len(gt_parent_index)):
                        if not gt_mask[gt_child].item():
                            continue
                        gt_parent = int(gt_parent_index[gt_child])
                        parent_total += 1
                        if gt_root_mask[gt_child]:
                            root_total += 1
                            if parent_ptr[gt_child] < 0:
                                root_correct += 1
                            continue
                        nonroot_total += 1
                        if gt_parent >= 0 and parent_ptr[gt_child] == gt_parent:
                            parent_correct += 1
                            nonroot_correct += 1
                            gt_edges.add(tuple(sorted((gt_parent, gt_child))))
                        elif gt_parent >= 0:
                            gt_edges.add(tuple(sorted((gt_parent, gt_child))))

                    metrics["parent_acc"] += parent_correct / max(parent_total, 1)
                    metrics["nonroot_parent_acc"] += nonroot_correct / max(nonroot_total, 1)
                    metrics["root_acc"] += root_correct / max(root_total, 1)
                    # Root precision/recall/F1 via pred_root vs gt_root_mask over active GT slots.
                    root_pred_mask = decoded.get("root_mask", decoded["parent_ptr"] < 0)
                    # Compute ambiguous coincident-position mask for this sample.
                    ambig_set = set()
                    if root_ambiguity_eps > 0.0:
                        pos_b = batch["joint_pos"][b]
                        valid_idx = torch.where(gt_mask)[0]
                        if valid_idx.numel() >= 2:
                            pos_v = pos_b[valid_idx]
                            dmat = torch.cdist(pos_v, pos_v)
                            root_v = torch.from_numpy(gt_root_mask)[valid_idx]
                            conflict = root_v.unsqueeze(0) != root_v.unsqueeze(1)
                            coincident = (dmat.cpu() < root_ambiguity_eps) & conflict
                            coincident.fill_diagonal_(False)
                            ambiguous_local = coincident.any(dim=1)
                            for idx in valid_idx[ambiguous_local].tolist():
                                ambig_set.add(idx)
                    metrics["ambiguous_root_nodes"] += len(ambig_set)
                    r_tp = sum(1 for i in range(len(gt_parent_index)) if gt_mask[i].item() and i not in ambig_set and gt_root_mask[i] and root_pred_mask[i])
                    r_fp = sum(1 for i in range(len(gt_parent_index)) if gt_mask[i].item() and i not in ambig_set and not gt_root_mask[i] and root_pred_mask[i])
                    r_fn = sum(1 for i in range(len(gt_parent_index)) if gt_mask[i].item() and i not in ambig_set and gt_root_mask[i] and not root_pred_mask[i])
                    r_prec = r_tp / max(r_tp + r_fp, 1)
                    r_rec = r_tp / max(r_tp + r_fn, 1)
                    r_f1 = 2.0 * r_prec * r_rec / max(r_prec + r_rec, 1e-8)
                    metrics["root_precision"] += r_prec
                    metrics["root_recall"] += r_rec
                    metrics["root_f1"] += r_f1
                    gt_active_total = int(gt_mask.float().sum().item())
                    selected_root_count = sum(1 for i in range(len(gt_parent_index)) if gt_mask[i].item() and root_pred_mask[i])
                    metrics["selected_root_count"] += float(selected_root_count)
                    metrics["gt_root_count"] += float(gt_root_count)
                    metrics["pred_root_ratio"] += float(selected_root_count / max(gt_active_total, 1))
                    metrics["gt_root_ratio"] += float(gt_root_count / max(gt_active_total, 1))
                    f1 = edge_metrics(pred_edges, gt_edges)
                    metrics["edge_precision"] += f1["precision"]
                    metrics["edge_recall"] += f1["recall"]
                    metrics["edge_f1"] += f1["f1"]
                    bone_metrics = bone_ratio_metrics(pred_pos[b], gt_pos[b], pred_edges, gt_edges)
                    metrics["bone_ratio_all"] += bone_metrics["all_pred_edges"]
                    metrics["bone_ratio_tp"] += bone_metrics["true_positive_edges"]
                    metrics["component_count"] += float(decoded["metadata"]["component_count"])
                    metrics["cycle_rate"] += float(decoded["metadata"]["cycle_rate"])
                    metrics["average_degree"] += float(decoded["metadata"]["average_degree"])
                    metrics["max_degree"] += float(decoded["metadata"]["max_degree"])
                    n += 1
                    continue

                row_ind, col_ind = hungarian_match(pred_pts, gt_pts)
                n_matched = len(row_ind)
                if n_matched > 0:
                    matched_pred_pts = pred_pts[row_ind]
                    matched_gt_pts = gt_pts[col_ind]
                    matched_dists = (matched_pred_pts - matched_gt_pts).norm(dim=-1)
                    metrics["mpjpe"] += float(matched_dists.mean().item())
                    metrics["count_ratio"] += float(n_pred / max(n_gt, 1))
                    metrics["spread"] = metrics.get("spread", 0.0) + float(compute_spread(pred_pts) / max(compute_spread(gt_pts), 1e-6))

                gt_to_pred = {int(col_ind[i]): int(row_ind[i]) for i in range(n_matched)}
                pred_to_gt = {int(row_ind[i]): int(col_ind[i]) for i in range(n_matched)}

                parent_ptr = decoded["parent_ptr"]
                root_pred_mask = decoded.get("root_mask", parent_ptr < 0)
                parent_correct = 0
                root_correct = 0
                parent_total = 0
                nonroot_total = 0
                nonroot_correct = 0
                root_total = 0
                for pred_slot, gt_slot in pred_to_gt.items():
                    parent_total += 1
                    gt_parent = int(gt_parent_index[gt_slot])
                    pred_parent = int(parent_ptr[pred_slot])
                    if gt_root_mask[gt_slot]:
                        root_total += 1
                        if root_pred_mask[pred_slot]:
                            root_correct += 1
                        continue
                    nonroot_total += 1
                    if gt_parent >= 0 and gt_parent in gt_to_pred and pred_parent == gt_to_pred[gt_parent]:
                        parent_correct += 1
                        nonroot_correct += 1

                metrics["parent_acc"] += parent_correct / max(parent_total, 1)
                metrics["nonroot_parent_acc"] += nonroot_correct / max(nonroot_total, 1)
                metrics["root_acc"] += root_correct / max(root_total, 1)
                selected_root_count = int(root_pred_mask.sum())
                metrics["selected_root_count"] += float(selected_root_count)
                metrics["gt_root_count"] += float(int(gt_root_mask[gt_mask.numpy()].sum()))
                metrics["pred_root_ratio"] += float(selected_root_count / max(int(gt_mask.sum().item()), 1))
                metrics["gt_root_ratio"] += float(int(gt_root_mask[gt_mask.numpy()].sum()) / max(int(gt_mask.sum().item()), 1))

                pred_edges = parent_edge_sets(parent_ptr, decoded["active_mask"])
                gt_edges = set()
                for gt_child in range(len(gt_parent_index)):
                    gt_parent = int(gt_parent_index[gt_child])
                    if gt_parent < 0 or gt_parent == gt_child:
                        continue
                    if gt_child in gt_to_pred and gt_parent in gt_to_pred:
                        a, b2 = sorted((gt_to_pred[gt_parent], gt_to_pred[gt_child]))
                        gt_edges.add((a, b2))

                f1 = edge_metrics(pred_edges, gt_edges)
                metrics["edge_precision"] += f1["precision"]
                metrics["edge_recall"] += f1["recall"]
                metrics["edge_f1"] += f1["f1"]

                bone_metrics = bone_ratio_metrics(pred_pos[b], gt_pos[b], pred_edges, gt_edges)
                metrics["bone_ratio_all"] += bone_metrics["all_pred_edges"]
                metrics["bone_ratio_tp"] += bone_metrics["true_positive_edges"]

                metrics["component_count"] += float(decoded["metadata"]["component_count"])
                metrics["cycle_rate"] += float(decoded["metadata"]["cycle_rate"])
                metrics["average_degree"] += float(decoded["metadata"]["average_degree"])
                metrics["max_degree"] += float(decoded["metadata"]["max_degree"])
                n += 1

    if n > 0 and "spread" in metrics:
        metrics["spread"] /= n
    out = {k: v / max(n, 1) for k, v in metrics.items() if k != "spread"}
    out["spread"] = metrics.get("spread", 0.0)
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate HyperBone static rig parent-pointer model")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--ckpt", "--checkpoint", dest="ckpt", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument("--pc-points", type=int, default=1024)
    parser.add_argument("--points-per-sample", type=int, default=None)
    parser.add_argument("--backbone", choices=["pointnet", "dgcnn"], default="dgcnn")
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--active-threshold", type=float, default=0.70)
    parser.add_argument("--decode-mode", choices=["parent_argmax", "parent_argmax_acyclic", "parent_mst_hybrid"], default="parent_argmax_acyclic")
    parser.add_argument("--max-degree", type=int, default=4)
    parser.add_argument("--enforce-root-per-component", action="store_true")
    parser.add_argument("--edge-alpha", type=float, default=1.0)
    parser.add_argument("--edge-beta", type=float, default=0.8)
    parser.add_argument("--edge-gamma", type=float, default=0.2)
    parser.add_argument("--edge-eta", type=float, default=0.1)
    parser.add_argument("--teacher-force-gt-nodes", action="store_true")
    parser.add_argument("--parent-head", choices=["slot", "pairwise"], default="slot")
    parser.add_argument("--no-backbone-for-gt-nodes", action="store_true")
    parser.add_argument("--root-feature-mode", choices=["none", "structural"], default="none")
    parser.add_argument("--root-bias-init", type=float, default=None)
    parser.add_argument("--root-decode-bias", type=float, default=0.0)
    parser.add_argument("--candidate-parent-k", type=int, default=None)
    parser.add_argument("--force-gt-parent-candidate", action="store_true")
    parser.add_argument("--two-stage-root-parent", action="store_true")
    parser.add_argument("--root-threshold", type=float, default=None,
                        help="Calibrated root decision threshold for two-stage decode.")
    parser.add_argument("--root-ambiguity-eps", type=float, default=0.0,
                        help="Exclude coincident root/non-root nodes within this distance from root metrics.")
    parser.add_argument("--root-decode-mode", choices=["threshold", "topk_budget", "ratio_budget"], default="threshold")
    parser.add_argument("--root-budget-ratio", type=float, default=None,
                        help="Budget ratio used by ratio_budget mode; if unset, falls back to a small default.")
    parser.add_argument("--output", "--out", dest="output", default=None)
    args = parser.parse_args()
    if args.points_per_sample is not None:
        args.pc_points = args.points_per_sample

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = AnymateStaticRigDataset(args.pt, f"{args.splits_dir}/{args.split}.jsonl", max_joints=args.max_nodes, pc_points=args.pc_points)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = HyperBoneStaticParentModel(
        in_channels=3,
        feat_dim=512,
        max_joints=args.max_nodes,
        predict_skinning=False,
        backbone=args.backbone,
        knn_k=args.knn_k,
        parent_head=getattr(args, "parent_head", "slot"),
        root_bias_init=getattr(args, "root_bias_init", None),
        root_feature_mode=getattr(args, "root_feature_mode", "none"),
    ).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)

    decode_cfg = ParentDecodeConfig(
        decode_mode=args.decode_mode,
        active_threshold=args.active_threshold,
        max_degree=args.max_degree,
        enforce_root_per_component=args.enforce_root_per_component,
        two_stage_root_parent=getattr(args, 'two_stage_root_parent', False),
        root_threshold=getattr(args, 'root_threshold', None),
        root_decode_mode=getattr(args, 'root_decode_mode', 'threshold'),
        root_budget_ratio=getattr(args, 'root_budget_ratio', None),
    )
    metrics = evaluate(model, loader, device, args.active_threshold, decode_cfg,
                       teacher_force_gt_nodes=args.teacher_force_gt_nodes,
                       no_backbone_for_gt_nodes=getattr(args, "no_backbone_for_gt_nodes", False),
                       root_decode_bias=args.root_decode_bias,
                       candidate_parent_k=args.candidate_parent_k,
                       force_gt_parent_candidate=args.force_gt_parent_candidate,
                       two_stage_root_parent=getattr(args, 'two_stage_root_parent', False),
                       root_ambiguity_eps=getattr(args, 'root_ambiguity_eps', 0.0),
                       root_decode_mode=getattr(args, 'root_decode_mode', 'threshold'),
                       root_budget_ratio=getattr(args, 'root_budget_ratio', None))
    print(json.dumps(metrics, indent=2))

    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
