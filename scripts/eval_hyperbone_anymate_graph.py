"""
Evaluate graph-mode rig prediction model.

Metrics: node F1, matched MPJPE, PCK, reproj error, edge F1, bone length error.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.models.hyperbone_rig_graph_frame import GraphTokenFrameModel
from hyperbone.rigs.graph_losses import hungarian_match
from hyperbone.rigs.topology import N_JOINT_TYPES, JOINT_TYPE_NAMES
from scripts.train_hyperbone_anymate_frame_graph import AnymateGraphDataset


def project_3d_to_2d(xyz: np.ndarray, K: np.ndarray, ext: np.ndarray) -> np.ndarray:
    N = xyz.shape[0]
    pts_h = np.hstack([xyz, np.ones((N, 1))])
    pts_cam = (ext @ pts_h.T).T[:, :3]
    proj_2d = np.zeros((N, 2))
    for i in range(N):
        if pts_cam[i, 2] > 0.01:
            p = K @ pts_cam[i]
            proj_2d[i, 0] = p[0] / p[2]
            proj_2d[i, 1] = p[1] / p[2]
    return proj_2d


def evaluate_full(model, dataset, device, max_samples=-1):
    model.eval()
    results = {
        "all_3d_errors": [],
        "all_2d_errors": [],
        "all_bone_errors": [],
        "n_matched": 0,
        "n_gt": 0,
        "n_pred_active": 0,
        "edge_tp": 0, "edge_fp": 0, "edge_fn": 0,
        "per_sample": [],
    }

    n = len(dataset) if max_samples < 0 else min(max_samples, len(dataset))

    with torch.no_grad():
        for idx in range(n):
            sample = dataset[idx]
            x = torch.cat([sample["rgb"], sample["mask"], sample["depth"]], dim=0).unsqueeze(0).to(device)
            pred = model(x)

            gt_xyz = sample["gt_xyz"].unsqueeze(0).to(device)
            gt_active = sample["gt_active"].unsqueeze(0).to(device)
            gt_vis = sample["gt_vis"].unsqueeze(0)
            gt_xy = sample["gt_xy"].numpy()
            gt_adj = sample["gt_adj"].numpy()
            K = sample["camera_K"].numpy()
            ext = sample["camera_ext"].numpy()

            matches = hungarian_match(pred["node_xyz"], gt_xyz, gt_active, pred["node_active"])
            pred_idx, gt_idx = matches[0]

            pred_xyz_np = pred["node_xyz"][0].cpu().numpy()
            pred_active_np = pred["node_active"][0].cpu().numpy()
            gt_xyz_np = sample["gt_xyz"].numpy()
            gt_active_np = sample["gt_active"].numpy()

            n_gt = int((gt_active_np > 0.5).sum())
            n_pred_active = int((pred_active_np > 0.5).sum())
            results["n_gt"] += n_gt
            results["n_pred_active"] += n_pred_active
            results["n_matched"] += len(pred_idx)

            # 3D error on matched
            sample_mpjpe = 0.0
            if len(pred_idx) > 0:
                p_idx = pred_idx.cpu().numpy()
                g_idx = gt_idx.cpu().numpy()
                errors_3d = np.linalg.norm(pred_xyz_np[p_idx] - gt_xyz_np[g_idx], axis=1)
                results["all_3d_errors"].extend(errors_3d.tolist())
                sample_mpjpe = float(errors_3d.mean())

                # 2D reprojection
                pred_2d = project_3d_to_2d(pred_xyz_np[p_idx], K, ext)
                gt_2d_matched = gt_xy[g_idx]
                vis_matched = gt_vis[0, g_idx].numpy() > 0.5
                if vis_matched.any():
                    errs_2d = np.linalg.norm(pred_2d[vis_matched] - gt_2d_matched[vis_matched], axis=1)
                    results["all_2d_errors"].extend(errs_2d.tolist())

                # Edge evaluation on matched subset
                M = len(p_idx)
                if M >= 2:
                    edge_pred = (pred["edge_logits"][0].cpu().numpy()[p_idx][:, p_idx] > 0).astype(float)
                    edge_gt = gt_adj[g_idx][:, g_idx]
                    tp = ((edge_pred > 0.5) & (edge_gt > 0.5)).sum()
                    fp = ((edge_pred > 0.5) & (edge_gt < 0.5)).sum()
                    fn = ((edge_pred < 0.5) & (edge_gt > 0.5)).sum()
                    results["edge_tp"] += int(tp)
                    results["edge_fp"] += int(fp)
                    results["edge_fn"] += int(fn)

                # Bone length error
                for i in range(M):
                    for j in range(M):
                        if gt_adj[g_idx[i], g_idx[j]] > 0.5:
                            gt_len = np.linalg.norm(gt_xyz_np[g_idx[i]] - gt_xyz_np[g_idx[j]])
                            pred_len = np.linalg.norm(pred_xyz_np[p_idx[i]] - pred_xyz_np[p_idx[j]])
                            if gt_len > 0.01:
                                results["all_bone_errors"].append(abs(pred_len - gt_len) / gt_len)

            results["per_sample"].append({
                "idx": idx, "mpjpe": sample_mpjpe,
                "n_gt": n_gt, "n_pred": n_pred_active, "n_matched": len(pred_idx),
            })

    return results


def summarize(results):
    e3d = np.array(results["all_3d_errors"]) if results["all_3d_errors"] else np.array([999.0])
    e2d = np.array(results["all_2d_errors"]) if results["all_2d_errors"] else np.array([])
    ebone = np.array(results["all_bone_errors"]) if results["all_bone_errors"] else np.array([])

    prec = results["n_matched"] / max(results["n_pred_active"], 1)
    rec = results["n_matched"] / max(results["n_gt"], 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    edge_prec = results["edge_tp"] / max(results["edge_tp"] + results["edge_fp"], 1)
    edge_rec = results["edge_tp"] / max(results["edge_tp"] + results["edge_fn"], 1)
    edge_f1 = 2 * edge_prec * edge_rec / max(edge_prec + edge_rec, 1e-8)

    return {
        "mpjpe": float(e3d.mean()),
        "mpjpe_median": float(np.median(e3d)),
        "pck_005": float((e3d < 0.05).mean()),
        "pck_010": float((e3d < 0.10).mean()),
        "pck_020": float((e3d < 0.20).mean()),
        "reproj_error_px_mean": float(e2d.mean()) if len(e2d) > 0 else None,
        "reproj_error_px_median": float(np.median(e2d)) if len(e2d) > 0 else None,
        "bone_length_error_pct": float(ebone.mean() * 100) if len(ebone) > 0 else None,
        "node_precision": prec,
        "node_recall": rec,
        "node_f1": f1,
        "edge_precision": edge_prec,
        "edge_recall": edge_rec,
        "edge_f1": edge_f1,
        "n_matched": results["n_matched"],
        "n_gt": results["n_gt"],
        "n_pred_active": results["n_pred_active"],
        "total_samples": len(results["per_sample"]),
    }


def make_overlays(model, dataset, output_dir, device, max_samples=20):
    import cv2
    output_dir.mkdir(parents=True, exist_ok=True)
    root_dir = dataset.root_dir

    with torch.no_grad():
        for idx in range(min(max_samples, len(dataset))):
            sample = dataset[idx]
            x = torch.cat([sample["rgb"], sample["mask"], sample["depth"]], dim=0).unsqueeze(0).to(device)
            pred = model(x)

            gt_xyz = sample["gt_xyz"].unsqueeze(0).to(device)
            gt_active = sample["gt_active"].unsqueeze(0).to(device)
            matches = hungarian_match(pred["node_xyz"], gt_xyz, gt_active, pred["node_active"])
            pred_idx, gt_idx = matches[0]

            gt_xy = sample["gt_xy"].numpy()
            K = sample["camera_K"].numpy()
            ext = sample["camera_ext"].numpy()
            pred_xyz_np = pred["node_xyz"][0].cpu().numpy()

            # Load image
            label = dataset.labels[idx]
            rgb_path = root_dir / label.get("rgb_path", "")
            if not rgb_path.exists():
                continue
            img = cv2.imread(str(rgb_path))
            if img is None:
                continue
            H, W = img.shape[:2]

            gt_active_np = sample["gt_active"].numpy()
            gt_vis_np = sample["gt_vis"].numpy()

            # Draw GT (green)
            for j in range(len(gt_active_np)):
                if gt_active_np[j] > 0.5 and gt_vis_np[j] > 0.5:
                    gx, gy = int(round(gt_xy[j, 0])), int(round(gt_xy[j, 1]))
                    if 0 <= gx < W and 0 <= gy < H:
                        cv2.circle(img, (gx, gy), 3, (0, 255, 0), -1)

            # Draw predicted active nodes (red)
            pred_active_np = pred["node_active"][0].cpu().numpy()
            pred_2d = project_3d_to_2d(pred_xyz_np, K, ext)
            for j in range(len(pred_active_np)):
                if pred_active_np[j] > 0.5:
                    px, py = int(round(pred_2d[j, 0])), int(round(pred_2d[j, 1]))
                    if 0 <= px < W and 0 <= py < H:
                        cv2.circle(img, (px, py), 2, (0, 0, 255), -1)

            # Draw match lines (yellow)
            if len(pred_idx) > 0:
                p_idx = pred_idx.cpu().numpy()
                g_idx = gt_idx.cpu().numpy()
                for pi, gi in zip(p_idx, g_idx):
                    if gt_vis_np[gi] > 0.5:
                        gx, gy = int(round(gt_xy[gi, 0])), int(round(gt_xy[gi, 1]))
                        px, py = int(round(pred_2d[pi, 0])), int(round(pred_2d[pi, 1]))
                        if 0 <= gx < W and 0 <= gy < H and 0 <= px < W and 0 <= py < H:
                            cv2.line(img, (gx, gy), (px, py), (0, 255, 255), 1)

            # Draw predicted edges (blue)
            edge_logits = pred["edge_logits"][0].cpu().numpy()
            active_nodes = np.where(pred_active_np > 0.5)[0]
            for i in active_nodes:
                for j in active_nodes:
                    if i < j and edge_logits[i, j] > 0:
                        pi = pred_2d[i]
                        pj = pred_2d[j]
                        x1, y1 = int(round(pi[0])), int(round(pi[1]))
                        x2, y2 = int(round(pj[0])), int(round(pj[1]))
                        if (0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H):
                            cv2.line(img, (x1, y1), (x2, y2), (255, 100, 0), 1)

            # Text
            errors_3d = np.linalg.norm(pred_xyz_np[pred_idx.cpu().numpy()] - sample["gt_xyz"].numpy()[gt_idx.cpu().numpy()], axis=1) if len(pred_idx) > 0 else np.array([0])
            mpjpe = float(errors_3d.mean())
            cv2.putText(img, f"MPJPE:{mpjpe:.3f} M:{len(pred_idx)} GT:{int(gt_active_np.sum())}",
                       (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cv2.putText(img, "Green=GT Red=Pred Blue=Edge", (3, H - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

            cv2.imwrite(str(output_dir / f"graph_pred_{idx:04d}.png"), img)

    print(f"[GraphEval] {min(max_samples, len(dataset))} overlays -> {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--make-overlays", action="store_true")
    parser.add_argument("--max-overlays", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max-nodes", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = GraphTokenFrameModel(in_channels=5, max_nodes=args.max_nodes,
                                  n_node_types=N_JOINT_TYPES, base_dim=64, n_query_layers=3).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    dataset = AnymateGraphDataset(args.dataset, resolution=args.resolution, max_nodes=args.max_nodes)
    print(f"[GraphEval] Model: {args.model}")
    print(f"[GraphEval] Samples: {len(dataset)}")

    t0 = time.time()
    results = evaluate_full(model, dataset, device)
    print(f"[GraphEval] Inference: {time.time()-t0:.1f}s")

    metrics = summarize(results)

    print(f"\n[GraphEval] MPJPE: {metrics['mpjpe']:.4f}")
    print(f"[GraphEval] PCK@0.05/0.10/0.20: {metrics['pck_005']:.3f} / {metrics['pck_010']:.3f} / {metrics['pck_020']:.3f}")
    print(f"[GraphEval] Node P/R/F1: {metrics['node_precision']:.3f} / {metrics['node_recall']:.3f} / {metrics['node_f1']:.3f}")
    print(f"[GraphEval] Edge P/R/F1: {metrics['edge_precision']:.3f} / {metrics['edge_recall']:.3f} / {metrics['edge_f1']:.3f}")
    if metrics["reproj_error_px_mean"] is not None:
        print(f"[GraphEval] Reproj: {metrics['reproj_error_px_mean']:.1f}px mean")
    if metrics["bone_length_error_pct"] is not None:
        print(f"[GraphEval] Bone error: {metrics['bone_length_error_pct']:.1f}%")

    if args.make_overlays:
        make_overlays(model, dataset, output_dir / "overlays", device, max_samples=args.max_overlays)

    # Save metrics
    with open(output_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Verdict
    reproj = metrics.get("reproj_error_px_mean")
    bone = metrics.get("bone_length_error_pct", 100)
    if reproj and reproj < 25 and bone < 15 and metrics["node_f1"] > 0.3:
        verdict = "GRAPH_MODEL_PASS"
    elif metrics["mpjpe"] < 0.5 or (reproj and reproj < 50):
        verdict = "GRAPH_MODEL_PARTIAL"
    else:
        verdict = "GRAPH_MODEL_FAIL"

    print(f"\n[GraphEval] VERDICT: {verdict}")
    metrics["verdict"] = verdict
    with open(output_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
