"""
Evaluate HyperBone Anymate Frame Model — metrics + prediction overlays.

Usage:
    python scripts/eval_hyperbone_anymate.py \
        --model outputs/models/hyperbone_anymate_frame_pilot/best_model.pt \
        --dataset outputs/anymate_clips_pilot/val.jsonl \
        --out outputs/models/hyperbone_anymate_frame_pilot/eval \
        --make-overlays
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

from hyperbone.datasets.anymate_clip_dataset import AnymateClipDataset
from scripts.train_hyperbone_anymate_frame import AnymateFrameModel


def load_model(model_path: str, device: torch.device, max_joints: int = 128):
    config_path = Path(model_path).parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        max_joints = cfg.get("max_joints", max_joints)
        in_channels = cfg.get("in_channels", 5)
    else:
        in_channels = 5

    model = AnymateFrameModel(in_channels=in_channels, max_joints=max_joints)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, max_joints


def project_3d_to_2d(xyz: np.ndarray, camera: dict) -> np.ndarray:
    """Project 3D world points to 2D image coords using camera params."""
    K = np.array(camera["K"])
    ext = np.array(camera["extrinsic"])
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


def evaluate(model, dataset, raw_labels: list, device: torch.device) -> dict:
    """Full evaluation with all metrics."""
    results = {
        "per_sample": [],
        "per_joint_errors": defaultdict(list),
        "all_3d_errors": [],
        "all_2d_errors": [],
        "all_bone_errors": [],
        "vis_correct": 0,
        "vis_total": 0,
    }

    # Index raw labels by (clip_key, frame_idx)
    label_index = {}
    for rl in raw_labels:
        key = (f"{rl.get('asset_id', '')}_{rl.get('animation_id', '')}", rl.get("frame_idx", -1))
        label_index[key] = rl

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            rgb = sample["rgb"].unsqueeze(0).to(device)
            mask = sample["mask"].unsqueeze(0).to(device)
            depth = sample["depth"].unsqueeze(0).to(device)
            x = torch.cat([rgb, mask, depth], dim=1)

            pred = model(x)
            pred_xyz = pred["joint_xyz"][0].cpu().numpy()
            pred_vis = pred["joint_vis"][0].cpu().numpy()

            gt_xyz = sample["joint_xyz"].numpy()
            gt_vis = sample["joint_vis"].numpy()
            gt_active = sample["joint_active"].numpy()
            gt_xy = sample["joint_xy"].numpy()
            frame_idx = sample["frame_idx"]
            clip_key = sample["clip_key"]

            rl = label_index.get((clip_key, frame_idx))
            camera = rl.get("camera") if rl else None

            eval_mask = (gt_active > 0) & (gt_vis > 0)

            # 3D errors
            errors_3d = np.linalg.norm(pred_xyz - gt_xyz, axis=1)
            valid_3d = errors_3d[eval_mask]
            results["all_3d_errors"].extend(valid_3d.tolist())

            for j_idx in np.where(eval_mask)[0]:
                results["per_joint_errors"][int(j_idx)].append(errors_3d[j_idx])

            # 2D reprojection
            reproj_err_mean = None
            if camera:
                pred_2d = project_3d_to_2d(pred_xyz, camera)
                errors_2d = np.linalg.norm(pred_2d - gt_xy, axis=1)
                valid_2d = errors_2d[eval_mask]
                results["all_2d_errors"].extend(valid_2d.tolist())
                reproj_err_mean = float(valid_2d.mean()) if len(valid_2d) > 0 else None

            # Visibility
            for j_idx in range(len(gt_active)):
                if gt_active[j_idx] > 0:
                    results["vis_total"] += 1
                    if (pred_vis[j_idx] > 0.5) == (gt_vis[j_idx] > 0.5):
                        results["vis_correct"] += 1

            # Bone length error
            active_idx = np.where(eval_mask)[0]
            if len(active_idx) >= 2:
                for i in range(len(active_idx) - 1):
                    j1, j2 = active_idx[i], active_idx[i + 1]
                    gt_len = np.linalg.norm(gt_xyz[j1] - gt_xyz[j2])
                    pred_len = np.linalg.norm(pred_xyz[j1] - pred_xyz[j2])
                    if gt_len > 0.01:
                        results["all_bone_errors"].append(abs(pred_len - gt_len) / gt_len)

            results["per_sample"].append({
                "idx": idx,
                "frame_idx": frame_idx,
                "clip_key": clip_key,
                "asset_id": rl.get("asset_id", "") if rl else "",
                "mpjpe": float(valid_3d.mean()) if len(valid_3d) > 0 else 0.0,
                "n_joints": int(eval_mask.sum()),
                "reproj_error_px": reproj_err_mean,
            })

    return results


def summarize(results: dict) -> dict:
    all_3d = np.array(results["all_3d_errors"])
    all_2d = np.array(results["all_2d_errors"]) if results["all_2d_errors"] else np.array([])
    all_bone = np.array(results["all_bone_errors"]) if results["all_bone_errors"] else np.array([])

    metrics = {
        "val_mpjpe": float(all_3d.mean()) if len(all_3d) > 0 else None,
        "val_mpjpe_median": float(np.median(all_3d)) if len(all_3d) > 0 else None,
        "pck_005": float((all_3d < 0.05).mean()) if len(all_3d) > 0 else 0.0,
        "pck_010": float((all_3d < 0.10).mean()) if len(all_3d) > 0 else 0.0,
        "pck_020": float((all_3d < 0.20).mean()) if len(all_3d) > 0 else 0.0,
        "reproj_error_px_mean": float(all_2d.mean()) if len(all_2d) > 0 else None,
        "reproj_error_px_median": float(np.median(all_2d)) if len(all_2d) > 0 else None,
        "bone_length_error_pct": float(all_bone.mean() * 100) if len(all_bone) > 0 else None,
        "visibility_accuracy": results["vis_correct"] / max(results["vis_total"], 1),
        "total_joints_evaluated": len(all_3d),
        "total_samples": len(results["per_sample"]),
    }

    # Per-joint table
    per_joint = {}
    for j_idx, errs in sorted(results["per_joint_errors"].items()):
        per_joint[f"joint_{j_idx}"] = {"mpjpe": float(np.mean(errs)), "count": len(errs)}
    metrics["per_joint"] = per_joint

    sorted_joints = sorted(per_joint.items(), key=lambda x: x[1]["mpjpe"], reverse=True)
    metrics["worst_5_joints"] = sorted_joints[:5]

    sorted_samples = sorted(results["per_sample"], key=lambda x: x["mpjpe"])
    metrics["best_3_samples"] = sorted_samples[:3]
    metrics["worst_3_samples"] = sorted_samples[-3:]

    return metrics


def make_overlays(model, dataset, raw_labels: list, output_dir: Path,
                  device: torch.device, max_samples: int = 30):
    """Generate overlay images: GT (green) vs predicted (red) joints."""
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    root_dir = Path(dataset.root_dir)

    label_index = {}
    for rl in raw_labels:
        key = (f"{rl.get('asset_id', '')}_{rl.get('animation_id', '')}", rl.get("frame_idx", -1))
        label_index[key] = rl

    n = min(max_samples, len(dataset))

    with torch.no_grad():
        for idx in range(n):
            sample = dataset[idx]
            rgb = sample["rgb"].unsqueeze(0).to(device)
            mask_t = sample["mask"].unsqueeze(0).to(device)
            depth_t = sample["depth"].unsqueeze(0).to(device)
            x = torch.cat([rgb, mask_t, depth_t], dim=1)

            pred = model(x)
            pred_xyz = pred["joint_xyz"][0].cpu().numpy()

            gt_active = sample["joint_active"].numpy()
            gt_vis = sample["joint_vis"].numpy()
            gt_xy = sample["joint_xy"].numpy()
            frame_idx = sample["frame_idx"]
            clip_key = sample["clip_key"]

            rl = label_index.get((clip_key, frame_idx))
            if not rl:
                continue
            camera = rl.get("camera")
            if not camera:
                continue

            # Load RGB image
            rgb_path = root_dir / rl["rgb_path"]
            if not rgb_path.exists():
                continue
            img = cv2.imread(str(rgb_path))
            if img is None:
                continue

            H, W = img.shape[:2]
            eval_mask = (gt_active > 0) & (gt_vis > 0)

            # Project predicted 3D → 2D
            pred_2d = project_3d_to_2d(pred_xyz, camera)

            # Draw GT (green) and Pred (red)
            for j_idx in np.where(eval_mask)[0]:
                gx, gy = int(round(gt_xy[j_idx, 0])), int(round(gt_xy[j_idx, 1]))
                px, py = int(round(pred_2d[j_idx, 0])), int(round(pred_2d[j_idx, 1]))

                if 0 <= gx < W and 0 <= gy < H:
                    cv2.circle(img, (gx, gy), 3, (0, 255, 0), -1)
                if 0 <= px < W and 0 <= py < H:
                    cv2.circle(img, (px, py), 3, (0, 0, 255), -1)
                if (0 <= gx < W and 0 <= gy < H and 0 <= px < W and 0 <= py < H):
                    cv2.line(img, (gx, gy), (px, py), (0, 255, 255), 1)

            # Metrics text
            errors = np.linalg.norm(pred_xyz - sample["joint_xyz"].numpy(), axis=1)
            mpjpe = float(errors[eval_mask].mean()) if eval_mask.any() else 0.0
            errs_2d = np.linalg.norm(pred_2d - gt_xy, axis=1)
            reproj = float(errs_2d[eval_mask].mean()) if eval_mask.any() else 0.0

            asset_short = rl.get("asset_id", "")[-20:]
            lines = [
                f"{asset_short} f{frame_idx}",
                f"MPJPE:{mpjpe:.3f} Reproj:{reproj:.1f}px",
                "Green=GT Red=Pred",
            ]
            for i, txt in enumerate(lines):
                cv2.putText(img, txt, (3, 12 + i * 13),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            cv2.imwrite(str(output_dir / f"pred_{idx:04d}.png"), img)

    print(f"[Eval] {n} overlays → {output_dir}")


def write_report(metrics: dict, output_dir: Path, model_dir: Path):
    """Write eval report (JSON + markdown)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training history
    hist_path = model_dir / "training_history.json"
    train_loss_start = train_loss_end = val_mpjpe_start = val_mpjpe_end = None
    if hist_path.exists():
        with open(hist_path) as f:
            hist = json.load(f)
        if hist:
            train_loss_start = hist[0].get("train_loss")
            train_loss_end = hist[-1].get("train_loss")
            val_mpjpe_start = hist[0].get("mpjpe")
            val_mpjpe_end = hist[-1].get("mpjpe")

    reproj = metrics.get("reproj_error_px_mean")
    bone_err = metrics.get("bone_length_error_pct", 100)
    loss_decreases = (train_loss_start is not None and train_loss_end is not None
                      and train_loss_end < train_loss_start)

    if reproj is not None and reproj < 25 and (bone_err is None or bone_err < 10):
        verdict = "FRAME_MODEL_PASS"
    elif loss_decreases and reproj is not None and reproj < 50:
        verdict = "FRAME_MODEL_PARTIAL"
    elif loss_decreases and metrics.get("val_mpjpe", 1) < 0.5:
        verdict = "FRAME_MODEL_PARTIAL"
    else:
        verdict = "FRAME_MODEL_FAIL"

    # JSON
    out_json = {
        "verdict": verdict,
        "train_loss_start": train_loss_start,
        "train_loss_end": train_loss_end,
        "val_mpjpe_start": val_mpjpe_start,
        "val_mpjpe_end": val_mpjpe_end,
        **{k: v for k, v in metrics.items()
           if k not in ("per_joint", "worst_5_joints", "best_3_samples", "worst_3_samples")},
        "worst_5_joints": [(n, d) for n, d in metrics.get("worst_5_joints", [])],
        "best_3_samples": metrics.get("best_3_samples", []),
        "worst_3_samples": metrics.get("worst_3_samples", []),
    }
    with open(output_dir / "eval_metrics.json", "w") as f:
        json.dump(out_json, f, indent=2)

    # Markdown
    md = f"""# HyperBone Anymate Frame Model — Evaluation

## Verdict: **{verdict}**

## Training
| Metric | Value |
|--------|-------|
| Train loss start | {f'{train_loss_start:.4f}' if train_loss_start else 'N/A'} |
| Train loss end | {f'{train_loss_end:.4f}' if train_loss_end else 'N/A'} |
| Val MPJPE start | {f'{val_mpjpe_start:.4f}' if val_mpjpe_start else 'N/A'} |
| Val MPJPE end | {f'{val_mpjpe_end:.4f}' if val_mpjpe_end else 'N/A'} |
| Loss decreases | {'YES' if loss_decreases else 'NO'} |

## Metrics
| Metric | Value |
|--------|-------|
| Val MPJPE | {f"{metrics['val_mpjpe']:.4f}" if metrics['val_mpjpe'] else 'N/A'} |
| Val MPJPE (median) | {f"{metrics['val_mpjpe_median']:.4f}" if metrics['val_mpjpe_median'] else 'N/A'} |
| PCK-3D @ 0.05 | {metrics['pck_005']:.3f} |
| PCK-3D @ 0.10 | {metrics['pck_010']:.3f} |
| PCK-3D @ 0.20 | {metrics['pck_020']:.3f} |
| 2D Reproj Error (mean) | {f'{reproj:.1f}' if reproj else 'N/A'}px |
| 2D Reproj Error (median) | {f"{metrics['reproj_error_px_median']:.1f}" if metrics['reproj_error_px_median'] else 'N/A'}px |
| Bone Length Error | {f'{bone_err:.1f}' if bone_err is not None else 'N/A'}% |
| Visibility Accuracy | {metrics['visibility_accuracy']:.3f} |
| Total Joints Evaluated | {metrics['total_joints_evaluated']} |
| Total Samples | {metrics['total_samples']} |

## Worst 5 Joints
| Joint | MPJPE | Count |
|-------|-------|-------|
"""
    for name, info in metrics.get("worst_5_joints", []):
        md += f"| {name} | {info['mpjpe']:.4f} | {info['count']} |\n"

    md += "\n## Best 3 Samples\n| Asset | Frame | MPJPE | Reproj |\n|-------|-------|-------|--------|\n"
    for s in metrics.get("best_3_samples", []):
        rp = f"{s['reproj_error_px']:.1f}px" if s.get('reproj_error_px') else "N/A"
        md += f"| {s['asset_id'][:25]} | {s['frame_idx']} | {s['mpjpe']:.4f} | {rp} |\n"

    md += "\n## Worst 3 Samples\n| Asset | Frame | MPJPE | Reproj |\n|-------|-------|-------|--------|\n"
    for s in metrics.get("worst_3_samples", []):
        rp = f"{s['reproj_error_px']:.1f}px" if s.get('reproj_error_px') else "N/A"
        md += f"| {s['asset_id'][:25]} | {s['frame_idx']} | {s['mpjpe']:.4f} | {rp} |\n"

    md += f"\n## Overlay Folder\n`{output_dir / 'overlays'}`\n"
    md += f"\n## Pass/Fail\n- Train loss decreases: {'YES' if loss_decreases else 'NO'}\n"
    md += f"- Reproj < 25px: {'YES' if reproj and reproj < 25 else 'NO'}\n"
    md += f"- Bone error < 10%: {'YES' if bone_err is not None and bone_err < 10 else 'NO'}\n"

    with open(output_dir / "eval_report.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[Eval] Report → {output_dir / 'eval_report.md'}")
    return verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--make-overlays", action="store_true")
    parser.add_argument("--max-overlays", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max-joints", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.out)
    model_dir = Path(args.model).parent

    print(f"[Eval] Model: {args.model}")
    print(f"[Eval] Dataset: {args.dataset}")

    model, max_joints = load_model(args.model, device, args.max_joints)
    dataset = AnymateClipDataset(
        args.dataset, clip_len=1, resolution=args.resolution,
        max_joints=max_joints, use_mask=True, use_depth=True,
    )
    print(f"[Eval] Samples: {len(dataset)}")

    # Load raw labels for camera info
    raw_labels = []
    with open(args.dataset) as f:
        for line in f:
            if line.strip():
                raw_labels.append(json.loads(line))

    t0 = time.time()
    results = evaluate(model, dataset, raw_labels, device)
    print(f"[Eval] Inference: {time.time()-t0:.1f}s")

    metrics = summarize(results)

    print(f"\n[Eval] Val MPJPE: {metrics['val_mpjpe']:.4f}")
    print(f"[Eval] PCK@0.05/0.10/0.20: {metrics['pck_005']:.3f} / {metrics['pck_010']:.3f} / {metrics['pck_020']:.3f}")
    if metrics["reproj_error_px_mean"] is not None:
        print(f"[Eval] Reproj: {metrics['reproj_error_px_mean']:.1f}px mean, {metrics['reproj_error_px_median']:.1f}px median")
    if metrics["bone_length_error_pct"] is not None:
        print(f"[Eval] Bone error: {metrics['bone_length_error_pct']:.1f}%")
    print(f"[Eval] Vis acc: {metrics['visibility_accuracy']:.3f}")

    if args.make_overlays:
        make_overlays(model, dataset, raw_labels, output_dir / "overlays",
                      device, max_samples=args.max_overlays)

    verdict = write_report(metrics, output_dir, model_dir)
    print(f"\n[Eval] VERDICT: {verdict}")


if __name__ == "__main__":
    main()
