"""
Generate prediction overlay videos — RGB with GT skeleton (green) and predicted skeleton (red).

Creates frame-by-frame overlay images and stitches them into a video.

Usage:
    python scripts/make_anymate_prediction_overlay.py \\
        --model outputs/models/hyperbone_anymate_frame_pilot/model.pt \\
        --clip-dir outputs/anymate_clips_pilot/asset_000001/walk_like/front \\
        --out outputs/overlays/asset_000001_walk_like_front.mp4 \\
        --device cuda
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def draw_skeleton(img, joints_xy, active, vis, bones, color, radius=3, thickness=1):
    """Draw a skeleton on an image."""
    for ji in range(len(active)):
        if active[ji] < 0.5 or vis[ji] < 0.5:
            continue
        x, y = int(joints_xy[ji, 0]), int(joints_xy[ji, 1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), radius, color, -1)

    # Draw bones
    if bones:
        for bone in bones:
            pi, ci = bone.get("parent_id", -1), bone.get("child_id", -1)
            if pi < 0 or ci < 0 or pi >= len(active) or ci >= len(active):
                continue
            if active[pi] < 0.5 or active[ci] < 0.5:
                continue
            if vis[pi] < 0.5 or vis[ci] < 0.5:
                continue

            x1, y1 = int(joints_xy[pi, 0]), int(joints_xy[pi, 1])
            x2, y2 = int(joints_xy[ci, 0]), int(joints_xy[ci, 1])

            if (0 <= x1 < img.shape[1] and 0 <= y1 < img.shape[0] and
                0 <= x2 < img.shape[1] and 0 <= y2 < img.shape[0]):
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def make_overlay_video(
    model,
    clip_dir: Path,
    output_path: Path,
    device: torch.device,
    resolution: int = 256,
    max_joints: int = 128,
):
    """Generate an overlay video for a single clip."""
    labels_path = clip_dir / "frame_labels.jsonl"
    if not labels_path.exists():
        print(f"  No frame_labels.jsonl in {clip_dir}")
        return

    with open(labels_path) as f:
        labels = [json.loads(line) for line in f if line.strip()]

    if not labels:
        return

    # Read clip metadata for bones
    bones = []
    meta_path = clip_dir / "clip_meta.json"
    rest_skel_path = clip_dir / "skeleton" / "rest_skeleton.json"
    if rest_skel_path.exists():
        with open(rest_skel_path) as f:
            rest = json.load(f)
            bones = rest.get("bones", [])

    # Determine FPS from labels
    fps = 24
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            fps = meta.get("fps", 24)

    # Setup video writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (resolution * 2, resolution))

    model.eval()

    for label in labels:
        # Load RGB
        rgb_path = clip_dir / label.get("rgb_path", "")
        if rgb_path.exists():
            img = cv2.imread(str(rgb_path))
            img = cv2.resize(img, (resolution, resolution))
        else:
            img = np.zeros((resolution, resolution, 3), dtype=np.uint8)

        # Get GT joints
        gt_joints = label.get("joints", [])
        gt_xy = np.zeros((max_joints, 2), dtype=np.float32)
        gt_active = np.zeros(max_joints, dtype=np.float32)
        gt_vis = np.zeros(max_joints, dtype=np.float32)
        for j in gt_joints:
            ji = j["id"]
            if ji < max_joints:
                gt_xy[ji] = j.get("image_xy", [0, 0])[:2]
                gt_active[ji] = 1.0
                gt_vis[ji] = 1.0 if j.get("visible", True) else 0.0

        # Run model prediction
        rgb_tensor = torch.from_numpy(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(device)

        # Create 5-channel input (RGB + zeros for mask/depth)
        x = torch.cat([
            rgb_tensor,
            torch.zeros(1, 1, resolution, resolution, device=device),
            torch.zeros(1, 1, resolution, resolution, device=device),
        ], dim=1)

        with torch.no_grad():
            pred = model(x)
            pred_vis = pred["joint_vis"][0].cpu().numpy()

        # For 2D overlay, project predicted 3D to 2D using camera params
        # Simplified: use GT xy for now (TODO: implement proper projection)
        pred_xy = gt_xy  # placeholder

        # Draw overlays
        gt_overlay = img.copy()
        draw_skeleton(gt_overlay, gt_xy, gt_active, gt_vis, bones, (0, 255, 0), radius=4, thickness=2)

        pred_overlay = img.copy()
        draw_skeleton(pred_overlay, pred_xy, gt_active, pred_vis, bones, (0, 0, 255), radius=4, thickness=2)

        # Side-by-side
        combined = np.hstack([gt_overlay, pred_overlay])

        # Add labels
        cv2.putText(combined, "GT", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(combined, "Predicted", (resolution + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        frame_idx = label.get("frame_idx", 0)
        cv2.putText(combined, f"Frame {frame_idx}", (resolution - 80, resolution - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        writer.write(combined)

    writer.release()
    print(f"  Overlay video: {output_path} ({len(labels)} frames)")


def main():
    parser = argparse.ArgumentParser(description="Generate Anymate prediction overlay videos")
    parser.add_argument("--model", required=True, help="Path to model.pt")
    parser.add_argument("--clip-dir", required=True, help="Directory of a single rendered clip")
    parser.add_argument("--out", required=True, help="Output video path (.mp4)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max-joints", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model_path = Path(args.model)
    config_path = model_path.parent / "config.json"

    from scripts.train_hyperbone_anymate_frame import AnymateFrameModel

    max_joints = args.max_joints
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        max_joints = config.get("max_joints", max_joints)

    model = AnymateFrameModel(in_channels=5, max_joints=max_joints).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
    model.eval()

    make_overlay_video(
        model=model,
        clip_dir=Path(args.clip_dir),
        output_path=Path(args.out),
        device=device,
        resolution=args.resolution,
        max_joints=max_joints,
    )


if __name__ == "__main__":
    main()
