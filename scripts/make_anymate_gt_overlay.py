"""
Generate GT overlay images/video — skeleton drawn on top of rendered frames.

Reads the dataset_index.jsonl produced by the pilot and creates overlays
showing projected joints and bones on the RGB frames.

Usage:
    python scripts/make_anymate_gt_overlay.py \
        --dataset outputs/anymate_clips_pilot/dataset_index.jsonl \
        --out outputs/anymate_clips_pilot/gt_overlays \
        --max-assets 5
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def draw_skeleton_on_frame(rgb: np.ndarray, joints: list,
                           bones: list = None, camera: dict = None,
                           joint_radius: int = 4, bone_thickness: int = 2) -> np.ndarray:
    """Draw projected joints and bones on an RGB frame."""
    import cv2

    overlay = rgb.copy()
    H, W = overlay.shape[:2]

    # Draw bones first (behind joints)
    if bones and camera:
        K = np.array(camera["K"])
        ext = np.array(camera["extrinsic"])

        for bone in bones:
            start_3d = np.array(bone["start_xyz"])
            end_3d = np.array(bone["end_xyz"])

            # Project both endpoints
            for pts_3d, color in [(start_3d, (100, 255, 100)), (end_3d, (100, 255, 100))]:
                pass

            # Project start and end to 2D
            start_h = np.append(start_3d, 1.0)
            end_h = np.append(end_3d, 1.0)
            start_cam = (ext @ start_h)[:3]
            end_cam = (ext @ end_h)[:3]

            if start_cam[2] > 0.01 and end_cam[2] > 0.01:
                s_proj = K @ start_cam
                e_proj = K @ end_cam
                sx = int(s_proj[0] / s_proj[2])
                sy = int(s_proj[1] / s_proj[2])
                ex = int(e_proj[0] / e_proj[2])
                ey = int(e_proj[1] / e_proj[2])

                if (0 <= sx < W and 0 <= sy < H and
                    0 <= ex < W and 0 <= ey < H):
                    cv2.line(overlay, (sx, sy), (ex, ey),
                             (0, 200, 0), bone_thickness)

    # Draw joints
    for j in joints:
        if not j.get("visible", False):
            continue
        xy = j.get("image_xy", [0, 0])
        px, py = int(round(xy[0])), int(round(xy[1]))
        if 0 <= px < W and 0 <= py < H:
            on_mask = j.get("on_mask", False)
            color = (0, 255, 0) if on_mask else (0, 0, 255)  # Green=on mask, Red=off mask
            cv2.circle(overlay, (px, py), joint_radius, color, -1)
            cv2.circle(overlay, (px, py), joint_radius + 1, (255, 255, 255), 1)

    return overlay


def make_overlays(dataset_path: str, output_dir: str, max_assets: int = 5,
                  make_video: bool = True):
    """Generate overlay images and optionally stitch into videos."""
    import cv2

    dataset_path = Path(dataset_path)
    root_dir = dataset_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read labels
    labels = []
    with open(dataset_path, "r") as f:
        for line in f:
            if line.strip():
                labels.append(json.loads(line))

    if not labels:
        print("[Overlay] ERROR: No labels found")
        return

    # Group by asset
    assets = defaultdict(list)
    for label in labels:
        assets[label["asset_id"]].append(label)

    asset_keys = list(assets.keys())[:max_assets]
    print(f"[Overlay] Processing {len(asset_keys)} assets, {len(labels)} total frames")

    all_overlay_paths = []
    video_paths = []

    for asset_id in asset_keys:
        asset_labels = sorted(assets[asset_id], key=lambda x: x["frame_idx"])
        safe_name = asset_labels[0]["rgb_path"].split("/")[0]
        asset_out = output_dir / safe_name
        asset_out.mkdir(parents=True, exist_ok=True)

        frame_paths = []

        for label in asset_labels:
            rgb_path = root_dir / label["rgb_path"]
            if not rgb_path.exists():
                continue

            rgb = cv2.imread(str(rgb_path))
            if rgb is None:
                continue

            overlay = draw_skeleton_on_frame(
                rgb,
                joints=label.get("joints", []),
                bones=label.get("bones", []),
                camera=label.get("camera"),
            )

            out_path = asset_out / f"overlay_{label['frame_idx']:04d}.png"
            cv2.imwrite(str(out_path), overlay)
            frame_paths.append(str(out_path))
            all_overlay_paths.append(out_path)

        # Make video from frames
        if make_video and frame_paths:
            video_path = output_dir / f"{safe_name}_overlay.mp4"
            fps = asset_labels[0].get("fps", 12)

            # Read first frame for dimensions
            first = cv2.imread(frame_paths[0])
            H, W = first.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (W, H))
            for fp in frame_paths:
                frame = cv2.imread(fp)
                if frame is not None:
                    writer.write(frame)
            writer.release()
            video_paths.append(video_path)
            print(f"  Video: {video_path} ({len(frame_paths)} frames)")

    # Summary
    print(f"\n[Overlay] Done!")
    print(f"  Overlay images: {len(all_overlay_paths)}")
    print(f"  Videos: {len(video_paths)}")
    print(f"  Output: {output_dir}")

    # Compute overlay quality stats
    n_green = 0
    n_red = 0
    for label in labels:
        for j in label.get("joints", []):
            if j.get("visible"):
                if j.get("on_mask"):
                    n_green += 1
                else:
                    n_red += 1

    total = n_green + n_red
    if total > 0:
        print(f"\n  Joint alignment quality:")
        print(f"    On-mask (green): {n_green} ({100*n_green/total:.1f}%)")
        print(f"    Off-mask (red):  {n_red} ({100*n_red/total:.1f}%)")
        if n_green / total < 0.7:
            print(f"    ⚠️  Low mask overlap — check camera projection!")
        else:
            print(f"    ✓ Good alignment")

    return all_overlay_paths


def main():
    parser = argparse.ArgumentParser(description="Generate GT skeleton overlays")
    parser.add_argument("--dataset", required=True, help="Path to dataset_index.jsonl")
    parser.add_argument("--out", default=None, help="Output directory for overlays")
    parser.add_argument("--max-assets", type=int, default=5, help="Max assets to process")
    parser.add_argument("--no-video", action="store_true", help="Skip video generation")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)

    output_dir = args.out or str(dataset_path.parent / "gt_overlays")

    make_overlays(
        dataset_path=str(dataset_path),
        output_dir=output_dir,
        max_assets=args.max_assets,
        make_video=not args.no_video,
    )


if __name__ == "__main__":
    main()
