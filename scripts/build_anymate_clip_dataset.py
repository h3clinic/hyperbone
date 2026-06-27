"""
Build Anymate Clip Dataset — iterate assets, generate/choose animations, render clips.

Orchestrates the full pipeline:
1. Read assets.jsonl index
2. For each asset: decide animation (use existing or generate synthetic)
3. Choose camera views
4. Call Blender to render each clip
5. Write dataset_index.jsonl

Usage:
    python scripts/build_anymate_clip_dataset.py \\
        --index outputs/anymate_index/assets.jsonl \\
        --out outputs/anymate_clips_pilot \\
        --max-assets 10 \\
        --motions-per-asset 1 \\
        --views-per-asset 1 \\
        --duration-sec 3 \\
        --fps 12 \\
        --resolution 256
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.rigs.schema import AssetRig, read_jsonl, write_jsonl


CAMERA_VIEWS = ["front", "side", "three_quarter", "top"]
MOTION_TYPES = ["idle_sway", "walk_like", "tail_swing", "hinge_sweep", "branch_bend", "generic_articulation"]


def find_blender() -> str:
    """Find Blender executable."""
    import shutil
    # Common paths
    candidates = [
        shutil.which("blender"),
        r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
        "/usr/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return "blender"  # Hope it's in PATH


def choose_motions(asset: dict, motions_per_asset: int) -> List[str]:
    """Choose which motions to apply to this asset."""
    has_anim = asset.get("has_animation", False)

    motions = []
    if has_anim:
        motions.append("original")

    # Add synthetic motions
    category = asset.get("category", "unknown")
    if category == "animal":
        preferred = ["walk_like", "tail_swing", "idle_sway"]
    elif category == "mechanical":
        preferred = ["hinge_sweep", "generic_articulation", "idle_sway"]
    elif category == "organic":
        preferred = ["branch_bend", "idle_sway", "generic_articulation"]
    else:
        preferred = ["idle_sway", "walk_like", "generic_articulation"]

    for m in preferred:
        if len(motions) >= motions_per_asset:
            break
        if m not in motions:
            motions.append(m)

    return motions[:motions_per_asset]


def render_clip(
    blender_path: str,
    asset_path: str,
    output_dir: str,
    motion: str,
    camera_view: str,
    duration_sec: float,
    fps: int,
    resolution: int,
) -> bool:
    """Call Blender to render a single clip."""
    script_path = str(Path(__file__).resolve().parent / "render_anymate_clip.py")

    cmd = [
        blender_path,
        "--background",
        "--python", script_path,
        "--",
        "--asset", asset_path,
        "--out", output_dir,
        "--duration-sec", str(duration_sec),
        "--fps", str(fps),
        "--resolution", str(resolution),
        "--motion", motion,
        "--camera-view", camera_view,
    ]

    if motion == "original":
        cmd.append("--use-existing-animation")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max per clip
        )
        if result.returncode == 0:
            return True
        else:
            print(f"    Blender error (exit {result.returncode}):")
            # Print last few lines of stderr
            stderr_lines = result.stderr.strip().split('\n')
            for line in stderr_lines[-5:]:
                print(f"      {line}")
            return False
    except subprocess.TimeoutExpired:
        print(f"    Blender timeout (>600s)")
        return False
    except FileNotFoundError:
        print(f"    ERROR: Blender not found at: {blender_path}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Build Anymate clip dataset")
    parser.add_argument("--index", required=True, help="Path to assets.jsonl")
    parser.add_argument("--out", required=True, help="Output directory for rendered clips")
    parser.add_argument("--max-assets", type=int, default=10, help="Max assets to process")
    parser.add_argument("--motions-per-asset", type=int, default=1, help="Motions per asset")
    parser.add_argument("--views-per-asset", type=int, default=1, help="Camera views per motion")
    parser.add_argument("--duration-sec", type=float, default=3.0, help="Clip duration")
    parser.add_argument("--fps", type=int, default=12, help="Frames per second")
    parser.add_argument("--resolution", type=int, default=256, help="Render resolution")
    parser.add_argument("--blender", default=None, help="Path to Blender executable")
    parser.add_argument("--skip-render", action="store_true", help="Only build index from existing renders")
    args = parser.parse_args()

    index_path = Path(args.index)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not index_path.exists():
        print(f"ERROR: Index not found: {index_path}")
        sys.exit(1)

    blender_path = args.blender or find_blender()
    print(f"[Dataset] Blender: {blender_path}")
    print(f"[Dataset] Index: {index_path}")
    print(f"[Dataset] Output: {output_dir}")

    # Load asset index
    assets = read_jsonl(index_path)

    # Filter to assets with skeletons
    rigged_assets = [a for a in assets if a.get("has_skeleton", False)]
    print(f"[Dataset] Total assets: {len(assets)}, Rigged: {len(rigged_assets)}")

    if args.max_assets > 0:
        rigged_assets = rigged_assets[:args.max_assets]

    print(f"[Dataset] Processing: {len(rigged_assets)} assets")
    print(f"[Dataset] Motions/asset: {args.motions_per_asset}")
    print(f"[Dataset] Views/motion: {args.views_per_asset}")
    print(f"[Dataset] Expected clips: {len(rigged_assets) * args.motions_per_asset * args.views_per_asset}")
    expected_frames = len(rigged_assets) * args.motions_per_asset * args.views_per_asset * int(args.duration_sec * args.fps)
    print(f"[Dataset] Expected frames: {expected_frames}")
    print()

    # Process assets
    dataset_records = []
    stats = {"processed": 0, "skipped": 0, "failed": 0, "frames": 0}

    t0 = time.time()

    for ai, asset in enumerate(rigged_assets):
        asset_id = asset["asset_id"]
        mesh_path = asset["mesh_path"]

        if not Path(mesh_path).exists():
            print(f"  [{ai+1}] SKIP {asset_id}: file not found ({mesh_path})")
            stats["skipped"] += 1
            continue

        motions = choose_motions(asset, args.motions_per_asset)
        views = CAMERA_VIEWS[:args.views_per_asset]

        for motion in motions:
            for view in views:
                clip_dir = output_dir / asset_id / motion / view
                clip_label_path = clip_dir / "frame_labels.jsonl"

                # Skip if already rendered
                if clip_label_path.exists() and not args.skip_render:
                    print(f"  [{ai+1}] EXISTS {asset_id}/{motion}/{view}")
                    # Add existing clip to dataset
                    existing_labels = read_jsonl(clip_label_path)
                    dataset_records.extend(existing_labels)
                    stats["frames"] += len(existing_labels)
                    stats["processed"] += 1
                    continue

                if args.skip_render:
                    if clip_label_path.exists():
                        existing_labels = read_jsonl(clip_label_path)
                        dataset_records.extend(existing_labels)
                        stats["frames"] += len(existing_labels)
                    continue

                print(f"  [{ai+1}] RENDER {asset_id}/{motion}/{view}...")
                success = render_clip(
                    blender_path=blender_path,
                    asset_path=mesh_path,
                    output_dir=str(clip_dir),
                    motion=motion,
                    camera_view=view,
                    duration_sec=args.duration_sec,
                    fps=args.fps,
                    resolution=args.resolution,
                )

                if success and clip_label_path.exists():
                    labels = read_jsonl(clip_label_path)
                    dataset_records.extend(labels)
                    stats["frames"] += len(labels)
                    stats["processed"] += 1
                else:
                    stats["failed"] += 1

    elapsed = time.time() - t0

    # Write dataset index
    dataset_index_path = output_dir / "dataset_index.jsonl"
    write_jsonl(dataset_index_path, dataset_records)

    # Split into train/val
    # Asset-level split: 80% train, 20% val
    asset_ids = sorted(set(r.get("asset_id", "") for r in dataset_records))
    split_idx = int(len(asset_ids) * 0.8)
    train_assets = set(asset_ids[:split_idx])
    val_assets = set(asset_ids[split_idx:])

    train_records = [r for r in dataset_records if r.get("asset_id", "") in train_assets]
    val_records = [r for r in dataset_records if r.get("asset_id", "") in val_assets]

    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "val.jsonl", val_records)

    # Summary
    print(f"\n{'='*60}")
    print(f"Dataset Build Complete")
    print(f"{'='*60}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Processed clips: {stats['processed']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Total frames: {stats['frames']}")
    print(f"  Train frames: {len(train_records)} ({len(train_assets)} assets)")
    print(f"  Val frames: {len(val_records)} ({len(val_assets)} assets)")
    print(f"  Index: {dataset_index_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
