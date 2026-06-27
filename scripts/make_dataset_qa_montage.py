"""Generate QA montage grids from pipeline overlays."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def make_grid(images, cols=4, cell_size=(320, 200)):
    """Arrange images into a grid."""
    if not images:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)

    rows_needed = (len(images) + cols - 1) // cols
    w, h = cell_size
    grid = np.zeros((rows_needed * h, cols * w, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        resized = cv2.resize(img, cell_size)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = resized

    return grid


def load_overlays(overlay_dir, max_count=24):
    """Load overlay PNGs from a directory."""
    images = []
    p = Path(overlay_dir)
    if not p.exists():
        return images
    for f in sorted(p.glob("overlay_*.png"))[:max_count]:
        img = cv2.imread(str(f))
        if img is not None:
            images.append(img)
    return images


def make_dataset_qa_montage(output_dir: str, montage_dir: str = None, max_per_grid: int = 24):
    """Create QA montage grids from a pipeline run."""
    out = Path(output_dir)
    mont = Path(montage_dir) if montage_dir else out / "qa_montage"
    mont.mkdir(parents=True, exist_ok=True)

    # Find overlay directories
    accepted_dirs = list(out.rglob("overlays"))
    rejected_dirs = list(out.rglob("rejected/overlays"))

    # Collect accepted overlays
    accepted_imgs = []
    for d in accepted_dirs:
        if "rejected" not in str(d):
            accepted_imgs.extend(load_overlays(d, max_per_grid))
    accepted_imgs = accepted_imgs[:max_per_grid]

    # Collect rejected overlays
    rejected_imgs = []
    for d in rejected_dirs:
        rejected_imgs.extend(load_overlays(d, max_per_grid))
    rejected_imgs = rejected_imgs[:max_per_grid]

    # Generate grids
    if accepted_imgs:
        grid = make_grid(accepted_imgs)
        path = mont / "accepted_grid.png"
        cv2.imwrite(str(path), grid)
        print(f"[Montage] Accepted grid ({len(accepted_imgs)} images): {path}")

    if rejected_imgs:
        grid = make_grid(rejected_imgs)
        path = mont / "rejected_grid.png"
        cv2.imwrite(str(path), grid)
        print(f"[Montage] Rejected grid ({len(rejected_imgs)} images): {path}")

    # Borderline: accepted with high bridge count or rejected with 2 components
    import json
    quality_files = list(out.rglob("quality.jsonl"))
    borderline_frames = []
    for qf in quality_files:
        for line in qf.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            rec = json.loads(line)
            bridges = rec.get("bridges_added", 0)
            accepted = rec.get("accepted", False)
            comp_after = rec.get("components_after_repair", 1)
            lcr = rec.get("largest_component_ratio", 1.0)

            is_borderline = False
            if accepted and bridges >= 4:
                is_borderline = True
            if not accepted and comp_after <= 2 and lcr >= 0.6:
                is_borderline = True

            if is_borderline:
                borderline_frames.append(rec)

    # Load borderline overlays
    borderline_imgs = []
    for rec in borderline_frames[:max_per_grid]:
        fi = rec.get("frame_idx", 0)
        oi = rec.get("object_id", 0)
        fname = f"overlay_{fi:06d}_obj{oi:03d}.png"
        # Search both accepted and rejected
        for d in accepted_dirs + rejected_dirs:
            candidate = d / fname
            if candidate.exists():
                img = cv2.imread(str(candidate))
                if img is not None:
                    borderline_imgs.append(img)
                    break

    if borderline_imgs:
        grid = make_grid(borderline_imgs)
        path = mont / "borderline_grid.png"
        cv2.imwrite(str(path), grid)
        print(f"[Montage] Borderline grid ({len(borderline_imgs)} images): {path}")

    if not (accepted_imgs or rejected_imgs):
        print("[Montage] No overlays found — nothing to montage.")
        return

    print(f"[Montage] Output: {mont}")


def main():
    parser = argparse.ArgumentParser(description="Generate QA montage grids")
    parser.add_argument("--dir", required=True, help="Pipeline output directory")
    parser.add_argument("--out", default=None, help="Montage output directory")
    parser.add_argument("--max", type=int, default=24, help="Max images per grid")
    args = parser.parse_args()
    make_dataset_qa_montage(args.dir, args.out, args.max)


if __name__ == "__main__":
    main()
