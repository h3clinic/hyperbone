"""SAM2 single-frame sanity test — isolates SAM2 mask generation from the full pipeline."""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_frame_test(image_path: str, output_dir: str, checkpoint: str,
                   model_cfg: str, device: str = "cuda", max_masks: int = 20):
    """Run SAM2 automatic mask generation on a single image."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    masks_dir = out / "masks"
    masks_dir.mkdir(exist_ok=True)
    overlays_dir = out / "overlays"
    overlays_dir.mkdir(exist_ok=True)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot read image: {image_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    frame_area = h * w
    print(f"[SAM2 Frame Test] Image: {image_path}")
    print(f"[SAM2 Frame Test] Size: {w}x{h} ({frame_area} pixels)")
    print(f"[SAM2 Frame Test] Checkpoint: {checkpoint}")
    print(f"[SAM2 Frame Test] Config: {model_cfg}")
    print(f"[SAM2 Frame Test] Device: {device}")
    print()

    # Import SAM2
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError as e:
        print(f"ERROR: SAM2 not available: {e}")
        print("Run: python scripts/check_sam2.py  for setup instructions.")
        sys.exit(1)

    # Build model
    print("[SAM2 Frame Test] Loading model...")
    t0 = time.time()
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    load_time = time.time() - t0
    print(f"[SAM2 Frame Test] Model loaded in {load_time:.1f}s")

    # Create mask generator
    generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        min_mask_region_area=100,
    )

    # Generate masks
    print("[SAM2 Frame Test] Generating masks...")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t1 = time.time()
    sam2_masks = generator.generate(rgb)
    gen_time = time.time() - t1
    print(f"[SAM2 Frame Test] Generated {len(sam2_masks)} masks in {gen_time:.1f}s")
    print()

    # Sort by area descending
    sam2_masks.sort(key=lambda x: x["area"], reverse=True)

    # Process and save
    records = []
    overlay = img.copy()

    for i, ann in enumerate(sam2_masks[:max_masks]):
        mask = ann["segmentation"].astype(np.uint8) * 255
        area = int(ann["area"])
        area_ratio = area / frame_area
        bbox = ann["bbox"]  # [x, y, w, h]
        iou = float(ann.get("predicted_iou", 0.0))
        stability = float(ann.get("stability_score", 0.0))

        # Check edge touching
        touches_edge = bool(
            mask[:2, :].any() or mask[-2:, :].any() or
            mask[:, :2].any() or mask[:, -2:].any()
        )

        # Save individual mask
        cv2.imwrite(str(masks_dir / f"mask_{i:03d}.png"), mask)

        # Draw on overlay
        color = np.random.randint(60, 255, 3).tolist()
        colored = np.zeros_like(img)
        colored[mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.4, 0)

        # Draw bbox
        x, y, bw, bh = [int(v) for v in bbox]
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(overlay, f"#{i} iou={iou:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        records.append({
            "object_id": i,
            "area": area,
            "area_ratio": round(area_ratio, 6),
            "bbox_xywh": [int(v) for v in bbox],
            "predicted_iou": round(iou, 4),
            "stability_score": round(stability, 4),
            "touches_edge": touches_edge,
        })

    # Save overlay
    cv2.imwrite(str(overlays_dir / "all_masks_overlay.png"), overlay)

    # Save JSONL
    jsonl_path = out / "sam2_masks.jsonl"
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # Summary statistics
    areas = [r["area_ratio"] for r in records]
    ious = [r["predicted_iou"] for r in records]
    stabs = [r["stability_score"] for r in records]
    edge_count = sum(1 for r in records if r["touches_edge"])

    print(f"[SAM2 Frame Test] Results ({len(records)} masks saved):")
    print(f"  Area ratios:     min={min(areas):.4f} max={max(areas):.4f} mean={np.mean(areas):.4f}")
    print(f"  IoU scores:      min={min(ious):.3f} max={max(ious):.3f} mean={np.mean(ious):.3f}")
    print(f"  Stability:       min={min(stabs):.3f} max={max(stabs):.3f} mean={np.mean(stabs):.3f}")
    print(f"  Edge-touching:   {edge_count}/{len(records)}")
    print(f"  Masks > 0.2%:    {sum(1 for a in areas if a >= 0.002)}")
    print(f"  Masks > 2%:      {sum(1 for a in areas if a >= 0.02)}")
    print(f"  Masks > 10%:     {sum(1 for a in areas if a >= 0.10)}")
    print()

    # Write summary
    summary = f"""# SAM2 Frame Test Summary

## Input
- Image: `{image_path}`
- Size: {w}x{h} ({frame_area} pixels)
- Checkpoint: `{checkpoint}`
- Config: `{model_cfg}`
- Device: {device}

## Timing
- Model load: {load_time:.1f}s
- Mask generation: {gen_time:.1f}s

## Masks Generated: {len(sam2_masks)} (saved top {len(records)})

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| Area ratio | {min(areas):.4f} | {max(areas):.4f} | {np.mean(areas):.4f} |
| Predicted IoU | {min(ious):.3f} | {max(ious):.3f} | {np.mean(ious):.3f} |
| Stability | {min(stabs):.3f} | {max(stabs):.3f} | {np.mean(stabs):.3f} |

## Edge Analysis
- Touching frame edge: {edge_count}/{len(records)}

## Quality Gate Prediction
- Masks passing min_area_ratio (0.002): {sum(1 for a in areas if a >= 0.002)}
- Masks passing max_area_ratio (0.80): {sum(1 for a in areas if a < 0.80)}
- Masks in valid range [0.002, 0.80]: {sum(1 for a in areas if 0.002 <= a < 0.80)}

## Output Paths
- Masks: `{masks_dir}`
- Overlay: `{overlays_dir}/all_masks_overlay.png`
- JSONL: `{jsonl_path}`
"""
    summary_path = out / "summary.md"
    summary_path.write_text(summary)
    print(f"[SAM2 Frame Test] Summary: {summary_path}")
    print(f"[SAM2 Frame Test] Overlay: {overlays_dir / 'all_masks_overlay.png'}")


def main():
    parser = argparse.ArgumentParser(description="SAM2 single-frame mask generation test")
    parser.add_argument("--image", required=True, help="Path to input image (PNG/JPG)")
    parser.add_argument("--out", default="outputs/sam2_frame_test", help="Output directory")
    parser.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_tiny.pt",
                        help="SAM2 checkpoint path")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml",
                        help="SAM2 model config path or name")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device (default: cuda)")
    parser.add_argument("--max-masks", type=int, default=20,
                        help="Max masks to save (default: 20)")

    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)

    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Download: curl -L https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt -o checkpoints/sam2.1_hiera_tiny.pt")
        sys.exit(1)

    run_frame_test(
        image_path=args.image,
        output_dir=args.out,
        checkpoint=args.checkpoint,
        model_cfg=args.model_cfg,
        device=args.device,
        max_masks=args.max_masks,
    )


if __name__ == "__main__":
    main()
