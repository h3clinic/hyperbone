"""Generate Fox bbox proposals from synthetic rendered video.

Uses background differencing to find the fox in each frame.
No DINO needed — works purely from synthetic clean background.
"""

import sys, json, argparse
import numpy as np
import cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.io.video import get_video_info, sample_frames


def estimate_bbox_background_diff(frame: np.ndarray, bg_color_range=30) -> dict:
    """Estimate object bbox by detecting non-background pixels.

    Assumes the background is relatively uniform (synthetic render).

    Returns dict with bbox_xywh and proposal_method.
    """
    h, w = frame.shape[:2]

    # Estimate background from corners (10% margins)
    margin_x, margin_y = max(1, w // 10), max(1, h // 10)
    corners = np.concatenate([
        frame[:margin_y, :margin_x].reshape(-1, 3),
        frame[:margin_y, -margin_x:].reshape(-1, 3),
        frame[-margin_y:, :margin_x].reshape(-1, 3),
        frame[-margin_y:, -margin_x:].reshape(-1, 3),
    ], axis=0)
    bg_mean = corners.mean(axis=0).astype(np.float32)

    # Threshold: pixels far from background
    diff = np.abs(frame.astype(np.float32) - bg_mean).max(axis=2)
    mask = (diff > bg_color_range).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find largest connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        # No foreground found — use central fallback
        return _central_fallback(w, h)

    # Skip label 0 (background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas) + 1
    x = stats[largest_idx, cv2.CC_STAT_LEFT]
    y = stats[largest_idx, cv2.CC_STAT_TOP]
    bw = stats[largest_idx, cv2.CC_STAT_WIDTH]
    bh = stats[largest_idx, cv2.CC_STAT_HEIGHT]

    # Expand by 10%
    expand_x = int(bw * 0.10)
    expand_y = int(bh * 0.10)
    x = max(0, x - expand_x)
    y = max(0, y - expand_y)
    bw = min(w - x, bw + 2 * expand_x)
    bh = min(h - y, bh + 2 * expand_y)

    if bw < 10 or bh < 10:
        return _central_fallback(w, h)

    return {
        "bbox_xywh": [int(x), int(y), int(bw), int(bh)],
        "proposal_method": "synthetic_background_bbox",
    }


def _central_fallback(w: int, h: int) -> dict:
    """Fallback: central 70% bbox."""
    bw = int(w * 0.7)
    bh = int(h * 0.7)
    x = (w - bw) // 2
    y = (h - bh) // 2
    return {
        "bbox_xywh": [x, y, bw, bh],
        "proposal_method": "central_fallback",
    }


def make_proposals(
    video_path: str,
    label: str = "fox",
    sample_fps: float = 5.0,
    output_path: str = None,
) -> list:
    """Generate proposals for all sampled frames."""
    info = get_video_info(video_path)
    print(f"[Proposals] Video: {info['path']}")
    print(f"[Proposals] Resolution: {info['width']}x{info['height']}")
    print(f"[Proposals] Sampling at {sample_fps} fps")

    proposals = []
    for frame_idx, ts, frame in sample_frames(video_path, sample_fps):
        result = estimate_bbox_background_diff(frame)

        proposal = {
            "frame_idx": frame_idx,
            "timestamp_sec": round(ts, 3),
            "object_id": 0,
            "label": label,
            "label_confidence": 1.0,
            "bbox_xywh": result["bbox_xywh"],
            "prompt": label,
            "proposal_method": result["proposal_method"],
        }
        proposals.append(proposal)

    print(f"[Proposals] Generated {len(proposals)} proposals")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for p in proposals:
                f.write(json.dumps(p) + "\n")
        print(f"[Proposals] Written: {out}")

    return proposals


def main():
    parser = argparse.ArgumentParser(description="Generate Fox proposals from synthetic video")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--label", default="fox", help="Object label")
    parser.add_argument("--out", required=True, help="Output proposals JSONL path")
    parser.add_argument("--sample-fps", type=float, default=5.0, help="Sampling rate")
    args = parser.parse_args()

    make_proposals(args.video, args.label, args.sample_fps, args.out)


if __name__ == "__main__":
    main()
