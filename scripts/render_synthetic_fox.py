"""Render a synthetic 5-second Fox-like video without Blender.

Creates a simple animated fox shape on a clean background for testing
the HyperBone pipeline when Blender is unavailable.
"""

import sys, json, argparse
import numpy as np
import cv2
from pathlib import Path


def render_synthetic_fox(
    output_video: str,
    fps: int = 24,
    duration_sec: float = 5.0,
    width: int = 640,
    height: int = 480,
    frames_dir: str = None,
):
    """Render a synthetic fox animation (no Blender required).

    Creates an animated fox-like quadruped shape moving and walking
    on a clean dark background.
    """
    out_path = Path(output_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if frames_dir:
        fdir = Path(frames_dir)
        fdir.mkdir(parents=True, exist_ok=True)
    else:
        fdir = None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    total_frames = int(fps * duration_sec)
    bg_color = (38, 38, 51)  # Dark blue-gray (matches Blender dark world)

    for i in range(total_frames):
        t = i / fps
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # Fox body center moves horizontally with slight bob
        cx = int(width * 0.3 + 80 * np.sin(t * 1.2))
        cy = int(height * 0.55 + 8 * np.sin(t * 2.4))

        # Body (elongated ellipse)
        body_rx, body_ry = 100, 35
        cv2.ellipse(frame, (cx, cy), (body_rx, body_ry), 0, 0, 360, (200, 130, 60), -1)

        # Head (smaller ellipse, offset forward)
        head_cx = cx + 95
        head_cy = cy - 25 + int(5 * np.sin(t * 3))
        cv2.ellipse(frame, (head_cx, head_cy), (28, 22), -10, 0, 360, (210, 140, 70), -1)

        # Ears
        ear_l = (head_cx - 12, head_cy - 22)
        ear_r = (head_cx + 8, head_cy - 25)
        cv2.ellipse(frame, ear_l, (8, 14), -5, 0, 360, (180, 110, 50), -1)
        cv2.ellipse(frame, ear_r, (8, 14), 5, 0, 360, (180, 110, 50), -1)

        # Snout
        snout_cx = head_cx + 22
        snout_cy = head_cy + 5
        cv2.ellipse(frame, (snout_cx, snout_cy), (12, 8), 0, 0, 360, (220, 160, 90), -1)

        # Tail (curved)
        tail_base_x = cx - 100
        tail_base_y = cy - 10
        tail_tip_x = tail_base_x - 40 + int(20 * np.sin(t * 4))
        tail_tip_y = tail_base_y - 50 + int(10 * np.cos(t * 4))
        tail_mid_x = (tail_base_x + tail_tip_x) // 2 - 15
        tail_mid_y = (tail_base_y + tail_tip_y) // 2 - 10
        pts = np.array([
            [tail_base_x, tail_base_y],
            [tail_mid_x, tail_mid_y],
            [tail_tip_x, tail_tip_y],
            [tail_tip_x + 10, tail_tip_y + 15],
        ], np.int32)
        cv2.polylines(frame, [pts], False, (220, 150, 70), 12)

        # Legs (4 legs with walking motion)
        leg_phase = t * 6  # Walking speed
        leg_positions = [
            (cx - 55, 1.0),   # front-left
            (cx - 25, 0.5),   # front-right
            (cx + 30, 0.0),   # back-left
            (cx + 60, 1.5),   # back-right
        ]
        for lx, phase_offset in leg_positions:
            ly_top = cy + body_ry - 5
            leg_swing = int(15 * np.sin(leg_phase + phase_offset * np.pi))
            ly_bottom = ly_top + 55 + int(10 * abs(np.sin(leg_phase + phase_offset * np.pi)))
            lx_bottom = lx + leg_swing
            cv2.line(frame, (lx, ly_top), (lx_bottom, ly_bottom), (170, 100, 45), 8)
            # Paw
            cv2.circle(frame, (lx_bottom, ly_bottom), 5, (150, 90, 40), -1)

        # Eye
        eye_x = head_cx + 5
        eye_y = head_cy - 5
        cv2.circle(frame, (eye_x, eye_y), 4, (30, 30, 30), -1)
        cv2.circle(frame, (eye_x + 1, eye_y - 1), 2, (255, 255, 255), -1)

        writer.write(frame)

        if fdir:
            cv2.imwrite(str(fdir / f"frame_{i+1:04d}.png"), frame)

    writer.release()
    print(f"[SyntheticRender] Written: {out_path} ({total_frames} frames, {duration_sec}s @ {fps}fps)")

    # Write manifest
    manifest = {
        "model_path": "synthetic (no Blender)",
        "animation_requested": "Walk",
        "animation_used": "synthetic_walk",
        "fps": fps,
        "duration_sec": duration_sec,
        "frame_count": total_frames,
        "resolution": [width, height],
        "camera_location": [0, 0, 0],
        "camera_rotation": [0, 0, 0],
        "light_location": [0, 0, 0],
        "output_video": str(out_path.resolve()),
        "frames_dir": str(fdir.resolve()) if fdir else None,
        "blender_available": False,
        "render_method": "opencv_synthetic",
    }

    manifest_path = out_path.parent / "fox_render_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[SyntheticRender] Manifest: {manifest_path}")

    return str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/synthetic/fox_5s.mp4")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 480])
    parser.add_argument("--frames-dir", default=None)
    args = parser.parse_args()

    render_synthetic_fox(
        args.out, args.fps, args.seconds,
        args.resolution[0], args.resolution[1],
        args.frames_dir,
    )


if __name__ == "__main__":
    main()
