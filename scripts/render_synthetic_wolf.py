"""Render a synthetic wolf-like quadruped video without Blender.

More detailed than the Fox renderer — includes body, head, legs with
walk cycle, tail wag, and proportions closer to a real wolf.
"""

import sys, json, argparse
import numpy as np
import cv2
from pathlib import Path


def render_synthetic_wolf(
    output_video: str,
    fps: int = 24,
    duration_sec: float = 5.0,
    width: int = 640,
    height: int = 480,
    frames_dir: str = None,
    animal_name: str = "wolf",
):
    """Render a synthetic wolf animation."""
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
    bg_color = (31, 31, 46)  # Dark background for clean differencing

    for i in range(total_frames):
        t = i / fps
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # Wolf center with horizontal walk motion
        cx = int(width * 0.4 + 60 * np.sin(t * 1.2))
        cy = int(height * 0.52 + 5 * np.sin(t * 2.0))

        # Body (large ellipse)
        body_rx, body_ry = 115, 48
        cv2.ellipse(frame, (cx, cy), (body_rx, body_ry), 0, 0, 360, (130, 130, 145), -1)

        # Shoulder hump
        cv2.ellipse(frame, (cx + 30, cy - 15), (40, 30), 0, 180, 360, (135, 135, 150), -1)

        # Head
        hx = cx + 105
        hy = cy - 25 + int(4 * np.sin(t * 2.5))
        cv2.ellipse(frame, (hx, hy), (32, 26), -8, 0, 360, (140, 140, 155), -1)

        # Snout (elongated)
        snout_x = hx + 28
        snout_y = hy + 6
        cv2.ellipse(frame, (snout_x, snout_y), (18, 10), 0, 0, 360, (150, 148, 160), -1)

        # Nose
        cv2.circle(frame, (snout_x + 14, snout_y), 4, (40, 40, 40), -1)

        # Eyes
        cv2.circle(frame, (hx + 8, hy - 8), 5, (50, 50, 50), -1)
        cv2.circle(frame, (hx + 9, hy - 9), 2, (200, 200, 100), -1)

        # Ears (pointed)
        ear1 = np.array([[hx - 10, hy - 22], [hx - 18, hy - 48], [hx - 2, hy - 30]], np.int32)
        ear2 = np.array([[hx + 8, hy - 24], [hx + 2, hy - 50], [hx + 16, hy - 32]], np.int32)
        cv2.fillPoly(frame, [ear1], (115, 115, 130))
        cv2.fillPoly(frame, [ear2], (115, 115, 130))

        # Tail (curved, wagging)
        tail_phase = t * 3.5
        tail_base_x = cx - 115
        tail_base_y = cy - 15
        tail_wag = int(20 * np.sin(tail_phase))
        tail_pts = np.array([
            [tail_base_x, tail_base_y],
            [tail_base_x - 30, tail_base_y - 30 + tail_wag],
            [tail_base_x - 50, tail_base_y - 55 + tail_wag],
            [tail_base_x - 55, tail_base_y - 70 + tail_wag],
        ], np.int32)
        cv2.polylines(frame, [tail_pts], False, (125, 125, 140), 12)

        # 4 legs with walk cycle
        walk_speed = 5.5
        leg_specs = [
            (cx - 65, 0.0, "front_left"),
            (cx - 30, 0.5, "front_right"),
            (cx + 45, 1.0, "back_left"),
            (cx + 80, 1.5, "back_right"),
        ]

        for lx, phase_offset, name in leg_specs:
            ly_top = cy + body_ry - 10
            phase = t * walk_speed + phase_offset * np.pi
            swing = int(15 * np.sin(phase))
            lift = int(12 * max(0, np.sin(phase)))

            # Upper leg
            knee_x = lx + swing // 2
            knee_y = ly_top + 35 - lift // 2
            cv2.line(frame, (lx, ly_top), (knee_x, knee_y), (115, 115, 130), 9)

            # Lower leg
            foot_x = lx + swing
            foot_y = knee_y + 35 - lift
            cv2.line(frame, (knee_x, knee_y), (foot_x, foot_y), (110, 110, 125), 7)

            # Paw
            cv2.ellipse(frame, (foot_x, foot_y + 3), (7, 5), 0, 0, 360, (100, 100, 115), -1)

        # Ground shadow (subtle)
        shadow_y = cy + body_ry + 60
        cv2.ellipse(frame, (cx, shadow_y), (100, 12), 0, 0, 360, (25, 25, 38), -1)

        writer.write(frame)
        if fdir:
            cv2.imwrite(str(fdir / f"frame_{i+1:04d}.png"), frame)

    writer.release()
    print(f"[SyntheticRender] Written: {out_path} ({total_frames} frames, {duration_sec}s @ {fps}fps)")

    # Manifest
    manifest = {
        "model_path": f"synthetic_{animal_name} (no Blender)",
        "animation_requested": "Walk",
        "animation_used": "synthetic_walk",
        "animation_source": "procedural",
        "fps": fps,
        "duration_sec": duration_sec,
        "frame_count": total_frames,
        "resolution": [width, height],
        "camera_location": [0, 0, 0],
        "light_location": [0, 0, 0],
        "output_video": str(out_path.resolve()),
        "frames_dir": str(fdir.resolve()) if fdir else None,
        "blender_available": False,
        "render_method": "opencv_synthetic",
        "source_asset": "Quaternius Animated Animal Pack (synthetic stand-in)",
        "license": "Public Domain / CC0",
        "animal_name": animal_name,
    }

    manifest_path = out_path.parent / f"{animal_name}_render_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[SyntheticRender] Manifest: {manifest_path}")

    return str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/synthetic_animals/wolf_5s.mp4")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 480])
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument("--animal", default="wolf")
    args = parser.parse_args()

    render_synthetic_wolf(
        args.out, args.fps, args.seconds,
        args.resolution[0], args.resolution[1],
        args.frames_dir, args.animal,
    )


if __name__ == "__main__":
    main()
