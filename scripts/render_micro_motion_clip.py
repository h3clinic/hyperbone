"""HyperBone Track C1: render a micro-motion clip (skeleton animation).

Renders an animated skeleton (optionally over the source mesh) with joint
trails, and exports MP4 (if an ffmpeg plugin is available) or GIF.

Usage:
    python scripts/render_micro_motion_clip.py \
        --clip outputs/track_c_micro_motion/clips/a0007_limb_swing.npz \
        --out  outputs/track_c_micro_motion/previews \
        --format mp4
    # with mesh:
    python scripts/render_micro_motion_clip.py --clip ... \
        --source datasets/anymate/Anymate_test.pt --out ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402
import imageio.v2 as imageio  # noqa: E402


def frame_to_rgb(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--source", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", default="mp4", choices=["mp4", "gif"])
    ap.add_argument("--trail", type=int, default=12, help="trail length in frames")
    ap.add_argument("--elev", type=float, default=12)
    ap.add_argument("--azim", type=float, default=-72)
    args = ap.parse_args()

    z = np.load(args.clip, allow_pickle=True)
    world = z["joints_world"].astype(np.float64)      # [T,J,3]
    edges = z["edges"]                                 # [E,2]
    T, J, _ = world.shape
    fps = int(z["fps"]); valid = bool(z["is_valid_motion"])
    preset = str(z["preset"]); ctype = str(z["corruption_type"])
    moving = set(z["moving_joint_ids"].tolist())
    name = str(z["name"]).split("/")[-1][:10]
    clip_id = str(z["clip_id"])

    sk_color = "#1a9850" if valid else "#d73027"
    status = "VALID" if valid else f"INVALID: {ctype}"

    mesh = None
    if args.source:
        import torch
        src = torch.load(args.source, map_location="cpu", weights_only=False)
        s = src[int(z["source_idx"])]
        m = s["mesh_pc"][:, :3].numpy()
        if m.shape[0] > 4000:
            m = m[np.random.RandomState(0).choice(m.shape[0], 4000, replace=False)]
        mesh = m

    allpts = world.reshape(-1, 3)
    mins, maxs = allpts.min(0), allpts.max(0)
    ctr = (mins + maxs) / 2; r = (maxs - mins).max() / 2 + 1e-6

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    trail_ids = sorted(moving) if moving else list(range(min(J, 6)))

    fig = plt.figure(figsize=(5.4, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    frames = []
    for t in range(T):
        ax.cla()
        if mesh is not None:
            ax.scatter(mesh[:, 0], mesh[:, 1], mesh[:, 2], c="#9aa7b3", s=2,
                       alpha=0.12, depthshade=False, linewidths=0)
        jp = world[t]
        # trails
        t0 = max(0, t - args.trail)
        for jid in trail_ids:
            seg = world[t0:t + 1, jid, :]
            if seg.shape[0] > 1:
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#fc8d59",
                        lw=1.3, alpha=0.7)
        # bones
        for (a, b) in edges:
            ax.plot([jp[a, 0], jp[b, 0]], [jp[a, 1], jp[b, 1]], [jp[a, 2], jp[b, 2]],
                    color=sk_color, lw=2.4)
        ax.scatter(jp[:, 0], jp[:, 1], jp[:, 2], c="#111111", s=12, depthshade=False)
        ax.set_xlim(ctr[0]-r, ctr[0]+r); ax.set_ylim(ctr[1]-r, ctr[1]+r)
        ax.set_zlim(ctr[2]-r, ctr[2]+r)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_title(f"{clip_id}\n{name} | {preset} | {status} | frame {t+1}/{T}",
                     fontsize=9)
        frames.append(frame_to_rgb(fig))
    plt.close(fig)

    stem = out_dir / clip_id
    wrote = None
    if args.format == "mp4":
        try:
            imageio.mimsave(str(stem) + ".mp4", frames, fps=fps, macro_block_size=None)
            wrote = str(stem) + ".mp4"
        except Exception as e:
            print(f"MP4 writer unavailable ({e}); falling back to GIF", flush=True)
    if wrote is None:
        imageio.mimsave(str(stem) + ".gif", frames, fps=fps, loop=0)
        wrote = str(stem) + ".gif"
    print(f"Wrote {wrote}  ({T} frames @ {fps}fps, {status})", flush=True)


if __name__ == "__main__":
    main()
