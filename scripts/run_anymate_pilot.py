"""
Anymate Pilot Execution — render 10 assets from the .pt dataset,
project joints, apply synthetic motion, validate synchronization.

Produces:
  outputs/anymate_clips_pilot/
    asset_XXXX/
      rgb/frame_XXXX.png      — rendered mesh/point-cloud
      mask/frame_XXXX.png     — object mask (binary)
      depth/frame_XXXX.npy    — depth map
    dataset_index.jsonl       — FramePoseLabel records for each frame
    sync_report.md            — synchronization validation results

Usage:
    python scripts/run_anymate_pilot.py --n-assets 10 --n-frames 30 --resolution 256
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.rigs.schema import (
    Joint, Bone, Skeleton, Camera, FramePoseLabel,
    write_jsonl, MotionSource,
)


def make_camera_orbit(center: np.ndarray, radius: float, elevation: float,
                      azimuth: float, resolution: int = 256) -> dict:
    """Create a camera looking at `center` from orbit position."""
    # Spherical to cartesian
    el_rad = np.radians(elevation)
    az_rad = np.radians(azimuth)
    eye = center + radius * np.array([
        np.cos(el_rad) * np.sin(az_rad),
        np.sin(el_rad),
        np.cos(el_rad) * np.cos(az_rad),
    ])

    # Look-at matrix
    forward = center - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-8)

    # Rotation matrix (world -> camera)
    R = np.stack([right, -up, forward], axis=0)  # 3x3

    # Translation
    t = -R @ eye  # 3x1

    # Extrinsic 4x4
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t

    # Intrinsic (simple pinhole)
    focal = resolution * 1.5  # ~60 degree FOV
    cx, cy = resolution / 2.0, resolution / 2.0
    K = np.array([
        [focal, 0, cx],
        [0, focal, cy],
        [0, 0, 1],
    ])

    return {
        "K": K,
        "extrinsic": extrinsic,
        "eye": eye,
        "center": center,
        "resolution": resolution,
    }


def project_joints_3d_to_2d(joints_3d: np.ndarray, K: np.ndarray,
                             extrinsic: np.ndarray) -> tuple:
    """Project 3D joints to 2D image coordinates.
    Returns (xy_2d [N,2], depths [N], visible [N])."""
    N = joints_3d.shape[0]
    # Homogeneous coordinates
    pts_h = np.hstack([joints_3d, np.ones((N, 1))])  # [N, 4]
    # Transform to camera space
    pts_cam = (extrinsic @ pts_h.T).T[:, :3]  # [N, 3]
    depths = pts_cam[:, 2]  # z in camera frame

    # Project
    valid = depths > 0.01
    xy_2d = np.zeros((N, 2))
    pts_proj = K @ pts_cam.T  # [3, N]
    for i in range(N):
        if valid[i]:
            xy_2d[i, 0] = pts_proj[0, i] / pts_proj[2, i]
            xy_2d[i, 1] = pts_proj[1, i] / pts_proj[2, i]

    return xy_2d, depths, valid


def render_mesh_zbuffer(verts: np.ndarray, faces: np.ndarray,
                        K: np.ndarray, extrinsic: np.ndarray,
                        resolution: int) -> tuple:
    """Software rasterizer: render mesh to RGB, mask, depth using z-buffer.
    Simple scanline approach — good enough for validation."""
    N_verts = verts.shape[0]

    # Transform vertices to camera space
    pts_h = np.hstack([verts[:, :3], np.ones((N_verts, 1))])
    pts_cam = (extrinsic @ pts_h.T).T[:, :3]

    # Project to 2D
    pts_proj = K @ pts_cam.T  # [3, N]
    xy = np.zeros((N_verts, 2))
    z_vals = pts_cam[:, 2]
    valid = z_vals > 0.01
    xy[valid, 0] = pts_proj[0, valid] / pts_proj[2, valid]
    xy[valid, 1] = pts_proj[1, valid] / pts_proj[2, valid]

    # Initialize buffers
    depth_buf = np.full((resolution, resolution), np.inf, dtype=np.float32)
    rgb_buf = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    mask_buf = np.zeros((resolution, resolution), dtype=np.uint8)

    # Normals for shading (if available in verts[:,3:6])
    has_normals = verts.shape[1] >= 6

    # Rasterize each triangle (simplified — vertex-level z-buffer)
    for face in faces:
        i0, i1, i2 = face[0], face[1], face[2]
        if not (valid[i0] and valid[i1] and valid[i2]):
            continue

        # Get 2D coords
        x0, y0 = xy[i0]
        x1, y1 = xy[i1]
        x2, y2 = xy[i2]

        # Bounding box
        min_x = max(0, int(min(x0, x1, x2)))
        max_x = min(resolution - 1, int(max(x0, x1, x2)))
        min_y = max(0, int(min(y0, y1, y2)))
        max_y = min(resolution - 1, int(max(y0, y1, y2)))

        if max_x - min_x > resolution // 2 or max_y - min_y > resolution // 2:
            continue  # Skip degenerate triangles

        # Simple: fill triangle using barycentric coordinates
        # For speed, only do vertex-level coloring (skip full scanline)
        # Just splat the 3 vertices
        for idx in [i0, i1, i2]:
            px, py = int(round(xy[idx, 0])), int(round(xy[idx, 1]))
            if 0 <= px < resolution and 0 <= py < resolution:
                z = z_vals[idx]
                if z < depth_buf[py, px]:
                    depth_buf[py, px] = z
                    mask_buf[py, px] = 255
                    if has_normals:
                        # Simple diffuse shading
                        n = verts[idx, 3:6]
                        n = n / (np.linalg.norm(n) + 1e-8)
                        light_dir = np.array([0.3, 0.7, 0.5])
                        light_dir = light_dir / np.linalg.norm(light_dir)
                        shade = max(0.2, float(np.dot(n, light_dir)))
                        c = int(shade * 200)
                        rgb_buf[py, px] = [c, c, c]
                    else:
                        rgb_buf[py, px] = [180, 180, 180]

    # Fill small holes in mask via dilation
    try:
        import cv2
        kernel = np.ones((3, 3), np.uint8)
        mask_buf = cv2.dilate(mask_buf, kernel, iterations=2)
        # Also fill rgb based on dilated mask
        rgb_dilated = cv2.dilate(rgb_buf, kernel, iterations=2)
        fill_mask = (mask_buf > 0) & (rgb_buf.sum(axis=2) == 0)
        rgb_buf[fill_mask] = rgb_dilated[fill_mask]
    except ImportError:
        pass

    depth_buf[depth_buf == np.inf] = 0.0
    return rgb_buf, mask_buf, depth_buf


def apply_synthetic_motion(joints: np.ndarray, conns: np.ndarray,
                           frame_idx: int, n_frames: int,
                           amplitude: float = 0.05) -> np.ndarray:
    """Apply a sinusoidal perturbation to joints (simulating motion).
    Parent joints move less, child joints move more."""
    t = frame_idx / max(n_frames - 1, 1)  # 0 to 1
    n_joints = joints.shape[0]

    # Compute depth from root
    depths = np.zeros(n_joints, dtype=int)
    for i in range(n_joints):
        parent = int(conns[i]) if i < len(conns) else 0
        if parent != i and 0 <= parent < n_joints:
            depths[i] = depths[parent] + 1

    max_depth = max(depths.max(), 1)
    moved = joints.copy()

    for i in range(n_joints):
        # Depth-scaled sinusoidal motion
        depth_factor = depths[i] / max_depth
        phase = 2 * np.pi * t + i * 0.5
        dx = amplitude * depth_factor * np.sin(phase)
        dy = amplitude * depth_factor * np.sin(phase + 1.2)
        dz = amplitude * depth_factor * np.sin(phase + 2.4) * 0.5
        moved[i] += np.array([dx, dy, dz])

    return moved


def apply_motion_to_mesh(mesh_verts: np.ndarray, joints_rest: np.ndarray,
                         joints_moved: np.ndarray,
                         skins_index: np.ndarray, skins_weight: np.ndarray) -> np.ndarray:
    """Apply joint motion to mesh via linear blend skinning."""
    n_verts = mesh_verts.shape[0]
    n_influences = skins_index.shape[1]
    n_joints = joints_rest.shape[0]

    # Compute per-joint displacement
    displacements = joints_moved - joints_rest  # [J, 3]

    # Apply LBS
    moved_verts = mesh_verts.copy()
    for v in range(n_verts):
        delta = np.zeros(3)
        w_sum = 0.0
        for k in range(n_influences):
            j_idx = int(skins_index[v, k])
            w = float(skins_weight[v, k])
            if j_idx < 0 or j_idx >= n_joints or w <= 0:
                continue
            delta += w * displacements[j_idx]
            w_sum += w
        if w_sum > 0:
            moved_verts[v, :3] += delta

    return moved_verts


def run_pilot(n_assets: int = 10, n_frames: int = 30, resolution: int = 256,
              dataset_path: str = "datasets/anymate/Anymate_test.pt",
              output_dir: str = "outputs/anymate_clips_pilot"):
    """Run the pilot: render assets, project joints, validate sync."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Pilot] Loading dataset: {dataset_path}")
    data = torch.load(dataset_path, weights_only=False)
    print(f"[Pilot] Loaded {len(data)} assets")

    # Select assets with reasonable joint counts and mesh sizes
    candidates = []
    for i, d in enumerate(data):
        j_num = d['joints_num']
        mesh_verts = d['mesh_pc'].shape[0]
        if 15 <= j_num <= 60 and mesh_verts >= 1000:
            candidates.append(i)
        if len(candidates) >= n_assets * 3:
            break

    # Pick evenly spaced from candidates
    step = max(1, len(candidates) // n_assets)
    selected = candidates[::step][:n_assets]
    print(f"[Pilot] Selected {len(selected)} assets from {len(candidates)} candidates")

    all_labels = []
    fps = 12

    for asset_idx, data_idx in enumerate(selected):
        d = data[data_idx]
        asset_id = d['name']
        safe_name = f"asset_{asset_idx:04d}"
        asset_dir = output_dir / safe_name

        joints_rest = d['joints'].numpy()[:d['joints_num']]
        conns = d['conns'].numpy()[:d['joints_num']]
        bones_np = d['bones'].numpy()[:d['bones_num']]
        mesh_verts = d['mesh_pc'].numpy()  # [V, 6] (xyz + normal)
        mesh_faces = d['mesh_face'].numpy()  # [F, 3]
        skins_idx = d['mesh_skins_index'].numpy()  # [V, 4]
        skins_wt = d['mesh_skins_weight'].numpy()  # [V, 4]

        n_joints = d['joints_num']
        n_bones = d['bones_num']

        # Center the asset
        center = mesh_verts[:, :3].mean(axis=0)
        extent = mesh_verts[:, :3].max(axis=0) - mesh_verts[:, :3].min(axis=0)
        cam_radius = max(extent) * 2.5

        print(f"  [{asset_idx+1}/{len(selected)}] {asset_id} — "
              f"joints={n_joints}, bones={n_bones}, "
              f"verts={mesh_verts.shape[0]}, faces={mesh_faces.shape[0]}")

        # Create output dirs
        (asset_dir / "rgb").mkdir(parents=True, exist_ok=True)
        (asset_dir / "mask").mkdir(parents=True, exist_ok=True)
        (asset_dir / "depth").mkdir(parents=True, exist_ok=True)

        for frame_idx in range(n_frames):
            # Camera orbits slowly
            azimuth = 30 + (frame_idx / n_frames) * 60  # 30° to 90°
            elevation = 20 + 10 * np.sin(2 * np.pi * frame_idx / n_frames)
            cam = make_camera_orbit(center, cam_radius, elevation, azimuth, resolution)

            # Apply synthetic motion to joints
            joints_moved = apply_synthetic_motion(
                joints_rest, conns, frame_idx, n_frames, amplitude=0.03)

            # Apply motion to mesh via LBS
            mesh_moved = apply_motion_to_mesh(
                mesh_verts, joints_rest, joints_moved, skins_idx, skins_wt)

            # Render
            rgb, mask, depth = render_mesh_zbuffer(
                mesh_moved, mesh_faces, cam["K"], cam["extrinsic"], resolution)

            # Project joints to 2D
            xy_2d, joint_depths, joint_visible = project_joints_3d_to_2d(
                joints_moved, cam["K"], cam["extrinsic"])

            # Save frames
            rgb_path = f"{safe_name}/rgb/frame_{frame_idx:04d}.png"
            mask_path = f"{safe_name}/mask/frame_{frame_idx:04d}.png"
            depth_path = f"{safe_name}/depth/frame_{frame_idx:04d}.npy"

            try:
                import cv2
                cv2.imwrite(str(output_dir / rgb_path), rgb)
                cv2.imwrite(str(output_dir / mask_path), mask)
            except ImportError:
                from PIL import Image
                Image.fromarray(rgb).save(str(output_dir / rgb_path))
                Image.fromarray(mask).save(str(output_dir / mask_path))

            np.save(str(output_dir / depth_path), depth)

            # Build label
            joint_labels = []
            for j_idx in range(n_joints):
                vis = bool(joint_visible[j_idx])
                # Check if projected point is in bounds and on mask
                px, py = int(round(float(xy_2d[j_idx, 0]))), int(round(float(xy_2d[j_idx, 1])))
                in_bounds = 0 <= px < resolution and 0 <= py < resolution
                on_mask = bool(in_bounds and mask[py, px] > 128) if vis else False

                joint_labels.append({
                    "id": int(j_idx),
                    "name": f"joint_{j_idx}",
                    "world_xyz": [float(x) for x in joints_moved[j_idx]],
                    "image_xy": [float(xy_2d[j_idx, 0]), float(xy_2d[j_idx, 1])],
                    "depth": float(joint_depths[j_idx]),
                    "visible": bool(vis and in_bounds),
                    "on_mask": bool(on_mask),
                })

            # Build bone labels
            bone_labels = []
            for b_idx in range(n_bones):
                start = bones_np[b_idx, :3]
                end = bones_np[b_idx, 3:]
                length = float(np.linalg.norm(end - start))
                bone_labels.append({
                    "id": int(b_idx),
                    "start_xyz": [float(x) for x in start],
                    "end_xyz": [float(x) for x in end],
                    "length": length,
                })

            label = {
                "asset_id": asset_id,
                "asset_idx": data_idx,
                "animation_id": "synthetic_sine",
                "motion_source": "procedural",
                "frame_idx": frame_idx,
                "timestamp_sec": round(frame_idx / fps, 4),
                "fps": fps,
                "rgb_path": rgb_path,
                "mask_path": mask_path,
                "depth_path": depth_path,
                "joints": joint_labels,
                "bones": bone_labels,
                "camera": {
                    "K": cam["K"].tolist(),
                    "extrinsic": cam["extrinsic"].tolist(),
                    "resolution": [resolution, resolution],
                    "eye": cam["eye"].tolist(),
                    "center": cam["center"].tolist(),
                },
            }
            all_labels.append(label)

        # Progress indicator
        if (asset_idx + 1) % 2 == 0:
            print(f"    ... {(asset_idx+1)*n_frames} frames rendered")

    # Write dataset index
    index_path = output_dir / "dataset_index.jsonl"
    with open(index_path, "w") as f:
        for label in all_labels:
            f.write(json.dumps(label) + "\n")

    print(f"\n[Pilot] Done! {len(all_labels)} frames written to {output_dir}")
    print(f"[Pilot] Dataset index: {index_path}")

    # Quick self-check
    n_visible = sum(
        1 for label in all_labels
        for j in label["joints"]
        if j["visible"]
    )
    n_on_mask = sum(
        1 for label in all_labels
        for j in label["joints"]
        if j.get("on_mask", False)
    )
    total_joints = sum(len(label["joints"]) for label in all_labels)
    print(f"\n[Pilot] Quick stats:")
    print(f"  Total joint observations: {total_joints}")
    print(f"  Visible (in-bounds): {n_visible} ({100*n_visible/max(total_joints,1):.1f}%)")
    print(f"  On mask: {n_on_mask} ({100*n_on_mask/max(n_visible,1):.1f}% of visible)")

    return index_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anymate Pilot Execution")
    parser.add_argument("--n-assets", type=int, default=10)
    parser.add_argument("--n-frames", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--dataset", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--output", default="outputs/anymate_clips_pilot")
    args = parser.parse_args()

    index_path = run_pilot(
        n_assets=args.n_assets,
        n_frames=args.n_frames,
        resolution=args.resolution,
        dataset_path=args.dataset,
        output_dir=args.output,
    )

    # Run sync validation
    print("\n" + "=" * 60)
    print("Running synchronization validation...")
    print("=" * 60)

    from scripts.validate_anymate_sync import validate_dataset, write_report
    result = validate_dataset(str(index_path), sample_count=50)

    report_path = Path(args.output) / "sync_report.md"
    write_report(result, report_path)

    print(f"\n[Pilot] Sync validation: {result['status']}")
    if result.get("failures"):
        for f in result["failures"]:
            print(f"  ❌ {f}")
    else:
        print("  ✓ All sync checks passed")

    for k, v in result.get("summary", {}).items():
        print(f"  {k}: {v}")
