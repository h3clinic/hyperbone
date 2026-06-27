"""
Render Anymate Clip — Blender script for frame-by-frame rendering with skeleton export.

Run via:
    blender --background --python scripts/render_anymate_clip.py -- \\
        --asset path/to/asset.glb \\
        --out outputs/anymate_clips/asset_000001/walk_like/cam_00 \\
        --duration-sec 3 \\
        --fps 24 \\
        --resolution 512 \\
        --motion walk_like \\
        --camera-view side

CRITICAL: For every rendered frame, the skeleton is exported AFTER animation evaluation.
    scene.frame_set(frame_idx)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    arm_eval = armature.evaluated_get(depsgraph)

This ensures: rendered frame timestamp == exported skeleton timestamp.
"""
import sys
import json
import math
import os
from pathlib import Path

# Parse arguments after "--"
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

import argparse
parser = argparse.ArgumentParser(description="Render Anymate clip in Blender")
parser.add_argument("--asset", required=True, help="Path to 3D asset (.glb/.gltf/.fbx)")
parser.add_argument("--out", required=True, help="Output directory for this clip")
parser.add_argument("--duration-sec", type=float, default=3.0, help="Clip duration in seconds")
parser.add_argument("--fps", type=int, default=24, help="Frames per second")
parser.add_argument("--resolution", type=int, default=512, help="Render resolution (square)")
parser.add_argument("--motion", default="idle_sway", help="Motion type for synthetic animation")
parser.add_argument("--camera-view", default="front", choices=["front", "side", "top", "three_quarter"])
parser.add_argument("--use-existing-animation", action="store_true", help="Use asset's own animation if available")
parser.add_argument("--animation-index", type=int, default=0, help="Which animation to use (if multiple)")
args = parser.parse_args(argv)

# Now we're inside Blender
import bpy
import bmesh
import mathutils
from mathutils import Vector, Matrix, Quaternion
import numpy as np

# Add project root to path for imports
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from hyperbone.rigs.blender_export import (
    extract_joints_at_frame,
    extract_rest_skeleton,
    get_camera_params,
    setup_render_passes,
)
from hyperbone.rigs.motion_synth import (
    generate_motion_curves,
    get_motion_preset,
    apply_motion_to_blender_armature,
)


def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    # Clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.armatures:
        if block.users == 0:
            bpy.data.armatures.remove(block)


def import_asset(asset_path: str):
    """Import a 3D asset into the scene."""
    ext = Path(asset_path).suffix.lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=asset_path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=asset_path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=asset_path)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def find_armature():
    """Find the first armature object in the scene."""
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            return obj
    return None


def find_mesh_objects():
    """Find all mesh objects in the scene."""
    return [obj for obj in bpy.data.objects if obj.type == 'MESH']


def setup_camera(view: str, target_center: Vector, target_size: float) -> bpy.types.Object:
    """Create and position camera based on view type."""
    cam_data = bpy.data.cameras.new(name="RenderCamera")
    cam_data.lens = 50  # 50mm focal length
    cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)

    distance = target_size * 2.5  # Camera distance based on object size

    if view == "front":
        cam_obj.location = target_center + Vector((0, -distance, target_size * 0.3))
    elif view == "side":
        cam_obj.location = target_center + Vector((distance, 0, target_size * 0.3))
    elif view == "top":
        cam_obj.location = target_center + Vector((0, 0, distance))
    elif view == "three_quarter":
        cam_obj.location = target_center + Vector((distance * 0.7, -distance * 0.7, target_size * 0.5))

    # Point camera at target
    direction = target_center - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    bpy.context.scene.camera = cam_obj
    return cam_obj


def setup_lighting():
    """Add basic 3-point lighting."""
    # Key light
    key_data = bpy.data.lights.new(name="KeyLight", type='SUN')
    key_data.energy = 3.0
    key_obj = bpy.data.objects.new("KeyLight", key_data)
    key_obj.rotation_euler = (math.radians(50), math.radians(10), math.radians(-30))
    bpy.context.scene.collection.objects.link(key_obj)

    # Fill light
    fill_data = bpy.data.lights.new(name="FillLight", type='SUN')
    fill_data.energy = 1.0
    fill_obj = bpy.data.objects.new("FillLight", fill_data)
    fill_obj.rotation_euler = (math.radians(40), math.radians(-20), math.radians(60))
    bpy.context.scene.collection.objects.link(fill_obj)


def get_scene_bounds():
    """Get bounding box center and size of all mesh objects."""
    min_corner = Vector((float('inf'), float('inf'), float('inf')))
    max_corner = Vector((float('-inf'), float('-inf'), float('-inf')))

    for obj in find_mesh_objects():
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_corner.x = min(min_corner.x, world_corner.x)
            min_corner.y = min(min_corner.y, world_corner.y)
            min_corner.z = min(min_corner.z, world_corner.z)
            max_corner.x = max(max_corner.x, world_corner.x)
            max_corner.y = max(max_corner.y, world_corner.y)
            max_corner.z = max(max_corner.z, world_corner.z)

    center = (min_corner + max_corner) / 2
    size = (max_corner - min_corner).length
    return center, max(size, 0.1)


def main():
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rgb").mkdir(exist_ok=True)
    (output_dir / "mask").mkdir(exist_ok=True)
    (output_dir / "depth").mkdir(exist_ok=True)
    (output_dir / "skeleton").mkdir(exist_ok=True)

    frame_count = int(args.duration_sec * args.fps)
    resolution = (args.resolution, args.resolution)

    print(f"[Render] Asset: {args.asset}")
    print(f"[Render] Output: {output_dir}")
    print(f"[Render] Frames: {frame_count} ({args.duration_sec}s @ {args.fps}fps)")
    print(f"[Render] Resolution: {resolution[0]}x{resolution[1]}")

    # Clear and import
    clear_scene()
    import_asset(args.asset)

    # Find armature
    armature = find_armature()
    if armature is None:
        print("[Render] WARNING: No armature found. Rendering mesh-only (no skeleton labels).")

    # Set object pass index for mask
    for mesh_obj in find_mesh_objects():
        mesh_obj.pass_index = 1

    # Setup scene
    center, size = get_scene_bounds()
    cam_obj = setup_camera(args.camera_view, center, size)
    setup_lighting()

    # Configure render
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 64
    scene.render.fps = args.fps
    scene.frame_start = 1
    scene.frame_end = frame_count

    setup_render_passes(scene, str(output_dir), resolution)

    # Setup animation
    motion_source = "none"
    if armature:
        if args.use_existing_animation and armature.animation_data and armature.animation_data.action:
            motion_source = "original"
            print(f"[Render] Using existing animation: {armature.animation_data.action.name}")
        else:
            # Generate synthetic motion
            motion_source = "synthetic_rig_motion"
            pose_bones = armature.pose.bones
            joint_names = [pb.name for pb in pose_bones]
            parent_ids = []
            name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
            for pb in pose_bones:
                parent_ids.append(name_to_idx.get(pb.parent.name) if pb.parent else None)

            params = get_motion_preset(
                args.motion,
                duration_sec=args.duration_sec,
                fps=args.fps,
            )
            rotations = generate_motion_curves(
                joint_count=len(joint_names),
                parent_ids=parent_ids,
                joint_names=joint_names,
                params=params,
            )
            apply_motion_to_blender_armature(armature, rotations, fps=args.fps, action_name=f"synth_{args.motion}")
            print(f"[Render] Applied synthetic motion: {args.motion}")

    # Export rest skeleton
    if armature:
        rest_joints, rest_bones = extract_rest_skeleton(armature)
        rest_skel = {"joints": rest_joints, "bones": rest_bones}
        with open(output_dir / "skeleton" / "rest_skeleton.json", "w") as f:
            json.dump(rest_skel, f, indent=2)

    # Render frame by frame
    camera_params = get_camera_params(cam_obj, resolution) if cam_obj else None
    frame_labels = []

    for frame_idx in range(1, frame_count + 1):
        # Set frame (this is the CRITICAL step — must happen before both render and export)
        scene.frame_set(frame_idx)

        # Render
        rgb_path = output_dir / "rgb" / f"frame_{frame_idx:06d}.png"
        scene.render.filepath = str(rgb_path)
        bpy.ops.render.render(write_still=True)

        # Export skeleton at this exact frame (AFTER frame_set)
        skeleton_data = None
        if armature:
            joints, bones = extract_joints_at_frame(
                armature, frame_idx, camera_obj=cam_obj, resolution=resolution
            )
            skeleton_data = {"joints": joints, "bones": bones}

            # Save per-frame skeleton
            skel_path = output_dir / "skeleton" / f"frame_{frame_idx:06d}.json"
            with open(skel_path, "w") as f:
                json.dump(skeleton_data, f)

        # Build frame label
        timestamp_sec = (frame_idx - 1) / args.fps
        label = {
            "asset_id": Path(args.asset).stem,
            "animation_id": args.motion if motion_source == "synthetic_rig_motion" else "original",
            "frame_idx": frame_idx,
            "timestamp_sec": round(timestamp_sec, 4),
            "rgb_path": f"rgb/frame_{frame_idx:06d}.png",
            "mask_path": f"mask/frame_{frame_idx:06d}.png",
            "depth_path": f"depth/frame_{frame_idx:06d}.exr",
            "camera": camera_params,
            "joints": joints if skeleton_data else [],
            "bones": bones if skeleton_data else [],
            "motion_source": motion_source,
        }
        frame_labels.append(label)

        if frame_idx % 10 == 0 or frame_idx == 1:
            print(f"[Render] Frame {frame_idx}/{frame_count} done")

    # Write frame labels JSONL
    labels_path = output_dir / "frame_labels.jsonl"
    with open(labels_path, "w") as f:
        for label in frame_labels:
            f.write(json.dumps(label) + "\n")

    # Write clip metadata
    clip_meta = {
        "asset_path": args.asset,
        "asset_id": Path(args.asset).stem,
        "motion_type": args.motion,
        "motion_source": motion_source,
        "camera_view": args.camera_view,
        "duration_sec": args.duration_sec,
        "fps": args.fps,
        "frame_count": frame_count,
        "resolution": list(resolution),
        "joint_count": len(armature.pose.bones) if armature else 0,
        "camera": camera_params,
    }
    with open(output_dir / "clip_meta.json", "w") as f:
        json.dump(clip_meta, f, indent=2)

    print(f"\n[Render] Complete: {frame_count} frames rendered")
    print(f"[Render] Labels: {labels_path}")
    print(f"[Render] Skeleton: {output_dir / 'skeleton'}")


if __name__ == "__main__":
    main()
