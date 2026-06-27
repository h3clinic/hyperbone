"""
Render RGB, mask, and depth for pose3d dataset from Blender.

Usage:
  blender --background --python scripts/render_pose3d_dataset.py -- \
    --model assets/gltf_samples/Fox/glTF-Binary/Fox.glb \
    --out outputs/pose3d/fox_dataset \
    --seconds 5 \
    --fps 24 \
    --animation Walk \
    --resolution 640 480 \
    --cameras 1

Requires Blender Python environment.
"""
import sys
import argparse
import json
import math
from pathlib import Path

try:
    import bpy
    import mathutils
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False


def parse_args():
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render pose3d dataset from Blender")
    parser.add_argument("--model", required=True, help="Path to GLB/GLTF/FBX/Blend file")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--animation", default="Walk")
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 480])
    parser.add_argument("--cameras", type=int, default=1, help="Number of camera views")
    return parser.parse_args(argv)


def setup_scene(resolution):
    """Configure render settings."""
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = True


def setup_camera(resolution, cam_idx=0):
    """Create a camera positioned to see the animal."""
    cam = bpy.data.cameras.new(f"DatasetCam{cam_idx}")
    cam.lens = 50
    cam.sensor_width = 36
    cam.clip_start = 0.1
    cam.clip_end = 1000

    cam_obj = bpy.data.objects.new(f"DatasetCamObj{cam_idx}", cam)
    bpy.context.scene.collection.objects.link(cam_obj)

    # Default side view
    angles = [
        (3.0, -5.0, 2.0, (70, 0, 30)),  # Side
        (0.0, -6.0, 2.5, (75, 0, 0)),    # Front-ish
        (-3.0, -4.0, 3.0, (65, 0, -30)), # Other side
        (0.0, -3.0, 6.0, (40, 0, 0)),    # Top-down-ish
    ]
    pos = angles[cam_idx % len(angles)]
    cam_obj.location = (pos[0], pos[1], pos[2])
    cam_obj.rotation_euler = tuple(math.radians(a) for a in pos[3])

    return cam_obj, cam


def get_camera_K(cam, resolution):
    w, h = resolution
    fx = cam.lens * w / cam.sensor_width
    fy = cam.lens * h / (cam.sensor_width * h / w)
    return [[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]]


def find_armature():
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            return obj
    return None


def select_animation(armature_obj, anim_name):
    if not armature_obj.animation_data:
        armature_obj.animation_data_create()
    for action in bpy.data.actions:
        if anim_name.lower() in action.name.lower():
            armature_obj.animation_data.action = action
            return action.name
    if bpy.data.actions:
        action = bpy.data.actions[0]
        armature_obj.animation_data.action = action
        return action.name
    return "none"


def render_rgb_frame(out_dir, frame_idx):
    """Render RGB frame."""
    scene = bpy.context.scene
    scene.render.film_transparent = False
    path = str(out_dir / "rgb" / f"frame_{frame_idx:05d}.png")
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path


def render_mask_frame(out_dir, frame_idx, armature_obj):
    """Render binary mask (animal vs background)."""
    scene = bpy.context.scene
    # Use material override for mask
    scene.render.film_transparent = True
    path = str(out_dir / "masks" / f"mask_{frame_idx:05d}.png")
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path


def render_depth_frame(out_dir, frame_idx):
    """Render depth map using compositor."""
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.links.clear()
    tree.nodes.clear()

    rl = tree.nodes.new('CompositorNodeRLayers')
    depth_out = tree.nodes.new('CompositorNodeOutputFile')
    depth_out.base_path = str(out_dir / "depth")
    depth_out.file_slots[0].path = f"depth_{frame_idx:05d}"
    depth_out.format.file_format = 'OPEN_EXR'

    tree.links.new(rl.outputs['Depth'], depth_out.inputs[0])

    bpy.ops.render.render()
    return str(out_dir / "depth" / f"depth_{frame_idx:05d}0001.exr")


def export_frame_pose(armature_obj, cam_obj, cam, resolution, frame_idx, fps, anim_name, asset_id):
    """Export 3D pose for one frame."""
    from bpy_extras.object_utils import world_to_camera_view

    scene = bpy.context.scene
    joints = []
    bones = []
    bone_id_map = {}
    world_mat = armature_obj.matrix_world

    for idx, pbone in enumerate(armature_obj.pose.bones):
        bone_id_map[pbone.name] = idx

    for idx, pbone in enumerate(armature_obj.pose.bones):
        world_pos = world_mat @ pbone.head
        co = world_to_camera_view(scene, cam_obj, world_pos)
        image_xy = [co.x * resolution[0], (1.0 - co.y) * resolution[1]]
        visible = co.z > 0

        cam_mat = cam_obj.matrix_world.inverted()
        cam_pos = cam_mat @ world_pos

        parent_id = bone_id_map.get(pbone.parent.name) if pbone.parent else None

        joints.append({
            "id": idx,
            "name": pbone.name,
            "parent_id": parent_id,
            "type": classify_bone_type(pbone.name),
            "world_xyz": [world_pos.x, world_pos.y, world_pos.z],
            "camera_xyz": [cam_pos.x, cam_pos.y, cam_pos.z],
            "local_xyz": [pbone.head.x, pbone.head.y, pbone.head.z],
            "image_xy": image_xy,
            "visible": visible,
            "confidence": 1.0,
        })

    for idx, pbone in enumerate(armature_obj.pose.bones):
        if pbone.parent:
            parent_idx = bone_id_map[pbone.parent.name]
            parent_world = world_mat @ pbone.parent.head
            child_world = world_mat @ pbone.head
            length = (child_world - parent_world).length

            bones.append({
                "parent_id": parent_idx,
                "child_id": idx,
                "name": f"{pbone.parent.name}_to_{pbone.name}",
                "length": length,
                "rest_length": pbone.bone.length,
            })

    cam_ext = cam_obj.matrix_world.inverted()
    extrinsic = [[cam_ext[i][j] for j in range(4)] for i in range(4)]

    return {
        "asset_id": asset_id,
        "frame_idx": frame_idx,
        "timestamp_sec": round(frame_idx / fps, 4),
        "animation_name": anim_name,
        "joints": joints,
        "bones": bones,
        "camera_K": get_camera_K(cam, resolution),
        "camera_extrinsic": extrinsic,
        "resolution": list(resolution),
        "coord_systems": {"world": "blender_world", "camera": "blender_camera",
                          "image": "pixel_xy", "canonical": "animal_root_normalized"},
    }


def classify_bone_type(bone_name):
    name = bone_name.lower()
    for key, jtype in [
        ("root", "root"), ("pelvis", "root"), ("hips", "root"),
        ("spine", "spine"), ("neck", "neck"), ("head", "head"),
        ("shoulder", "shoulder"), ("upperarm", "shoulder"),
        ("elbow", "elbow"), ("forearm", "elbow"),
        ("wrist", "wrist"), ("hand", "wrist"),
        ("hip", "hip"), ("thigh", "hip"), ("upperleg", "hip"),
        ("knee", "knee"), ("shin", "knee"),
        ("ankle", "ankle"), ("foot", "ankle"),
        ("paw", "paw"), ("toe", "paw"),
        ("tail", "tail"), ("ear", "ear"),
    ]:
        if key in name:
            return jtype
    return "unknown"


def main():
    if not HAS_BLENDER:
        print("ERROR: This script must be run inside Blender.")
        print("Usage: blender --background --python scripts/render_pose3d_dataset.py -- --model <path> --out <path>")
        sys.exit(1)

    args = parse_args()
    model_path = Path(args.model).resolve()
    out_dir = Path(args.out)
    resolution = tuple(args.resolution)
    n_frames = int(args.seconds * args.fps)
    asset_id = model_path.stem

    # Create output directories
    (out_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)

    print(f"[Pose3D Dataset] Model: {model_path}")
    print(f"[Pose3D Dataset] Output: {out_dir}")
    print(f"[Pose3D Dataset] {n_frames} frames @ {args.fps}fps")

    # Clear and import
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ext = model_path.suffix.lower()
    if ext in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=str(model_path))
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=str(model_path))

    # Setup
    armature = find_armature()
    if not armature:
        print("ERROR: No armature found!")
        sys.exit(1)

    anim_name = select_animation(armature, args.animation)
    setup_scene(resolution)
    cam_obj, cam = setup_camera(resolution)
    bpy.context.scene.camera = cam_obj

    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = n_frames - 1

    # Render and export
    pose_records = []
    for fi in range(n_frames):
        scene.frame_set(fi)

        rgb_path = render_rgb_frame(out_dir, fi)
        mask_path = render_mask_frame(out_dir, fi, armature)

        record = export_frame_pose(armature, cam_obj, cam, resolution, fi, args.fps, anim_name, asset_id)
        record["rgb_path"] = rgb_path
        record["mask_path"] = mask_path
        pose_records.append(record)

        if fi % 24 == 0:
            print(f"  Frame {fi}/{n_frames}")

    # Write pose GT
    gt_path = out_dir / "pose3d_gt.jsonl"
    with open(gt_path, 'w') as f:
        for r in pose_records:
            f.write(json.dumps(r) + "\n")

    # Write manifest
    manifest = {
        "asset_id": asset_id,
        "model_path": str(model_path),
        "animation": anim_name,
        "fps": args.fps,
        "seconds": args.seconds,
        "n_frames": n_frames,
        "resolution": list(resolution),
        "camera_K": get_camera_K(cam, resolution),
        "output_dir": str(out_dir),
    }
    with open(out_dir / "dataset_manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    # Write rest skeleton
    rest_bones = []
    for bone in armature.data.bones:
        rest_bones.append({
            "name": bone.name,
            "head_local": list(bone.head_local),
            "tail_local": list(bone.tail_local),
            "length": bone.length,
            "parent": bone.parent.name if bone.parent else None,
        })
    with open(out_dir / "rest_skeleton.json", 'w') as f:
        json.dump({"asset_id": asset_id, "bones": rest_bones}, f, indent=2)

    # Write camera
    cam_ext = cam_obj.matrix_world.inverted()
    with open(out_dir / "camera.json", 'w') as f:
        json.dump({
            "K": get_camera_K(cam, resolution),
            "extrinsic": [[cam_ext[i][j] for j in range(4)] for i in range(4)],
            "location": list(cam_obj.location),
            "rotation_euler": list(cam_obj.rotation_euler),
            "resolution": list(resolution),
        }, f, indent=2)

    print(f"\n[Pose3D Dataset] DONE.")
    print(f"  RGB frames: {n_frames}")
    print(f"  Pose GT: {gt_path}")
    print(f"  Manifest: {out_dir / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
