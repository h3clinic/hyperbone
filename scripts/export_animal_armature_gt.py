"""
Export animal armature ground-truth 3D pose from Blender.

Usage:
  blender --background --python scripts/export_animal_armature_gt.py -- \
    --model assets/gltf_samples/Fox/glTF-Binary/Fox.glb \
    --out outputs/pose3d/fox_armature_gt.jsonl \
    --seconds 5 \
    --fps 24 \
    --animation Walk \
    --resolution 640 480

Requires Blender Python environment.
"""
import sys
import argparse
import json
import math
from pathlib import Path

# ----------- Blender imports (only available inside Blender) -----------
try:
    import bpy
    import mathutils
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False


def parse_args():
    # Blender passes args after '--'
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Export animal armature GT from Blender")
    parser.add_argument("--model", required=True, help="Path to GLB/GLTF/FBX/Blend file")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--seconds", type=float, default=5.0, help="Duration to export")
    parser.add_argument("--fps", type=int, default=24, help="Frame rate")
    parser.add_argument("--animation", default="Walk", help="Animation/action name")
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 480],
                        help="Render resolution WxH")
    parser.add_argument("--camera-manifest", default=None,
                        help="Output camera manifest JSON path")
    return parser.parse_args(argv)


def setup_camera(resolution):
    """Create or get camera and configure it."""
    cam = bpy.data.cameras.get("PoseCamera")
    if not cam:
        cam = bpy.data.cameras.new("PoseCamera")
    cam.lens = 50  # 50mm focal length
    cam.sensor_width = 36  # 35mm sensor
    cam.clip_start = 0.1
    cam.clip_end = 1000

    cam_obj = bpy.data.objects.get("PoseCameraObj")
    if not cam_obj:
        cam_obj = bpy.data.objects.new("PoseCameraObj", cam)
        bpy.context.scene.collection.objects.link(cam_obj)

    # Position camera to see the animal from the side
    cam_obj.location = (3.0, -5.0, 2.0)
    cam_obj.rotation_euler = (math.radians(70), 0, math.radians(30))

    bpy.context.scene.camera = cam_obj
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.resolution_percentage = 100

    return cam_obj, cam


def get_camera_intrinsics(cam, resolution):
    """Compute 3x3 camera intrinsic matrix K."""
    w, h = resolution
    focal_mm = cam.lens
    sensor_w = cam.sensor_width
    sensor_h = sensor_w * h / w  # Derive from aspect ratio

    fx = focal_mm * w / sensor_w
    fy = focal_mm * h / sensor_h
    cx = w / 2.0
    cy = h / 2.0

    K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    return K


def get_camera_extrinsic(cam_obj):
    """Get 4x4 world-to-camera transform."""
    # Blender camera looks down -Z in camera space
    mat = cam_obj.matrix_world.inverted()
    return [[mat[i][j] for j in range(4)] for i in range(4)]


def project_3d_to_2d(point_3d, cam_obj, cam, resolution):
    """Project a 3D world point to 2D image coordinates."""
    from bpy_extras.object_utils import world_to_camera_view
    scene = bpy.context.scene
    co = world_to_camera_view(scene, cam_obj, mathutils.Vector(point_3d))
    # co.x and co.y are normalized [0,1], co.z is depth
    x = co.x * resolution[0]
    y = (1.0 - co.y) * resolution[1]  # Flip Y
    visible = co.z > 0
    return (x, y), visible


def classify_bone_type(bone_name):
    """Guess joint type from bone name."""
    name = bone_name.lower()
    type_map = {
        "root": "root", "pelvis": "root", "hips": "root",
        "spine": "spine", "back": "spine",
        "neck": "neck",
        "head": "head", "jaw": "head", "skull": "head",
        "shoulder": "shoulder", "clavicle": "shoulder", "upperarm": "shoulder",
        "elbow": "elbow", "forearm": "elbow",
        "wrist": "wrist", "hand": "wrist",
        "hip": "hip", "thigh": "hip", "upperleg": "hip",
        "knee": "knee", "lowerleg": "knee", "shin": "knee",
        "ankle": "ankle", "foot": "ankle",
        "paw": "paw", "toe": "paw",
        "tail": "tail",
        "ear": "ear",
    }
    for key, jtype in type_map.items():
        if key in name:
            return jtype
    return "unknown"


def find_armature():
    """Find the armature object in the scene."""
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            return obj
    return None


def select_animation(armature_obj, anim_name):
    """Select animation by name, fallback to first available."""
    if not armature_obj.animation_data:
        armature_obj.animation_data_create()

    # Try NLA tracks first
    available_actions = [a.name for a in bpy.data.actions]
    print(f"[Armature GT] Available actions: {available_actions}")

    # Find requested animation
    for action in bpy.data.actions:
        if anim_name.lower() in action.name.lower():
            armature_obj.animation_data.action = action
            print(f"[Armature GT] Selected action: {action.name}")
            return action.name

    # Fallback to first action
    if bpy.data.actions:
        action = bpy.data.actions[0]
        armature_obj.animation_data.action = action
        print(f"[Armature GT] Fallback to action: {action.name}")
        return action.name

    print("[Armature GT] WARNING: No animations found!")
    return "none"


def export_armature_frame(armature_obj, cam_obj, cam, resolution, frame_idx, fps, anim_name, asset_id):
    """Export one frame of armature pose data."""
    timestamp = frame_idx / fps

    joints = []
    bones = []
    bone_id_map = {}

    # Get pose bones (animated positions)
    pose_bones = armature_obj.pose.bones
    world_mat = armature_obj.matrix_world

    for idx, pbone in enumerate(pose_bones):
        bone_id_map[pbone.name] = idx

    for idx, pbone in enumerate(pose_bones):
        # World position of bone head
        world_pos = world_mat @ pbone.head
        world_xyz = (world_pos.x, world_pos.y, world_pos.z)

        # Camera-space position
        cam_mat = cam_obj.matrix_world.inverted()
        cam_pos = cam_mat @ world_pos
        camera_xyz = (cam_pos.x, cam_pos.y, cam_pos.z)

        # Local position (relative to parent)
        local_pos = pbone.head
        local_xyz = (local_pos.x, local_pos.y, local_pos.z)

        # Project to 2D
        image_xy, visible = project_3d_to_2d(world_xyz, cam_obj, cam, resolution)

        # Parent
        parent_id = bone_id_map.get(pbone.parent.name) if pbone.parent else None

        joints.append({
            "id": idx,
            "name": pbone.name,
            "parent_id": parent_id,
            "type": classify_bone_type(pbone.name),
            "world_xyz": list(world_xyz),
            "camera_xyz": list(camera_xyz),
            "local_xyz": list(local_xyz),
            "image_xy": list(image_xy),
            "visible": visible,
            "confidence": 1.0,
        })

    # Bones (parent-child connections)
    for idx, pbone in enumerate(pose_bones):
        if pbone.parent:
            parent_idx = bone_id_map[pbone.parent.name]
            parent_world = world_mat @ pbone.parent.head
            child_world = world_mat @ pbone.head
            length = (child_world - parent_world).length

            # Rotation quaternion
            local_quat = pbone.rotation_quaternion if pbone.rotation_mode == 'QUATERNION' else pbone.matrix_basis.to_quaternion()
            world_quat = (world_mat @ pbone.matrix).to_quaternion()

            bones.append({
                "parent_id": parent_idx,
                "child_id": idx,
                "name": f"{pbone.parent.name}_to_{pbone.name}",
                "length": length,
                "rest_length": pbone.bone.length,
                "local_rotation_quat": [local_quat.x, local_quat.y, local_quat.z, local_quat.w],
                "world_rotation_quat": [world_quat.x, world_quat.y, world_quat.z, world_quat.w],
            })

    # Camera data
    K = get_camera_intrinsics(cam, resolution)
    extrinsic = get_camera_extrinsic(cam_obj)

    record = {
        "asset_id": asset_id,
        "frame_idx": frame_idx,
        "timestamp_sec": round(timestamp, 4),
        "animation_name": anim_name,
        "joints": joints,
        "bones": bones,
        "camera_K": K,
        "camera_extrinsic": extrinsic,
        "resolution": list(resolution),
        "bbox_xywh": None,
        "mask_path": None,
        "rgb_path": None,
        "depth_path": None,
        "coord_systems": {
            "world": "blender_world",
            "camera": "blender_camera",
            "image": "pixel_xy",
            "canonical": "animal_root_normalized",
        },
    }

    # Compute bbox from projected joints
    visible_pts = [(j["image_xy"][0], j["image_xy"][1]) for j in joints if j["visible"]]
    if visible_pts:
        xs = [p[0] for p in visible_pts]
        ys = [p[1] for p in visible_pts]
        margin = 20
        x = max(0, min(xs) - margin)
        y = max(0, min(ys) - margin)
        w = min(resolution[0], max(xs) + margin) - x
        h = min(resolution[1], max(ys) + margin) - y
        record["bbox_xywh"] = [int(x), int(y), int(w), int(h)]

    return record


def main():
    if not HAS_BLENDER:
        print("ERROR: This script must be run inside Blender.")
        print("Usage: blender --background --python scripts/export_animal_armature_gt.py -- --model <path> --out <path>")
        sys.exit(1)

    args = parse_args()
    model_path = Path(args.model).resolve()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    resolution = tuple(args.resolution)
    n_frames = int(args.seconds * args.fps)
    asset_id = model_path.stem

    print(f"[Armature GT] Model: {model_path}")
    print(f"[Armature GT] Output: {out_path}")
    print(f"[Armature GT] Frames: {n_frames} @ {args.fps} fps ({args.seconds}s)")
    print(f"[Armature GT] Resolution: {resolution}")

    # Clear scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import model
    ext = model_path.suffix.lower()
    if ext in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=str(model_path))
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=str(model_path))
    elif ext == '.blend':
        with bpy.data.libraries.load(str(model_path)) as (data_from, data_to):
            data_to.objects = data_from.objects
        for obj in data_to.objects:
            bpy.context.scene.collection.objects.link(obj)
    else:
        print(f"ERROR: Unsupported format: {ext}")
        sys.exit(1)

    print(f"[Armature GT] Imported {len(bpy.data.objects)} objects")

    # Find armature
    armature = find_armature()
    if not armature:
        print("ERROR: No armature found in model!")
        sys.exit(1)

    print(f"[Armature GT] Armature: {armature.name}, bones: {len(armature.pose.bones)}")

    # Setup camera
    cam_obj, cam = setup_camera(resolution)

    # Select animation
    anim_name = select_animation(armature, args.animation)

    # Configure scene
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = n_frames - 1
    scene.render.fps = args.fps

    # Export frames
    records = []
    for fi in range(n_frames):
        scene.frame_set(fi)
        record = export_armature_frame(
            armature, cam_obj, cam, resolution,
            fi, args.fps, anim_name, asset_id
        )
        records.append(record)
        if fi % 24 == 0:
            print(f"  Frame {fi}/{n_frames}")

    # Write JSONL
    with open(out_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[Armature GT] Wrote {len(records)} frames to {out_path}")

    # Write rest skeleton
    rest_path = out_path.parent / f"{asset_id}_rest_skeleton.json"
    rest_bones = []
    for bone in armature.data.bones:
        rest_bones.append({
            "name": bone.name,
            "head_local": list(bone.head_local),
            "tail_local": list(bone.tail_local),
            "length": bone.length,
            "parent": bone.parent.name if bone.parent else None,
            "type": classify_bone_type(bone.name),
        })
    with open(rest_path, 'w') as f:
        json.dump({"asset_id": asset_id, "bones": rest_bones}, f, indent=2)
    print(f"[Armature GT] Rest skeleton: {rest_path}")

    # Write camera manifest
    if args.camera_manifest:
        manifest_path = Path(args.camera_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "asset_id": asset_id,
            "model_path": str(model_path),
            "animation": anim_name,
            "fps": args.fps,
            "seconds": args.seconds,
            "n_frames": n_frames,
            "resolution": list(resolution),
            "camera_K": get_camera_intrinsics(cam, resolution),
            "camera_location": list(cam_obj.location),
            "camera_rotation_euler": list(cam_obj.rotation_euler),
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"[Armature GT] Camera manifest: {manifest_path}")

    print(f"\n[Armature GT] DONE. Joint count: {len(armature.pose.bones)}")


if __name__ == "__main__":
    main()
