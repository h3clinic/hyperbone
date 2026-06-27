"""Render 5 seconds of a public animated animal using Blender.

Usage (run inside Blender):
  blender --background --python scripts/render_public_animal_5sec.py -- \
    --model assets/quaternius_animals/Wolf.gltf \
    --out outputs/synthetic_animals/wolf_5s.mp4 \
    --seconds 5 --fps 24 --resolution 640 480 --animation Walk
"""

import sys
import json
import math
from pathlib import Path

# Parse args after "--"
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Path to model file (.gltf/.glb/.fbx/.blend)")
parser.add_argument("--out", required=True, help="Output video path (.mp4)")
parser.add_argument("--seconds", type=float, default=5.0)
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--resolution", nargs=2, type=int, default=[640, 480])
parser.add_argument("--animation", default="Walk", help="Animation name to use")
parser.add_argument("--frames-dir", default=None, help="Save individual frames here")
args = parser.parse_args(argv)

try:
    import bpy
    import mathutils
except ImportError:
    print("ERROR: This script must be run inside Blender.")
    print("  blender --background --python scripts/render_public_animal_5sec.py -- --model ... --out ...")
    sys.exit(1)

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import model
model_path = str(Path(args.model).resolve())
ext = Path(model_path).suffix.lower()

if ext in (".gltf", ".glb"):
    bpy.ops.import_scene.gltf(filepath=model_path)
elif ext == ".fbx":
    bpy.ops.import_scene.fbx(filepath=model_path)
elif ext == ".obj":
    bpy.ops.import_scene.obj(filepath=model_path)
elif ext == ".blend":
    # Append all objects from the blend file
    with bpy.data.libraries.load(model_path) as (data_from, data_to):
        data_to.objects = data_from.objects
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.scene.collection.objects.link(obj)
else:
    print(f"ERROR: Unsupported format: {ext}")
    sys.exit(1)

print(f"[Render] Imported: {model_path}")

# Find armature and animations
armature = None
for obj in bpy.context.scene.objects:
    if obj.type == 'ARMATURE':
        armature = obj
        break

available_animations = []
if bpy.data.actions:
    available_animations = [a.name for a in bpy.data.actions]
    print(f"[Render] Available animations: {available_animations}")

# Select animation by priority
animation_used = None
animation_source = "asset"
preferred = [args.animation, "Walk", "Run", "Gallop", "Trot", "Idle"]

for name in preferred:
    matches = [a for a in available_animations if name.lower() in a.lower()]
    if matches:
        animation_used = matches[0]
        break

if not animation_used and available_animations:
    animation_used = available_animations[0]
    print(f"[Render] No preferred animation found, using: {animation_used}")

if not animation_used:
    animation_source = "fallback"
    print("[Render] No animations found. Using static pose with rotation.")

if animation_used and armature:
    action = bpy.data.actions[animation_used]
    if armature.animation_data is None:
        armature.animation_data_create()
    armature.animation_data.action = action
    print(f"[Render] Using animation: {animation_used}")

# Setup scene
scene = bpy.context.scene
scene.render.fps = args.fps
total_frames = int(args.seconds * args.fps)
scene.frame_start = 1
scene.frame_end = total_frames
scene.render.resolution_x = args.resolution[0]
scene.render.resolution_y = args.resolution[1]
scene.render.resolution_percentage = 100

# High-contrast flat background
world = bpy.data.worlds.new("FlatWorld")
scene.world = world
world.use_nodes = True
bg_node = world.node_tree.nodes["Background"]
bg_node.inputs[0].default_value = (0.12, 0.12, 0.18, 1.0)

# Compute bounds
min_co = [float('inf')] * 3
max_co = [float('-inf')] * 3
for obj in bpy.context.scene.objects:
    if obj.type in ('MESH', 'ARMATURE'):
        for corner in obj.bound_box:
            wc = obj.matrix_world @ mathutils.Vector(corner)
            for i in range(3):
                min_co[i] = min(min_co[i], wc[i])
                max_co[i] = max(max_co[i], wc[i])

center = [(min_co[i] + max_co[i]) / 2 for i in range(3)]
size = max(max_co[i] - min_co[i] for i in range(3))
cam_distance = size * 2.5

# Camera
bpy.ops.object.camera_add(
    location=(center[0] + cam_distance * 0.6, center[1] - cam_distance * 0.8, center[2] + size * 0.4)
)
camera = bpy.context.object
camera.name = "AnimalCam"
scene.camera = camera

direction = mathutils.Vector(center) - camera.location
rot_quat = direction.to_track_quat('-Z', 'Y')
camera.rotation_euler = rot_quat.to_euler()

# Light
bpy.ops.object.light_add(type='SUN', location=(center[0], center[1] - size * 2, center[2] + size * 3))
light = bpy.context.object
light.data.energy = 4.0

# Output paths
out_path = Path(args.out).resolve()
out_path.parent.mkdir(parents=True, exist_ok=True)

frames_dir = Path(args.frames_dir) if args.frames_dir else out_path.parent / (out_path.stem + "_frames")
frames_dir.mkdir(parents=True, exist_ok=True)

# If no animation, add slow rotation as fallback
if animation_source == "fallback":
    for obj in bpy.context.scene.objects:
        if obj.type in ('MESH', 'ARMATURE'):
            obj.rotation_euler = (0, 0, 0)
            obj.keyframe_insert(data_path="rotation_euler", frame=1)
            obj.rotation_euler = (0, 0, math.radians(360))
            obj.keyframe_insert(data_path="rotation_euler", frame=total_frames)
            break

# Render frames
scene.render.image_settings.file_format = 'PNG'
for frame in range(scene.frame_start, scene.frame_end + 1):
    scene.frame_set(frame)
    scene.render.filepath = str(frames_dir / f"frame_{frame:04d}.png")
    bpy.ops.render.render(write_still=True)

print(f"[Render] Frames: {frames_dir}")

# Encode video
import subprocess
ffmpeg_cmd = [
    "ffmpeg", "-y", "-framerate", str(args.fps),
    "-i", str(frames_dir / "frame_%04d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
    str(out_path),
]
try:
    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    print(f"[Render] Video: {out_path}")
except (subprocess.CalledProcessError, FileNotFoundError):
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(animation=True)
    print(f"[Render] Video (Blender): {out_path}")

# Manifest
manifest = {
    "model_path": model_path,
    "animation_requested": args.animation,
    "animation_used": animation_used,
    "animation_source": animation_source,
    "available_animations": available_animations,
    "fps": args.fps,
    "duration_sec": args.seconds,
    "frame_count": total_frames,
    "resolution": args.resolution,
    "camera_location": list(camera.location),
    "light_location": list(light.location),
    "output_video": str(out_path),
    "frames_dir": str(frames_dir),
    "source_asset": "Quaternius Animated Animal Pack",
    "license": "Public Domain / CC0",
}

manifest_path = out_path.parent / f"{out_path.stem}_render_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"[Render] Manifest: {manifest_path}")
print(f"[Render] Done. {total_frames} frames @ {args.fps}fps")
