"""Render 5 seconds of a glTF animated model using Blender.

Usage (run inside Blender):
  blender --background --python scripts/render_gltf_5sec.py -- \
    --model assets/gltf_samples/Fox/glTF-Binary/Fox.glb \
    --out outputs/synthetic/fox_5s.mp4 \
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
parser.add_argument("--model", required=True, help="Path to .glb file")
parser.add_argument("--out", required=True, help="Output video path (.mp4)")
parser.add_argument("--seconds", type=float, default=5.0)
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--resolution", nargs=2, type=int, default=[640, 480])
parser.add_argument("--animation", default="Walk", help="Animation name to use")
parser.add_argument("--frames-dir", default=None, help="Also save individual frames here")
args = parser.parse_args(argv)

try:
    import bpy
except ImportError:
    print("ERROR: This script must be run inside Blender.")
    print("  blender --background --python scripts/render_gltf_5sec.py -- --model ... --out ...")
    sys.exit(1)

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import glTF
model_path = str(Path(args.model).resolve())
bpy.ops.import_scene.gltf(filepath=model_path)
print(f"[Render] Imported: {model_path}")

# Find armature and available animations
armature = None
for obj in bpy.context.scene.objects:
    if obj.type == 'ARMATURE':
        armature = obj
        break

available_animations = []
if bpy.data.actions:
    available_animations = [a.name for a in bpy.data.actions]
    print(f"[Render] Available animations: {available_animations}")

# Select animation
animation_used = None
if args.animation and args.animation in available_animations:
    animation_used = args.animation
elif available_animations:
    animation_used = available_animations[0]
    print(f"[Render] Requested '{args.animation}' not found, using fallback: {animation_used}")

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

# Create world with solid background
world = bpy.data.worlds.new("CleanWorld")
scene.world = world
world.use_nodes = True
bg_node = world.node_tree.nodes["Background"]
bg_node.inputs[0].default_value = (0.15, 0.15, 0.2, 1.0)  # Dark blue-gray

# Compute bounding box of imported objects
min_co = [float('inf')] * 3
max_co = [float('-inf')] * 3
for obj in bpy.context.scene.objects:
    if obj.type in ('MESH', 'ARMATURE'):
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ bpy.mathutils.Vector(corner) if hasattr(bpy, 'mathutils') else None
            if world_corner is None:
                import mathutils
                world_corner = obj.matrix_world @ mathutils.Vector(corner)
            for i in range(3):
                min_co[i] = min(min_co[i], world_corner[i])
                max_co[i] = max(max_co[i], world_corner[i])

center = [(min_co[i] + max_co[i]) / 2 for i in range(3)]
size = max(max_co[i] - min_co[i] for i in range(3))
cam_distance = size * 2.5

# Create camera
bpy.ops.object.camera_add(
    location=(center[0] + cam_distance * 0.7, center[1] - cam_distance, center[2] + size * 0.5)
)
camera = bpy.context.object
camera.name = "FoxCam"
scene.camera = camera

# Point camera at center
import mathutils
direction = mathutils.Vector(center) - camera.location
rot_quat = direction.to_track_quat('-Z', 'Y')
camera.rotation_euler = rot_quat.to_euler()

# Create light
bpy.ops.object.light_add(
    type='SUN',
    location=(center[0] + size, center[1] - size, center[2] + size * 2)
)
light = bpy.context.object
light.data.energy = 3.0

# Setup output
out_path = Path(args.out).resolve()
out_path.parent.mkdir(parents=True, exist_ok=True)

# Also save individual frames
frames_dir = Path(args.frames_dir) if args.frames_dir else out_path.parent / "fox_frames"
frames_dir.mkdir(parents=True, exist_ok=True)

# Render frames
scene.render.image_settings.file_format = 'PNG'
for frame in range(scene.frame_start, scene.frame_end + 1):
    scene.frame_set(frame)
    scene.render.filepath = str(frames_dir / f"frame_{frame:04d}.png")
    bpy.ops.render.render(write_still=True)

print(f"[Render] Frames saved: {frames_dir}")

# Encode video with ffmpeg if available
import subprocess
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-framerate", str(args.fps),
    "-i", str(frames_dir / "frame_%04d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "-crf", "18",
    str(out_path),
]
try:
    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    print(f"[Render] Video: {out_path}")
except (subprocess.CalledProcessError, FileNotFoundError):
    # Try Blender's built-in encoding as fallback
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(animation=True)
    print(f"[Render] Video (Blender encode): {out_path}")

# Write render manifest
manifest = {
    "model_path": model_path,
    "animation_requested": args.animation,
    "animation_used": animation_used,
    "fps": args.fps,
    "duration_sec": args.seconds,
    "frame_count": total_frames,
    "resolution": args.resolution,
    "camera_location": list(camera.location),
    "camera_rotation": list(camera.rotation_euler),
    "light_location": list(light.location),
    "output_video": str(out_path),
    "frames_dir": str(frames_dir),
}

manifest_path = out_path.parent / "fox_render_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"[Render] Manifest: {manifest_path}")
print(f"[Render] Done. {total_frames} frames, {args.seconds}s @ {args.fps}fps")
