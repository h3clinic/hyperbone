"""
Export Fox.glb true 3D armature ground truth per frame (NO Blender required).

Uses pygltflib to extract the actual bone positions from the glTF skinned mesh
and evaluates the skeletal animation at each frame. This produces the internal
armature joint positions that HyperBone3D should eventually predict.

Usage:
  python scripts/export_fox_armature_gt_nobl.py \
    --model assets/gltf_samples/Fox/glTF-Binary/Fox.glb \
    --out output/fox3d/fox_armature_gt.jsonl \
    --seconds 5 --fps 24 --animation Walk
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import numpy as np
import pygltflib
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def read_accessor(glb, blob, acc_idx):
    acc = glb.accessors[acc_idx]
    bv = glb.bufferViews[acc.bufferView]
    offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    count = acc.count
    type_sizes = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
    comp_sizes = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
    np_types = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
                5123: np.uint16, 5125: np.uint32, 5126: np.float32}
    n_components = type_sizes[acc.type]
    dtype = np_types[acc.componentType]
    comp_size = comp_sizes[acc.componentType]
    stride = bv.byteStride
    if stride and stride > 0:
        data = np.zeros((count, n_components), dtype=dtype)
        for i in range(count):
            start = offset + i * stride
            end = start + n_components * comp_size
            data[i] = np.frombuffer(blob[start:end], dtype=dtype)
    else:
        total = count * n_components
        data = np.frombuffer(blob[offset:offset + total * comp_size], dtype=dtype)
        if n_components > 1:
            data = data.reshape(count, n_components)
    return data


def quat_to_matrix(q):
    x, y, z, w = q
    m = np.eye(4)
    m[0, 0] = 1 - 2*(y*y + z*z)
    m[0, 1] = 2*(x*y - z*w)
    m[0, 2] = 2*(x*z + y*w)
    m[1, 0] = 2*(x*y + z*w)
    m[1, 1] = 1 - 2*(x*x + z*z)
    m[1, 2] = 2*(y*z - x*w)
    m[2, 0] = 2*(x*z - y*w)
    m[2, 1] = 2*(y*z + x*w)
    m[2, 2] = 1 - 2*(x*x + y*y)
    return m


def translation_matrix(t):
    m = np.eye(4)
    m[:3, 3] = t
    return m


def scale_matrix(s):
    m = np.eye(4)
    m[0, 0], m[1, 1], m[2, 2] = s[0], s[1], s[2]
    return m


def slerp(q0, q1, t):
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return q0 + t * (q1 - q0)
    theta = np.arccos(np.clip(dot, -1, 1))
    sin_theta = np.sin(theta)
    if sin_theta < 1e-6:
        return q0
    return (np.sin((1 - t) * theta) / sin_theta) * q0 + (np.sin(t * theta) / sin_theta) * q1


def node_local_transform(node):
    m = np.eye(4)
    if node.matrix:
        return np.array(node.matrix).reshape(4, 4).T
    if node.scale:
        m = scale_matrix(node.scale) @ m
    if node.rotation:
        m = quat_to_matrix(node.rotation) @ m
    if node.translation:
        m = translation_matrix(node.translation) @ m
    return m


def sample_animation(glb, blob, anim_idx, t):
    anim = glb.animations[anim_idx]
    overrides = {}
    for channel in anim.channels:
        sampler = anim.samplers[channel.sampler]
        times = read_accessor(glb, blob, sampler.input).flatten()
        values = read_accessor(glb, blob, sampler.output)
        duration = times[-1]
        t_wrapped = t % duration if duration > 0 else 0
        idx = np.searchsorted(times, t_wrapped, side='right') - 1
        idx = max(0, min(idx, len(times) - 2))
        t0, t1 = times[idx], times[idx + 1]
        alpha = (t_wrapped - t0) / (t1 - t0) if t1 > t0 else 0.0
        v0, v1 = values[idx], values[idx + 1]
        if channel.target.path == 'rotation':
            val = slerp(v0, v1, alpha)
        else:
            val = v0 * (1 - alpha) + v1 * alpha
        node_idx = channel.target.node
        if node_idx not in overrides:
            overrides[node_idx] = {}
        overrides[node_idx][channel.target.path] = val
    return overrides


def compute_joint_world_transforms(glb, joint_indices, overrides):
    """Compute world transform for each joint node."""
    joint_world = {}

    def get_world(node_idx, visited=None):
        if visited is None:
            visited = set()
        if node_idx in joint_world:
            return joint_world[node_idx]
        if node_idx in visited:
            return np.eye(4)
        visited.add(node_idx)

        node = glb.nodes[node_idx]
        local = node_local_transform(node)

        if node_idx in overrides:
            ov = overrides[node_idx]
            local = np.eye(4)
            s = ov.get('scale', node.scale or [1, 1, 1])
            r = ov.get('rotation', node.rotation or [0, 0, 0, 1])
            tr = ov.get('translation', node.translation or [0, 0, 0])
            local = translation_matrix(tr) @ quat_to_matrix(r) @ scale_matrix(s)

        parent_idx = None
        for jidx in joint_indices:
            n = glb.nodes[jidx]
            if n.children and node_idx in n.children:
                parent_idx = jidx
                break

        if parent_idx is not None:
            parent_world = get_world(parent_idx, visited)
            world = parent_world @ local
        else:
            world = local

        joint_world[node_idx] = world
        return world

    for jidx in joint_indices:
        get_world(jidx)
    return joint_world


def perspective_project(xyz, fov_deg=45, w=640, h=480):
    """Project single 3D point using the same camera as render."""
    fov = np.radians(fov_deg)
    aspect = w / h
    f = 1.0 / np.tan(fov / 2)

    cam_pos = np.array([100.0, 50.0, 0.0])
    cam_target = np.array([0.0, 35.0, 0.0])
    cam_up = np.array([0.0, 1.0, 0.0])

    forward = cam_target - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, cam_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    view = np.eye(4)
    view[0, :3] = right
    view[1, :3] = up
    view[2, :3] = -forward
    view[:3, 3] = -view[:3, :3] @ cam_pos

    p_homo = np.array([*xyz, 1.0])
    p_cam = view @ p_homo
    cam_xyz = p_cam[:3].tolist()

    x = p_cam[0] * f / aspect
    y = p_cam[1] * f
    z = -p_cam[2]
    z = max(z, 0.01)

    px = (x / z * 0.5 + 0.5) * w
    py = (1.0 - (y / z * 0.5 + 0.5)) * h

    return (round(px, 2), round(py, 2)), cam_xyz


def infer_joint_type(name: str) -> str:
    """Infer joint type from bone name."""
    n = name.lower()
    if 'root' in n or n == 'b_root_00':
        return "root"
    if 'spine' in n or 'body' in n:
        return "spine"
    if 'neck' in n:
        return "neck"
    if 'head' in n:
        return "head"
    if 'shoulder' in n or 'upperarm' in n:
        return "shoulder"
    if 'forearm' in n or 'elbow' in n:
        return "elbow"
    if 'hand' in n or 'wrist' in n or 'paw' in n:
        return "paw"
    if 'hip' in n or 'thigh' in n or 'upperleg' in n:
        return "hip"
    if 'shin' in n or 'lowerleg' in n or 'knee' in n:
        return "knee"
    if 'foot' in n or 'ankle' in n:
        return "ankle"
    if 'tail' in n:
        return "tail"
    if 'ear' in n:
        return "ear"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Export Fox armature GT (no Blender)")
    parser.add_argument("--model", default="assets/gltf_samples/Fox/glTF-Binary/Fox.glb")
    parser.add_argument("--out", default="output/fox3d/fox_armature_gt.jsonl")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--animation", default="Walk")
    parser.add_argument("--resolution", type=int, nargs=2, default=[640, 480])
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    glb = pygltflib.GLTF2().load(args.model)
    blob = glb.binary_blob()

    # Find animation by name
    anim_idx = 0
    for i, anim in enumerate(glb.animations):
        if anim.name and anim.name.lower() == args.animation.lower():
            anim_idx = i
            break
    anim = glb.animations[anim_idx]
    print(f"Animation: '{anim.name}' (index {anim_idx})")

    # Get skeleton
    skin = glb.skins[0]
    joint_indices = skin.joints
    print(f"Skeleton: {len(joint_indices)} joints")

    # Build parent map
    parent_map = {}
    for jidx in joint_indices:
        node = glb.nodes[jidx]
        if node.children:
            for child_idx in node.children:
                if child_idx in joint_indices:
                    parent_map[child_idx] = jidx

    # Joint info
    joint_names = {}
    for jidx in joint_indices:
        node = glb.nodes[jidx]
        joint_names[jidx] = node.name or f"joint_{jidx}"

    # Export frames
    n_frames = int(args.seconds * args.fps)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = args.resolution

    print(f"Exporting {n_frames} frames at {args.fps} fps...")

    records = []
    rest_joints = []

    with open(out_path, 'w') as f:
        for fi in range(n_frames):
            t = fi / args.fps
            overrides = sample_animation(glb, blob, anim_idx, t)
            joint_world = compute_joint_world_transforms(glb, joint_indices, overrides)

            joints_data = []
            for local_id, jidx in enumerate(joint_indices):
                world_mat = joint_world[jidx]
                world_xyz = world_mat[:3, 3].tolist()
                image_xy, cam_xyz = perspective_project(world_xyz, w=w, h=h)
                name = joint_names[jidx]

                parent_local_id = None
                if jidx in parent_map:
                    parent_local_id = joint_indices.index(parent_map[jidx])

                joints_data.append({
                    "id": local_id,
                    "name": name,
                    "parent_id": parent_local_id,
                    "type": infer_joint_type(name),
                    "world_xyz": [round(v, 4) for v in world_xyz],
                    "camera_xyz": [round(v, 4) for v in cam_xyz],
                    "image_xy": [round(v, 2) for v in image_xy],
                    "visible": True,
                })

            # Build bone list
            bones_data = []
            for local_id, jidx in enumerate(joint_indices):
                if jidx in parent_map:
                    parent_local_id = joint_indices.index(parent_map[jidx])
                    parent_name = joint_names[parent_map[jidx]]
                    child_name = joint_names[jidx]
                    bones_data.append({
                        "parent_id": parent_local_id,
                        "child_id": local_id,
                        "name": f"{parent_name}_to_{child_name}",
                    })

            record = {
                "frame_idx": fi,
                "timestamp_sec": round(t, 4),
                "animation": anim.name,
                "asset_id": "Fox.glb",
                "joints": joints_data,
                "bones": bones_data,
                "resolution": [w, h],
                "bbox_xywh": None,  # Could compute from projected joints
            }

            # Compute bbox from projected joints
            xs = [j["image_xy"][0] for j in joints_data if j["visible"]]
            ys = [j["image_xy"][1] for j in joints_data if j["visible"]]
            if xs and ys:
                margin = 20
                bx = max(0, int(min(xs)) - margin)
                by = max(0, int(min(ys)) - margin)
                bx2 = min(w, int(max(xs)) + margin)
                by2 = min(h, int(max(ys)) + margin)
                record["bbox_xywh"] = [bx, by, bx2 - bx, by2 - by]

            f.write(json.dumps(record) + "\n")
            records.append(record)

    print(f"Exported {len(records)} frames -> {out_path}")
    print(f"  Joints per frame: {len(joint_indices)}")
    print(f"  Bones per frame: {len(bones_data)}")

    # Write rest skeleton
    rest_path = out_path.parent / "fox_rest_skeleton.json"
    # Use frame 0 as rest pose reference
    rest_record = records[0]
    rest_skeleton = {
        "asset_id": "Fox.glb",
        "joint_count": len(joint_indices),
        "bone_count": len(bones_data),
        "joints": rest_record["joints"],
        "bones": rest_record["bones"],
        "animation": anim.name,
    }
    with open(rest_path, 'w') as f:
        json.dump(rest_skeleton, f, indent=2)
    print(f"  Rest skeleton: {rest_path}")


if __name__ == "__main__":
    main()
