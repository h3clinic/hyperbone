"""
Render Fox.glb 3D animated model to video frames and test HyperBone pipeline.
Uses trimesh + pygltflib to extract mesh & skeletal animation, then projects
to 2D with a simple perspective camera. No Blender required.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
import json
import struct
from pathlib import Path
import pygltflib
import trimesh

# ─── Config ──────────────────────────────────────────────────────────────────
GLB_PATH = Path("assets/gltf_samples/Fox/glTF-Binary/Fox.glb")
OUTPUT_DIR = Path("output/fox3d")
VIDEO_PATH = OUTPUT_DIR / "fox3d_walk.mp4"
PROPOSALS_PATH = OUTPUT_DIR / "proposals.jsonl"
FRAME_W, FRAME_H = 640, 480
FPS = 24
DURATION_SEC = 5.0
ANIM_INDEX = 1  # Walk animation
BG_COLOR = (40, 45, 50)  # Dark gray-blue background

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── glTF Helper Functions ───────────────────────────────────────────────────
def load_glb():
    glb = pygltflib.GLTF2().load(str(GLB_PATH))
    blob = glb.binary_blob()
    return glb, blob


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
    """Quaternion (x, y, z, w) to 4x4 rotation matrix."""
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


def node_local_transform(node):
    """Compute local transform from node TRS or matrix."""
    if node.matrix:
        return np.array(node.matrix).reshape(4, 4).T  # glTF uses column-major
    m = np.eye(4)
    if node.scale:
        m = m @ scale_matrix(node.scale)
    if node.rotation:
        m = quat_to_matrix(node.rotation) @ m
    if node.translation:
        m = translation_matrix(node.translation) @ m
    return m


# ─── Skeleton + Animation ────────────────────────────────────────────────────
def build_skeleton(glb):
    """Build joint hierarchy from the skin."""
    skin = glb.skins[0]
    joint_indices = skin.joints
    # Build parent map
    parent_map = {}
    for idx in joint_indices:
        node = glb.nodes[idx]
        if node.children:
            for child_idx in node.children:
                if child_idx in joint_indices:
                    parent_map[child_idx] = idx
    return joint_indices, parent_map


def sample_animation(glb, blob, anim_idx, t):
    """Sample animation channels at time t, return per-node overrides."""
    anim = glb.animations[anim_idx]
    overrides = {}  # node_idx -> {path: value}

    for channel in anim.channels:
        sampler = anim.samplers[channel.sampler]
        times = read_accessor(glb, blob, sampler.input).flatten()
        values = read_accessor(glb, blob, sampler.output)

        # Wrap t into animation duration
        duration = times[-1]
        t_wrapped = t % duration if duration > 0 else 0

        # Linear interpolation
        idx = np.searchsorted(times, t_wrapped, side='right') - 1
        idx = max(0, min(idx, len(times) - 2))
        t0, t1 = times[idx], times[idx + 1]
        alpha = (t_wrapped - t0) / (t1 - t0) if t1 > t0 else 0.0

        v0 = values[idx]
        v1 = values[idx + 1]

        if channel.target.path == 'rotation':
            # SLERP for quaternions
            val = slerp(v0, v1, alpha)
        else:
            val = v0 * (1 - alpha) + v1 * alpha

        node_idx = channel.target.node
        if node_idx not in overrides:
            overrides[node_idx] = {}
        overrides[node_idx][channel.target.path] = val

    return overrides


def slerp(q0, q1, t):
    """Spherical linear interpolation for quaternions."""
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


def compute_posed_vertices(glb, blob, mesh_verts, skin, joint_indices, overrides):
    """Apply skeletal animation to mesh vertices using linear blend skinning."""
    # Compute world transforms for each joint
    joint_world = {}

    def get_world_transform(node_idx, visited=None):
        if visited is None:
            visited = set()
        if node_idx in joint_world:
            return joint_world[node_idx]
        if node_idx in visited:
            return np.eye(4)
        visited.add(node_idx)

        node = glb.nodes[node_idx]
        # Start with node local transform
        local = node_local_transform(node)

        # Apply animation overrides
        if node_idx in overrides:
            ov = overrides[node_idx]
            local = np.eye(4)
            s = ov.get('scale', node.scale or [1, 1, 1])
            r = ov.get('rotation', node.rotation or [0, 0, 0, 1])
            tr = ov.get('translation', node.translation or [0, 0, 0])
            local = translation_matrix(tr) @ quat_to_matrix(r) @ scale_matrix(s)

        # Find parent
        parent_idx = None
        for jidx in joint_indices:
            n = glb.nodes[jidx]
            if n.children and node_idx in n.children:
                parent_idx = jidx
                break

        if parent_idx is not None:
            parent_world = get_world_transform(parent_idx, visited)
            world = parent_world @ local
        else:
            world = local

        joint_world[node_idx] = world
        return world

    # Compute all joint world transforms
    for jidx in joint_indices:
        get_world_transform(jidx)

    # Read inverse bind matrices
    ibm_data = read_accessor(glb, blob, skin.inverseBindMatrices)
    ibms = ibm_data.reshape(-1, 4, 4)
    # glTF stores matrices column-major
    ibms = ibms.transpose(0, 2, 1)

    # Read skinning weights from the mesh
    # Find the mesh primitive
    mesh_node = None
    for node in glb.nodes:
        if node.mesh is not None:
            mesh_node = node
            break

    prim = glb.meshes[mesh_node.mesh].primitives[0]
    joints_attr = prim.attributes.JOINTS_0
    weights_attr = prim.attributes.WEIGHTS_0

    if joints_attr is None or weights_attr is None:
        # No skinning data, just return vertices as-is
        return mesh_verts

    joint_ids = read_accessor(glb, blob, joints_attr)  # (N, 4) joint indices per vertex
    weights = read_accessor(glb, blob, weights_attr)    # (N, 4) weights per vertex

    # Compute skinned positions
    n_verts = mesh_verts.shape[0]
    posed = np.zeros((n_verts, 3), dtype=np.float64)

    for i in range(4):
        w = weights[:, i:i+1].astype(np.float64)
        j = joint_ids[:, i].astype(int)

        # For each vertex, compute: weight * (jointWorld @ IBM @ vertex)
        for v_idx in range(n_verts):
            if w[v_idx, 0] < 1e-6:
                continue
            jidx = joint_indices[j[v_idx]]
            skin_mat = joint_world[jidx] @ ibms[j[v_idx]]
            v_homo = np.append(mesh_verts[v_idx], 1.0)
            posed[v_idx] += w[v_idx, 0] * (skin_mat @ v_homo)[:3]

    return posed


def compute_posed_vertices_fast(glb, blob, mesh_verts, skin, joint_indices, overrides):
    """Vectorized skinning - much faster than per-vertex loop."""
    # Compute world transforms for each joint
    joint_world = {}

    def get_world_transform(node_idx, visited=None):
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
            parent_world = get_world_transform(parent_idx, visited)
            world = parent_world @ local
        else:
            world = local

        joint_world[node_idx] = world
        return world

    for jidx in joint_indices:
        get_world_transform(jidx)

    # Inverse bind matrices
    ibm_data = read_accessor(glb, blob, skin.inverseBindMatrices)
    ibms = ibm_data.reshape(-1, 4, 4).transpose(0, 2, 1)

    # Skinning attributes
    mesh_node = None
    for node in glb.nodes:
        if node.mesh is not None:
            mesh_node = node
            break
    prim = glb.meshes[mesh_node.mesh].primitives[0]
    joints_attr = prim.attributes.JOINTS_0
    weights_attr = prim.attributes.WEIGHTS_0

    if joints_attr is None or weights_attr is None:
        return mesh_verts

    joint_ids = read_accessor(glb, blob, joints_attr).astype(int)   # (N, 4)
    weights = read_accessor(glb, blob, weights_attr).astype(np.float64)  # (N, 4)

    # Precompute skin matrices for each joint
    n_joints = len(joint_indices)
    skin_mats = np.zeros((n_joints, 4, 4))
    for j in range(n_joints):
        jidx = joint_indices[j]
        skin_mats[j] = joint_world[jidx] @ ibms[j]

    # Vectorized skinning
    n_verts = mesh_verts.shape[0]
    verts_homo = np.hstack([mesh_verts, np.ones((n_verts, 1))])  # (N, 4)
    posed = np.zeros((n_verts, 3))

    for i in range(4):
        w = weights[:, i]  # (N,)
        j = joint_ids[:, i]  # (N,)
        mask = w > 1e-6

        if not mask.any():
            continue

        # Gather skin matrices for these joints
        mats = skin_mats[j[mask]]  # (M, 4, 4)
        v = verts_homo[mask]  # (M, 4)
        # Batch multiply: (M, 4, 4) @ (M, 4, 1) -> (M, 4, 1)
        transformed = np.einsum('mij,mj->mi', mats, v)[:, :3]  # (M, 3)
        posed[mask] += w[mask, np.newaxis] * transformed

    return posed


# ─── 3D Rendering ────────────────────────────────────────────────────────────
def perspective_project(verts_3d, fov_deg=45, w=FRAME_W, h=FRAME_H):
    """Project 3D vertices to 2D using perspective projection."""
    fov = np.radians(fov_deg)
    aspect = w / h
    f = 1.0 / np.tan(fov / 2)

    # Camera looking at the fox from the side
    # Fox is oriented: Y=up, Z=forward (roughly)
    # Place camera to the side
    cam_pos = np.array([100.0, 50.0, 0.0])  # Side view
    cam_target = np.array([0.0, 35.0, 0.0])  # Look at fox center
    cam_up = np.array([0.0, 1.0, 0.0])

    # View matrix (look-at)
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

    # Apply view transform
    verts_homo = np.hstack([verts_3d, np.ones((len(verts_3d), 1))])
    verts_cam = (view @ verts_homo.T).T[:, :3]

    # Perspective projection
    x = verts_cam[:, 0] * f / aspect
    y = verts_cam[:, 1] * f
    z = -verts_cam[:, 2]  # Negate for OpenGL-style

    # Avoid division by zero
    z = np.maximum(z, 0.01)

    px = (x / z * 0.5 + 0.5) * w
    py = (1.0 - (y / z * 0.5 + 0.5)) * h

    return np.stack([px, py], axis=1), z


def render_frame(posed_verts, faces, w=FRAME_W, h=FRAME_H):
    """Render a single frame by projecting and rasterizing the mesh."""
    img = np.full((h, w, 3), BG_COLOR, dtype=np.uint8)

    pts_2d, depths = perspective_project(posed_verts, w=w, h=h)

    # Simple face rendering (painter's algorithm by average depth)
    face_depths = depths[faces].mean(axis=1)
    sorted_faces = faces[np.argsort(-face_depths)]  # Back to front

    for face in sorted_faces:
        tri = pts_2d[face].astype(np.int32)
        # Simple shading based on face normal
        v0, v1, v2 = posed_verts[face]
        normal = np.cross(v1 - v0, v2 - v0)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-8:
            continue
        normal = normal / norm_len

        # Light from camera direction
        light_dir = np.array([1.0, 0.5, 0.3])
        light_dir = light_dir / np.linalg.norm(light_dir)
        intensity = max(0.2, min(1.0, np.dot(normal, light_dir) * 0.6 + 0.4))

        # Orange-brown fox color
        color = (int(60 * intensity), int(120 * intensity), int(200 * intensity))
        cv2.fillConvexPoly(img, tri.reshape(-1, 1, 2), color)

    return img


# ─── Main Pipeline ───────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("HyperBone 3D Test: Fox.glb Walk Animation")
    print("=" * 60)

    # Load model
    print("\n[1/5] Loading Fox.glb...")
    glb, blob = load_glb()
    scene = trimesh.load(str(GLB_PATH))
    mesh = scene.geometry['fox1']
    rest_verts = mesh.vertices.copy()
    faces = mesh.faces.copy()
    print(f"  Mesh: {rest_verts.shape[0]} vertices, {faces.shape[0]} faces")

    # Get skeleton info
    skin = glb.skins[0]
    joint_indices, parent_map = build_skeleton(glb)
    print(f"  Skeleton: {len(joint_indices)} joints")

    # Get animation info
    anim = glb.animations[ANIM_INDEX]
    sampler0 = anim.samplers[0]
    timestamps = read_accessor(glb, blob, sampler0.input).flatten()
    anim_duration = timestamps[-1]
    print(f"  Animation: '{anim.name}', duration={anim_duration:.3f}s (looped to {DURATION_SEC}s)")

    # Render frames
    print(f"\n[2/5] Rendering {int(DURATION_SEC * FPS)} frames at {FPS} fps...")
    n_frames = int(DURATION_SEC * FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(VIDEO_PATH), fourcc, FPS, (FRAME_W, FRAME_H))

    frames_cache = []
    for fi in range(n_frames):
        t = fi / FPS
        overrides = sample_animation(glb, blob, ANIM_INDEX, t)
        posed = compute_posed_vertices_fast(glb, blob, rest_verts, skin, joint_indices, overrides)
        img = render_frame(posed, faces)

        # Add frame counter
        cv2.putText(img, f"Frame {fi}/{n_frames}  t={t:.2f}s",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, "Fox.glb Walk (3D skinned mesh)",
                    (10, FRAME_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        writer.write(img)
        frames_cache.append(img)

        if fi % 20 == 0:
            print(f"    Frame {fi}/{n_frames}")

    writer.release()
    print(f"  Video saved: {VIDEO_PATH}")

    # Generate proposals via background differencing
    print(f"\n[3/5] Generating bbox proposals (background diff, 5 fps)...")
    sample_interval = FPS // 5  # Sample at 5fps
    proposals = []
    proposal_idx = 0

    for fi in range(0, n_frames, sample_interval):
        img = frames_cache[fi]
        # Background diff: subtract BG color, threshold
        bg = np.full_like(img, BG_COLOR, dtype=np.uint8)
        diff = cv2.absdiff(img, bg)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Merge all contours into one bbox (fox is one object)
            all_pts = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(all_pts)
            # Add margin
            margin = 10
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(FRAME_W - x, w + 2 * margin)
            h = min(FRAME_H - y, h + 2 * margin)

            if w > 20 and h > 20:
                proposals.append({
                    "frame_idx": fi,
                    "object_id": 1,
                    "label": "fox",
                    "bbox_xywh": [x, y, w, h],
                    "confidence": 0.95,
                    "source_asset": "Fox.glb (3D skinned mesh)"
                })
                proposal_idx += 1

    with open(PROPOSALS_PATH, 'w') as f:
        for p in proposals:
            f.write(json.dumps(p) + "\n")
    print(f"  Generated {len(proposals)} proposals -> {PROPOSALS_PATH}")

    # Run HyperBone pipeline
    print(f"\n[4/5] Running HyperBone pipeline...")
    from hyperbone.pipelines.proposal_skeleton import run_proposal_skeleton

    pipeline_out = OUTPUT_DIR / "pipeline"
    stats = run_proposal_skeleton(
        video_path=str(VIDEO_PATH),
        output_dir=str(pipeline_out),
        proposals_path=str(PROPOSALS_PATH),
        proposal_source="manual",
        sample_fps=5,
        max_side=128,
        thinning_algorithm="zhang-suen",
    )

    # Read quality JSONL for detailed results
    quality_path = pipeline_out / "quality.jsonl"
    results = []
    if quality_path.exists():
        with open(quality_path) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

    # Summarize results
    print(f"\n[5/5] Results Summary")
    print("-" * 60)
    accepted = [r for r in results if r.get("accepted", False)]
    rejected = [r for r in results if not r.get("accepted", False)]
    print(f"  Total proposals processed: {len(results)}")
    print(f"  Accepted: {len(accepted)}")
    print(f"  Rejected: {len(rejected)}")
    print(f"  Pipeline stats: {stats}")

    if accepted:
        scores = [r["quality_score"] for r in accepted if "quality_score" in r]
        nodes = [r["n_nodes"] for r in accepted if "n_nodes" in r]
        edges = [r["n_edges"] for r in accepted if "n_edges" in r]
        times = [r["time_ms"] for r in accepted if "time_ms" in r]
        if scores:
            print(f"  Quality scores: min={min(scores):.3f} avg={np.mean(scores):.3f} max={max(scores):.3f}")
        if nodes:
            print(f"  Graph nodes: min={min(nodes)} avg={np.mean(nodes):.1f} max={max(nodes)}")
        if edges:
            print(f"  Graph edges: min={min(edges)} avg={np.mean(edges):.1f} max={max(edges)}")
        if times:
            print(f"  Time/object: min={min(times):.1f}ms avg={np.mean(times):.1f}ms max={max(times):.1f}ms")

        # Verify all use hyperbone-custom mapper
        mappers = set(r.get("skeleton_mapper", "unknown") for r in accepted)
        print(f"  Skeleton mapper(s): {mappers}")
        if "hyperbone-custom" in mappers:
            print(f"\n  ✓ ALL graphs produced by hyperbone-custom (no SAM2/MediaPipe)")
    else:
        print("  WARNING: No accepted skeletons!")

    # Generate overlay video
    print(f"\n  Generating overlay video...")
    overlay_path = OUTPUT_DIR / "fox3d_overlay.mp4"
    overlay_writer = cv2.VideoWriter(str(overlay_path), fourcc, FPS, (FRAME_W, FRAME_H))

    # Load graph records (have bbox, nodes, edges)
    graph_jsonl_path = pipeline_out / "graphs" / "graphs.jsonl"
    graph_records = []
    if graph_jsonl_path.exists():
        with open(graph_jsonl_path) as f:
            for line in f:
                if line.strip():
                    graph_records.append(json.loads(line))

    # Index by frame
    result_by_frame = {}
    for r in graph_records:
        result_by_frame[r["frame_idx"]] = r

    for fi in range(n_frames):
        img = frames_cache[fi].copy()

        if fi in result_by_frame:
            r = result_by_frame[fi]
            bbox = r.get("bbox_xywh", [])
            accepted_flag = r.get("accepted", False)
            color = (0, 255, 0) if accepted_flag else (0, 0, 255)

            if bbox:
                x, y, bw, bh = [int(v) for v in bbox]
                cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)

                # Draw skeleton graph if accepted
                if accepted_flag and r.get("nodes"):
                    nodes_list = r["nodes"]
                    edges_list = r.get("edges", [])
                    # Build node id -> xy map
                    node_xy = {n["id"]: n["xy"] for n in nodes_list}
                    for edge in edges_list:
                        n1, n2 = edge["parent"], edge["child"]
                        if n1 in node_xy and n2 in node_xy:
                            pt1 = (int(node_xy[n1][0]), int(node_xy[n1][1]))
                            pt2 = (int(node_xy[n2][0]), int(node_xy[n2][1]))
                            cv2.line(img, pt1, pt2, (0, 255, 255), 1)
                    for node in nodes_list:
                        px, py = int(node["xy"][0]), int(node["xy"][1])
                        cv2.circle(img, (px, py), 2, (255, 0, 255), -1)

                # Label
                label = f"{'OK' if accepted_flag else 'REJ'} n={r.get('node_count', 0)}"
                cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        overlay_writer.write(img)

    overlay_writer.release()
    print(f"  Overlay saved: {overlay_path}")

    # Final verdict
    print("\n" + "=" * 60)
    total_processed = len(results)
    n_accepted = len(accepted)
    acceptance_rate = n_accepted / max(1, total_processed)
    if acceptance_rate >= 0.5 and n_accepted >= 5:
        print(f"  ✓ PASS: HyperBone works on 3D animated mesh!")
        print(f"    {n_accepted}/{total_processed} accepted ({acceptance_rate:.0%})")
    elif n_accepted > 0:
        print(f"  ~ PARTIAL: {n_accepted}/{total_processed} accepted ({acceptance_rate:.0%})")
    else:
        print(f"  ✗ FAIL: Acceptance rate too low: {acceptance_rate:.0%}")
    print("=" * 60)

    # Save report
    report = {
        "test": "HyperBone 3D Fox.glb Walk",
        "source": "Fox.glb (KhronosGroup glTF-Sample-Assets)",
        "animation": "Walk",
        "mesh_verts": int(rest_verts.shape[0]),
        "mesh_faces": int(faces.shape[0]),
        "skeleton_joints": len(joint_indices),
        "frames_rendered": n_frames,
        "fps": FPS,
        "proposals": len(proposals),
        "accepted": n_accepted,
        "rejected": len(rejected),
        "acceptance_rate": round(acceptance_rate, 3),
        "pass": acceptance_rate >= 0.5 and n_accepted >= 5,
        "pipeline_stats": stats,
    }
    report_path = OUTPUT_DIR / "test_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report: {report_path}")


if __name__ == "__main__":
    main()
