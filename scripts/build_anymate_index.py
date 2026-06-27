"""
Build Anymate Index — scan an Anymate dataset root and catalog all assets.

Scans for .glb/.gltf/.fbx/.obj files with associated rig/skinning data.
Detects: mesh availability, skeleton, skinning weights, animation clips.
Outputs assets.jsonl index + report.md summary.

Usage:
    python scripts/build_anymate_index.py --anymate-root D:\\datasets\\Anymate --out outputs/anymate_index --max-assets 100
"""
import argparse
import json
import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.rigs.schema import AssetRig, AnimationClip, write_jsonl


MESH_EXTENSIONS = {".glb", ".gltf", ".fbx", ".obj", ".ply", ".stl"}
RIG_EXTENSIONS = {".glb", ".gltf", ".fbx"}  # Formats that can contain rigs


def scan_asset_file(file_path: Path, asset_id: str) -> AssetRig:
    """Inspect a single 3D asset file for rig/skinning/animation data."""
    ext = file_path.suffix.lower()

    asset = AssetRig(
        asset_id=asset_id,
        mesh_path=str(file_path),
        rig_path=str(file_path),
        category="unknown",
    )

    if ext in (".glb", ".gltf"):
        _inspect_gltf(file_path, asset)
    elif ext == ".fbx":
        _inspect_fbx(file_path, asset)
    elif ext in (".obj", ".ply", ".stl"):
        # Mesh-only formats — no rig possible
        asset.has_skeleton = False
        asset.has_skinning = False
        asset.has_animation = False
        _get_basic_mesh_stats(file_path, asset)
    else:
        pass

    return asset


def _inspect_gltf(file_path: Path, asset: AssetRig):
    """Inspect a glTF/GLB file for skeleton, skinning, and animation."""
    try:
        import pygltflib
        gltf = pygltflib.GLTF2().load(str(file_path))

        # Mesh stats
        total_verts = 0
        if gltf.meshes:
            for mesh in gltf.meshes:
                for prim in mesh.primitives:
                    if prim.attributes.POSITION is not None:
                        accessor = gltf.accessors[prim.attributes.POSITION]
                        total_verts += accessor.count
        asset.vertex_count = total_verts

        # Skeleton (skins contain joint references)
        if gltf.skins:
            skin = gltf.skins[0]
            joint_indices = skin.joints or []
            asset.has_skeleton = len(joint_indices) > 0
            asset.joint_count = len(joint_indices)
            asset.bone_count = max(0, len(joint_indices) - 1)
            asset.has_skinning = skin.inverseBindMatrices is not None
        else:
            asset.has_skeleton = False
            asset.has_skinning = False

        # Animations
        if gltf.animations:
            asset.has_animation = True
            for anim in gltf.animations:
                # Estimate duration from sampler input accessor
                max_time = 0.0
                for sampler in anim.samplers:
                    input_acc = gltf.accessors[sampler.input]
                    if hasattr(input_acc, 'max') and input_acc.max:
                        max_time = max(max_time, input_acc.max[0])

                clip = AnimationClip(
                    name=anim.name or f"action_{len(asset.animations)}",
                    duration_sec=max_time,
                    fps=24,
                    frame_count=int(max_time * 24),
                    motion_source="original",
                )
                asset.animations.append(clip)
        else:
            asset.has_animation = False

    except ImportError:
        # pygltflib not installed — try trimesh fallback
        _inspect_with_trimesh(file_path, asset)
    except Exception as e:
        print(f"  WARNING: Failed to inspect {file_path.name}: {e}")


def _inspect_fbx(file_path: Path, asset: AssetRig):
    """Inspect an FBX file. Requires trimesh or pyfbx."""
    _inspect_with_trimesh(file_path, asset)


def _inspect_with_trimesh(file_path: Path, asset: AssetRig):
    """Fallback inspection using trimesh."""
    try:
        import trimesh
        scene = trimesh.load(str(file_path), process=False)

        if isinstance(scene, trimesh.Scene):
            total_verts = sum(
                g.vertices.shape[0] for g in scene.geometry.values()
                if hasattr(g, 'vertices')
            )
            asset.vertex_count = total_verts

            # Check for skeleton data in scene graph
            if hasattr(scene, 'graph') and scene.graph.transforms:
                # Heuristic: if there are named transforms with bone-like names
                nodes = list(scene.graph.transforms.node_data.keys()) if hasattr(scene.graph.transforms, 'node_data') else []
                bone_like = [n for n in nodes if any(k in n.lower() for k in ['bone', 'joint', 'armature', 'skeleton'])]
                if bone_like:
                    asset.has_skeleton = True
                    asset.joint_count = len(bone_like)
                    asset.bone_count = max(0, len(bone_like) - 1)

        elif isinstance(scene, trimesh.Trimesh):
            asset.vertex_count = scene.vertices.shape[0]

    except Exception as e:
        print(f"  WARNING: trimesh inspection failed for {file_path.name}: {e}")


def _get_basic_mesh_stats(file_path: Path, asset: AssetRig):
    """Get basic vertex count for mesh-only formats."""
    try:
        import trimesh
        mesh = trimesh.load(str(file_path), process=False)
        if hasattr(mesh, 'vertices'):
            asset.vertex_count = mesh.vertices.shape[0]
    except Exception:
        pass


def _guess_category(file_path: Path) -> str:
    """Guess asset category from path components."""
    parts = str(file_path).lower()
    if any(k in parts for k in ["animal", "creature", "dog", "cat", "horse", "bird", "fish"]):
        return "animal"
    elif any(k in parts for k in ["human", "character", "person", "body"]):
        return "humanoid"
    elif any(k in parts for k in ["vehicle", "car", "robot", "mech", "machine"]):
        return "mechanical"
    elif any(k in parts for k in ["plant", "tree", "flower", "organic"]):
        return "organic"
    else:
        return "unknown"


def build_index(anymate_root: Path, max_assets: int = 0) -> list:
    """Scan the Anymate root directory and build the asset index."""
    print(f"[Index] Scanning: {anymate_root}")

    # Find all mesh files
    mesh_files = []
    for ext in MESH_EXTENSIONS:
        mesh_files.extend(anymate_root.rglob(f"*{ext}"))
        mesh_files.extend(anymate_root.rglob(f"*{ext.upper()}"))

    # Deduplicate
    mesh_files = sorted(set(mesh_files))

    if max_assets > 0:
        mesh_files = mesh_files[:max_assets]

    print(f"[Index] Found {len(mesh_files)} mesh files")

    assets = []
    for i, fpath in enumerate(mesh_files):
        asset_id = f"asset_{i:06d}"
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[Index] Processing {i+1}/{len(mesh_files)}: {fpath.name}")

        asset = scan_asset_file(fpath, asset_id)
        asset.category = _guess_category(fpath)
        assets.append(asset)

    return assets


def write_report(assets: list, output_dir: Path):
    """Write a summary report."""
    total = len(assets)
    has_skeleton = sum(1 for a in assets if a.has_skeleton)
    has_skinning = sum(1 for a in assets if a.has_skinning)
    has_animation = sum(1 for a in assets if a.has_animation)

    categories = Counter(a.category for a in assets)
    joint_counts = [a.joint_count for a in assets if a.has_skeleton]
    vertex_counts = [a.vertex_count for a in assets if a.vertex_count > 0]

    report = f"""# Anymate Index Report

## Summary
- Total assets scanned: {total}
- Has skeleton: {has_skeleton} ({100*has_skeleton/max(total,1):.1f}%)
- Has skinning: {has_skinning} ({100*has_skinning/max(total,1):.1f}%)
- Has animation: {has_animation} ({100*has_animation/max(total,1):.1f}%)
- Needs synthetic motion: {has_skeleton - has_animation}

## Categories
"""
    for cat, count in categories.most_common():
        report += f"- {cat}: {count}\n"

    if joint_counts:
        report += f"""
## Joint Count Distribution
- Min: {min(joint_counts)}
- Max: {max(joint_counts)}
- Mean: {sum(joint_counts)/len(joint_counts):.1f}
- Median: {sorted(joint_counts)[len(joint_counts)//2]}
"""

    if vertex_counts:
        report += f"""
## Vertex Count Distribution
- Min: {min(vertex_counts)}
- Max: {max(vertex_counts)}
- Mean: {sum(vertex_counts)/len(vertex_counts):.0f}
- Median: {sorted(vertex_counts)[len(vertex_counts)//2]}
"""

    report_path = output_dir / "report.md"
    report_path.write_text(report)
    print(f"[Index] Report written to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Build Anymate asset index")
    parser.add_argument("--anymate-root", required=True, help="Root directory of Anymate dataset")
    parser.add_argument("--out", default="outputs/anymate_index", help="Output directory")
    parser.add_argument("--max-assets", type=int, default=0, help="Max assets to index (0 = all)")
    args = parser.parse_args()

    anymate_root = Path(args.anymate_root)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not anymate_root.exists():
        print(f"ERROR: Anymate root not found: {anymate_root}")
        sys.exit(1)

    t0 = time.time()
    assets = build_index(anymate_root, max_assets=args.max_assets)
    elapsed = time.time() - t0

    # Write index
    index_path = output_dir / "assets.jsonl"
    write_jsonl(index_path, [a.to_index_row() for a in assets])
    print(f"[Index] Wrote {len(assets)} assets to: {index_path}")

    # Write report
    write_report(assets, output_dir)

    print(f"[Index] Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
