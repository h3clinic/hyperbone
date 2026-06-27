"""
Inspect a single Anymate asset — print joint count, bones, skinning, animations, mesh stats.

Usage:
    python scripts/inspect_anymate_asset.py --asset-id asset_000001 --index outputs/anymate_index/assets.jsonl
    python scripts/inspect_anymate_asset.py --file path/to/asset.glb
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.rigs.schema import AssetRig, read_jsonl


def inspect_from_index(asset_id: str, index_path: Path):
    """Look up asset in index and print details."""
    records = read_jsonl(index_path)
    match = None
    for rec in records:
        if rec.get("asset_id") == asset_id:
            match = rec
            break

    if match is None:
        print(f"ERROR: Asset '{asset_id}' not found in {index_path}")
        sys.exit(1)

    asset = AssetRig.from_dict(match)
    print_asset_info(asset)

    # Also try to load and inspect the actual file
    mesh_path = Path(match.get("mesh_path", ""))
    if mesh_path.exists():
        inspect_file_detailed(mesh_path)


def inspect_file_detailed(file_path: Path):
    """Deep inspection of a 3D asset file."""
    print(f"\n{'='*50}")
    print(f"Detailed inspection: {file_path.name}")
    print(f"{'='*50}")

    ext = file_path.suffix.lower()

    if ext in (".glb", ".gltf"):
        _inspect_gltf_detailed(file_path)
    elif ext == ".fbx":
        _inspect_fbx_detailed(file_path)
    else:
        print(f"  Format: {ext} (mesh-only, no rig data)")
        _inspect_trimesh(file_path)


def _inspect_gltf_detailed(file_path: Path):
    """Detailed glTF inspection."""
    try:
        import pygltflib
        gltf = pygltflib.GLTF2().load(str(file_path))

        print(f"\n  Meshes: {len(gltf.meshes) if gltf.meshes else 0}")
        if gltf.meshes:
            for i, mesh in enumerate(gltf.meshes):
                prim_verts = 0
                for prim in mesh.primitives:
                    if prim.attributes.POSITION is not None:
                        acc = gltf.accessors[prim.attributes.POSITION]
                        prim_verts += acc.count
                print(f"    [{i}] {mesh.name or 'unnamed'}: {prim_verts} vertices, {len(mesh.primitives)} primitives")

        print(f"\n  Skins: {len(gltf.skins) if gltf.skins else 0}")
        if gltf.skins:
            for i, skin in enumerate(gltf.skins):
                joints = skin.joints or []
                print(f"    [{i}] {skin.name or 'unnamed'}: {len(joints)} joints")
                print(f"        Has inverse bind matrices: {skin.inverseBindMatrices is not None}")

                # Print joint names
                if joints and gltf.nodes:
                    print(f"        Joint hierarchy:")
                    for ji, node_idx in enumerate(joints[:20]):
                        node = gltf.nodes[node_idx]
                        print(f"          [{ji}] {node.name or f'node_{node_idx}'}")
                    if len(joints) > 20:
                        print(f"          ... and {len(joints)-20} more")

        print(f"\n  Animations: {len(gltf.animations) if gltf.animations else 0}")
        if gltf.animations:
            for i, anim in enumerate(gltf.animations):
                # Duration
                max_time = 0
                for sampler in anim.samplers:
                    input_acc = gltf.accessors[sampler.input]
                    if hasattr(input_acc, 'max') and input_acc.max:
                        max_time = max(max_time, input_acc.max[0])
                print(f"    [{i}] {anim.name or 'unnamed'}: {max_time:.2f}s, {len(anim.channels)} channels")

        # Warnings
        print(f"\n  Warnings:")
        warnings = []
        if not gltf.skins:
            warnings.append("No skeleton/rig found")
        if gltf.skins and not gltf.animations:
            warnings.append("Has skeleton but NO animations — needs synthetic motion")
        if gltf.skins and gltf.skins[0].joints and len(gltf.skins[0].joints) > 128:
            warnings.append(f"High joint count ({len(gltf.skins[0].joints)}) — may need simplification")
        if not warnings:
            warnings.append("None — asset looks good")
        for w in warnings:
            print(f"    ⚠ {w}")

    except ImportError:
        print("  pygltflib not installed. Install: pip install pygltflib")
        _inspect_trimesh(file_path)
    except Exception as e:
        print(f"  ERROR: {e}")


def _inspect_fbx_detailed(file_path: Path):
    """FBX inspection via trimesh."""
    print("  Format: FBX")
    _inspect_trimesh(file_path)


def _inspect_trimesh(file_path: Path):
    """Inspect with trimesh."""
    try:
        import trimesh
        scene = trimesh.load(str(file_path), process=False)

        if isinstance(scene, trimesh.Scene):
            print(f"\n  Scene with {len(scene.geometry)} geometry objects")
            total_verts = 0
            total_faces = 0
            for name, geom in scene.geometry.items():
                if hasattr(geom, 'vertices'):
                    v = geom.vertices.shape[0]
                    f = geom.faces.shape[0] if hasattr(geom, 'faces') else 0
                    total_verts += v
                    total_faces += f
                    if v > 1000:
                        print(f"    {name}: {v} verts, {f} faces")
            print(f"  Total: {total_verts} vertices, {total_faces} faces")
        elif isinstance(scene, trimesh.Trimesh):
            print(f"  Single mesh: {scene.vertices.shape[0]} vertices, {scene.faces.shape[0]} faces")
            print(f"  Bounds: {scene.bounds}")
    except Exception as e:
        print(f"  trimesh inspection failed: {e}")


def print_asset_info(asset: AssetRig):
    """Print formatted asset info."""
    print(f"\n{'='*50}")
    print(f"Asset: {asset.asset_id}")
    print(f"{'='*50}")
    print(f"  Mesh path:      {asset.mesh_path}")
    print(f"  Vertices:       {asset.vertex_count}")
    print(f"  Category:       {asset.category}")
    print(f"  Has skeleton:   {asset.has_skeleton}")
    print(f"  Joint count:    {asset.joint_count}")
    print(f"  Bone count:     {asset.bone_count}")
    print(f"  Has skinning:   {asset.has_skinning}")
    print(f"  Has animation:  {asset.has_animation}")
    if asset.animations:
        print(f"  Animations:")
        for anim in asset.animations:
            print(f"    - {anim.name}: {anim.duration_sec:.2f}s ({anim.frame_count} frames, {anim.motion_source})")


def main():
    parser = argparse.ArgumentParser(description="Inspect a single Anymate asset")
    parser.add_argument("--asset-id", help="Asset ID to look up in index")
    parser.add_argument("--index", help="Path to assets.jsonl index file")
    parser.add_argument("--file", help="Direct path to a 3D asset file")
    args = parser.parse_args()

    if args.file:
        fpath = Path(args.file)
        if not fpath.exists():
            print(f"ERROR: File not found: {fpath}")
            sys.exit(1)
        inspect_file_detailed(fpath)
    elif args.asset_id and args.index:
        inspect_from_index(args.asset_id, Path(args.index))
    else:
        parser.print_help()
        print("\nProvide either --file or both --asset-id and --index")
        sys.exit(1)


if __name__ == "__main__":
    main()
