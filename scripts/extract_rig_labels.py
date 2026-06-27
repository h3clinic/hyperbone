"""
Extract graph labels from rigged 3D assets.

Usage:
  python scripts/extract_rig_labels.py \
    --asset assets/animated_animal_pack/Fox.glb \
    --animation Walk \
    --out outputs/label_forge/fox_walk/ \
    --resolution 512 \
    --frames 120

Outputs:
  outputs/label_forge/fox_walk/labels.jsonl    (GraphLabel per frame)
  outputs/label_forge/fox_walk/frames/         (rendered RGB if available)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.labels.schema import GraphLabel, save_graph_labels
from hyperbone.labels.from_rig import (
    extract_joints_from_gltf,
    joints_to_graph_label,
    RigExtractionConfig,
    project_joints_to_2d,
)


# Default joint name → semantic mapping for common quadrupeds
QUADRUPED_JOINT_MAP = {
    # Fox.glb / common quadruped naming
    "b_Root_00": "root",
    "b_Hip_01": "root",
    "b_Spine01_02": "spine_1",
    "b_Spine02_03": "spine_2",
    "b_Neck_04": "neck",
    "b_Head_05": "head",
    "b_LeftUpperArm_09": "front_left_shoulder",
    "b_LeftForeArm_010": "front_left_elbow",
    "b_LeftHand_011": "front_left_hoof",
    "b_RightUpperArm_06": "front_right_shoulder",
    "b_RightForeArm_07": "front_right_elbow",
    "b_RightHand_08": "front_right_hoof",
    "b_LeftUpLeg_015": "rear_left_hip",
    "b_LeftLeg_016": "rear_left_knee",
    "b_LeftFoot_017": "rear_left_hoof",
    "b_RightUpLeg_018": "rear_right_hip",
    "b_RightLeg_019": "rear_right_knee",
    "b_RightFoot_020": "rear_right_hoof",
    "b_Tail01_012": "tail_base",
    "b_Tail02_013": "tail_tip",
}


def main():
    parser = argparse.ArgumentParser(description="Extract graph labels from rigged 3D assets")
    parser.add_argument("--asset", required=True, help="Path to GLB/FBX file")
    parser.add_argument("--animation", default=None, help="Animation name to sample")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution for projection")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames to extract")
    parser.add_argument("--joint-map", default=None, help="JSON file with joint_name→semantic mapping")
    args = parser.parse_args()

    asset_path = Path(args.asset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not asset_path.exists():
        print(f"Error: asset not found: {asset_path}")
        sys.exit(1)

    # Load joint mapping
    if args.joint_map:
        with open(args.joint_map) as f:
            joint_map = json.load(f)
    else:
        joint_map = QUADRUPED_JOINT_MAP

    # Configuration
    config = RigExtractionConfig(
        joint_name_map=joint_map,
        image_width=args.resolution,
        image_height=args.resolution,
    )

    print(f"Asset: {asset_path}")
    print(f"Animation: {args.animation or 'rest pose'}")
    print(f"Resolution: {args.resolution}x{args.resolution}")

    # Extract joints
    joints = extract_joints_from_gltf(asset_path)
    print(f"Extracted {len(joints)} joints from armature")

    # Map to semantic names
    mapped = 0
    for j in joints:
        if j.name in joint_map:
            mapped += 1
    print(f"Mapped {mapped}/{len(joints)} joints to semantic names")

    # Generate graph labels per frame
    labels: list[GraphLabel] = []

    for frame_idx in range(args.frames):
        # For static rest pose, all frames are the same
        # For animated, would need to evaluate animation at each frame time
        graph = joints_to_graph_label(
            joints=joints,
            config=config,
            sample_id=f"{asset_path.stem}_{args.animation or 'rest'}_{frame_idx:04d}",
            frame_index=frame_idx,
        )
        graph.metadata["asset"] = str(asset_path.name)
        graph.metadata["animation"] = args.animation
        graph.metadata["frame"] = frame_idx
        labels.append(graph)

    # Save
    labels_path = out_dir / "labels.jsonl"
    save_graph_labels(labels, labels_path)

    # Summary
    if labels:
        sample = labels[0]
        node_types = {}
        for n in sample.nodes:
            t = n.node_type.value
            node_types[t] = node_types.get(t, 0) + 1
        print(f"\nSaved {len(labels)} graph labels → {labels_path}")
        print(f"Nodes per frame: {sample.node_count()}")
        print(f"Edges per frame: {sample.edge_count()}")
        print(f"Node types: {node_types}")
        print(f"Sources: {[s for s in sample.nodes[0].label_sources.keys()] if sample.nodes else []}")


if __name__ == "__main__":
    main()
