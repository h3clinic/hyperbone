"""
LabelForge Dataset Pack builder.

Generates a multi-source HyperBone training dataset using all available
LabelForge label sources.

Usage:
  python scripts/build_labelforge_dataset.py ^
    --out outputs/labelforge_v0 ^
    --procedural-count 1000 ^
    --include-procedural-branches ^
    --include-procedural-leaves ^
    --include-procedural-hinges ^
    --include-procedural-rocks ^
    --include-rigged-assets ^
    --include-medial-axis ^
    --include-motion-articulation
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.labels.schema import (
    GraphLabel,
    LabelSource,
    save_graph_labels,
    load_graph_labels,
)
from hyperbone.labels.fusion import FusionConfig, fuse_graph_labels
from hyperbone.labels.review_queue import (
    ReviewQueueConfig,
    generate_review_queue,
    save_review_queue,
)
from hyperbone.labels.procedural.branches import BranchParams, generate_branch_graph
from hyperbone.labels.procedural.leaves import LeafParams, generate_leaf_graph
from hyperbone.labels.procedural.hinge_objects import HingeParams, generate_hinge_graph
from hyperbone.labels.procedural.rocks import RockParams, generate_rock_graph
from hyperbone.labels.from_medial_axis import MedialAxisConfig, extract_medial_axis_graph
from hyperbone.labels.from_motion import (
    MotionArticulationConfig,
    detect_articulations_from_tracks,
)


def generate_procedural_branches(count: int, seed: int = 0) -> list[GraphLabel]:
    """Generate branch/tree graphs with varying complexity."""
    graphs = []
    for i in range(count):
        params = BranchParams(
            max_depth=random.randint(2, 5),
            branch_prob=random.uniform(0.5, 0.9),
            min_length=random.uniform(0.2, 0.5),
            max_length=random.uniform(0.8, 2.0),
            angle_range=(random.uniform(15, 30), random.uniform(40, 70)),
            radius_decay=random.uniform(0.5, 0.8),
            initial_radius=random.uniform(0.05, 0.2),
            seed=seed + i,
        )
        g = generate_branch_graph(params)
        g.sample_id = f"proc_branch_{i:04d}"
        g.metadata["source"] = "procedural_branch"
        g.metadata["generator_seed"] = seed + i
        graphs.append(g)
    return graphs


def generate_procedural_leaves(count: int, seed: int = 1000) -> list[GraphLabel]:
    """Generate leaf vein graphs."""
    graphs = []
    for i in range(count):
        params = LeafParams(
            midrib_length=random.uniform(2.0, 5.0),
            num_secondary_veins=random.randint(3, 8),
            secondary_angle=random.uniform(30, 60),
            secondary_length_ratio=random.uniform(0.4, 0.8),
            tertiary_prob=random.uniform(0.2, 0.6),
            tertiary_length_ratio=random.uniform(0.2, 0.5),
            midrib_segments=random.randint(5, 12),
            seed=seed + i,
        )
        g = generate_leaf_graph(params)
        g.sample_id = f"proc_leaf_{i:04d}"
        g.metadata["source"] = "procedural_leaf"
        g.metadata["generator_seed"] = seed + i
        graphs.append(g)
    return graphs


def generate_procedural_hinges(count: int, seed: int = 2000) -> list[GraphLabel]:
    """Generate articulated hinge/chain graphs."""
    graphs = []
    for i in range(count):
        num_seg = random.randint(2, 6)
        angles = [random.uniform(-1.5, 1.5) for _ in range(num_seg - 1)]
        params = HingeParams(
            num_segments=num_seg,
            segment_length=random.uniform(0.5, 2.0),
            joint_angles=angles,
            seed=seed + i,
        )
        g = generate_hinge_graph(params)
        g.sample_id = f"proc_hinge_{i:04d}"
        g.metadata["source"] = "procedural_hinge"
        g.metadata["generator_seed"] = seed + i
        graphs.append(g)
    return graphs


def generate_procedural_rocks(count: int, seed: int = 3000) -> list[GraphLabel]:
    """Generate rock/ridge graphs."""
    graphs = []
    for i in range(count):
        params = RockParams(
            num_ridges=random.randint(2, 7),
            ridge_nodes_per_ridge=random.randint(2, 5),
            radius=random.uniform(0.5, 2.0),
            seed=seed + i,
        )
        g = generate_rock_graph(params)
        g.sample_id = f"proc_rock_{i:04d}"
        g.metadata["source"] = "procedural_rock"
        g.metadata["generator_seed"] = seed + i
        graphs.append(g)
    return graphs


def generate_synthetic_motion_labels(count: int, seed: int = 4000) -> list[GraphLabel]:
    """Generate motion-derived articulation labels from synthetic sequences."""
    graphs = []
    rng = np.random.default_rng(seed)

    for i in range(count):
        T = rng.integers(15, 40)
        N = rng.integers(60, 150)
        num_parts = rng.integers(2, 5)

        # Generate multi-part synthetic motion
        tracks = np.zeros((T, N, 2))
        points_per_part = N // num_parts
        pivot_x = 100.0

        initial_pos = rng.uniform(10, 190, size=(N, 2))

        for t in range(T):
            tracks[t] = initial_pos.copy()
            for part_idx in range(num_parts):
                start = part_idx * points_per_part
                end = start + points_per_part if part_idx < num_parts - 1 else N
                # Each part rotates differently
                angle = t * rng.uniform(0.02, 0.1) * (part_idx + 1)
                cx = initial_pos[start:end, 0].mean()
                cy = initial_pos[start:end, 1].mean()
                dx = initial_pos[start:end, 0] - cx
                dy = initial_pos[start:end, 1] - cy
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                tracks[t, start:end, 0] = cx + dx * cos_a - dy * sin_a
                tracks[t, start:end, 1] = cy + dx * sin_a + dy * cos_a

        config = MotionArticulationConfig(
            min_frames=5,
            motion_threshold=0.05,
            max_clusters=min(num_parts + 1, 6),
        )
        g = detect_articulations_from_tracks(
            tracks, config=config, sample_id=f"motion_synth_{i:04d}"
        )
        g.metadata["source"] = "motion_articulation"
        g.metadata["synthetic"] = True
        graphs.append(g)

    return graphs


def generate_medial_axis_labels(count: int, seed: int = 5000) -> list[GraphLabel]:
    """Generate medial-axis labels from synthetic masks."""
    graphs = []
    rng = np.random.default_rng(seed)

    for i in range(count):
        H, W = 200, 200
        mask = np.zeros((H, W), dtype=np.uint8)

        shape_type = rng.choice(["bar", "cross", "L", "T", "Y", "blob"])

        if shape_type == "bar":
            y1, y2 = sorted(rng.integers(20, 180, size=2))
            x_center = rng.integers(60, 140)
            thickness = rng.integers(8, 25)
            mask[y1:y2, x_center - thickness // 2:x_center + thickness // 2] = 1

        elif shape_type == "cross":
            cx, cy = 100, 100
            t = rng.integers(8, 20)
            arm = rng.integers(40, 80)
            mask[cy - arm:cy + arm, cx - t // 2:cx + t // 2] = 1
            mask[cy - t // 2:cy + t // 2, cx - arm:cx + arm] = 1

        elif shape_type == "L":
            t = rng.integers(10, 20)
            mask[40:160, 50:50 + t] = 1
            mask[160 - t:160, 50:150] = 1

        elif shape_type == "T":
            t = rng.integers(10, 20)
            mask[40:150, 95:95 + t] = 1
            mask[40:40 + t, 40:160] = 1

        elif shape_type == "Y":
            t = rng.integers(8, 15)
            # Stem
            mask[100:180, 95:95 + t] = 1
            # Left branch
            for j in range(50):
                x = 95 - j
                y = 100 - j
                if 0 <= x < W - t and 0 <= y < H:
                    mask[y:y + t, x:x + t] = 1
            # Right branch
            for j in range(50):
                x = 95 + t + j
                y = 100 - j
                if 0 <= x < W and 0 <= y < H:
                    mask[y:y + t, x:x + t] = 1

        else:  # blob
            cy, cx = rng.integers(60, 140, size=2)
            for y in range(H):
                for x in range(W):
                    r = rng.uniform(30, 60)
                    if (y - cy) ** 2 + (x - cx) ** 2 < r ** 2:
                        mask[y, x] = 1

        config = MedialAxisConfig(min_branch_length=5, simplify_tolerance=2.0)
        g = extract_medial_axis_graph(mask, config, sample_id=f"medial_{shape_type}_{i:04d}")
        g.metadata["source"] = "medial_axis"
        g.metadata["shape_type"] = shape_type
        graphs.append(g)

    return graphs


def extract_rigged_asset_labels(
    asset_dir: Path,
    max_frames: int = 120,
    sample_fps: int = 12,
) -> list[GraphLabel]:
    """Extract labels from any available rigged GLB assets, multi-frame."""
    from hyperbone.labels.from_rig import (
        extract_joints_from_gltf,
        joints_to_graph_label,
        RigExtractionConfig,
    )

    graphs = []
    glb_files = list(asset_dir.rglob("*.glb")) + list(asset_dir.rglob("*.GLB"))

    if not glb_files:
        print(f"  No GLB files found in {asset_dir}")
        return graphs

    # Multiple camera viewpoints for diversity
    import math
    cameras = []
    for elevation in [0.5, 1.0, 1.8]:  # 3 heights
        for angle_deg in range(0, 360, 30):  # 12 azimuth
            angle = math.radians(angle_deg)
            dist = 3.0
            cameras.append({
                "position": np.array([dist * math.sin(angle), elevation, dist * math.cos(angle)]),
                "target": np.array([0.0, 0.5, 0.0]),
            })
    # Also add close-up shots (distance = 2.0)
    for angle_deg in range(0, 360, 60):  # 6 close-ups
        angle = math.radians(angle_deg)
        cameras.append({
            "position": np.array([2.0 * math.sin(angle), 1.0, 2.0 * math.cos(angle)]),
            "target": np.array([0.0, 0.5, 0.0]),
        })

    for glb_path in glb_files[:10]:
        try:
            joints = extract_joints_from_gltf(glb_path)
            if not joints:
                continue

            num_frames = min(max_frames, len(cameras) * 4)
            for frame_idx in range(num_frames):
                cam = cameras[frame_idx % len(cameras)]
                config = RigExtractionConfig(
                    camera_position=cam["position"],
                    camera_target=cam["target"],
                )
                g = joints_to_graph_label(
                    joints=joints,
                    config=config,
                    sample_id=f"rig_{glb_path.stem}_f{frame_idx:04d}",
                )
                g.metadata["source"] = "rigged_asset"
                g.metadata["asset"] = glb_path.name
                g.metadata["camera_idx"] = frame_idx % len(cameras)
                g.metadata["frame"] = frame_idx
                graphs.append(g)

            print(f"  Extracted {len(joints)} joints × {num_frames} frames from {glb_path.name}")
        except Exception as e:
            print(f"  Failed {glb_path.name}: {e}")

    return graphs


def validate_graph(g: GraphLabel) -> bool:
    """Validate a graph has minimum required fields."""
    if not g.nodes:
        return False
    for n in g.nodes:
        if n.node_type is None:
            return False
        if not n.label_sources:
            return False
    for e in g.edges:
        if e.edge_type is None:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Build LabelForge Dataset Pack")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--procedural-count", type=int, default=1000,
                        help="Total procedural samples (split across types)")
    parser.add_argument("--include-procedural-branches", action="store_true")
    parser.add_argument("--include-procedural-leaves", action="store_true")
    parser.add_argument("--include-procedural-hinges", action="store_true")
    parser.add_argument("--include-procedural-rocks", action="store_true")
    parser.add_argument("--include-rigged-assets", action="store_true")
    parser.add_argument("--include-medial-axis", action="store_true")
    parser.add_argument("--include-motion-articulation", action="store_true")
    parser.add_argument("--medial-axis-count", type=int, default=100)
    parser.add_argument("--motion-count", type=int, default=50)
    parser.add_argument("--asset-dir", default="assets/",
                        help="Directory containing rigged assets")
    parser.add_argument("--rigged-assets-dir", default=None,
                        help="Override for rigged assets directory")
    parser.add_argument("--rigged-max-frames", type=int, default=120)
    parser.add_argument("--rigged-sample-fps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    from collections import Counter
    from hyperbone.labels.acceptance import (
        AcceptanceConfig,
        assess_graph_quality,
        classify_graph,
    )

    out_dir = Path(args.out)
    graphs_dir = out_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(exist_ok=True)

    all_graphs: list[GraphLabel] = []
    t0 = time.time()

    # Count how many procedural types are enabled
    proc_types = []
    if args.include_procedural_branches:
        proc_types.append("branches")
    if args.include_procedural_leaves:
        proc_types.append("leaves")
    if args.include_procedural_hinges:
        proc_types.append("hinges")
    if args.include_procedural_rocks:
        proc_types.append("rocks")

    if not proc_types and not args.include_rigged_assets and not args.include_medial_axis and not args.include_motion_articulation:
        proc_types = ["branches", "leaves", "hinges", "rocks"]

    per_type = args.procedural_count // max(len(proc_types), 1)

    # === PROCEDURAL ===
    print("=== Generating procedural labels ===")

    if "branches" in proc_types:
        print(f"  Branches: {per_type}...")
        graphs = generate_procedural_branches(per_type, seed=args.seed)
        valid = [g for g in graphs if validate_graph(g)]
        all_graphs.extend(valid)
        print(f"    Generated {len(valid)} valid branch graphs")

    if "leaves" in proc_types:
        print(f"  Leaves: {per_type}...")
        graphs = generate_procedural_leaves(per_type, seed=args.seed + 1000)
        valid = [g for g in graphs if validate_graph(g)]
        all_graphs.extend(valid)
        print(f"    Generated {len(valid)} valid leaf graphs")

    if "hinges" in proc_types:
        print(f"  Hinges: {per_type}...")
        graphs = generate_procedural_hinges(per_type, seed=args.seed + 2000)
        valid = [g for g in graphs if validate_graph(g)]
        all_graphs.extend(valid)
        print(f"    Generated {len(valid)} valid hinge graphs")

    if "rocks" in proc_types:
        print(f"  Rocks: {per_type}...")
        graphs = generate_procedural_rocks(per_type, seed=args.seed + 3000)
        valid = [g for g in graphs if validate_graph(g)]
        all_graphs.extend(valid)
        print(f"    Generated {len(valid)} valid rock graphs")

    # === RIGGED ASSETS ===
    if args.include_rigged_assets:
        print("\n=== Extracting rigged asset labels ===")
        asset_dir = Path(args.rigged_assets_dir or args.asset_dir)
        if asset_dir.exists():
            rig_graphs = extract_rigged_asset_labels(
                asset_dir,
                max_frames=args.rigged_max_frames,
                sample_fps=args.rigged_sample_fps,
            )
            all_graphs.extend(rig_graphs)
            print(f"  Total rigged: {len(rig_graphs)} graphs")
        else:
            print(f"  Asset dir not found: {asset_dir}")

    # === MEDIAL AXIS ===
    if args.include_medial_axis:
        print(f"\n=== Generating medial-axis labels ({args.medial_axis_count}) ===")
        ma_graphs = generate_medial_axis_labels(args.medial_axis_count, seed=args.seed + 5000)
        valid = [g for g in ma_graphs if validate_graph(g)]
        all_graphs.extend(valid)
        print(f"  Generated {len(valid)} valid medial-axis graphs")

    # === MOTION ARTICULATION ===
    if args.include_motion_articulation:
        print(f"\n=== Generating motion-articulation labels ({args.motion_count}) ===")
        motion_graphs = generate_synthetic_motion_labels(args.motion_count, seed=args.seed + 4000)
        valid = [g for g in motion_graphs if validate_graph(g)]
        all_graphs.extend(valid)
        print(f"  Generated {len(valid)} valid motion-articulation graphs")

    # === SAVE RAW ===
    print(f"\n=== Saving {len(all_graphs)} raw graphs ===")
    save_graph_labels(all_graphs, graphs_dir / "all_graphs.jsonl")

    # === FUSION ===
    print("\n=== Running label fusion ===")
    fusion_config = FusionConfig(
        merge_distance=0.05,
        min_confidence=0.20,
        min_sources_for_uncertain=1,
    )

    fused_graphs = []
    for g in all_graphs:
        fused, report = fuse_graph_labels([g], fusion_config)
        fused.sample_id = g.sample_id
        fused.image_path = g.image_path
        fused.metadata = {**g.metadata, "fused": True}
        fused_graphs.append(fused)

    save_graph_labels(fused_graphs, graphs_dir / "fused_graphs.jsonl")
    print(f"  Fused: {len(fused_graphs)} graphs")

    # === ACCEPTANCE CALIBRATION ===
    print("\n=== Running acceptance calibration ===")
    accept_config = AcceptanceConfig()

    trainable_graphs = []
    review_graphs_list = []
    rejected_graphs_list = []
    rigged_graphs = []
    procedural_graphs_list = []
    medial_graphs = []
    motion_graphs_list = []

    for g in fused_graphs:
        quality = assess_graph_quality(g, accept_config)
        classification = classify_graph(g, quality, accept_config)
        g.metadata["quality"] = quality.to_dict()
        g.metadata["classification"] = classification

        if classification == "trainable":
            trainable_graphs.append(g)
        elif classification == "review":
            review_graphs_list.append(g)
        else:
            rejected_graphs_list.append(g)

        # Source-specific splits
        source = g.metadata.get("source", "")
        if "rigged" in source:
            rigged_graphs.append(g)
        elif "procedural" in source:
            procedural_graphs_list.append(g)
        elif "medial" in source:
            medial_graphs.append(g)
        elif "motion" in source:
            motion_graphs_list.append(g)

    # Save splits
    save_graph_labels(trainable_graphs, graphs_dir / "trainable_graphs.jsonl")
    save_graph_labels(review_graphs_list, graphs_dir / "review_graphs.jsonl")
    save_graph_labels(rejected_graphs_list, graphs_dir / "rejected_graphs.jsonl")
    save_graph_labels(rigged_graphs, graphs_dir / "rigged_animals.jsonl")
    save_graph_labels(procedural_graphs_list, graphs_dir / "procedural_graphs.jsonl")
    save_graph_labels(medial_graphs, graphs_dir / "medial_axis_graphs.jsonl")
    save_graph_labels(motion_graphs_list, graphs_dir / "motion_graphs.jsonl")

    print(f"  Trainable: {len(trainable_graphs)}")
    print(f"  Review:    {len(review_graphs_list)}")
    print(f"  Rejected:  {len(rejected_graphs_list)}")
    print(f"  Rigged:    {len(rigged_graphs)}")

    # === REVIEW QUEUE ===
    print("\n=== Generating review queue ===")
    review_config = ReviewQueueConfig(low_confidence_threshold=0.60)
    all_review_items = []
    for g in fused_graphs:
        items = generate_review_queue(g, review_config)
        all_review_items.extend(items)

    save_review_queue(all_review_items, out_dir / "review_queue.jsonl")
    print(f"  Review items: {len(all_review_items)}")

    # === STATISTICS ===
    total_nodes = sum(g.node_count() for g in fused_graphs)
    total_edges = sum(g.edge_count() for g in fused_graphs)

    node_types = Counter()
    edge_types = Counter()
    sources = Counter()
    for g in fused_graphs:
        for n in g.nodes:
            node_types[n.node_type.value] += 1
        for e in g.edges:
            edge_types[e.edge_type.value] += 1
        src = g.metadata.get("source", "unknown")
        sources[src] += 1

    # Confidence histogram
    all_confs = [n.confidence for g in fused_graphs for n in g.nodes]
    conf_hist = np.histogram(all_confs, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])[0]

    # === MANIFEST ===
    manifest = {
        "version": "labelforge_v0.5",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "total_graphs": len(fused_graphs),
        "trainable_graphs": len(trainable_graphs),
        "review_graphs": len(review_graphs_list),
        "rejected_graphs": len(rejected_graphs_list),
        "rigged_animal_graphs": len(rigged_graphs),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "review_queue_count": len(all_review_items),
        "node_types": dict(node_types),
        "edge_types": dict(edge_types),
        "sources": dict(sources),
        "confidence_histogram": {
            "bins": ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
            "counts": conf_hist.tolist(),
        },
        "paths": {
            "all_graphs": "graphs/all_graphs.jsonl",
            "trainable_graphs": "graphs/trainable_graphs.jsonl",
            "review_graphs": "graphs/review_graphs.jsonl",
            "rejected_graphs": "graphs/rejected_graphs.jsonl",
            "rigged_animals": "graphs/rigged_animals.jsonl",
            "fused_graphs": "graphs/fused_graphs.jsonl",
            "review_queue": "review_queue.jsonl",
            "overlays": "overlays/",
        },
    }
    with open(out_dir / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # === PRINT REPORT ===
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"LabelForge Dataset Pack v0.5 — COMPLETE")
    print(f"{'='*60}")
    print(f"  Output:          {out_dir}")
    print(f"  Total graphs:    {len(fused_graphs)}")
    print(f"  Trainable:       {len(trainable_graphs)}")
    print(f"  Review:          {len(review_graphs_list)}")
    print(f"  Rejected:        {len(rejected_graphs_list)}")
    print(f"  Rigged animal:   {len(rigged_graphs)}")
    print(f"  Total nodes:     {total_nodes}")
    print(f"  Total edges:     {total_edges}")
    print(f"  Review queue:    {len(all_review_items)}")
    print(f"  Time:            {elapsed:.1f}s")
    print(f"\n  Confidence histogram:")
    for label, count in zip(manifest["confidence_histogram"]["bins"],
                             manifest["confidence_histogram"]["counts"]):
        print(f"    {label}: {count}")
    print(f"\n  Sources:")
    for src, cnt in sources.most_common():
        print(f"    {src}: {cnt}")
    print(f"\n  Node types:")
    for nt, cnt in node_types.most_common():
        print(f"    {nt}: {cnt}")
    print(f"\n  Edge types:")
    for et, cnt in edge_types.most_common():
        print(f"    {et}: {cnt}")

    # Verdict
    review_pct = len(review_graphs_list) / max(len(fused_graphs), 1)
    if len(trainable_graphs) < 500:
        verdict = "FAIL"
    elif len(trainable_graphs) < 2000:
        verdict = "PARTIAL"
    elif len(rigged_graphs) >= 300 and review_pct < 0.40:
        verdict = "READY_FOR_SMOKE_TRAINING"
    elif len(rigged_graphs) >= 1000:
        verdict = "READY_FOR_JOINT_TRAINING"
    else:
        verdict = "PARTIAL (need more rigged animals or review too high)"

    print(f"\n  Verdict: {verdict}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
