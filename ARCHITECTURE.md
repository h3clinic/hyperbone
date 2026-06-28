# HyperBone — Architecture Specification

## Problem Statement

Extract stable 2D/3D skeleton graphs from arbitrary videos, then export those graphs into a 3D training format. Human pose tools alone will not solve leaves/rocks/animals.

## Core Insight

The strongest stack is not one "skeleton model." It is **segmentation + point tracking + depth/geometry + graph extraction**, with separate schemas for humans, animals, and generic objects.

**Strategy:**
1. Build a pseudo-label factory first.
2. Train HyperBone to imitate/refine the pseudo-labels.
3. Export every skeleton as a time-varying 3D graph.

Trying to make one model directly discover joints from raw videos will fail or become un-debuggable. A staged CV pipeline is required.

---

## 1. Core Architecture

```
Video frame / clip
   ↓
object detection + segmentation
   ↓
object tracking
   ↓
depth / camera / 3D point estimation
   ↓
2D skeleton graph
   ↓
3D lifted skeleton graph
   ↓
temporal smoothing + constraints
   ↓
export: JSONL / NPZ / BVH / FBX / glTF
```

Use existing models for the first pseudo-labeling pass:
- **SAM 2** — promptable image and video segmentation with streaming memory
- **CoTracker** — tracking dense or selected points through video
- **Depth Anything V2 / Video Depth Anything** — monocular depth and temporal depth consistency
- **VGGT / DUSt3R / MASt3R** — camera parameters, point maps, multi-view geometry

---

## 2. Skeleton Types (Do NOT Use One)

| Object type | Representation | Starting joint/node count |
|---|---|---|
| Human | semantic articulated skeleton / SMPL-style | 17, 24, or 52+ |
| Animal | semantic quadruped/multispecies skeleton | 12–30 |
| Leaf / plant | medial-axis branch graph | 8–64 adaptive |
| Rock | shape skeleton / medial graph, not real joints | 4–32 adaptive |
| Generic object | structural graph + optional rig bones | 8–64 adaptive |

### Per-type tools:
- **Humans:** WHAM, 4DHumans/HMR2, MotionBERT
- **Animals:** Animal3D (26-keypoint/SMAL), SLEAP, DeepLabCut, BANMo, MagicPony
- **Leaves/rocks/shapes:** DeepSkeleton, AMAT, DeepFlux (medial-axis/object skeleton extraction)

---

## 3. Data Schema

Universal graph format per frame:

```json
{
  "video_id": "abc123",
  "frame_idx": 120,
  "timestamp_sec": 4.0,
  "object_id": 3,
  "object_class": "leaf",
  "coord_systems": {
    "image": "pixel_xy",
    "camera": "meters_xyz",
    "world": "meters_xyz",
    "canonical": "object_bbox_normalized"
  },
  "nodes": [
    {
      "id": 0,
      "type": "root|joint|endpoint|branch|ridge|center",
      "semantic": "pelvis|wrist|leaf_tip|null",
      "xy": [410.2, 218.7],
      "camera_xyz": [0.12, -0.03, 2.41],
      "world_xyz": [1.20, 0.45, 0.91],
      "canonical_xyz": [0.00, 0.00, 0.00],
      "radius_norm": 0.08,
      "confidence": 0.91
    }
  ],
  "edges": [
    {
      "parent": 0,
      "child": 1,
      "type": "bone|branch|ridge|medial_axis",
      "length_norm": 0.31,
      "dof": "ball|hinge|rigid|deformable",
      "confidence": 0.87
    }
  ],
  "quality": {
    "mask_iou_track": 0.82,
    "depth_confidence": 0.74,
    "temporal_stability": 0.79,
    "accepted": true
  }
}
```

### Coordinate spaces:
- `image_xy` — pixel coordinates for visual debugging
- `camera_xyz` — 3D coordinates relative to camera
- `canonical_xyz` — normalized object coordinates for training
- `world_xyz` — global scene coordinates if camera estimation is reliable

**Train mostly on canonical coordinates** — videos have different cameras, scales, focal lengths.

---

## 4. Normalized Features

For every object:

```
edge_length_norm = edge_3d_length / object_bbox_3d_diagonal
node_radius_norm = local_medial_radius / object_bbox_3d_diagonal
parent_angle     = angle(parent_edge, child_edge)
branch_order     = depth from root
symmetry_score   = left/right or branch-pair symmetry
motion_norm      = node_velocity / object_bbox_3d_diagonal
stiffness_score  = temporal deformation variance
```

### Humans/animals:
- bone length ratios, joint angles, limb symmetry, contact points, velocity/acceleration, root-relative pose

### Leaves/plants:
- stem-to-tip path length, branching angle, branch radius, leaf midrib curve, vein endpoints

### Rocks:
- ridge graph, medial thickness, convex-lobe structure
- No articulation unless motion proves deformation
- Output structural axes, NOT fake rig joints

---

## 5. Pseudo-Labeling Pipeline

### Stage A — Object Segmentation
SAM 2 / Grounded SAM 2 → mask[t], bbox[t], object_id, track_confidence

### Stage B — Point Tracking
CoTracker → tracked_points[t, n, 2], visibility[t, n], track_confidence[t, n]

### Stage C — Depth / 3D Lift
- Video Depth Anything → long-video depth consistency
- VGGT → camera params, point maps, 3D point tracks
- DUSt3R / MASt3R → reconstruction/matching for multi-view

### Stage D — Skeleton Graph Extraction
- Humans: WHAM / 4DHumans → SMPL / joints / global trajectory
- Animals: SLEAP/DeepLabCut keypoints + Animal3D/SMAL prior
- Arbitrary objects: mask → medial axis → graph pruning → radius estimation → temporal tracking

### Stage E — Graph Cleanup
- Remove tiny branches
- Merge close nodes
- Enforce edge continuity
- Reject unstable depth
- Reject inconsistent masks
- Smooth node trajectories
- Lock rigid structures when deformation is not real

---

## 6. Model Design — HyperBone-v0

Supervised model trained on pseudo-labels.

### Architecture:

```
Inputs:
  RGB frame or short clip
  segmentation mask
  depth map
  optional tracked point features

Backbone:
  ViT / ConvNeXt / Swin encoder

Heads:
  1. skeleton heatmap
  2. endpoint heatmap
  3. junction heatmap
  4. radius/thickness map
  5. node set decoder (DETR-style, variable count)
  6. edge adjacency decoder
  7. 3D offset/depth refinement head
  8. confidence head
```

### Output format:
```
node_tokens: max 64 or 128
node_mask:   active/inactive
edge_matrix: N × N
node_xyz:    N × 3
node_type:   N classes
edge_type:   N × N classes
```

### Losses:
- skeleton heatmap loss
- endpoint/junction focal loss
- node coordinate L1/Huber loss
- edge BCE loss
- Chamfer distance (predicted vs pseudo-label graph)
- temporal smoothness loss
- bone/branch length consistency loss
- 2D reprojection loss
- depth consistency loss

### Version roadmap:

| Version | Input | Output |
|---|---|---|
| v0 | 1 frame + mask + depth | 2D/3D graph |
| v1 | 8–16 frame clip | temporally stable graph sequence |
| v2 | video clip + point tracks | animation-ready skeleton w/ rest pose + transforms |

---

## 7. Export Format

### ML training: `.jsonl` and `.npz`

```
hyperbone_dataset/
  videos/
  frames/
  masks/
  depth/
  tracks/
  graphs_jsonl/
  graphs_npz/
  blender_exports/
```

Each `.npz` stores:
```
nodes_xyz        # [T, N, 3]
nodes_xy         # [T, N, 2]
node_active      # [T, N]
node_type        # [T, N]
edges            # [T, N, N]
edge_type        # [T, N, N]
edge_length_norm # [T, N, N]
radius_norm      # [T, N]
confidence       # [T, N]
camera_K         # [T, 3, 3]
camera_extrinsic # [T, 4, 4]
object_transform # [T, 4, 4]
```

### Animation export:
- Humans → BVH/FBX
- Arbitrary graphs → glTF with custom node metadata or Blender armature bones

---

## 8. Research Stack

| Area | Tool |
|---|---|
| Video segmentation | SAM 2 / Grounded SAM 2 |
| Point tracking | CoTracker / CoTracker3 |
| Monocular depth | Depth Anything V2 / Video Depth Anything |
| 3D reconstruction | VGGT, DUSt3R, MASt3R |
| Human 3D skeletons | WHAM, 4DHumans, MotionBERT |
| Animal pose | SLEAP, DeepLabCut, Animal3D, SMAL |
| Articulated animals/objects | BANMo, MagicPony, DreaMo |
| Generic object skeletons | DeepSkeleton, AMAT, DeepFlux |
| Mesh rigging | RigNet, Anymate, Category-Agnostic Neural Object Rigging |

---

## 9. Build Plan

### Milestone 1 — Pseudo-Label Factory (no training)

Input: video → Output: per-object skeleton JSONL + Blender preview

Steps:
1. Shot splitting
2. Frame sampling
3. Object segmentation (SAM 2)
4. Tracking (CoTracker)
5. Depth (Video Depth Anything / VGGT)
6. Mask skeletonization
7. Graph pruning
8. 3D lifting
9. JSONL export
10. Blender preview export

**Target:** 1,000 videos, 100,000 object tracks, 1M accepted skeleton frames, 1,000 manually reviewed samples

### Milestone 2 — HyperBone-v0
Single-frame graph prediction (RGB + mask + depth → skeleton graph)

### Milestone 3 — HyperBone-v1
Clip-based temporal stabilization (8–16 frames → stable graph sequence)

### Milestone 4 — HyperBone-v2
Rest-pose + animation decomposition (rest skeleton, per-frame transforms, deformation confidence, rigid/articulated/deformable classification)

---

## 10. Differentiator

The winning angle is NOT "we made a skeleton model."

**HyperBone learns structural graphs across biological, mechanical, and natural objects:** humans, animals, leaves, rocks, tools, and arbitrary deformable shapes.

This requires separating:
- **semantic joints** → humans/animals
- **medial skeletons** → leaves/rocks/shapes
- **animation rigs** → 3D meshes/dynamic objects
- **temporal graph motion** → video-derived skeleton animation

**Immediate next step:** the skeleton-label factory. Once that works, training HyperBone is straightforward. Without it, the model has no reliable target and will hallucinate fake bones.

---

## 11. v4.1 Claim Boundary (Topology Extraction)

**Status:** v4.1 is frozen and validated.

### What v4.1 solves (Track A)

HyperBone v4.1 recovers skeleton topology from known joint positions when mesh skinning context is available:
- Input: rigged/skinned 3D asset with GT joint positions and mesh skinning weights.
- Output: undirected edge adjacency (which joints are connected).
- Test F1: 0.888, sel/GT: 1.008, cycles: 0.00.
- Method: skinning-aware hybrid cost MST (density + neural + skinning_cosine + max_shared_weight).
- Dataset: Anymate static rig dataset (2,808 assets, 572 test).
- Ablation: density(0.632) → v3.1 neural(0.756) → skin_cos(0.826) / max_shared_wt(0.806) → v4.1 combined(0.888).

### What v4.1 does NOT solve (Track B)

- Unrigged video or raw object meshes (no skinning signal).
- Direct joint discovery from raw geometry without GT joint positions.
- Full skeleton inference end-to-end from mesh alone.

The ceiling for Track B (geometry-only, no skinning) is approximately F1=0.77. This is a separate, open research problem.

### Locked config

See `configs/topology/hyperbone_v4_1_default.json` for the frozen hyperparameters.

### Production CLI

`scripts/predict_topology_v41.py` — takes a rigged/skinned asset, outputs adjacency JSON and optional overlay PNG.

### Remaining failure mode

Dominant FN source is branch--chain edges (2,489 FN). Any v4.2 work should narrow-target this bucket only, with pass target F1>0.90 and no cycle explosion.
