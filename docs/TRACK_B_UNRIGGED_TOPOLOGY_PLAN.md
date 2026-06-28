# Track B: Unrigged Topology Extraction — Planning Document

**Status:** Planning only. No implementation yet.
**Branch:** `track-b-unrigged-topology`
**Prerequisite:** Track A (v4.1) is frozen and released on `main`.

---

## Context

### Track A: Solved

HyperBone v4.1 solves known-node topology extraction for rigged/skinned Anymate-style assets.

- Test F1: 0.888
- Method: skinning-aware hybrid cost MST
- Input: known joint positions + mesh skinning weights
- Tag: `hyperbone-v4.1-skinned-topology`

The skinning signal (skinning_cosine ROC-AUC=0.946, max_shared_weight ROC-AUC=0.939) is what lifted v4.1 from v3.1's 0.756 to 0.888. This signal does not exist for unrigged assets.

### Track B: Unresolved

Unrigged topology extraction — recovering skeleton graphs from raw geometry, point clouds, or video without skinning weights.

- Current ceiling: F1 ~0.756 (v3.1 neural hybrid, no skinning)
- Geometry-only baseline: F1 ~0.632 (density MST)
- The gap between 0.756 and 0.888 is almost entirely explained by skinning features

---

## Problem Definition

**Input:** 3D mesh or point cloud with known joint positions, but NO skinning weights.

**Output:** Undirected edge adjacency (which joints are connected).

**Why this is harder:** Without skinning weights, the system has no direct signal about which joints share mesh influence. It must infer connectivity from:
- Spatial relationships between joints
- Local mesh geometry around joints
- Learned neural features from the topology scorer

---

## What Has Been Tried and Failed

These approaches are confirmed dead. Do not revisit.

### v3.2: Branch-Aware MST with Branch Insertion
- Attempted to add extra edges at high-degree nodes after MST
- Result: increased FP without meaningful FN reduction
- Root cause: branch insertion without reliable branch detection creates noise

### v3.3: Node Degree Predictor
- Trained a degree classifier to predict node degree (endpoint/chain/branch)
- Result: plateaued far below the branch-F1 requirement
- Root cause: degree prediction from geometry alone lacks the discriminative power needed to guide topology decisions
- The degree predictor cannot reliably distinguish chain nodes from branch nodes in the feature space

---

## What Signals Could Replace Skinning Weights

The core question: what geometry-derived features approximate the edge separability that skinning_cosine and max_shared_weight provide?

### Candidate signals to investigate (not yet validated)

1. **Mesh geodesic distance**
   - Geodesic path length along the mesh surface between joint neighborhoods
   - Hypothesis: connected joints have shorter geodesic paths than Euclidean distance alone suggests
   - Risk: expensive to compute; may not separate branch edges from non-edges

2. **Medial axis / skeleton from mesh**
   - Extract the medial axis of the mesh, then match joints to medial axis nodes
   - Hypothesis: the medial axis provides a topology prior even without skinning
   - Risk: medial axis is noisy and sensitive to mesh quality; may not align with rig topology

3. **Local mesh curvature around joints**
   - Mean/Gaussian curvature in the neighborhood of each joint
   - Connected joints may share curvature characteristics along the limb
   - Risk: curvature is a weak signal compared to direct skinning influence

4. **Mesh cross-section analysis**
   - For each candidate edge, sample the mesh cross-section perpendicular to the edge direction
   - Connected edges should pass through "limb-like" cross-sections
   - Risk: requires clean mesh geometry; fragile on noisy or non-manifold meshes

5. **Learned mesh-to-joint attention**
   - Train a network to predict which mesh vertices are "associated" with each joint, without skinning supervision
   - Use predicted associations as a proxy for skinning influence vectors
   - Risk: this is essentially learning to predict skinning weights — circular without a training signal

6. **Multi-scale density features**
   - Point density at multiple radii around each joint
   - Joint pairs in the same limb may share density profiles
   - Risk: low discriminative power alone; likely needs combination with other features

### Open question

Is there a geometry-only signal with ROC-AUC > 0.85 for edge classification? If not, Track B may require a fundamentally different approach (e.g., learning topology end-to-end rather than scoring candidate edges).

---

## Evaluation Protocol

Same two-stage cached eval as Track A:
- Stage A: cache neural logits + geometry features per sample
- Stage B: sweep optimizer hyperparameters on val, confirm top configs on test
- Composite score: `edge_f1 - 0.10*|sel/gt - 1| - 0.02*cycles`

### Pass criteria for Track B v1

- Minimum: F1 > 0.80 on test (meaningful improvement over v3.1's 0.756)
- Strong: F1 > 0.85
- Must not regress on easy buckets (chain--chain, chain--endpoint)
- Zero cycles (MST constraint)

---

## Recommended Approach

1. **Signal audit first.** Before building anything, run a feature audit similar to `scripts/audit_mesh_topology_features.py`. For each candidate geometry signal, compute ROC-AUC on the edge classification task. Only pursue signals with AUC > 0.80.

2. **One signal at a time.** Add the strongest geometry signal to v3.1's neural hybrid cost function (replacing the skinning terms). Measure delta. Do not combine multiple weak signals hoping they'll compensate — that was the lesson from v3.2/v3.3.

3. **Separate dataset split.** Track B should be evaluated on assets with skinning removed (or a separate unrigged dataset). Do not evaluate Track B by running v4.1 without skinning — that conflates "no signal" with "wrong signal."

4. **No blurring.** Track B results do not retroactively change Track A claims. v4.1 is for rigged/skinned assets. If Track B reaches parity, that is a separate result with a separate claim.

---

## Non-Goals for Track B

- Joint position discovery (Track B assumes known joints, same as Track A)
- End-to-end mesh-to-skeleton prediction
- Video-based skeleton extraction (separate pipeline, different input modality)
- Improving Track A results (v4.1 is frozen)

---

## Signal Audit Results (2026-06-28)

**Conclusion: hand-crafted geometry features are dead for Track B.**

All 13 geometry features tested. Zero exceeded the 0.80 AUC threshold:

| Rank | Feature | ROC-AUC |
|------|---------|---------|
| 1 | euclidean_dist | 0.716 |
| 2 | relative_edge_length | 0.713 |
| 3 | geodesic_dist | 0.618 |
| 4 | tube_density | 0.568 |
| 5 | curvature_diff | 0.567 |
| 6-13 | density/normal/cross-section | 0.52-0.56 |

Reference: skinning_cosine = 0.946, max_shared_weight = 0.939.

Script: `scripts/audit_unrigged_geometry_topology_features.py`
Report: `outputs/models/hyperbone_track_b_geometry_audit/geometry_signal_audit.json`

---

## Track B v1: Geometry-Only Student (Teacher-Student Distillation)

The audit killed feature engineering. The remaining path:

**Use Track A (v4.1) as teacher.** Train a geometry-only student that infers topology from local mesh patches without skinning at inference.

### Architecture

```
Per-joint:
  PointNetLite encodes local mesh patch (xyz + normals) -> joint_mesh_token

Per-edge:
  concat(token_i, token_j, corridor_token, 16-dim geom_feats) -> MLP -> edge score
```

### Inputs (NO skinning)

For each candidate edge (i, j):
- Local mesh patch around joint i (64 pts, xyz + normals)
- Local mesh patch around joint j (64 pts, xyz + normals)
- Corridor mesh patch between i and j (64 pts, xyz + normals)
- 16-dim geometric edge features (distance, density, direction, ranks)

### Targets

- GT edge label (binary)
- Optional: v4.1 teacher cost/score for distillation

### Files

- Model: `hyperbone/models/geometry_edge_student.py`
- Prototype trainer: `scripts/train_geometry_student_prototype.py`

### Pass criteria

- Prototype: ROC-AUC meaningfully above Euclidean baseline (0.716)
- Full-scale: F1 > 0.77 (must beat v3.1 neural-only ceiling)
- Strong pass: F1 >= 0.85

### Decision gate

If the 200-sample prototype cannot beat Euclidean distance AUC by > 0.02, stop before scaling. The representation change is not working.
