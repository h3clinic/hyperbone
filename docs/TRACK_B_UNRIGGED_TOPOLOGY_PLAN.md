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

### Prototype result (2026-06-28)

**Decision gate: PASSED.**

| Metric | Euclidean | Student | Delta |
|--------|-----------|---------|-------|
| ROC-AUC | 0.716 | 0.905 | +0.190 |
| PR-AUC | 0.269 | 0.621 | +0.351 |

Best epoch: 17/30. Reference: skinning_cosine ROC=0.946.

---

## Track B v1 Full Training (2026-06-28)

Pipeline: train on train, select on val, test once.

### Files

- Cache builder: `scripts/build_geometry_student_cache.py`
- Full trainer: `scripts/train_geometry_student_full.py`
- Topology eval: `scripts/eval_geometry_student_topology.py`

### Eval methods

- distance_only: Kruskal MST on -distance
- density: midpoint density - distance
- student_only: Kruskal MST on student edge scores
- student_dist_hybrid: student score - weight * relative_distance

### Pass criteria

- Minimum: topology F1 > 0.77 (beats v3.1 node-only)
- Strong: F1 >= 0.82
- Research pass: F1 >= 0.85

### v1 Result (2026-06-28): BELOW THRESHOLD

Training: 50 epochs on 4464 train samples (1.6M edges), best epoch 30.

| Metric | Baseline | Student | Delta |
|--------|----------|---------|-------|
| Val ROC-AUC | 0.727 | 0.873 | +0.147 |
| Val PR-AUC | 0.288 | 0.561 | +0.273 |

Topology (test, 572 samples, val-tuned dist_weight=10.0):

| Method | F1 |
|--------|------|
| distance_only | 0.628 |
| density | 0.596 |
| student_only | 0.638 |
| student_dist_hybrid | 0.678 |

**Verdict: BELOW THRESHOLD (F1=0.678, needed >0.77)**

The edge classifier learned real signal (ROC=0.873), but BCE edge training does not translate to topology recovery. The MST decoder needs near-perfect ranking of the top N-1 edges, not just good average ranking. The training objective is wrong for topology.

---

## Track B v1.1: Topology-Aware Teacher Distillation (2026-06-28)

v1 proved geometry signal exists. v1.1 fixes the objective mismatch.

### Changes from v1

1. **Training targets**: v4.1 teacher edge selection + scores (skinning used as supervision only)
2. **Loss function**: per-sample ranking loss + teacher distillation, not independent BCE
3. **Checkpoint selection**: val topology F1, not val PR-AUC
4. **Student input**: still geometry-only (no skinning)

### Loss components

- A: GT BCE (edge label supervision)
- B: Teacher BCE (predict v4.1-selected edges)
- C: Pairwise ranking (score(pos) > score(hard_neg) + margin)
- D: Listwise/top-k (top E predicted edges match GT/teacher set)

### Files

- Teacher cache augmenter: `scripts/augment_cache_with_teacher.py`
- v1.1 trainer: `scripts/train_geometry_student_v11.py`
- Topology eval: `scripts/eval_geometry_student_topology.py` (reused)

### Pass criteria (same as v1)

- Minimum: F1 > 0.77
- Strong: F1 >= 0.82
- Research pass: F1 >= 0.85

### v1.1 Result (2026-06-28): BELOW THRESHOLD

Best val topology F1: 0.670 (epoch 25/50).

| Method | F1 |
|--------|------|
| distance_only | 0.628 |
| density | 0.596 |
| student_only | 0.649 |
| student_dist_hybrid | 0.662 |

**Verdict: BELOW THRESHOLD (F1=0.662, needed >0.77)**

Teacher distillation + ranking loss on independent edge scoring made student_only slightly better but made hybrid decode worse. The score distribution became less compatible with MST. The architecture — not the loss — is the bottleneck.

---

## Track B v2: Edge-Graph Message Passing Student (2026-06-28)

v1/v1.1 proved the bottleneck is independent edge scoring, not the loss function. v2 replaces per-edge MLP with edge-graph message passing.

### Architecture

```
Per-edge initial embedding:
  PointNetLite(patch_i) || PointNetLite(patch_j) || CorridorNet(corridor) || geom_feats
  → Linear projection → edge_dim

Line graph construction:
  Two edge-nodes connected if they share a joint endpoint

Message passing (3 rounds):
  Pre-norm → max-aggregation → gated residual update
  Max-agg preserves discriminative signal (mean-agg collapsed it)

Score head:
  MLP(edge_dim → 1) → per-edge logit with graph context
```

### Key design decisions

1. **Max aggregation** instead of mean: mean over ~30 neighbors washed out signal
2. **Gated residual**: learned gate controls how much context to mix in
3. **Pre-norm**: LayerNorm before message passing, not after residual
4. **Gradient accumulation (4 steps)**: per-sample SGD was too noisy for 305K params
5. **Warmup (3 epochs)**: prevents early divergence with lr=3e-3

### Files

- Model: `hyperbone/models/geometry_edge_graph_student.py`
- One-batch overfit: `scripts/overfit_edge_graph_one_batch.py`
- Full trainer: `scripts/train_edge_graph_student_v2.py`

### Training

- 305,092 params
- 50 epochs, lr=3e-3 with warmup + cosine annealing
- BCE + pairwise ranking loss (weight=2.0, margin=1.0)
- Gradient accumulation (4 steps)
- Best epoch: 46, best val topology F1: 0.795

### v2 Result (2026-06-28): MINIMUM PASS

| Method | F1 |
|--------|------|
| distance_only | 0.628 |
| density | 0.577 |
| student_only | **0.802** |
| student_dist_hybrid | 0.788 |

Degree-bucketed F1 (student_only):

| Bucket | N | F1 |
|--------|---|------|
| tiny(<=10) | 5 | 0.775 |
| small(11-25) | 79 | 0.790 |
| medium(26-50) | 253 | 0.841 |
| large(>50) | 235 | 0.766 |

**Verdict: MINIMUM PASS (F1=0.802, needed >0.77)**

- student_only (0.802) beats student_dist_hybrid (0.788): the model's scores are strong enough that distance penalty hurts
- Surpassed v3.1 node-only (~0.756) by +0.046
- Surpassed v1 BCE (0.678) by +0.124
- Surpassed v1.1 distilled (0.662) by +0.140
- Zero cycles, 1 component per sample (MST constraint)
- Not yet strong pass (0.82) or research pass (0.85)

---

## Track B v2.1: Edge-Graph + Teacher Distillation (2026-06-29)

v2 proved the edge-graph architecture works. v1.1 showed teacher distillation
failed on the *independent* scorer — but the failure was the architecture, not
the teacher. v2.1 retries teacher distillation on the *edge-graph* model.

### Changes from v2

Same architecture, warm-started from the v2 checkpoint. Added two teacher
losses on top of GT BCE + hard-negative ranking:

- **Teacher BCE**: predict v4.1-teacher-selected edges (weight 0.5)
- **Score distillation**: KL(teacher || student) on softmax score distributions,
  temperature 2.0 (weight 1.0)

Teacher labels/scores come from `cache_*_v11.pt` (v4.1 skinning-aware teacher).
Skinning is used only as a training target — never as student input.

### Files

- Trainer: `scripts/train_edge_graph_student_v21.py`
- Eval: `scripts/eval_edge_graph_student_topology.py` (reused)

### Training

- Warm start from `best_student_v2.pt`, lr=1e-3, 60 epochs, cosine + 3ep warmup
- Best epoch: 49, best val topology F1: 0.818

### v2.1 Result (2026-06-29): improved, still MINIMUM PASS

| Method | F1 |
|--------|------|
| distance_only | 0.628 |
| density | 0.577 |
| student_only | **0.818** |
| student_dist_hybrid | 0.803 |

Degree-bucketed F1 (student_only):

| Bucket | N | F1 | vs v2 |
|--------|---|------|------|
| tiny(<=10) | 5 | 0.772 | -0.003 |
| small(11-25) | 79 | 0.797 | +0.007 |
| medium(26-50) | 253 | 0.848 | +0.007 |
| large(>50) | 235 | 0.793 | +0.027 |

**Verdict: MINIMUM PASS (F1=0.818, needed >0.77; strong pass 0.82 not reached)**

- Teacher distillation on the edge-graph model added +0.016 test F1 over v2
- Val F1 (0.818) == test F1 (0.818): no overfitting
- Biggest gain on large skeletons (>50 joints): +0.027 — the teacher helps most
  where independent geometry is hardest
- Still 0.002 below strong pass. Next: cheap probability-space ensemble (v2.2)

---

## Track B v2.2: Score Ensemble of v2 + v2.1 (2026-06-29)

Cheapest shot at strong pass. Ensemble per-edge logits:
`ensemble = alpha * logits_v2 + (1-alpha) * logits_v21`, decode student_only
MST. Alpha swept 0.0..1.0 (step 0.05) on VALIDATION only, then test once at
the selected alpha.

### Files

- Eval: `scripts/eval_edge_graph_ensemble.py`

### Result (2026-06-29): did NOT reach strong pass

| Quantity | Value |
|----------|-------|
| Best alpha (val-selected) | 0.40 |
| Val F1 @ best alpha | 0.8202 |
| **Test F1 @ best alpha** | **0.8198** |
| Ensemble delta vs best single model | +0.0020 |
| test precision / recall | 0.818 / 0.822 |
| test cycles / components | 0.0 / 1.0 |

Harness self-check: endpoint alphas reproduce committed per-model test F1s
(alpha=1.0 → 0.8024 ≈ v2; alpha=0.0 → 0.8178 ≈ v2.1).

**Verdict: MINIMUM PASS (test F1=0.8198). Strong pass (0.820) NOT reached — 0.0002 short.**

- Val crossed 0.820 (0.8202) but TEST — the criterion that counts — did not
- Only +0.002 over pure v2.1: v2 and v2.1 are too correlated (v2.1 warm-started
  from v2) for ensembling to add much
- Honest call: this is not a strong pass. Proceed to v2.3 (from-scratch teacher)

---

## Track B v2.3: From-Scratch Teacher Distillation (2026-06-29) — RESEARCH PASS

The v2.1 insight, pushed: teacher distillation works on the edge-graph model,
but warm-starting from v2 anchored it in v2's weaker basin. v2.3 trains the SAME
architecture and losses from RANDOM init, with teacher losses active from epoch 1,
higher lr (3e-3), longer schedule (100ep cap), early stopping (patience 20).

### Files

- Trainer: `scripts/train_edge_graph_student_v23.py`
- Test eval: `scripts/eval_edge_graph_student_topology.py`
- Error analysis: `scripts/analyze_edge_graph_errors.py`
- N-way ensemble (built, unused — v2.3B passed outright): `scripts/eval_edge_graph_ensemble_nway.py`
- Frozen config: `configs/topology/hyperbone_track_b_v23_research.json`
- Report: `outputs/models/hyperbone_track_b_student_v23B/report.md`

### Variant sweep (selected by VAL topology F1 only)

| Variant | rank / teacher_bce / distill | Val F1 | Epoch |
|---------|------------------------------|--------|-------|
| A | 2.0 / 0.5 / 0.25 | 0.8531 | 93 |
| **B (selected)** | **2.0 / 1.0 / 0.25** | **0.8637** | 85 |
| C | 3.0 / 0.5 / 0.5 | 0.8409 | 86 |

Higher teacher-BCE weight (1.0) won. Only variant B was tested (test-once).

### v2.3B Result (2026-06-29): RESEARCH PASS

| Method | Test F1 |
|--------|---------|
| distance_only | 0.628 |
| density | 0.577 |
| student_dist_hybrid | 0.842 |
| **student_only** | **0.8569** |

Degree-bucketed test F1 (student_only), by joint count:

| Bucket | N | F1 | vs v2.1 |
|--------|---|------|------|
| tiny(<=10) | 5 | 0.772 | 0.000 |
| small(11-25) | 79 | 0.818 | +0.021 |
| medium(26-50) | 253 | 0.876 | +0.028 |
| large(>50) | 235 | 0.851 | +0.058 |

By GT max-degree: deg≤3 0.751 (n=19), deg4-5 0.888 (n=313), deg6-8 0.845 (n=214),
deg≥9 0.659 (n=26). High-degree hubs remain the hardest.

**Verdict: RESEARCH PASS (test F1=0.8569 ≥ 0.85). Also clears strong pass (0.82).**

- Val 0.8637 → test 0.8569: normal ~0.007 generalization gap (metric not inflated)
- From-scratch teacher (+0.055 over v2) >> warm-start teacher v2.1 (+0.016 over v2):
  the teacher signal is most valuable when it shapes training from epoch 1
- Biggest win on the historically hardest bucket, large skeletons: +0.058 vs v2.1
- Residual errors: high-degree hubs and long true edges (FN skew long 0.364 vs
  FP short 0.208) → next architecture is joint-node ⨯ edge-node bipartite graph

### Claim audit (4 independent read-only auditors, all CLEAN / high confidence)

1. **Metric correctness**: trainer val-F1 == committed test-eval student_only F1
   (identical scoring, MST tie-break, edge_prf, averaging). No inflation.
2. **Skinning leakage**: no skinning field reaches student input; teacher labels
   used only in loss, never passed to `model()`.
3. **Selection discipline**: all selection on val; test evaluated once; reference
   test numbers not used to select.
4. **Split integrity**: SHA256-deterministic asset split, 0 overlaps across
   train(4464)/val(562)/test(572).

### Bottom line

Track B (unrigged, geometry-only topology) reached a research pass: **test F1 =
0.8569**, geometry-only, no skinning at inference. Track A (rigged, v4.1 F1=0.888)
remains a separate, skinning-based result — the two are not to be conflated.
