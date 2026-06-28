# HyperBone v4.1 Release Notes

**Tag:** `hyperbone-v4.1-skinned-topology`
**Commit:** `d3539ac`
**Date:** 2026-06-28
**Track:** A (rigged/skinned asset topology extraction)

## Summary

v4.1 introduces skinning-aware hybrid cost MST for skeleton topology extraction from rigged 3D assets. Given known joint positions and mesh skinning weights, v4.1 recovers the undirected edge adjacency (which joints are connected) with F1 = 0.888 on the Anymate test set.

## Test Results

| Metric | Value |
|--------|-------|
| Edge F1 | 0.888 |
| Edge Precision | 0.885 |
| Edge Recall | 0.891 |
| Selected/GT ratio | 1.008 |
| Cycles | 0.00 |
| Test samples | 572 |
| Median F1 | 0.909 |
| Std F1 | 0.106 |

## Ablation (Test Set)

| Method | F1 | Delta |
|--------|-----|-------|
| Density MST (baseline) | 0.632 | — |
| v3.1 Neural Hybrid | 0.756 | +0.124 |
| Skinning Cosine Only | 0.826 | +0.194 |
| Max Shared Weight Only | 0.806 | +0.174 |
| **v4.1 Combined** | **0.888** | **+0.256** |

## Method

Skinning-aware hybrid cost MST. The optimizer scores candidate edges using four signals:

1. **Distance** — density-normalized pairwise distance (penalizes long edges)
2. **Neural** — learned edge logits from the topology edge scorer network
3. **Skinning cosine** — cosine similarity of mesh vertex influence vectors between joint pairs
4. **Max shared weight** — maximum skinning weight shared between joint pairs

Kruskal MST selects edges greedily by composite score without creating cycles.

## Frozen Config

```json
{
  "method": "hybrid_skinning_cost_mst",
  "optimizer": "density_normalized_mst",
  "candidate_k": 16,
  "neural_weight": 1.0,
  "skin_cosine_weight": 4.0,
  "shared_weight": 4.0,
  "distance_weight": 1.0,
  "degree_penalty": 0.05,
  "long_edge_penalty": 0.25,
  "mutual_bonus": 0.2,
  "max_degree": 4
}
```

Config file: `configs/topology/hyperbone_v4_1_default.json`

## Known Limitations

1. **Track A only.** Requires mesh skinning weights (bone-indexed). Does not work on unrigged video, raw point clouds, or meshes without skinning data.

2. **Dominant failure mode.** Branch-to-chain false negatives account for 2,489 of 2,913 total FN (85%). Branch-to-branch edges also underperform (F1 = 0.478). These are the only worthwhile targets for v4.2.

3. **Worst-case samples.** 5 samples have F1 < 0.45, typically assets with complex branching topology (50+ joints). The minimum F1 is 0.118.

4. **Track B ceiling.** Without skinning features, the geometry-only ceiling is approximately F1 = 0.77. This is a separate, open research problem.

## Degree-Bucketed Performance

| Edge Type | F1 | Recall | FP | FN |
|-----------|-----|--------|-----|------|
| chain--chain | 0.934 | 0.985 | 1,490 | 178 |
| chain--endpoint | 0.910 | 0.961 | 810 | 207 |
| branch--endpoint | 0.846 | 0.989 | 195 | 6 |
| branch--chain | 0.802 | 0.686 | 193 | **2,489** |
| branch--branch | 0.478 | 0.440 | 20 | 28 |
| endpoint--endpoint | 0.232 | 0.906 | 312 | 5 |

## Files

| File | Purpose |
|------|---------|
| `configs/topology/hyperbone_v4_1_default.json` | Frozen hyperparameters |
| `hyperbone/rigs/skinning_topology_features.py` | Skinning feature extractor |
| `hyperbone/rigs/topology_optimizers.py` | v4.1 optimizer (hybrid_skinning_cost_optimize) |
| `scripts/predict_topology_v41.py` | Production CLI |
| `scripts/eval_skinning_topology_v41.py` | Evaluation pipeline (sweep + test) |
| `scripts/analyze_v41_errors.py` | Error analysis + ablation |
| `tests/test_topology_v41.py` | Regression tests (18/18 passing) |
| `ARCHITECTURE.md` Section 11 | Claim boundary |

## Reproduce

```bash
# Run regression tests (no GPU required for synthetic tests)
python -m pytest tests/test_topology_v41.py -v

# Run full evaluation (requires dataset + GPU)
python scripts/eval_skinning_topology_v41.py

# Run error analysis on cached test results
python scripts/analyze_v41_errors.py

# Predict on a single sample
python scripts/predict_topology_v41.py --sample-idx 42 --overlay

# Predict on full test split
python scripts/predict_topology_v41.py --split test --overlay
```

## Prerequisites

- Anymate dataset: `datasets/anymate/Anymate_test.pt`
- Model checkpoint: `outputs/models/hyperbone_anymate_static_v2.16_topology_full/best_model.pt`
- Split files: `outputs/anymate_local_dev/splits/{train,val,test}.jsonl`

## What This Does NOT Claim

- HyperBone can infer skeletons from raw unrigged objects.
- HyperBone discovers joint positions from geometry alone.
- v4.1 generalizes beyond the Anymate dataset without retraining.
- The skinning features transfer to assets without standard bone-indexed skinning.

## What This Does Claim

HyperBone v4.1 recovers skeleton topology from known joint positions when mesh skinning context is available. It is a validated, reproducible result on the Anymate static rig dataset for rigged/skinned 3D assets.
