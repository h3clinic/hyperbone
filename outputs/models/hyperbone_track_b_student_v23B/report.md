# Track B (Unrigged, Geometry-Only Topology) — Research Pass

**Model:** v2.3 variant B — from-scratch teacher-distilled edge-graph student
**Date:** 2026-06-29
**Status:** RESEARCH PASS (test topology F1 = 0.8569 ≥ 0.85)

## Claim boundary

This is the **unrigged, geometry-only** topology result. The student receives
**no skinning** at inference — only mesh geometry (point-cloud patches + normals)
and joint positions. Skinning is used **only** as the v4.1 teacher's training
target. This result must **not** be conflated with the rigged Track A result
(v4.1 F1 = 0.888, which uses skinning).

Audited clean (high confidence) on all four claim-invalidating dimensions:
metric correctness, skinning leakage, selection discipline, split integrity.

## Headline

| Method | Test F1 | Prec | Rec |
|--------|---------|------|-----|
| distance_only (baseline) | 0.6282 | 0.626 | 0.631 |
| density | 0.5769 | 0.575 | 0.579 |
| student_dist_hybrid | 0.8420 | 0.840 | 0.845 |
| **student_only** | **0.8569** | **0.855** | **0.860** |

- Zero cycles, 1 component per sample (valid trees via Kruskal MST).
- student_only beats the distance-hybrid decode: the learned geometry score is
  strong enough on its own that adding a distance penalty hurts.

## Progression (all geometry-only, test topology F1)

| Version | Approach | Test F1 |
|---------|----------|---------|
| distance_only | non-learned baseline | 0.628 |
| v3.1 | node-only hybrid | ~0.756 |
| v1 | independent edge BCE | 0.678 |
| v1.1 | independent edge + warm teacher distill | 0.662 |
| v2 | edge-graph message passing | 0.802 |
| v2.1 | edge-graph + warm-start teacher distill | 0.818 |
| v2.2 | ensemble v2 + v2.1 | 0.8198 |
| **v2.3B** | **edge-graph + from-scratch teacher distill** | **0.8569** |

The decisive move: applying teacher distillation **from random init** instead of
warm-starting from v2. Warm-start anchored the model in v2's weaker basin
(+0.016); from-scratch with teacher losses active from epoch 1 found a far better
minimum (+0.055 over v2). Original from-scratch v2 *without* teacher only reached
0.795, so the teacher signal contributes ~+0.06 when it shapes training from the
start.

## Variant selection (by validation F1 only, then test once)

| Variant | rank / teacher_bce / distill | Val F1 | Epoch |
|---------|------------------------------|--------|-------|
| A | 2.0 / 0.5 / 0.25 | 0.8531 | 93 |
| **B (selected)** | **2.0 / 1.0 / 0.25** | **0.8637** | 85 |
| C | 3.0 / 0.5 / 0.5 | 0.8409 | 86 |

Higher teacher-BCE weight (1.0) won. Only variant B was evaluated on test
(test-once discipline). Val 0.8637 → test 0.8569 is a normal ~0.007
generalization gap.

## Degree-bucketed test F1 (student_only)

By joint count:

| Bucket | N | F1 |
|--------|---|-----|
| tiny (≤10) | 5 | 0.772 |
| small (11–25) | 79 | 0.818 |
| medium (26–50) | 253 | 0.876 |
| large (>50) | 235 | 0.851 |

By GT max-degree (hub-heaviness):

| Bucket | N | F1 |
|--------|---|-----|
| deg ≤3 | 19 | 0.751 |
| deg 4–5 | 313 | 0.888 |
| deg 6–8 | 214 | 0.845 |
| deg ≥9 | 26 | 0.659 |

The largest historical weak spot — large skeletons (>50 joints) — improved from
0.766 (v2) → 0.793 (v2.1) → **0.851** (v2.3B).

## Residual error modes

- **High-degree hubs** (GT max-degree ≥9, n=26) remain hardest at F1 0.659. MST
  struggles to attach many bones to a single hub joint from geometry alone.
- **Long true edges** are missed: false negatives skew to longer candidate edges
  (mean length percentile 0.364) while false positives skew short (0.208) — the
  model misses long bones and over-connects near neighbors.
- Worst samples are hub-heavy (max-degree 10–14).

These point at the next architecture (if pushed further): explicit joint-degree
modeling / a joint-node ⨯ edge-node bipartite graph so hub structure is
represented directly rather than inferred per-edge.

## Reproduce

```
# Train (variant B config)
python scripts/train_edge_graph_student_v23.py \
  --train-cache outputs/models/hyperbone_track_b_student/cache_train_v11.pt \
  --val-cache   outputs/models/hyperbone_track_b_student/cache_val_v11.pt \
  --epochs 100 --lr 3e-3 --warmup-epochs 5 --patience 20 \
  --rank-weight 2.0 --teacher-bce-weight 1.0 --distill-weight 0.25 \
  --tag v23B --out outputs/models/hyperbone_track_b_student_v23B

# Test once (val-selected winner)
python scripts/eval_edge_graph_student_topology.py \
  --test-cache outputs/models/hyperbone_track_b_student/cache_test_v11.pt \
  --ckpt outputs/models/hyperbone_track_b_student_v23B/best_student_v23.pt \
  --edge-dim 128 --n-mp-rounds 3
```

Frozen config: `configs/topology/hyperbone_track_b_v23_research.json`
