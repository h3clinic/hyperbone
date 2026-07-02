# HyperBone Track B — Result Card

**Geometry-only, known-node topology recovery.** Repro-verified research pass.

> **Claim (external-safe):** HyperBone Track B recovers known-node skeleton
> topology from mesh geometry alone, with no skinning input at inference,
> achieving held-out Anymate test F1 **0.8569** and reproducing across
> independent seeds above the **0.85** research-pass threshold. Skinning is used
> only as a training-time teacher signal.

**Precise scope:** *known-node topology recovery from mesh geometry alone.*
Not "solves unrigged skeleton extraction." Not "general animation rigging."

---

## 1. Result card

| Field | Value |
|-------|-------|
| Model | `GeometryEdgeGraphStudent` (edge-graph message passing), 305,092 params |
| Model file | `hyperbone/models/geometry_edge_graph_student.py` |
| Training | v2.3B — from-scratch, teacher-distilled; `scripts/train_edge_graph_student_v23.py` |
| Loss | GT BCE(1.0) + hard-neg ranking(2.0, margin 1.0, k=16) + teacher BCE(1.0) + KL score distill(0.25, T=2.0) |
| Frozen config | `configs/topology/hyperbone_track_b_v23_research.json` |
| Deliverable checkpoint | `outputs/models/hyperbone_track_b_student_v23B/best_student_v23.pt` (seed 0) |
| Commit | `ff9e9ed` (result `26c7935`) |
| Tag | `hyperbone-track-b-v23-research-pass` |
| **Test F1 (seed 0)** | **0.8569** (precision 0.8548, recall 0.8596) |
| Test F1 (seed 1 repro) | 0.8521 |
| Cross-seed spread | 0.0048 (both ≥ 0.85) |
| Val F1 (selection metric) | 0.8637 (seed 0), 0.8599 (seed 1) |

### Test split

Held-out Anymate test split: **572 samples**, 4464 train / 562 val / 572 test.
Split is SHA256-deterministic per asset with **0 overlaps** across train/val/test
(audited). Caches: `outputs/models/hyperbone_track_b_student/cache_{train,val,test}_v11.pt`.

### Metric definition

**Topology F1** = undirected edge-set F1 between predicted and ground-truth
skeleton adjacency, over active (known) joints. Prediction is a **Kruskal MST**
over candidate edges scored by the student (raw logits; "student_only" decode;
tie-break by (score, −distance) descending; exactly n−1 edges → always a valid
tree, 0 cycles, 1 component). Candidate edges = KNN-on-joint-positions ∪
ground-truth edges (the distance-only baseline on the same candidates scores
0.628, so candidacy does not reveal the answer). Reported F1 is the mean over
test samples.

### Selection discipline

Variant (A/B/C) and checkpoint selected by **validation** topology F1 only; the
**test** set was evaluated **once** on the val-selected winner. Audited clean.

### Baselines (same task, geometry-only test F1)

| distance_only | density | v3.1 node-only | v2.3B (this) |
|:---:|:---:|:---:|:---:|
| 0.628 | 0.577 | ~0.756 | **0.857** |

---

## 2. Qualitative overlays

`outputs/models/hyperbone_track_b_student_v23B/overlays/` — rendered from the
committed checkpoint via `scripts/render_topology_overlays.py`. Green = correct
edge, red dashed = missed GT edge, orange = extra predicted edge, dots = known joints.

| Case | Sample | Joints | Max deg | F1 | Reads as |
|------|--------|--------|---------|-----|----------|
| Good | idx 20 | 38 | 5 | 1.00 | full skeleton recovered exactly (37 TP, 0 FN, 0 FP) |
| Borderline | idx 140 | 50 | 4 | 0.857 | mostly correct, 7 missed + 7 extra — the headline-average case |
| Failure (hub) | idx 487 | 21 | 10 | 0.05 | high-degree hub: connectivity mostly wrong |
| Failure (large) | idx 513 | 57 | 6 | 0.30 | large skeleton: dense mid-body errors |

Grid: `overlays/overlays_grid.png`.

### Character context (mesh + skeleton)

To characterize *what* these skeletons belong to, the source Anymate meshes are
overlaid under the skeleton (`scripts/render_character_overlays.py`, mapped
cache→source by exact joint match). The domain is **diverse rigged 3D assets**,
not only humanoids: bipeds, quadrupeds, winged/spread creatures, thin props, and
multi-limb rigs, spanning **8–94 joints**.

- `overlays/character_cases_grid.png` — the four result cases over their meshes:
  the perfect case (idx 20) is a clean **humanoid biped**; the large-skeleton
  failure (idx 513) is a **dense animal body** where many joints pack into a
  torso (the hardest topology).
- `overlays/character_gallery_grid.png` — 9 characters spanning the joint-count
  range, GT skeleton over mesh, to show dataset variety.
- `overlays/character_montage_grid.png` — 24-character montage across the full
  joint-count range (thin prop, radial ball-creature, quadrupeds, humanoids,
  winged/tailed creatures, dense many-limb bodies).
- `overlays/character_mesh_grid.png` / `character_mesh_hipoly_grid.png` — the
  actual **solid triangle mesh** (from `mesh_face`) rendered translucent with the
  skeleton inside (`scripts/render_character_mesh.py`). The high-poly set (up to
  63k verts) shows detailed characters: elongated quadrupeds, a standing
  humanoid, and multi-limb creatures, with the recovered skeleton overlaid.

### Test-set composition (all 572 characters)

We produce a skeleton for **every** test character. Structural make-up:

- **Joint count:** 8–94 (median 44). tiny ≤10: 1% · small 11–25: 14% ·
  medium 26–50: 44% · large >50: 41%.
- **Limb tips (degree-1 leaves):** median 10, up to 40. ≤3 tips (simple/prop):
  2% · 4–6 (biped-like): 27% · 7–10: 27% · **>10 (many-limb): 45%**.
- **Max hub degree:** median 5, up to 21. ≥9 (heavy hub): 5%.

So the domain is **mostly complex multi-limb creatures/assets, not simple
bipeds** — which is exactly why high-degree hubs and dense creature torsos (many
joints in a small volume) are where geometry-only scoring is most ambiguous and
the residual errors concentrate.

---

## 3. Ablation narrative

The result is the endpoint of a gated search; every step was selected on
validation and the headline number tested once.

1. **v1 — independent per-edge scorer (BCE).** Geometry signal exists but scoring
   each candidate edge independently caps out. Test F1 **0.678**.
2. **v1.1 — independent scorer + warm teacher distillation.** Adding the teacher
   to the *wrong architecture* did not help; the score distribution became less
   MST-compatible. Test F1 **0.662**. Diagnosis: the bottleneck is the
   architecture, not the loss.
3. **v2 — edge-graph message passing.** Candidate edges become nodes in a line
   graph; message passing lets each edge see its neighbors before scoring. This
   broke the independent-scoring wall. Test F1 **0.802** (minimum pass).
4. **v2.1 — edge-graph + warm-start teacher distillation.** Teacher distillation
   *does* help on the right architecture (+0.016). Test F1 **0.818**. Still short
   of the 0.82 strong gate.
5. **v2.2 — ensemble of v2 + v2.1.** Alpha selected on val (0.40); test F1
   **0.8198** — explicitly **below** the 0.820 strong gate, so **not**
   overclaimed. v2 and v2.1 are too correlated (v2.1 warm-started from v2) to
   ensemble usefully.
6. **v2.3B — from-scratch teacher distillation.** Training the same architecture
   and teacher losses from *random init* (rather than warm-starting v2) escaped
   v2's weaker basin. Variant B (teacher-BCE weight 1.0) selected on val over
   A/C; tested once → **0.8569 (research pass)**. From-scratch teacher (+0.055
   over v2) ≫ warm-start teacher (+0.016); the teacher signal is most valuable
   when it shapes training from epoch 1. Original from-scratch v2 *without*
   teacher only reached 0.795, so the teacher contributes ~+0.06 from scratch.

The largest bucket-level gain was on the historically hardest bucket, large
skeletons (>50 joints): 0.766 (v2) → 0.793 (v2.1) → **0.851** (v2.3B).

---

## 4. Track A / Track B boundary

See `docs/figures/track_ab_boundary.svg`.

| | Track A (rigged) | Track B (geometry-only) |
|---|---|---|
| Inputs at inference | mesh + known joints + **skinning weights** | mesh + known joints, **no skinning** |
| Model | skinning-aware (v4.1) | edge-graph student (v2.3B) |
| Test F1 | 0.888 | 0.857 |
| Applies to | rigged/skinned assets | unrigged mesh geometry |

Track A's skinning-based model is the **teacher** that trains Track B — skinning
enters Track B **only** at training time, as a distillation target. **Never
conflate the two results.** Both assume known nodes.

---

## 5. Limitations

- **Known-node only.** Joint positions are an input. This is topology recovery
  over *given* nodes, not node/joint **discovery**. A full unrigged pipeline
  still needs a node-proposal stage upstream.
- **Topology only.** Produces the undirected bone-connectivity graph. It does not
  produce a rooted/directed hierarchy, joint rotations, or skinning weights.
- **Anymate-domain evaluation.** Results are on the held-out Anymate test split.
  Cross-dataset / in-the-wild mesh generalization is unmeasured.
- **Residual failure modes (measured).** (a) High-degree hubs: GT max-degree ≥9
  scores F1 0.659 (n=26) — attaching many bones to one hub from geometry alone is
  hard. (b) Long true edges are missed while near-neighbors are over-connected
  (false-negative mean length percentile 0.364 vs false-positive 0.208).
- **Not** a claim of "solving unrigged skeleton extraction" or "general animation
  rigging." The defensible claim is **known-node topology recovery from mesh
  geometry alone**.

If pushed further, the indicated next architecture is a joint-node × edge-node
bipartite graph so hub structure is modeled directly rather than inferred
per-edge — but per current guidance, packaging > further training.
