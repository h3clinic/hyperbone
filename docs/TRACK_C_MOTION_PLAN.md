# HyperBone Track C — Motion Prior (and the road to scene structure)

Track B recovers **static** known-node topology from mesh geometry (test F1
0.8569, repro-verified). Track C learns **how recovered skeletons move**. The
strategic thesis beyond C is a *structural taxonomy* — different object classes
need different priors (skeletons for characters, branch graphs for trees, surface
fields for terrain, volume fields for clouds) — but we build the grounded ladder
first, not the cinematic end state.

## Ladder (do not skip rungs)

1. **Static mesh topology** — Track B (done, research pass).
2. **Local joint micro-motion** — Track C1 (this milestone).
3. **Motion plausibility / mobility model** — Track C2/C3.
4. **Real animation clips + retargeting** — Track C4/C5.
5. **Rendered video → structure + motion** — Track D.
6. **Universal scene structure** (characters/trees/ground/clouds…) — Track E.

Start controlled (ground-truth joint motion), not from rendered movies — full
RGB is too noisy for the first motion model.

## Structural taxonomy (Track E target — not built yet)

| Object | Representation |
|--------|----------------|
| human / animal / creature | articulated skeleton graph |
| tree / coral / lightning | branch graph + deformable foliage |
| ground / terrain / walls | surface / terrain mesh |
| cloud / fog / smoke | volumetric density field |
| grass / cloth / flags | deformable surface / strand field |
| water / fire / smoke | flow field |
| vehicle / machine | rigid-part mechanical graph |

The eventual model: `object mask → structure class → structure parameters`. Do
**not** force trees/clouds into skeleton logic.

---

## Track C1 — Procedural micro-motion dataset (IMPLEMENTED 2026-07-02)

Generate controlled motion on existing Anymate skeletons with exact labels.

### Files
- Generator: `scripts/generate_micro_motion_dataset.py`
- Renderer: `scripts/render_micro_motion_clip.py`
- Output: `outputs/track_c_micro_motion/` (`clips/*.npz`, `index.json`, `summary.json`)

### Method
Anymate `conns` gives the parent array → forward-kinematics skeleton (root,
parents, children, topo order, rest offsets, bone lengths; cycles broken by
promoting a node to root). **Valid** motions are driven by pure local rotations
about parents, so bone lengths are preserved to machine precision.

**8 valid presets:** `single_joint_bend`, `limb_swing`, `spine_bend`,
`tail_sway`, `wing_flap`, `root_sway`, `generic_breathing`, `tremor_micro_jitter`.

**8 corrupted instances (6 types):** `bone_length_scale_error`, `detached_child`,
`wrong_parent_motion`, `temporal_jitter` (×2 amplitudes),
`impossible_large_rotation` (×2 amplitudes), `swapped_limb_motion`. Length-based
corruptions change bone lengths; articulation-based ones
(`impossible_large_rotation`, `swapped_limb_motion`) stay length-preserving but
are physically implausible — so a plausibility model must learn more than a
length check.

### Clip format (`.npz`)
`asset_idx`, `source_idx`, `name`, `clip_id`, `preset`, `fps`, `num_frames`,
`joints_rest[J,3]`, `joints_world[T,J,3]`, `local_rot_quat[T,J,4]` (xyzw),
`edges[E,2]`, `parents[J]`, `bone_lengths[J]`, `moving_joint_ids`,
`motion_amplitude`, `motion_frequency`, `is_valid_motion`, `corruption_type`,
`mesh_vertex_count`, `joint_count`.

### Smoke result (20 assets, fps 24, 3 s → T=72)
- 160 valid + 160 corrupted (≥100 each ✓)
- **max bone-length deviation on valid clips = 4.4e-16** (machine precision ✓)
- Corruption deviations: scale 0.14, detached 0.06, jitter 0.10, wrong-parent
  0.17, swapped 0.01, impossible-rotation 0.00 (length-preserving but implausible)
- Renderer produced MP4 previews + a valid/invalid filmstrip

### Acceptance (all met)
✓ generates `.npz` dataset · ✓ ≥100 valid + ≥100 corrupted in smoke ·
✓ bone lengths constant on valid clips · ✓ corrupted clips labeled with type ·
✓ renderer produces a preview · ✓ no neural training · ✓ no RGB-movie training.

### Full run
`python scripts/generate_micro_motion_dataset.py --source datasets/anymate/Anymate_test.pt --split outputs/anymate_local_dev/splits/test.jsonl --num-assets 100 --out outputs/track_c_micro_motion`
(100 assets → 800 valid + 800 corrupted; use `--num-assets 125` for the
1000/1000 gate below.)

---

---

## Track C2 — Mobility / motion-readiness feature layer (IMPLEMENTED 2026-07-02)

Interpretable annotation of *why* a motion is (im)plausible, feeding C3.

### Files
- Extractor: `hyperbone/motion/mobility.py`
- Batch annotate: `scripts/annotate_micro_motion.py` → `features.npz` + `mobility_report.json`
- Dataset: `outputs/track_c_micro_motion_1000/` (1000 valid + 1000 invalid, 125 assets)

### What it computes (pure geometry, no net)
- **per-joint `mobility_type`:** fixed | hinge | ball | flexible_chain | root
  (from observed bone swing + rotation-axis DOF + chain membership)
- **per-edge `motion_role`:** rigid_bone | flexible_branch | soft_field_proxy
  (soft = bone length not preserved → deformation, a hook for the Track E taxonomy)
- **per-clip features:** bone_length_deviation, mean_bone_length_deviation,
  max/mean joint_angle_range, max angular velocity/acceleration,
  parent_child_consistency, motion_symmetry, temporal_smoothness, motion_energy,
  moving_joint_fraction. Plus localization (max-length-dev edge, max-angle joint).

### Separability (mean feature by class, 1000/1000 set)

| class | bone_len_dev | max_angle_range | motion_symmetry | temporal_smoothness |
|-------|:---:|:---:|:---:|:---:|
| valid | 0.000 | 0.61 | 0.35 | 0.94 |
| bone_length_scale_error | 1.03 | – | – | – |
| temporal_jitter | 3.92 | 2.75 | – | 0.52 |
| **impossible_large_rotation** | **0.00** | **2.50** | – | – |
| **swapped_limb_motion** | **0.01** | – | **0.09** | – |

**The scientific point holds:** the two length-preserving corruptions
(`impossible_large_rotation`, `swapped_limb_motion`) have bone_length_deviation
indistinguishable from valid, yet are separated by **angle range** and
**symmetry** respectively. A bone-length checker would miss both; the mobility
layer does not. All four rules-separability checks pass.

### Scaled dataset
`--pt/--valid-per-asset/--corrupt-per-asset` added to the generator; 125 assets →
**1000 valid + 1000 invalid**, valid bone-length deviation 5.3e-15.

---

## Next milestones

- **C3 — Motion plausibility scorer:** valid vs corrupted classifier on the C2
  features. First real **Track C gate:** validity F1 ≥ 0.90, corruption_type
  macro F1 ≥ 0.75, and it must catch the length-preserving implausibilities.
  Compare rules-only vs learned. Immediately useful: scores whether a Track B
  skeleton is *motion-ready*, not just geometrically right.
- **C4/C5 — Real mocap (CMU/Mixamo/AMASS) + retargeting** onto Anymate skeletons.
- **D — Rendered video → structure + motion** (only after joint-motion works).
- **E — Universal scene structure** (the taxonomy above).

Success is **not** "the movie looks cool." Success is measurable structure/motion
recovery with exact labels from the renderer.
