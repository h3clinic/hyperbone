"""HyperBone Track C2 — motion mobility / motion-readiness feature layer.

Given a micro-motion clip (joints over time + skeleton), compute an
interpretable annotation of *why* a motion is (im)plausible:

  * per-joint mobility_type:  fixed | hinge | ball | flexible_chain | root
  * per-edge  motion_role:    rigid_bone | flexible_branch | soft_field_proxy
  * per-clip feature vector: joint angle range/velocity/acceleration,
    bone-length deviation, parent-child consistency, motion symmetry,
    temporal smoothness, motion energy.

Design goal: these features must separate not only length-breaking corruptions
(scale error, detached child, jitter) but also LENGTH-PRESERVING implausibilities
(impossible rotations -> huge angle range; swapped limbs -> broken symmetry).
That is the test of a real motion-plausibility layer vs a bone-length checker.

Pure geometry; no skinning, no neural net.
"""
from __future__ import annotations

import numpy as np

MOBILITY_TYPES = ["fixed", "hinge", "ball", "flexible_chain", "root"]
MOTION_ROLES = ["rigid_bone", "flexible_branch", "soft_field_proxy"]

FEATURE_NAMES = [
    "bone_length_deviation",       # max relative bone-length change (KEY)
    "mean_bone_length_deviation",
    "max_joint_angle_range",       # catches impossible_large_rotation
    "mean_joint_angle_range",
    "max_angular_velocity",
    "max_angular_acceleration",
    "parent_child_consistency",    # ~1 when every bone stays rigid
    "motion_symmetry",             # low/negative under swapped_limb_motion
    "temporal_smoothness",         # low under temporal_jitter
    "motion_energy",               # overall amount of motion
    "moving_joint_fraction",
]

_EPS = 1e-8


def _angle_series(dirs: np.ndarray) -> np.ndarray:
    """Angle (rad) of each frame's unit dir vs frame 0. dirs [T,3] -> [T]."""
    d0 = dirs[0]
    dots = np.clip(dirs @ d0, -1.0, 1.0)
    return np.arccos(dots)


def _dof_ratio(dirs: np.ndarray) -> float:
    """2nd/1st eigenvalue of the direction covariance. ~0 => hinge (1D arc),
    larger => ball (2D patch)."""
    if dirs.shape[0] < 3:
        return 0.0
    c = np.cov(dirs.T)
    w = np.sort(np.linalg.eigvalsh(c))[::-1]
    if w[0] <= _EPS:
        return 0.0
    return float(w[1] / (w[0] + _EPS))


def edge_series(world: np.ndarray, edges: np.ndarray):
    """Per-edge bone vectors, lengths, unit dirs. world [T,J,3], edges [E,2]."""
    p = edges[:, 0]; c = edges[:, 1]
    vec = world[:, c, :] - world[:, p, :]          # [T,E,3]
    length = np.linalg.norm(vec, axis=2)            # [T,E]
    unit = vec / (length[:, :, None] + _EPS)
    return vec, length, unit


def annotate_clip(world: np.ndarray, edges: np.ndarray, parents: np.ndarray,
                  joints_rest: np.ndarray, fps: int,
                  fixed_thresh: float = 0.03) -> dict:
    """Compute mobility annotation + per-clip features for one clip."""
    T, J, _ = world.shape
    dt = 1.0 / float(fps)
    E = edges.shape[0]

    bone_med = 1.0
    if E:
        rest_len = np.linalg.norm(joints_rest[edges[:, 1]] - joints_rest[edges[:, 0]], axis=1)
        pos = rest_len[rest_len > _EPS]
        bone_med = float(np.median(pos)) if pos.size else 1.0

    # ---- per-edge quantities ----
    per_edge_len_dev = np.zeros(E)
    per_edge_angle_range = np.zeros(E)
    per_edge_max_vel = np.zeros(E)
    per_edge_max_acc = np.zeros(E)
    per_edge_dof = np.zeros(E)
    edge_theta = np.zeros((T, E)) if E else np.zeros((T, 0))

    if E:
        _, length, unit = edge_series(world, edges)
        l0 = length[0]
        # relative deviation, floored denominator so near-zero degenerate
        # bones (duplicate joints) don't inflate the metric
        denom = np.maximum(np.abs(l0), 1e-3 * bone_med)
        per_edge_len_dev = np.max(np.abs(length - l0[None, :]), axis=0) / denom
        for e in range(E):
            th = _angle_series(unit[:, e, :])
            edge_theta[:, e] = th
            per_edge_angle_range[e] = float(th.max() - th.min())
            vel = np.gradient(th) / dt
            acc = np.gradient(vel) / dt
            per_edge_max_vel[e] = float(np.abs(vel).max())
            per_edge_max_acc[e] = float(np.abs(acc).max())
            per_edge_dof[e] = _dof_ratio(unit[:, e, :])

    # map edges to child joint (bone p->c is "owned" by c)
    joint_angle_range = np.zeros(J)
    joint_dof = np.zeros(J)
    child_edge = {int(edges[e, 1]): e for e in range(E)}
    for c, e in child_edge.items():
        joint_angle_range[c] = per_edge_angle_range[e]
        joint_dof[c] = per_edge_dof[e]

    moving = joint_angle_range > fixed_thresh

    # ---- per-joint mobility_type ----
    mobility = np.empty(J, dtype=object)
    for j in range(J):
        if parents[j] < 0:
            mobility[j] = "root"
        elif not moving[j]:
            mobility[j] = "fixed"
        elif joint_dof[j] < 0.20:
            mobility[j] = "hinge"
        else:
            mobility[j] = "ball"
    # upgrade chain members: joint moves, its parent moves, exactly one child that moves
    children = {j: [] for j in range(J)}
    for j in range(J):
        if parents[j] >= 0:
            children[int(parents[j])].append(j)
    for j in range(J):
        if mobility[j] in ("hinge", "ball"):
            p = int(parents[j])
            par_moves = p >= 0 and moving[p]
            mv_children = [c for c in children[j] if moving[c]]
            if par_moves and len(mv_children) >= 1:
                mobility[j] = "flexible_chain"

    # ---- per-edge motion_role ----
    role = np.empty(E, dtype=object)
    for e in range(E):
        c = int(edges[e, 1])
        if per_edge_len_dev[e] > 0.02:
            role[e] = "soft_field_proxy"          # length not preserved -> deformation
        elif mobility[c] == "flexible_chain":
            role[e] = "flexible_branch"
        else:
            role[e] = "rigid_bone"

    # ---- per-clip scalar features ----
    bld = float(per_edge_len_dev.max()) if E else 0.0
    mean_bld = float(per_edge_len_dev.mean()) if E else 0.0
    max_ar = float(per_edge_angle_range.max()) if E else 0.0
    mean_ar = float(per_edge_angle_range.mean()) if E else 0.0
    max_vel = float(per_edge_max_vel.max()) if E else 0.0
    max_acc = float(per_edge_max_acc.max()) if E else 0.0
    consistency = float(np.mean(1.0 - np.minimum(per_edge_len_dev, 1.0))) if E else 1.0

    symmetry = _motion_symmetry(edge_theta, edges, parents, joints_rest, moving)
    smooth = _temporal_smoothness(world, bone_med, fps)
    # motion energy: mean per-joint path length / bone_med
    if T > 1:
        step = np.linalg.norm(np.diff(world, axis=0), axis=2)   # [T-1,J]
        energy = float(step.sum(axis=0).mean() / (bone_med + _EPS))
    else:
        energy = 0.0
    moving_frac = float(moving.mean())

    features = {
        "bone_length_deviation": bld,
        "mean_bone_length_deviation": mean_bld,
        "max_joint_angle_range": max_ar,
        "mean_joint_angle_range": mean_ar,
        "max_angular_velocity": max_vel,
        "max_angular_acceleration": max_acc,
        "parent_child_consistency": consistency,
        "motion_symmetry": symmetry,
        "temporal_smoothness": smooth,
        "motion_energy": energy,
        "moving_joint_fraction": moving_frac,
    }

    # localization: most suspicious joint / edge
    culprit_edge = int(np.argmax(per_edge_len_dev)) if E else -1
    culprit_joint_angle = int(np.argmax(joint_angle_range)) if J else -1

    return {
        "features": features,
        "mobility_type": mobility.tolist(),
        "motion_role": role.tolist(),
        "per_edge_len_dev": per_edge_len_dev.tolist(),
        "per_joint_angle_range": joint_angle_range.tolist(),
        "localization": {
            "max_len_dev_edge": culprit_edge,
            "max_len_dev_value": float(per_edge_len_dev.max()) if E else 0.0,
            "max_angle_joint": culprit_joint_angle,
            "max_angle_value": max_ar,
        },
    }


def _motion_symmetry(edge_theta, edges, parents, joints_rest, moving):
    """Mean correlation of angle series between mirror-paired moving bones."""
    E = edges.shape[0]
    if E == 0 or edge_theta.shape[0] < 3:
        return 0.0
    # rest offset per child joint
    child = edges[:, 1]
    off = joints_rest[edges[:, 1]] - joints_rest[edges[:, 0]]   # [E,3]
    mag = np.linalg.norm(off, axis=1)
    corrs = []
    used = set()
    for a in range(E):
        if a in used or not moving[int(child[a])]:
            continue
        best = -1; bestscore = 1e9
        for b in range(E):
            if b == a or b in used or not moving[int(child[b])]:
                continue
            # mirror in x: opposite x sign, similar |offset|, similar y,z
            if off[a, 0] * off[b, 0] >= 0:
                continue
            dmag = abs(mag[a] - mag[b]) / (mag[a] + mag[b] + _EPS)
            dyz = np.linalg.norm(off[a, 1:] - off[b, 1:]) / (mag[a] + _EPS)
            score = dmag + dyz
            if score < bestscore and score < 0.4:
                bestscore = score; best = b
        if best >= 0:
            used.add(a); used.add(best)
            ta, tb = edge_theta[:, a], edge_theta[:, b]
            if ta.std() > _EPS and tb.std() > _EPS:
                corrs.append(float(np.corrcoef(ta, tb)[0, 1]))
    if not corrs:
        return 0.0
    return float(np.mean(corrs))


def _temporal_smoothness(world, bone_med, fps):
    """1/(1+normalized mean jerk). Low under temporal jitter."""
    T = world.shape[0]
    if T < 4:
        return 1.0
    jerk = np.diff(world, n=3, axis=0)               # [T-3,J,3]
    jmag = np.linalg.norm(jerk, axis=2)              # [T-3,J]
    jn = jmag.mean() / (bone_med + _EPS)
    return float(1.0 / (1.0 + jn))
