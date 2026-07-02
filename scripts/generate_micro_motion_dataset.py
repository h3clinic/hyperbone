"""HyperBone Track C1: procedural micro-motion dataset generation.

Generates controlled motion sequences on existing Anymate skeletons so the
model can begin learning how recovered skeletons move. Valid clips are driven
by forward kinematics (pure rotations about parent joints), so bone lengths are
preserved exactly. Corrupted clips deliberately break motion realism and are
labeled with a corruption type.

No neural training here. No rendered RGB. Ground-truth joint motion only.

Usage:
    python scripts/generate_micro_motion_dataset.py \
        --source datasets/anymate/Anymate_test.pt \
        --split  outputs/anymate_local_dev/splits/test.jsonl \
        --num-assets 100 --fps 24 --seconds 3 \
        --out outputs/track_c_micro_motion
    # smoke:
    python scripts/generate_micro_motion_dataset.py ... --smoke
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot


# ----------------------------- skeleton -----------------------------

@dataclass
class Skeleton:
    J: int
    joints: np.ndarray          # [J,3] rest positions
    parents: np.ndarray         # [J] parent idx, -1 for root
    roots: list                 # root joint ids
    children: dict              # j -> list of children
    topo: list                  # topo order (parents before children)
    offset: np.ndarray          # [J,3] rest offset from parent (0 for roots)
    bone_len: np.ndarray        # [J] |offset|, 0 for roots
    depth: np.ndarray           # [J] hops from its root
    edges: np.ndarray           # [E,2] parent-child undirected edges


def build_skeleton(joints: np.ndarray, conns: np.ndarray) -> Skeleton:
    J = joints.shape[0]
    parents = np.full(J, -1, dtype=np.int64)
    for c in range(J):
        p = int(conns[c])
        if 0 <= p < J and p != c:
            parents[c] = p
    roots = [c for c in range(J) if parents[c] < 0]

    # topo order guaranteeing parent-before-child; break cycles by promoting
    # an unresolved node to a root.
    processed = set(roots)
    topo = list(roots)
    while len(topo) < J:
        progress = False
        for c in range(J):
            if c in processed:
                continue
            p = int(parents[c])
            if p < 0 or p in processed:
                processed.add(c); topo.append(c); progress = True
        if not progress:
            for c in range(J):
                if c not in processed:
                    parents[c] = -1; roots.append(c)
                    processed.add(c); topo.append(c)
                    break

    children = defaultdict(list)
    for c in range(J):
        if parents[c] >= 0:
            children[parents[c]].append(c)
    depth = np.zeros(J, dtype=np.int64)
    for n in topo:  # parents precede children in topo
        if parents[n] >= 0:
            depth[n] = depth[parents[n]] + 1
    offset = np.zeros((J, 3), dtype=np.float64)
    bone_len = np.zeros(J, dtype=np.float64)
    for c in range(J):
        if parents[c] >= 0:
            offset[c] = joints[c] - joints[parents[c]]
            bone_len[c] = np.linalg.norm(offset[c])
    edges = np.array([[parents[c], c] for c in range(J) if parents[c] >= 0],
                     dtype=np.int64) if (parents >= 0).any() else np.zeros((0, 2), np.int64)
    return Skeleton(J, joints.astype(np.float64), parents, roots, dict(children),
                    topo, offset, bone_len, depth, edges)


def leaves_of(sk: Skeleton):
    deg = np.zeros(sk.J, dtype=int)
    for c in range(sk.J):
        if sk.parents[c] >= 0:
            deg[c] += 1; deg[sk.parents[c]] += 1
    return [c for c in range(sk.J) if deg[c] == 1 and c not in sk.roots]


def path_to_root(sk: Skeleton, j: int):
    path = [j]
    while sk.parents[path[-1]] >= 0:
        path.append(int(sk.parents[path[-1]]))
    return path  # j ... root


def spine_chain(sk: Skeleton):
    """Root -> deepest leaf path."""
    lv = leaves_of(sk)
    if not lv:
        return sk.topo[:1]
    deep = max(lv, key=lambda j: sk.depth[j])
    return list(reversed(path_to_root(sk, deep)))  # root..leaf


def limb_segment(sk: Skeleton, leaf: int):
    """From a leaf, walk up until a branch point (node with >1 child).
    Returns (proximal_joint, chain_joints leaf..proximal)."""
    chain = [leaf]
    cur = leaf
    while True:
        p = int(sk.parents[cur])
        if p < 0:
            break
        chain.append(p)
        if len(sk.children.get(p, [])) > 1 or sk.parents[p] < 0:
            # p is a branch/root -> proximal is `cur` (child of the branch)
            return cur, chain[:-1] if len(chain) > 1 else chain
        cur = p
    return chain[-1], chain


def perp_axis(sk: Skeleton, j: int, rng):
    """A unit axis roughly perpendicular to the bone at joint j."""
    if sk.parents[j] >= 0 and sk.bone_len[j] > 1e-6:
        bd = sk.offset[j] / sk.bone_len[j]
    else:
        bd = np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, 0.0, 1.0])
    ax = np.cross(bd, up)
    if np.linalg.norm(ax) < 1e-3:
        ax = np.cross(bd, np.array([1.0, 0.0, 0.0]))
    if np.linalg.norm(ax) < 1e-3:
        ax = np.array([1.0, 0.0, 0.0])
    return ax / (np.linalg.norm(ax) + 1e-9)


# ----------------------------- forward kinematics -----------------------------

def fk(sk: Skeleton, rotvec: np.ndarray, root_trans: np.ndarray,
       scale: np.ndarray = None) -> np.ndarray:
    """Vectorized FK over frames.
      rotvec: [T,J,3] local rotation vectors (per joint, per frame)
      root_trans: [T,3] translation applied to roots
      scale: [T,J] optional per-joint bone scale (breaks length if != 1)
    Returns world [T,J,3].
    """
    T = rotvec.shape[0]
    world = np.zeros((T, sk.J, 3), dtype=np.float64)
    Rg = [None] * sk.J
    for j in sk.topo:
        Rl = Rot.from_rotvec(rotvec[:, j, :])       # length-T
        if sk.parents[j] < 0:
            Rg[j] = Rl
            world[:, j, :] = sk.joints[j][None, :] + root_trans
        else:
            p = int(sk.parents[j])
            Rg[j] = Rg[p] * Rl
            off = sk.offset[j]
            if scale is not None:
                off = off[None, :] * scale[:, j][:, None]
                world[:, j, :] = world[:, p, :] + Rg[p].apply(off)
            else:
                world[:, j, :] = world[:, p, :] + Rg[p].apply(off)
    return world


def quats_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    T, J, _ = rotvec.shape
    q = np.zeros((T, J, 4), dtype=np.float32)
    for j in range(J):
        q[:, j, :] = Rot.from_rotvec(rotvec[:, j, :]).as_quat()  # [x,y,z,w]
    return q


def bone_length_series(sk: Skeleton, world: np.ndarray) -> np.ndarray:
    """[T,E] bone lengths per frame for parent-child edges."""
    if sk.edges.shape[0] == 0:
        return np.zeros((world.shape[0], 0))
    p = sk.edges[:, 0]; c = sk.edges[:, 1]
    return np.linalg.norm(world[:, c, :] - world[:, p, :], axis=2)


# ----------------------------- valid presets -----------------------------

def times(T, fps):
    return np.arange(T) / float(fps)


def preset_rotvec(sk: Skeleton, preset: str, T, fps, rng):
    """Return (rotvec[T,J,3], root_trans[T,3], moving_ids, amp, freq)."""
    rv = np.zeros((T, sk.J, 3))
    rt = np.zeros((T, 3))
    tt = times(T, fps)
    moving = []
    amp = 0.0
    freq = 0.0

    def osc(a, f, phase=0.0):
        return a * np.sin(2 * np.pi * f * tt + phase)

    if preset == "single_joint_bend":
        cand = [j for j in range(sk.J) if sk.parents[j] >= 0]
        j = int(rng.choice(cand)) if cand else 0
        amp, freq = 0.45, 1.0
        rv[:, j, :] = perp_axis(sk, j, rng)[None, :] * osc(amp, freq)[:, None]
        moving = [j]

    elif preset == "limb_swing":
        lv = leaves_of(sk)
        if lv:
            leaf = int(rng.choice(lv))
            prox, _ = limb_segment(sk, leaf)
            j = prox
        else:
            j = sk.topo[-1]
        amp, freq = 0.55, 0.8
        rv[:, j, :] = perp_axis(sk, j, rng)[None, :] * osc(amp, freq)[:, None]
        moving = [j]

    elif preset == "spine_bend":
        chain = [j for j in spine_chain(sk) if sk.parents[j] >= 0]
        amp, freq = 0.22, 0.6
        for k, j in enumerate(chain):
            ph = 0.4 * k
            rv[:, j, :] = perp_axis(sk, j, rng)[None, :] * osc(amp, freq, ph)[:, None]
        moving = chain

    elif preset == "tail_sway":
        lv = leaves_of(sk)
        # tail = longest limb chain not equal to spine
        spine = set(spine_chain(sk))
        best = None; bestlen = -1
        for leaf in lv:
            _, chain = limb_segment(sk, leaf)
            if leaf in spine:
                continue
            if len(chain) > bestlen:
                bestlen = len(chain); best = chain
        chain = [j for j in (best or []) if sk.parents[j] >= 0]
        amp, freq = 0.3, 1.2
        for k, j in enumerate(chain):
            rv[:, j, :] = perp_axis(sk, j, rng)[None, :] * osc(amp, freq, 0.6 * k)[:, None]
        moving = chain if chain else []
        if not moving:  # fallback
            return preset_rotvec(sk, "single_joint_bend", T, fps, rng)

    elif preset == "wing_flap":
        lv = leaves_of(sk)
        # find a symmetric pair by opposite-sign x of proximal offset
        proxs = []
        for leaf in lv:
            prox, _ = limb_segment(sk, leaf)
            proxs.append(prox)
        proxs = list(dict.fromkeys(proxs))
        pair = None
        for a in range(len(proxs)):
            for b in range(a + 1, len(proxs)):
                ja, jb = proxs[a], proxs[b]
                xa = sk.offset[ja][0]; xb = sk.offset[jb][0]
                if xa * xb < 0 and abs(abs(xa) - abs(xb)) < 0.25 * (abs(xa) + abs(xb) + 1e-6):
                    pair = (ja, jb); break
            if pair:
                break
        if pair is None and len(proxs) >= 2:
            pair = (proxs[0], proxs[1])
        amp, freq = 0.6, 2.0
        if pair:
            ja, jb = pair
            ax = perp_axis(sk, ja, rng)
            rv[:, ja, :] = ax[None, :] * osc(amp, freq)[:, None]
            rv[:, jb, :] = -ax[None, :] * osc(amp, freq)[:, None]  # mirrored
            moving = [ja, jb]
        else:
            return preset_rotvec(sk, "limb_swing", T, fps, rng)

    elif preset == "root_sway":
        amp, freq = 0.15, 0.5
        ax = np.array([0.0, 0.0, 1.0])
        for r in sk.roots:
            rv[:, r, :] = ax[None, :] * osc(amp, freq)[:, None]
        moving = list(sk.roots)

    elif preset == "generic_breathing":
        amp, freq = 0.05, 0.35
        near = [j for j in range(sk.J) if 0 < sk.depth[j] <= 2]
        for j in near:
            rv[:, j, :] = perp_axis(sk, j, rng)[None, :] * osc(amp, freq)[:, None]
        # slight root vertical bob (length-preserving global shift)
        rt[:, 2] = osc(0.02 * (sk.bone_len[sk.bone_len > 0].mean() if (sk.bone_len > 0).any() else 1.0), freq)
        moving = near

    elif preset == "tremor_micro_jitter":
        amp, freq = 0.04, 7.0
        cand = [j for j in range(sk.J) if sk.parents[j] >= 0]
        for j in cand:
            ph = rng.uniform(0, 2 * np.pi)
            ax = perp_axis(sk, j, rng)
            rv[:, j, :] = ax[None, :] * osc(amp, freq, ph)[:, None]
        moving = cand
    else:
        raise ValueError(preset)

    return rv, rt, moving, amp, freq


VALID_PRESETS = ["single_joint_bend", "limb_swing", "spine_bend", "tail_sway",
                 "wing_flap", "root_sway", "generic_breathing", "tremor_micro_jitter"]


# ----------------------------- corruptions -----------------------------

CORRUPTIONS = [
    ("bone_length_scale_error", {}),
    ("detached_child", {}),
    ("wrong_parent_motion", {}),
    ("temporal_jitter", {"sigma": 0.06}),
    ("impossible_large_rotation", {"amp": 2.6}),
    ("swapped_limb_motion", {}),
    ("temporal_jitter", {"sigma": 0.15}),          # variant
    ("impossible_large_rotation", {"amp": 3.0}),   # variant
]


def make_corruption(sk: Skeleton, ctype, params, T, fps, rng):
    """Return (world[T,J,3], rotvec[T,J,3], moving_ids, amp, freq, ctype)."""
    tt = times(T, fps)
    med = sk.bone_len[sk.bone_len > 0].mean() if (sk.bone_len > 0).any() else 1.0

    if ctype == "bone_length_scale_error":
        rv, rt, moving, amp, freq = preset_rotvec(sk, "limb_swing", T, fps, rng)
        scale = np.ones((T, sk.J))
        # oscillate lengths of a subset of bones
        cand = [j for j in range(sk.J) if sk.parents[j] >= 0]
        pick = rng.choice(cand, size=max(1, len(cand)//3), replace=False)
        for j in pick:
            scale[:, j] = 1.0 + 0.35 * np.sin(2*np.pi*1.0*tt + rng.uniform(0, 6))
        world = fk(sk, rv, rt, scale=scale)
        return world, rv, list(pick), 0.35, 1.0

    if ctype == "detached_child":
        rv, rt, moving, amp, freq = preset_rotvec(sk, "single_joint_bend", T, fps, rng)
        world = fk(sk, rv, rt)
        cand = [j for j in range(sk.J) if sk.parents[j] >= 0]
        j = int(rng.choice(cand))
        # drift joint j (and its subtree) away from parent -> attachment broken
        drift = np.outer(np.linspace(0, 1, T), rng.normal(size=3))
        drift = drift / (np.linalg.norm(drift[-1]) + 1e-9) * (0.8 * med)
        sub = subtree(sk, j)
        world[:, sub, :] += drift[:, None, :]
        return world, rv, [j], 0.8, 0.0

    if ctype == "wrong_parent_motion":
        # rotate a subtree about a NON-parent pivot
        cand = [j for j in range(sk.J) if sk.parents[j] >= 0 and sk.children.get(j)]
        base = fk(sk, np.zeros((T, sk.J, 3)), np.zeros((T, 3)))  # rest, static
        if not cand:
            return make_corruption(sk, "detached_child", {}, T, fps, rng)
        j = int(rng.choice(cand))
        sub = subtree(sk, j)
        pivots = [p for p in range(sk.J) if p != sk.parents[j] and p != j]
        pivot = int(rng.choice(pivots)) if pivots else int(sk.roots[0])
        amp, freq = 0.5, 1.0
        ang = amp * np.sin(2*np.pi*freq*tt)
        ax = perp_axis(sk, j, rng)
        world = base.copy()
        pv = sk.joints[pivot]
        for t in range(T):
            Rt = Rot.from_rotvec(ax * ang[t])
            world[t, sub, :] = pv + Rt.apply(sk.joints[sub] - pv)
        return world, np.zeros((T, sk.J, 3)), [j], amp, freq

    if ctype == "temporal_jitter":
        rv, rt, moving, amp, freq = preset_rotvec(sk, "limb_swing", T, fps, rng)
        world = fk(sk, rv, rt)
        sigma = params.get("sigma", 0.06) * med
        world = world + rng.normal(scale=sigma, size=world.shape)
        return world, rv, moving, params.get("sigma", 0.06), freq

    if ctype == "impossible_large_rotation":
        rv = np.zeros((T, sk.J, 3)); rt = np.zeros((T, 3))
        cand = [j for j in range(sk.J) if sk.parents[j] >= 0 and sk.children.get(j)]
        j = int(rng.choice(cand)) if cand else sk.topo[-1]
        amp = params.get("amp", 2.6); freq = 1.0
        rv[:, j, :] = perp_axis(sk, j, rng)[None, :] * (amp * np.sin(2*np.pi*freq*tt))[:, None]
        world = fk(sk, rv, rt)  # lengths preserved, but physically implausible fold
        return world, rv, [j], amp, freq

    if ctype == "swapped_limb_motion":
        lv = leaves_of(sk)
        proxs = list(dict.fromkeys(limb_segment(sk, l)[0] for l in lv))
        base = fk(sk, np.zeros((T, sk.J, 3)), np.zeros((T, 3)))
        if len(proxs) < 2:
            return make_corruption(sk, "detached_child", {}, T, fps, rng)
        ja, jb = [int(x) for x in rng.choice(proxs, size=2, replace=False)]
        sa, sb = subtree(sk, ja), subtree(sk, jb)
        amp, freq = 0.5, 1.0
        ang = amp * np.sin(2*np.pi*freq*tt)
        axa, axb = perp_axis(sk, ja, rng), perp_axis(sk, jb, rng)
        world = base.copy()
        # apply limb A's rotation to limb B's joints about B's parent, and vice versa
        pa = sk.joints[sk.parents[ja]]; pb = sk.joints[sk.parents[jb]]
        for t in range(T):
            Ra = Rot.from_rotvec(axa * ang[t]); Rb = Rot.from_rotvec(axb * ang[t])
            world[t, sb, :] = pb + Ra.apply(sk.joints[sb] - pb)   # B moves with A's rot
            world[t, sa, :] = pa + Rb.apply(sk.joints[sa] - pa)   # A moves with B's rot
        return world, np.zeros((T, sk.J, 3)), [ja, jb], amp, freq

    raise ValueError(ctype)


def subtree(sk: Skeleton, j: int):
    out = []
    dq = deque([j])
    while dq:
        n = dq.popleft(); out.append(n)
        for ch in sk.children.get(n, []):
            dq.append(ch)
    return np.array(out, dtype=np.int64)


# ----------------------------- driver -----------------------------

def save_clip(out_dir, rec):
    p = out_dir / "clips" / f"{rec['clip_id']}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **rec)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--num-assets", type=int, default=100)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.smoke:
        args.num_assets = min(args.num_assets, 20)
    T = int(round(args.fps * args.seconds))
    out_dir = Path(args.out); (out_dir / "clips").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Track C1 micro-motion  T={T} fps={args.fps} assets={args.num_assets}", flush=True)
    with open(args.split) as f:
        split_lines = [json.loads(l) for l in f]
    print("Loading source...", flush=True)
    source = torch.load(args.source, map_location="cpu", weights_only=False)

    # select assets with enough joints, spread across joint-count range
    usable = [(k, sl) for k, sl in enumerate(split_lines) if sl.get("joints_num", 0) >= 4]
    usable.sort(key=lambda ks: ks[1]["joints_num"])
    if len(usable) > args.num_assets:
        pick = [usable[round(t*(len(usable)-1)/(args.num_assets-1))] for t in range(args.num_assets)]
    else:
        pick = usable

    index = []
    n_valid = n_corrupt = 0
    max_len_dev_valid = 0.0
    for ai, (cache_idx, sl) in enumerate(pick):
        src = source[sl["idx"]]
        J = int(src["joints_num"])
        joints = src["joints"][:J].numpy().astype(np.float64)
        conns = src["conns"][:J].numpy()
        sk = build_skeleton(joints, conns)
        rest_len = bone_length_series(sk, joints[None])[0]  # [E]
        base = dict(asset_idx=cache_idx, source_idx=int(sl["idx"]),
                    name=str(sl.get("name", "")),
                    joints_rest=sk.joints.astype(np.float32),
                    edges=sk.edges.astype(np.int64),
                    parents=sk.parents.astype(np.int64),
                    bone_lengths=sk.bone_len.astype(np.float32),
                    joint_count=J, mesh_vertex_count=int(sl.get("mesh_verts", 0)),
                    fps=args.fps, num_frames=T)

        for preset in VALID_PRESETS:
            rv, rt, moving, amp, freq = preset_rotvec(sk, preset, T, args.fps, rng)
            world = fk(sk, rv, rt)
            bl = bone_length_series(sk, world)  # [T,E]
            dev = float(np.abs(bl - rest_len[None]).max()) if bl.size else 0.0
            max_len_dev_valid = max(max_len_dev_valid, dev)
            rec = dict(base, clip_id=f"a{ai:04d}_{preset}", preset=preset,
                       joints_world=world.astype(np.float32),
                       local_rot_quat=quats_from_rotvec(rv),
                       moving_joint_ids=np.array(moving, dtype=np.int64),
                       motion_amplitude=float(amp), motion_frequency=float(freq),
                       is_valid_motion=True, corruption_type="")
            save_clip(out_dir, rec); n_valid += 1
            index.append(dict(clip_id=rec["clip_id"], preset=preset, valid=True,
                              asset_idx=cache_idx, joints=J, max_bone_dev=dev))

        for ci, (ctype, params) in enumerate(CORRUPTIONS):
            world, rv, moving, amp, freq = make_corruption(sk, ctype, params, T, args.fps, rng)
            bl = bone_length_series(sk, world)
            dev = float(np.abs(bl - rest_len[None]).max()) if bl.size else 0.0
            rec = dict(base, clip_id=f"a{ai:04d}_corrupt{ci}_{ctype}",
                       preset=f"corrupt:{ctype}",
                       joints_world=world.astype(np.float32),
                       local_rot_quat=quats_from_rotvec(rv),
                       moving_joint_ids=np.array(moving, dtype=np.int64),
                       motion_amplitude=float(amp), motion_frequency=float(freq),
                       is_valid_motion=False, corruption_type=ctype)
            save_clip(out_dir, rec); n_corrupt += 1
            index.append(dict(clip_id=rec["clip_id"], preset=f"corrupt:{ctype}",
                              valid=False, corruption_type=ctype,
                              asset_idx=cache_idx, joints=J, max_bone_dev=dev))

        if (ai + 1) % 10 == 0:
            print(f"  {ai+1}/{len(pick)} assets  valid={n_valid} corrupt={n_corrupt}", flush=True)

    summary = dict(
        n_assets=len(pick), n_valid=n_valid, n_corrupt=n_corrupt,
        fps=args.fps, num_frames=T, seconds=args.seconds,
        valid_presets=VALID_PRESETS,
        corruption_types=sorted(set(c for c, _ in CORRUPTIONS)),
        max_bone_length_deviation_valid=max_len_dev_valid,
        bone_length_preserved_valid=bool(max_len_dev_valid < 1e-4),
    )
    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n== Summary ==", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nSaved -> {out_dir}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
