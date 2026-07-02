"""Track B: solid character mesh + skeleton.

Renders the actual source triangle mesh (mesh_face over mesh_pc) as a
translucent surface with the skeleton inside, so you can see the real
character the skeleton came from. Maps cache->source by exact joint match.

Usage:
    python scripts/render_character_mesh.py \
        --cache outputs/models/hyperbone_track_b_student/cache_test_v11.pt \
        --ckpt  outputs/models/hyperbone_track_b_student_v23B/best_student_v23.pt \
        --source datasets/anymate/Anymate_test.pt \
        --split outputs/anymate_local_dev/splits/test.jsonl \
        --indices 20,140,487,513 --grid-cols 2 \
        --out outputs/models/hyperbone_track_b_student_v23B/overlays
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.models.geometry_edge_graph_student import (
    GeometryEdgeGraphStudent, build_line_graph,
)
from hyperbone.rigs.undirected_topology import edge_prf


@dataclass
class EC:
    i: int
    j: int
    dist: float
    score: float


def kruskal(n_nodes, active, edges):
    max_edges = max(len(active) - 1, 0)
    mask = torch.zeros(n_nodes, n_nodes, dtype=torch.bool)
    if len(active) <= 1:
        return mask
    parent = {n: n for n in active}; rnk = {n: 0 for n in active}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rnk[ra] < rnk[rb]:
            parent[ra] = rb
        elif rnk[ra] > rnk[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra; rnk[ra] += 1
        return True

    sel = 0
    for e in sorted(edges, key=lambda x: (x.score, -x.dist), reverse=True):
        if sel >= max_edges:
            break
        if find(e.i) == find(e.j):
            continue
        if union(e.i, e.j):
            mask[e.i, e.j] = True; mask[e.j, e.i] = True; sel += 1
    return mask


@torch.no_grad()
def predict(model, sample, device):
    pi = sample["patch_i"].to(device); pj = sample["patch_j"].to(device)
    co = sample["corridor"].to(device); gf = sample["geom_feats"].to(device)
    lg = build_line_graph(sample["edge_pairs"]).to(device)
    scores = model(pi, pj, co, gf, lg).cpu()
    am = sample["active_mask"]; active = torch.where(am)[0].tolist()
    jp = sample["joint_pos"]; pairs = sample["edge_pairs"]
    cands = [EC(int(pairs[e, 0]), int(pairs[e, 1]),
               float(torch.norm(jp[int(pairs[e, 0])] - jp[int(pairs[e, 1])])),
               float(scores[e])) for e in range(sample["n_edges"])]
    return kruskal(am.shape[0], active, cands)


def equal_aspect(ax, pts):
    mins, maxs = pts.min(0), pts.max(0)
    ctr = (mins + maxs) / 2; r = (maxs - mins).max() / 2 + 1e-6
    ax.set_xlim(ctr[0]-r, ctr[0]+r); ax.set_ylim(ctr[1]-r, ctr[1]+r)
    ax.set_zlim(ctr[2]-r, ctr[2]+r)


def add_mesh(ax, verts, faces, max_faces, alpha):
    if faces.shape[0] > max_faces:
        sub = np.random.RandomState(0).choice(faces.shape[0], max_faces, replace=False)
        faces = faces[sub]
    tris = verts[faces]  # [F,3,3]
    # simple normal-based shading
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True); ln[ln == 0] = 1
    n = n / ln
    ls = np.array([0.3, 0.4, 0.85]); ls = ls / np.linalg.norm(ls)
    shade = np.clip(np.abs(n @ ls), 0.25, 1.0)
    base = np.array([0.62, 0.69, 0.78])
    cols = np.clip(base[None, :] * (0.55 + 0.6 * shade[:, None]), 0, 1)
    facecolors = np.concatenate([cols, np.full((cols.shape[0], 1), alpha)], axis=1)
    coll = Poly3DCollection(tris, facecolors=facecolors, edgecolors="none")
    ax.add_collection3d(coll)


def render(ax, sample, verts, faces, pred, max_faces, mesh_alpha):
    add_mesh(ax, verts, faces, max_faces, mesh_alpha)
    am = sample["active_mask"]; active = torch.where(am)[0].tolist()
    jp = sample["joint_pos"].numpy()
    gt = sample["gt_adj"].bool(); pm = pred.bool() if pred is not None else None
    pts = jp[active]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="#0b0b0b", s=16, depthshade=False)

    def draw(i, j, c, ls_, lw):
        ax.plot([jp[i, 0], jp[j, 0]], [jp[i, 1], jp[j, 1]], [jp[i, 2], jp[j, 2]],
                color=c, linestyle=ls_, linewidth=lw)
    for a in range(len(active)):
        for b in range(a + 1, len(active)):
            i, j = active[a], active[b]
            g = bool(gt[i, j])
            if pm is None:
                if g: draw(i, j, "#128a3a", "-", 2.6)
            else:
                p = bool(pm[i, j])
                if g and p: draw(i, j, "#0f8f3a", "-", 2.8)
                elif g and not p: draw(i, j, "#d73027", "--", 2.4)
                elif p and not g: draw(i, j, "#fc8d59", "-", 2.0)
    equal_aspect(ax, verts)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=12, azim=-72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--indices", default="20,140,487,513")
    ap.add_argument("--gt-only", action="store_true")
    ap.add_argument("--plain", action="store_true", help="suppress good/borderline case labels")
    ap.add_argument("--grid-cols", type=int, default=2)
    ap.add_argument("--mesh-alpha", type=float, default=0.45)
    ap.add_argument("--max-faces", type=int, default=40000)
    ap.add_argument("--edge-dim", type=int, default=128)
    ap.add_argument("--n-mp-rounds", type=int, default=3)
    ap.add_argument("--tag", default="character_mesh")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    model = GeometryEdgeGraphStudent(
        patch_in_channels=6, patch_out_dim=64, geom_feat_dim=16,
        edge_dim=args.edge_dim, n_mp_rounds=args.n_mp_rounds, dropout=0.0).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()

    cached = torch.load(args.cache, map_location="cpu", weights_only=False)
    with open(args.split) as f:
        split_lines = [json.loads(l) for l in f]
    print("Loading source meshes (large)...", flush=True)
    source = torch.load(args.source, map_location="cpu", weights_only=False)

    indices = [int(x) for x in args.indices.split(",")]
    labels = {} if args.plain else {0: "good", 1: "borderline", 2: "failure (hub)", 3: "failure (large)"}
    cols = args.grid_cols; rows = math.ceil(len(indices) / cols)
    fig = plt.figure(figsize=(5.4 * cols, 5.0 * rows))

    for k, idx in enumerate(indices):
        sample = cached[idx]; n_j = int(sample["active_mask"].sum())
        src_idx = split_lines[idx]["idx"]; name = split_lines[idx].get("name", "?")
        src = source[src_idx]
        cj = sample["joint_pos"][:n_j].numpy()
        sj = src["joints"][:src["joints_num"]].numpy()
        if not (sj.shape == cj.shape and np.allclose(sj, cj, atol=1e-4)):
            for sl in split_lines:
                cand = source[sl["idx"]]["joints"][:sl["joints_num"]].numpy()
                if cand.shape == cj.shape and np.allclose(cand, cj, atol=1e-4):
                    src_idx = sl["idx"]; name = sl.get("name", "?"); src = source[src_idx]; break
        verts = src["mesh_pc"][:, :3].numpy().astype(np.float32)
        faces = src["mesh_face"].numpy().astype(np.int64)
        faces = faces[(faces < verts.shape[0]).all(axis=1)]

        pred = None if args.gt_only else predict(model, sample, device)
        prf = None if pred is None else edge_prf(pred, sample["gt_adj"], sample["active_mask"])
        gt = sample["gt_adj"].bool(); am = sample["active_mask"]
        maxdeg = int(gt[am][:, am].sum(1).max())
        short = name.split("/")[-1][:10] if isinstance(name, str) else str(name)

        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        render(ax, sample, verts, faces, pred, args.max_faces, args.mesh_alpha)
        pfx = f"[{labels[k]}] " if k in labels else ""
        ttl = f"{pfx}{short} | {n_j} joints, max-deg {maxdeg}"
        if prf is not None:
            ttl += f" | F1={prf['f1']:.2f}"
        ax.set_title(ttl, fontsize=10)
        print(f"idx {idx} -> src {src_idx} ({short}) n_j={n_j} verts={verts.shape[0]} "
              f"faces={faces.shape[0]}"
              + ("" if prf is None else f" F1={prf['f1']:.3f}"), flush=True)

        fig1 = plt.figure(figsize=(6.5, 6.5))
        ax1 = fig1.add_subplot(111, projection="3d")
        render(ax1, sample, verts, faces, pred, args.max_faces, args.mesh_alpha)
        ax1.set_title(ttl, fontsize=11)
        fig1.tight_layout()
        lbl = labels[k].split()[0] if k in labels else "char"
        fig1.savefig(out_dir / f"{args.tag}_{k}_{lbl}_idx{idx}.png", dpi=135)
        plt.close(fig1)

    legend = [
        Line2D([0],[0], marker='s', color='w', markerfacecolor="#9aa7be", markersize=10, label="character mesh"),
        Line2D([0],[0], color="#0f8f3a", lw=2.8, label="correct (TP)"),
        Line2D([0],[0], color="#d73027", lw=2.4, ls="--", label="missed GT (FN)"),
        Line2D([0],[0], color="#fc8d59", lw=2.0, label="extra predicted (FP)"),
        Line2D([0],[0], marker='o', color='w', markerfacecolor="#0b0b0b", markersize=6, label="joint"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=len(legend), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.004))
    fig.suptitle("Track B — solid character mesh + skeleton (test set, seed-0)", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_dir / f"{args.tag}_grid.png", dpi=125)
    plt.close(fig)
    print(f"\nSaved -> {out_dir / (args.tag + '_grid.png')}", flush=True)


if __name__ == "__main__":
    main()
