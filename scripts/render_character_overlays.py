"""Track B: character-mesh + skeleton overlays.

Renders the actual source point cloud under the skeleton so the shapes
behind the topology results can be characterized. Maps each geometry-cache
sample back to its source Anymate mesh via the test split ordering
(verified by exact joint match).

Two modes:
  * result cases (default): GT vs predicted edges over mesh for given --indices
  * gallery (--gallery N): N characters spanning the joint-count range,
    GT skeleton over mesh, to characterize dataset variety (--gt-only)

Usage:
    python scripts/render_character_overlays.py \
        --cache outputs/models/hyperbone_track_b_student/cache_test_v11.pt \
        --ckpt  outputs/models/hyperbone_track_b_student_v23B/best_student_v23.pt \
        --source datasets/anymate/Anymate_test.pt \
        --split outputs/anymate_local_dev/splits/test.jsonl \
        --gallery 9 --gt-only --grid-cols 3 --tag gallery \
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


def kruskal(n_nodes, active_nodes, edges):
    max_edges = max(len(active_nodes) - 1, 0)
    mask = torch.zeros(n_nodes, n_nodes, dtype=torch.bool)
    if len(active_nodes) <= 1:
        return mask
    parent = {n: n for n in active_nodes}
    rnk = {n: 0 for n in active_nodes}

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


def surface_pts(src, max_pts):
    pc = src.get("pc") if "pc" in src else None
    arr = (pc if pc is not None else src["mesh_pc"])[:, :3].numpy()
    if arr.shape[0] > max_pts:
        sub = np.random.RandomState(0).choice(arr.shape[0], max_pts, replace=False)
        arr = arr[sub]
    return arr


def render_panel(ax, sample, src, pred, gt_only, max_pts):
    mesh = surface_pts(src, max_pts)
    am = sample["active_mask"]; active = torch.where(am)[0].tolist()
    jp = sample["joint_pos"].numpy()
    gt = sample["gt_adj"].bool()
    pm = pred.bool() if pred is not None else None

    ax.scatter(mesh[:, 0], mesh[:, 1], mesh[:, 2], c="#9aa7b3", s=3,
               alpha=0.25, depthshade=False, linewidths=0)
    pts = jp[active]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="#111111", s=12, depthshade=False)

    def draw(i, j, c, ls, lw):
        ax.plot([jp[i, 0], jp[j, 0]], [jp[i, 1], jp[j, 1]], [jp[i, 2], jp[j, 2]],
                color=c, linestyle=ls, linewidth=lw)
    for a in range(len(active)):
        for b in range(a + 1, len(active)):
            i, j = active[a], active[b]
            g = bool(gt[i, j])
            if gt_only:
                if g:
                    draw(i, j, "#1a9850", "-", 2.2)
            else:
                p = bool(pm[i, j])
                if g and p: draw(i, j, "#1a9850", "-", 2.4)
                elif g and not p: draw(i, j, "#d73027", "--", 2.2)
                elif p and not g: draw(i, j, "#fc8d59", "-", 1.8)
    equal_aspect(ax, mesh)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=12, azim=-72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--indices", default="20,140,487,513")
    ap.add_argument("--gallery", type=int, default=0,
                    help="if >0, auto-select N cache samples spanning joint-count range")
    ap.add_argument("--gt-only", action="store_true")
    ap.add_argument("--grid-cols", type=int, default=2)
    ap.add_argument("--edge-dim", type=int, default=128)
    ap.add_argument("--n-mp-rounds", type=int, default=3)
    ap.add_argument("--mesh-points", type=int, default=6000)
    ap.add_argument("--tag", default="characters")
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
    print(f"cache={len(cached)}  split_lines={len(split_lines)}", flush=True)

    labels = {}
    if args.gallery > 0:
        njs = [(i, int(s["active_mask"].sum())) for i, s in enumerate(cached)]
        njs.sort(key=lambda x: x[1])
        pick = [njs[round(t * (len(njs) - 1) / (args.gallery - 1))][0]
                for t in range(args.gallery)]
        indices = pick
    else:
        indices = [int(x) for x in args.indices.split(",")]
        labels = {0: "good", 1: "borderline", 2: "failure (hub)", 3: "failure (large)"}

    print("Loading source meshes (large)...", flush=True)
    source = torch.load(args.source, map_location="cpu", weights_only=False)

    cols = args.grid_cols
    rows = math.ceil(len(indices) / cols)
    fig = plt.figure(figsize=(5.0 * cols, 4.6 * rows))
    for k, idx in enumerate(indices):
        sample = cached[idx]
        n_j = int(sample["active_mask"].sum())
        src_idx = split_lines[idx]["idx"]; name = split_lines[idx].get("name", "?")
        src = source[src_idx]
        cj = sample["joint_pos"][:n_j].numpy()
        sj = src["joints"][:src["joints_num"]].numpy()
        match = sj.shape == cj.shape and np.allclose(sj, cj, atol=1e-4)
        if not match:
            for sl in split_lines:
                cand = source[sl["idx"]]["joints"][:sl["joints_num"]].numpy()
                if cand.shape == cj.shape and np.allclose(cand, cj, atol=1e-4):
                    src_idx = sl["idx"]; name = sl.get("name", "?"); src = source[src_idx]
                    match = True; break

        pred = None if args.gt_only else predict(model, sample, device)
        prf = None if pred is None else edge_prf(pred, sample["gt_adj"], sample["active_mask"])
        gt = sample["gt_adj"].bool(); am = sample["active_mask"]
        maxdeg = int(gt[am][:, am].sum(1).max())
        short = name.split("/")[-1][:10] if isinstance(name, str) else str(name)

        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        render_panel(ax, sample, src, pred, args.gt_only, args.mesh_points)
        if args.gt_only:
            ax.set_title(f"{short} | {n_j} joints, max-deg {maxdeg}", fontsize=9)
            print(f"idx {idx} -> src {src_idx} ({short}) match={match} n_j={n_j} maxdeg={maxdeg}", flush=True)
        else:
            ax.set_title(f"[{labels.get(k,'')}] {short} | {n_j} joints, "
                         f"max-deg {maxdeg} | F1={prf['f1']:.2f}", fontsize=9)
            print(f"idx {idx} -> src {src_idx} ({short}) match={match} n_j={n_j} "
                  f"F1={prf['f1']:.3f}", flush=True)

    if args.gt_only:
        legend = [
            Line2D([0],[0], marker='o', color='w', markerfacecolor="#9aa7b3", markersize=7, label="mesh surface"),
            Line2D([0],[0], color="#1a9850", lw=2.2, label="ground-truth skeleton"),
            Line2D([0],[0], marker='o', color='w', markerfacecolor="#111111", markersize=6, label="joint"),
        ]
        title = "Track B — character mesh + ground-truth skeleton (test set)"
    else:
        legend = [
            Line2D([0],[0], marker='o', color='w', markerfacecolor="#9aa7b3", markersize=7, label="mesh surface"),
            Line2D([0],[0], color="#1a9850", lw=2.4, label="correct (TP)"),
            Line2D([0],[0], color="#d73027", lw=2.2, ls="--", label="missed GT (FN)"),
            Line2D([0],[0], color="#fc8d59", lw=1.8, label="extra predicted (FP)"),
            Line2D([0],[0], marker='o', color='w', markerfacecolor="#111111", markersize=6, label="joint"),
        ]
        title = "Track B — character mesh + predicted vs GT skeleton (test set, seed-0)"
    fig.legend(handles=legend, loc="lower center", ncol=len(legend), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.004))
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_dir / f"{args.tag}_grid.png", dpi=125)
    plt.close(fig)
    print(f"\nSaved -> {out_dir / (args.tag + '_grid.png')}", flush=True)


if __name__ == "__main__":
    main()
