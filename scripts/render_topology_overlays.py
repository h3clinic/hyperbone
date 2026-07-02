"""Track B: qualitative topology overlays for the result card.

Renders skeleton overlays for selected test samples: joints as dots,
edges classified as TP (correct), FN (missed GT), FP (extra predicted).
Reproducible from the committed checkpoint.

Usage:
    python scripts/render_topology_overlays.py \
        --cache outputs/models/hyperbone_track_b_student/cache_test_v11.pt \
        --ckpt  outputs/models/hyperbone_track_b_student_v23B/best_student_v23.pt \
        --indices 20,140,487,513 \
        --out outputs/models/hyperbone_track_b_student_v23B/overlays
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.models.geometry_edge_graph_student import (
    GeometryEdgeGraphStudent,
    build_line_graph,
)
from hyperbone.rigs.undirected_topology import edge_prf


@dataclass
class EdgeCandidate:
    i: int
    j: int
    dist: float
    score: float


def kruskal_mst_from_scores(n_nodes, active_nodes, edges, max_edges=None):
    if max_edges is None:
        max_edges = max(len(active_nodes) - 1, 0)
    edge_mask = torch.zeros(n_nodes, n_nodes, dtype=torch.bool)
    if len(active_nodes) <= 1 or max_edges <= 0:
        return edge_mask
    parent = {n: n for n in active_nodes}
    rank = {n: 0 for n in active_nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    selected = 0
    for e in sorted(edges, key=lambda x: (x.score, -x.dist), reverse=True):
        if selected >= max_edges:
            break
        if find(e.i) == find(e.j):
            continue
        if union(e.i, e.j):
            edge_mask[e.i, e.j] = True
            edge_mask[e.j, e.i] = True
            selected += 1
    return edge_mask


@torch.no_grad()
def predict_mask(model, sample, device):
    pi = sample["patch_i"].to(device)
    pj = sample["patch_j"].to(device)
    co = sample["corridor"].to(device)
    gf = sample["geom_feats"].to(device)
    lg = build_line_graph(sample["edge_pairs"]).to(device)
    scores = model(pi, pj, co, gf, lg).cpu()

    active_mask = sample["active_mask"]
    active_nodes = torch.where(active_mask)[0].tolist()
    n_nodes = active_mask.shape[0]
    jp = sample["joint_pos"]
    pairs = sample["edge_pairs"]
    cands = []
    for e in range(sample["n_edges"]):
        i, j = int(pairs[e, 0]), int(pairs[e, 1])
        d = float(torch.norm(jp[i] - jp[j]))
        cands.append(EdgeCandidate(i=i, j=j, dist=d, score=float(scores[e])))
    return kruskal_mst_from_scores(n_nodes, active_nodes, cands)


def set_equal_aspect(ax, pts):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ctr = (mins + maxs) / 2
    r = (maxs - mins).max() / 2 + 1e-6
    ax.set_xlim(ctr[0]-r, ctr[0]+r)
    ax.set_ylim(ctr[1]-r, ctr[1]+r)
    ax.set_zlim(ctr[2]-r, ctr[2]+r)


def render_case(ax, sample, pred_mask, title):
    am = sample["active_mask"]
    active = torch.where(am)[0].tolist()
    jp = sample["joint_pos"].numpy()
    gt = sample["gt_adj"].bool()
    pred = pred_mask.bool()
    pts = jp[active]

    # joints
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="#333333", s=18, depthshade=False)

    def draw(i, j, color, style, lw):
        ax.plot([jp[i, 0], jp[j, 0]], [jp[i, 1], jp[j, 1]],
                [jp[i, 2], jp[j, 2]], color=color, linestyle=style, linewidth=lw)

    tp = fn = fp = 0
    for a_idx in range(len(active)):
        for b_idx in range(a_idx + 1, len(active)):
            i, j = active[a_idx], active[b_idx]
            g, p = bool(gt[i, j]), bool(pred[i, j])
            if g and p:
                draw(i, j, "#1a9850", "-", 2.2); tp += 1      # correct
            elif g and not p:
                draw(i, j, "#d73027", "--", 2.0); fn += 1     # missed GT
            elif p and not g:
                draw(i, j, "#fc8d59", "-", 1.6); fp += 1      # extra pred
    set_equal_aspect(ax, pts)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=12, azim=-72)
    return tp, fn, fp


def main():
    parser = argparse.ArgumentParser(description="Render Track B topology overlays")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--indices", default="20,140,487,513")
    parser.add_argument("--edge-dim", type=int, default=128)
    parser.add_argument("--n-mp-rounds", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = GeometryEdgeGraphStudent(
        patch_in_channels=6, patch_out_dim=64, geom_feat_dim=16,
        edge_dim=args.edge_dim, n_mp_rounds=args.n_mp_rounds, dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()

    cached = torch.load(args.cache, map_location="cpu", weights_only=False)
    indices = [int(x) for x in args.indices.split(",")]

    labels = {0: "good", 1: "borderline", 2: "failure (hub)", 3: "failure (large)"}
    fig = plt.figure(figsize=(14, 12))
    for k, idx in enumerate(indices):
        sample = cached[idx]
        pred_mask = predict_mask(model, sample, device)
        prf = edge_prf(pred_mask, sample["gt_adj"], sample["active_mask"])
        n_j = int(sample["active_mask"].sum())
        gt = sample["gt_adj"].bool()
        am = sample["active_mask"]
        maxdeg = int(gt[am][:, am].sum(dim=1).max())
        tag = labels.get(k, "")
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        tp, fn, fp = render_case(
            ax, sample, pred_mask,
            f"[{tag}] idx {idx} | {n_j} joints, max-deg {maxdeg} | F1={prf['f1']:.2f}")
        print(f"idx {idx} ({tag}): F1={prf['f1']:.3f}  TP={tp} FN={fn} FP={fp}", flush=True)

        # also save an individual figure
        fig1 = plt.figure(figsize=(6, 6))
        ax1 = fig1.add_subplot(111, projection="3d")
        render_case(ax1, sample, pred_mask,
                    f"[{tag}] idx {idx} | {n_j} joints | F1={prf['f1']:.2f}")
        fig1.tight_layout()
        fig1.savefig(out_dir / f"overlay_{k}_{tag.split()[0]}_idx{idx}.png", dpi=130)
        plt.close(fig1)

    # legend on the combined figure
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="#1a9850", lw=2.2, label="correct (TP)"),
        Line2D([0], [0], color="#d73027", lw=2.0, ls="--", label="missed GT (FN)"),
        Line2D([0], [0], color="#fc8d59", lw=1.6, label="extra predicted (FP)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#333333",
               markersize=6, label="joint (known node)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Track B — geometry-only topology overlays (test set, seed-0 checkpoint)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_dir / "overlays_grid.png", dpi=130)
    plt.close(fig)
    print(f"\nSaved overlays -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
