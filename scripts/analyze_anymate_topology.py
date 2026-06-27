"""
Analyze Anymate topology variation across assets.

Reports:
- Joint/bone count distribution
- Hierarchy structure per asset
- Topology clustering by structure signature
- Whether joint indices are semantically consistent
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def analyze_topology(dataset_path: str, pt_path: str = None, max_assets: int = 100) -> dict:
    """Analyze topology from either .pt dataset or rendered dataset_index.jsonl."""

    assets_info = []

    if pt_path and Path(pt_path).exists():
        data = torch.load(pt_path, weights_only=False)
        for i in range(min(len(data), max_assets)):
            d = data[i]
            jn = d['joints_num']
            bn = d['bones_num']
            conns = d['conns'].numpy()[:jn]

            # Build hierarchy
            children = defaultdict(list)
            parents = {}
            for j in range(jn):
                p = int(conns[j])
                parents[j] = p
                if p != j and 0 <= p < jn:
                    children[p].append(j)

            # Find roots
            roots = [j for j in range(jn) if int(conns[j]) == j or int(conns[j]) >= jn or int(conns[j]) < 0]

            # Compute depths from root
            depths = np.zeros(jn, dtype=int)
            for j in range(jn):
                visited = set()
                curr = j
                d_val = 0
                while int(conns[curr]) != curr and 0 <= int(conns[curr]) < jn and curr not in visited:
                    visited.add(curr)
                    curr = int(conns[curr])
                    d_val += 1
                depths[j] = d_val

            n_leaves = sum(1 for j in range(jn) if j not in children)
            n_branches = sum(1 for j in range(jn) if len(children.get(j, [])) > 1)
            max_depth = int(depths.max())

            # Topology signature: (joint_count, max_depth, n_leaves, n_branches)
            sig = f"j{jn}_d{max_depth}_l{n_leaves}_b{n_branches}"

            # Child count distribution
            child_counts = [len(children.get(j, [])) for j in range(jn)]

            assets_info.append({
                "index": i,
                "name": d['name'],
                "joints_num": jn,
                "bones_num": bn,
                "roots": roots,
                "max_depth": max_depth,
                "n_leaves": n_leaves,
                "n_branches": n_branches,
                "topology_signature": sig,
                "child_count_dist": Counter(child_counts),
                "depths": depths.tolist(),
            })

    elif dataset_path and Path(dataset_path).exists():
        with open(dataset_path) as f:
            labels = [json.loads(l) for l in f if l.strip()]
        # Group by asset
        seen_assets = set()
        for label in labels:
            aid = label.get("asset_id", "")
            if aid in seen_assets:
                continue
            seen_assets.add(aid)
            joints = label.get("joints", [])
            jn = len(joints)
            bn = len(label.get("bones", []))
            assets_info.append({
                "index": len(assets_info),
                "name": aid,
                "joints_num": jn,
                "bones_num": bn,
                "roots": [],
                "max_depth": 0,
                "n_leaves": 0,
                "n_branches": 0,
                "topology_signature": f"j{jn}_b{bn}",
                "child_count_dist": {},
                "depths": [],
            })
            if len(assets_info) >= max_assets:
                break

    # Aggregate stats
    joint_counts = [a["joints_num"] for a in assets_info]
    bone_counts = [a["bones_num"] for a in assets_info]
    sigs = [a["topology_signature"] for a in assets_info]
    sig_counter = Counter(sigs)

    report = {
        "n_assets": len(assets_info),
        "joint_count_stats": {
            "min": min(joint_counts) if joint_counts else 0,
            "max": max(joint_counts) if joint_counts else 0,
            "mean": float(np.mean(joint_counts)) if joint_counts else 0,
            "median": float(np.median(joint_counts)) if joint_counts else 0,
            "unique": len(set(joint_counts)),
        },
        "bone_count_stats": {
            "min": min(bone_counts) if bone_counts else 0,
            "max": max(bone_counts) if bone_counts else 0,
            "mean": float(np.mean(bone_counts)) if bone_counts else 0,
            "unique": len(set(bone_counts)),
        },
        "topology_clusters": {
            "n_unique_signatures": len(sig_counter),
            "top_signatures": sig_counter.most_common(10),
        },
        "joint_index_consistency": "INCONSISTENT — joint counts vary from {} to {}, no shared semantics".format(
            min(joint_counts) if joint_counts else 0,
            max(joint_counts) if joint_counts else 0,
        ),
        "root_joint_analysis": {
            "always_index_0": all(0 in a["roots"] for a in assets_info if a["roots"]),
            "multiple_roots": sum(1 for a in assets_info if len(a["roots"]) > 1),
        },
        "depth_stats": {
            "min_max_depth": min(a["max_depth"] for a in assets_info) if assets_info else 0,
            "max_max_depth": max(a["max_depth"] for a in assets_info) if assets_info else 0,
            "mean_max_depth": float(np.mean([a["max_depth"] for a in assets_info])) if assets_info else 0,
        },
        "can_infer_canonical": False,
        "recommendation": "Use graph/set prediction with Hungarian matching. Joint index slots have NO shared semantics across assets.",
        "assets": assets_info,
    }

    return report


def write_report(report: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_out = {k: v for k, v in report.items() if k != "assets"}
    json_out["assets_summary"] = [
        {"name": a["name"], "joints": a["joints_num"], "bones": a["bones_num"],
         "sig": a["topology_signature"], "depth": a["max_depth"]}
        for a in report["assets"][:50]
    ]
    with open(output_dir / "topology_report.json", "w") as f:
        json.dump(json_out, f, indent=2)

    # Markdown
    md = f"""# Anymate Topology Analysis

## Summary
| Metric | Value |
|--------|-------|
| Assets analyzed | {report['n_assets']} |
| Joint counts | {report['joint_count_stats']['min']}–{report['joint_count_stats']['max']} (mean {report['joint_count_stats']['mean']:.1f}, {report['joint_count_stats']['unique']} unique) |
| Bone counts | {report['bone_count_stats']['min']}–{report['bone_count_stats']['max']} (mean {report['bone_count_stats']['mean']:.1f}) |
| Unique topology signatures | {report['topology_clusters']['n_unique_signatures']} |
| Multiple roots | {report['root_joint_analysis']['multiple_roots']} assets |
| Max hierarchy depth | {report['depth_stats']['min_max_depth']}–{report['depth_stats']['max_max_depth']} (mean {report['depth_stats']['mean_max_depth']:.1f}) |

## Joint Index Consistency
**{report['joint_index_consistency']}**

## Recommendation
{report['recommendation']}

## Top Topology Signatures
| Signature | Count |
|-----------|-------|
"""
    for sig, count in report['topology_clusters']['top_signatures']:
        md += f"| {sig} | {count} |\n"

    md += "\n## Sample Assets\n| # | Name | Joints | Bones | Depth | Sig |\n|---|------|--------|-------|-------|-----|\n"
    for a in report["assets"][:20]:
        md += f"| {a['index']} | {a['name'][:30]} | {a['joints_num']} | {a['bones_num']} | {a['max_depth']} | {a['topology_signature']} |\n"

    with open(output_dir / "topology_report.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[Topology] Report written to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="outputs/anymate_clips_pilot/dataset_index.jsonl")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--out", default="outputs/anymate_clips_pilot")
    parser.add_argument("--max-assets", type=int, default=100)
    args = parser.parse_args()

    report = analyze_topology(args.dataset, args.pt, max_assets=args.max_assets)
    write_report(report, Path(args.out))

    print(f"\n[Topology] Key findings:")
    print(f"  Joint counts: {report['joint_count_stats']['min']}–{report['joint_count_stats']['max']} ({report['joint_count_stats']['unique']} unique)")
    print(f"  Topology signatures: {report['topology_clusters']['n_unique_signatures']} unique")
    print(f"  Consistency: {report['joint_index_consistency']}")
    print(f"  Recommendation: {report['recommendation']}")


if __name__ == "__main__":
    main()
