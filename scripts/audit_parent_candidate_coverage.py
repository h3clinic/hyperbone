from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.datasets.anymate_static_dataset import AnymateStaticRigDataset
from hyperbone.rigs.parent_candidates import build_parent_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit GT parent coverage under kNN candidate restriction")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument("--points-per-sample", type=int, default=1024)
    parser.add_argument("--ks", default="4,8,12,16")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output", "--out", dest="output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = AnymateStaticRigDataset(args.pt, f"{args.splits_dir}/{args.split}.jsonl", max_joints=args.max_nodes, pc_points=args.points_per_sample)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]
    rows = []

    for k in ks:
        active_children = 0
        nonroot_children = 0
        covered = 0
        forced = 0
        candidate_sum = 0.0
        root_sum = 0.0
        distance_ranks = []
        batch_count = 0

        for batch in loader:
            if args.max_batches is not None and batch_count >= args.max_batches:
                break
            batch = {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in batch.items()}
            batch_count += 1
            candidate_info = build_parent_candidates(
                batch["joint_pos"],
                batch["joint_active"] > 0.5,
                batch["parent_index"].long(),
                k=k,
                include_root=True,
                force_gt_parent=False,
            )
            active_mask = batch["joint_active"] > 0.5
            root_mask = batch["root_mask"] > 0.5
            parent_index = batch["parent_index"].long()

            active_children += int(active_mask.sum().item())
            nonroot_children += int(((active_mask > 0.5) & (~root_mask)).sum().item())
            covered += int(candidate_info["gt_parent_in_candidates"].sum().item())
            forced += int(candidate_info["forced_gt_parent"].sum().item())
            candidate_sum += float(candidate_info["candidate_mask"].sum(dim=-1).float().mean().item())
            root_sum += float(candidate_info["candidate_mask"][:, :, -1].float().mean().item())

            for b in range(batch["joint_pos"].shape[0]):
                active_b = active_mask[b]
                for child in torch.where(active_b)[0].tolist():
                    if bool(root_mask[b, child].item()):
                        continue
                    gt_parent = int(parent_index[b, child].item())
                    if gt_parent < 0 or gt_parent == child or not bool(active_b[gt_parent].item()):
                        continue
                    pool = torch.where(active_b)[0]
                    pool = pool[pool != child]
                    if pool.numel() == 0:
                        continue
                    dists = torch.norm(batch["joint_pos"][b, pool] - batch["joint_pos"][b, child].unsqueeze(0), dim=-1)
                    order = torch.argsort(dists, descending=False)
                    rank = int((pool[order] == gt_parent).nonzero(as_tuple=True)[0].item()) + 1
                    distance_ranks.append(rank)

        ranks = np.asarray(distance_ranks, dtype=np.float32)
        rows.append(
            {
                "k": k,
                "gt_parent_candidate_coverage": covered / max(nonroot_children, 1),
                "forced_inclusion_rate": forced / max(nonroot_children, 1),
                "avg_candidate_count": candidate_sum / max(len(loader), 1),
                "avg_root_candidate_count": root_sum / max(len(loader), 1),
                "mean_parent_distance_rank": float(ranks.mean()) if ranks.size else 0.0,
                "median_parent_distance_rank": float(np.median(ranks)) if ranks.size else 0.0,
                "p90_parent_distance_rank": float(np.percentile(ranks, 90)) if ranks.size else 0.0,
            }
        )

    print(json.dumps(rows, indent=2))
    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()