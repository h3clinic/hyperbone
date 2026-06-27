"""Diagnostic: check whether GT root and GT non-root nodes share identical
positions in the fixed overfit batch. If so, a position-only root head cannot
separate them, which caps root F1 below the sanity gate.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.datasets.anymate_static_dataset import AnymateStaticRigDataset


def main() -> None:
    pt = "datasets/anymate/Anymate_test.pt"
    splits_dir = "outputs/anymate_local_dev/splits"
    ds = AnymateStaticRigDataset(pt, f"{splits_dir}/train.jsonl", max_joints=128, pc_points=1024)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, drop_last=True)
    batch = next(iter(loader))

    joint_pos = batch["joint_pos"]
    active = batch["joint_active"] > 0.5
    root = batch["root_mask"] > 0.5
    parent = batch["parent_index"].long()

    B = joint_pos.shape[0]
    total_collisions = 0
    near_collisions = 0
    for b in range(B):
        valid = torch.where(active[b])[0]
        pos = joint_pos[b]
        for i_idx in range(len(valid)):
            for j_idx in range(i_idx + 1, len(valid)):
                i = int(valid[i_idx].item())
                j = int(valid[j_idx].item())
                d = float(torch.norm(pos[i] - pos[j]).item())
                conflicting = bool(root[b, i].item()) != bool(root[b, j].item())
                if not conflicting:
                    continue
                if d < 1e-6:
                    total_collisions += 1
                    print(f"[EXACT] batch={b} nodes=({i},{j}) dist={d:.2e} "
                          f"root=({int(root[b,i].item())},{int(root[b,j].item())}) "
                          f"parent=({int(parent[b,i].item())},{int(parent[b,j].item())})")
                elif d < 1e-3:
                    near_collisions += 1
                    print(f"[NEAR ] batch={b} nodes=({i},{j}) dist={d:.2e} "
                          f"root=({int(root[b,i].item())},{int(root[b,j].item())})")

    print()
    print(f"Exact-position conflicting root/non-root pairs: {total_collisions}")
    print(f"Near-position (<1e-3) conflicting pairs:        {near_collisions}")

    # Also report per-batch root counts.
    for b in range(B):
        n_active = int(active[b].sum().item())
        n_root = int((root[b] & active[b]).sum().item())
        print(f"batch={b} active={n_active} roots={n_root}")

    # Rank non-root nodes by distance to nearest GT-root node (across all batches).
    # The persistent root FP is likely a non-root node sitting very close to a root.
    print()
    print("Closest non-root -> nearest GT-root distances (smallest 15):")
    rows = []
    for b in range(B):
        valid = torch.where(active[b])[0]
        root_idx = [int(i.item()) for i in valid if bool(root[b, i].item())]
        nonroot_idx = [int(i.item()) for i in valid if not bool(root[b, i].item())]
        if not root_idx or not nonroot_idx:
            continue
        pos = joint_pos[b]
        root_pos = pos[root_idx]
        for j in nonroot_idx:
            d = float(torch.cdist(pos[j].unsqueeze(0), root_pos).min().item())
            rows.append((d, b, j))
    rows.sort(key=lambda r: r[0])
    for d, b, j in rows[:15]:
        print(f"batch={b} nonroot_node={j} dist_to_nearest_root={d:.4e}")


if __name__ == "__main__":
    main()

