"""
Train HyperBone Anymate Graph Frame Model — topology-aware rig prediction.

Uses Hungarian matching to handle variable skeleton topology across assets.

Usage:
    python scripts/train_hyperbone_anymate_frame_graph.py \
        --dataset outputs/anymate_clips_pilot/train.jsonl \
        --val-dataset outputs/anymate_clips_pilot/val.jsonl \
        --out outputs/models/hyperbone_anymate_frame_graph_pilot \
        --epochs 30 --batch-size 8 --resolution 256 --max-nodes 128 \
        --device cuda --amp --workers 0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.models.hyperbone_rig_graph_frame import GraphTokenFrameModel
from hyperbone.rigs.graph_losses import compute_graph_loss, hungarian_match
from hyperbone.rigs.topology import (
    compute_adjacency, compute_joint_types, compute_rest_bone_lengths,
    N_JOINT_TYPES,
)


class AnymateGraphDataset(Dataset):
    """Dataset for graph-mode training. Provides adjacency + types + bone lengths."""

    def __init__(self, index_path: str, resolution: int = 256, max_nodes: int = 128):
        self.resolution = resolution
        self.max_nodes = max_nodes
        self.root_dir = Path(index_path).parent

        self.labels = []
        with open(index_path) as f:
            for line in f:
                if line.strip():
                    self.labels.append(json.loads(line))

        # Precompute per-asset topology (adjacency, types, bone lengths)
        self._precompute_topology()

    def _precompute_topology(self):
        """Compute adjacency/types from dataset labels (uses bones + joints)."""
        # Group by asset to avoid recomputation
        self.asset_topology = {}
        for label in self.labels:
            aid = label.get("asset_id", "")
            if aid in self.asset_topology:
                continue

            joints = label.get("joints", [])
            n_joints = len(joints)

            # Build parent array from bones
            bones = label.get("bones", [])
            # Infer connectivity: for each joint find closest bone endpoint
            # Since we don't have explicit parent indices in the rendered dataset,
            # approximate from bone start/end positions
            conns = np.zeros(n_joints, dtype=np.int32)
            joint_xyz = np.array([j.get("world_xyz", [0, 0, 0]) for j in joints])

            # Use bone data to infer parent-child relationships
            adj = np.zeros((n_joints, n_joints), dtype=np.float32)
            for bone in bones:
                start = np.array(bone.get("start_xyz", [0, 0, 0]))
                end = np.array(bone.get("end_xyz", [0, 0, 0]))
                # Find closest joints to start and end
                if n_joints > 0:
                    dist_start = np.linalg.norm(joint_xyz - start, axis=1)
                    dist_end = np.linalg.norm(joint_xyz - end, axis=1)
                    j_start = int(dist_start.argmin())
                    j_end = int(dist_end.argmin())
                    if j_start != j_end and dist_start[j_start] < 0.05 and dist_end[j_end] < 0.05:
                        adj[j_start, j_end] = 1.0
                        adj[j_end, j_start] = 1.0
                        conns[j_end] = j_start  # j_start is parent of j_end

            # Compute types and bone lengths
            types = compute_joint_types(conns, n_joints)
            bone_lengths = compute_rest_bone_lengths(joint_xyz, conns, n_joints)

            self.asset_topology[aid] = {
                "adj": adj,
                "types": types,
                "bone_lengths": bone_lengths,
                "conns": conns,
            }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        import cv2
        label = self.labels[idx]
        aid = label.get("asset_id", "")
        joints = label.get("joints", [])
        n_joints = len(joints)

        # Load image
        rgb_path = self.root_dir / label.get("rgb_path", "")
        mask_path = self.root_dir / label.get("mask_path", "")
        depth_path = self.root_dir / label.get("depth_path", "")

        rgb = self._load_rgb(rgb_path)
        mask = self._load_mask(mask_path)
        depth = self._load_depth(depth_path)

        # Joint targets (padded to max_nodes)
        gt_xyz = np.zeros((self.max_nodes, 3), dtype=np.float32)
        gt_xy = np.zeros((self.max_nodes, 2), dtype=np.float32)
        gt_active = np.zeros(self.max_nodes, dtype=np.float32)
        gt_vis = np.zeros(self.max_nodes, dtype=np.float32)

        for j in joints:
            ji = j.get("id", 0)
            if ji >= self.max_nodes:
                continue
            gt_active[ji] = 1.0
            gt_xyz[ji] = j.get("world_xyz", [0, 0, 0])[:3]
            gt_xy[ji] = j.get("image_xy", [0, 0])[:2]
            gt_vis[ji] = 1.0 if j.get("visible", False) else 0.0

        # Topology targets (padded)
        topo = self.asset_topology.get(aid, {})
        gt_adj = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
        gt_types = np.zeros(self.max_nodes, dtype=np.int64)
        gt_bone_lengths = np.zeros(self.max_nodes, dtype=np.float32)

        if topo:
            adj = topo["adj"]
            n = min(adj.shape[0], self.max_nodes)
            gt_adj[:n, :n] = adj[:n, :n]
            gt_types[:n] = topo["types"][:n]
            gt_bone_lengths[:n] = topo["bone_lengths"][:n]

        # Camera
        camera = label.get("camera", {})
        K = np.array(camera.get("K", np.eye(3)), dtype=np.float32)
        ext = np.array(camera.get("extrinsic", np.eye(4)), dtype=np.float32)

        return {
            "rgb": rgb,
            "mask": mask,
            "depth": depth,
            "gt_xyz": torch.from_numpy(gt_xyz),
            "gt_xy": torch.from_numpy(gt_xy),
            "gt_active": torch.from_numpy(gt_active),
            "gt_vis": torch.from_numpy(gt_vis),
            "gt_adj": torch.from_numpy(gt_adj),
            "gt_types": torch.from_numpy(gt_types),
            "gt_bone_lengths": torch.from_numpy(gt_bone_lengths),
            "camera_K": torch.from_numpy(K),
            "camera_ext": torch.from_numpy(ext),
        }

    def _load_rgb(self, path):
        import cv2
        if path.exists():
            img = cv2.imread(str(path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.resolution, self.resolution))
                return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        return torch.zeros(3, self.resolution, self.resolution)

    def _load_mask(self, path):
        import cv2
        if path.exists():
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (self.resolution, self.resolution))
                return torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        return torch.zeros(1, self.resolution, self.resolution)

    def _load_depth(self, path):
        if path.exists() and path.suffix == ".npy":
            d = np.load(str(path))
            import cv2
            d = cv2.resize(d, (self.resolution, self.resolution))
            valid = d[d > 0]
            if len(valid) > 0:
                d = d / (valid.max() + 1e-8)
            return torch.from_numpy(d.astype(np.float32)).unsqueeze(0)
        return torch.zeros(1, self.resolution, self.resolution)


def evaluate_graph(model, dataloader, device):
    """Quick eval: node precision/recall + matched MPJPE."""
    model.eval()
    all_mpjpe = []
    all_matched = 0
    all_gt_nodes = 0
    all_pred_active = 0

    with torch.no_grad():
        for batch in dataloader:
            x = torch.cat([batch["rgb"], batch["mask"], batch["depth"]], dim=1).to(device)
            pred = model(x)

            gt_xyz = batch["gt_xyz"].to(device)
            gt_active = batch["gt_active"].to(device)

            matches = hungarian_match(pred["node_xyz"], gt_xyz, gt_active, pred["node_active"])

            B = x.shape[0]
            for b in range(B):
                pred_idx, gt_idx = matches[b]
                n_gt = int((gt_active[b] > 0.5).sum().item())
                n_pred_active = int((pred["node_active"][b] > 0.5).sum().item())

                all_gt_nodes += n_gt
                all_pred_active += n_pred_active
                all_matched += len(pred_idx)

                if len(pred_idx) > 0:
                    errors = torch.norm(
                        pred["node_xyz"][b, pred_idx] - gt_xyz[b, gt_idx], dim=-1)
                    all_mpjpe.extend(errors.cpu().tolist())

    mpjpe = float(np.mean(all_mpjpe)) if all_mpjpe else 999.0
    precision = all_matched / max(all_pred_active, 1)
    recall = all_matched / max(all_gt_nodes, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    errors_arr = np.array(all_mpjpe) if all_mpjpe else np.array([999.0])
    pck_010 = float((errors_arr < 0.10).mean())
    pck_020 = float((errors_arr < 0.20).mean())

    return {
        "mpjpe": mpjpe,
        "pck_010": pck_010,
        "pck_020": pck_020,
        "node_precision": precision,
        "node_recall": recall,
        "node_f1": f1,
        "n_matched": all_matched,
        "n_gt": all_gt_nodes,
        "n_pred_active": all_pred_active,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--val-dataset", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[GraphTrain] Dataset: {args.dataset}")
    print(f"[GraphTrain] Output: {output_dir}")
    print(f"[GraphTrain] Device: {device}")

    train_ds = AnymateGraphDataset(args.dataset, resolution=args.resolution, max_nodes=args.max_nodes)
    print(f"[GraphTrain] Train: {len(train_ds)} samples")

    val_ds = None
    if args.val_dataset and Path(args.val_dataset).exists():
        val_ds = AnymateGraphDataset(args.val_dataset, resolution=args.resolution, max_nodes=args.max_nodes)
        print(f"[GraphTrain] Val: {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers) if val_ds else None

    model = GraphTokenFrameModel(
        in_channels=5, max_nodes=args.max_nodes,
        n_node_types=N_JOINT_TYPES, base_dim=64, n_query_layers=3,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[GraphTrain] Params: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    best_val_mpjpe = float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_pos = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            x = torch.cat([batch["rgb"], batch["mask"], batch["depth"]], dim=1).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=args.amp):
                pred = model(x)
                losses = compute_graph_loss(
                    pred,
                    gt_xyz=batch["gt_xyz"].to(device),
                    gt_active=batch["gt_active"].to(device),
                    gt_vis=batch["gt_vis"].to(device),
                    gt_adj=batch["gt_adj"].to(device),
                    gt_types=batch["gt_types"].to(device),
                    gt_bone_lengths=batch["gt_bone_lengths"].to(device),
                    gt_xy=batch["gt_xy"].to(device),
                    camera_K=batch["camera_K"].to(device),
                    camera_ext=batch["camera_ext"].to(device),
                )

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += losses["total"].item()
            epoch_pos += losses["pos_loss"]
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_pos = epoch_pos / max(n_batches, 1)
        elapsed = time.time() - t0

        log = f"[Epoch {epoch:3d}/{args.epochs}] loss={avg_loss:.4f} pos={avg_pos:.4f} time={elapsed:.1f}s"

        val_metrics = {}
        if val_loader:
            val_metrics = evaluate_graph(model, val_loader, device)
            log += f" | val_mpjpe={val_metrics['mpjpe']:.4f} f1={val_metrics['node_f1']:.3f} pck@0.20={val_metrics['pck_020']:.3f}"
            if val_metrics["mpjpe"] < best_val_mpjpe:
                best_val_mpjpe = val_metrics["mpjpe"]
                torch.save(model.state_dict(), output_dir / "best_model.pt")

        print(log)
        history.append({"epoch": epoch, "train_loss": avg_loss, "pos_loss": avg_pos, **val_metrics})

    # Save
    torch.save(model.state_dict(), output_dir / "model.pt")
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "max_nodes": args.max_nodes, "resolution": args.resolution,
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "param_count": param_count, "train_samples": len(train_ds),
            "val_samples": len(val_ds) if val_ds else 0,
            "best_val_mpjpe": best_val_mpjpe,
            "n_node_types": N_JOINT_TYPES,
        }, f, indent=2)

    print(f"\n[GraphTrain] Done. Best val MPJPE: {best_val_mpjpe:.4f}")


if __name__ == "__main__":
    main()
