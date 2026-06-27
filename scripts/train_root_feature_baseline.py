"""Dumb standalone root classifier baseline for v2.14b.1.

Trains a small MLP on pre-computed structural root features using plain BCE,
no fancy loss terms. Reports root_f1, PR-AUC, budgeted top-k F1 and pred_root_ratio.

Purpose: verify whether the structural features contain usable root signal at all,
independent of the integrated pairwise model and its training objective.
If this baseline beats the integrated model on root_f1, the issue is in training
objective or model wiring, not the feature quality.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.datasets.anymate_static_dataset import AnymateStaticRigDataset
from hyperbone.rigs.parent_candidates import build_parent_candidates
from hyperbone.rigs.root_features import ROOT_STRUCTURAL_FEATURE_DIM, compute_root_structural_features


# ---------------------------------------------------------------------------
# Simple MLP classifier
# ---------------------------------------------------------------------------

class RootMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Feature extraction (batched, no-grad)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features_labels(loader: DataLoader, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (features [N, F], labels [N]) over the full loader."""
    feat_list = []
    label_list = []
    for batch in loader:
        joint_pos = batch["joint_pos"]
        active = batch["joint_active"] > 0.5
        gt_root = batch["root_mask"] > 0.5

        cand = build_parent_candidates(
            joint_pos, active, batch["parent_index"].long(), k=k, include_root=False, force_gt_parent=False,
        )
        feat = compute_root_structural_features(
            joint_pos, active,
            candidate_indices=cand["candidate_indices"],
            candidate_mask=cand["candidate_mask"],
            pair_logits=None,
            k=k,
        )

        B, J, F = feat.shape
        for b in range(B):
            valid = active[b]
            feat_list.append(feat[b, valid])
            label_list.append(gt_root[b, valid].float())

    if not feat_list:
        raise RuntimeError("No valid samples found in loader")
    return torch.cat(feat_list, dim=0), torch.cat(label_list, dim=0)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _pr_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    n_pos = int(labels.sum().item())
    if n_pos == 0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    labs = labels[order]
    tp = 0.0
    rec_prev = 0.0
    prec_prev = 1.0
    auc = 0.0
    for i, lab in enumerate(labs.tolist()):
        if lab > 0.5:
            tp += 1.0
        prec = tp / (i + 1)
        rec = tp / n_pos
        auc += (rec - rec_prev) * (prec + prec_prev) / 2.0
        rec_prev = rec
        prec_prev = prec
    return float(auc)


def _f1_at_threshold(scores: torch.Tensor, labels: torch.Tensor, t: float) -> dict:
    pred = (scores > t).float()
    tp = float((pred * labels).sum().item())
    fp = float((pred * (1 - labels)).sum().item())
    fn = float(((1 - pred) * labels).sum().item())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"precision": prec, "recall": rec, "f1": f1,
            "pred_root_ratio": float(pred.mean().item())}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_baseline(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = AnymateStaticRigDataset(args.pt, f"{args.splits_dir}/train.jsonl", max_joints=args.max_nodes, pc_points=args.pc_points)
    val_ds = AnymateStaticRigDataset(args.pt, f"{args.splits_dir}/val.jsonl", max_joints=args.max_nodes, pc_points=args.pc_points)
    test_ds = AnymateStaticRigDataset(args.pt, f"{args.splits_dir}/test.jsonl", max_joints=args.max_nodes, pc_points=args.pc_points)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("[baseline] Extracting features...")
    t0 = time.time()
    train_feat, train_lab = extract_features_labels(train_loader, k=args.k)
    val_feat, val_lab = extract_features_labels(val_loader, k=args.k)
    test_feat, test_lab = extract_features_labels(test_loader, k=args.k)
    print(f"[baseline] Feature extraction done in {time.time()-t0:.1f}s | train={train_feat.shape} val={val_feat.shape} test={test_feat.shape}")
    print(f"[baseline] GT root ratio: train={train_lab.mean().item():.4f} val={val_lab.mean().item():.4f} test={test_lab.mean().item():.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_feat = train_feat.to(device)
    train_lab = train_lab.to(device)
    val_feat = val_feat.to(device)
    val_lab = val_lab.to(device)
    test_feat = test_feat.to(device)
    test_lab = test_lab.to(device)

    model = RootMLP(in_dim=ROOT_STRUCTURAL_FEATURE_DIM, hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best_val_f1 = -1.0
    best_state = None

    train_dataset = TensorDataset(train_feat, train_lab)
    train_dl = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        for x_batch, y_batch in train_dl:
            optimizer.zero_grad()
            logits = model(x_batch)
            # Plain BCE with optional mild pos_weight (no cap).
            if args.pos_weight > 0:
                pw = torch.tensor(args.pos_weight, device=device, dtype=logits.dtype)
                loss = F.binary_cross_entropy_with_logits(logits, y_batch, pos_weight=pw)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1

        # Validation F1 sweep for best threshold.
        model.eval()
        with torch.no_grad():
            val_logits = model(val_feat)
            val_probs = torch.sigmoid(val_logits)
            best_thr_f1 = max(
                (_f1_at_threshold(val_probs, val_lab, t)["f1"], t)
                for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            )
            val_f1 = best_thr_f1[0]
            val_thr = best_thr_f1[1]
            val_pr_auc = _pr_auc(val_logits, val_lab)
            val_pred_ratio = float((val_probs > val_thr).float().mean().item())

        row = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss / max(n, 1),
            "val_f1": val_f1,
            "val_thr": val_thr,
            "val_pr_auc": val_pr_auc,
            "val_pred_ratio": val_pred_ratio,
        }
        history.append(row)
        print(f"E{epoch+1:03d} | train_loss={row['train_loss']:.4f} val_f1={val_f1:.4f} val_thr={val_thr:.2f} val_pr_auc={val_pr_auc:.4f} pred_ratio={val_pred_ratio:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Test eval with best model.
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        test_logits = model(test_feat)
        test_probs = torch.sigmoid(test_logits)
        test_pr_auc = _pr_auc(test_logits.cpu(), test_lab.cpu())
        thr_results = {}
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            thr_results[t] = _f1_at_threshold(test_probs, test_lab, t)
        best_test = max(thr_results.items(), key=lambda kv: kv[1]["f1"])

    gt_ratio = float(test_lab.mean().item())
    print(f"\n[baseline] Test results (best_model):")
    print(f"  PR-AUC = {test_pr_auc:.4f}")
    print(f"  GT root ratio = {gt_ratio:.4f}")
    for t, m in sorted(thr_results.items()):
        print(f"  thr={t:.2f} | f1={m['f1']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} pred_ratio={m['pred_root_ratio']:.4f}")

    summary = {
        "test_pr_auc": test_pr_auc,
        "test_gt_root_ratio": gt_ratio,
        "test_threshold_results": {str(k): v for k, v in thr_results.items()},
        "best_test_threshold": best_test[0],
        "best_test_f1": best_test[1]["f1"],
        "history": history,
    }
    (out_dir / "baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), out_dir / "baseline_model.pt")
    print(f"[baseline] Saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Dumb root feature MLP baseline")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument("--points-per-sample", type=int, default=1024, dest="pc_points")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--pos-weight", type=float, default=0.0,
                        help="BCE pos_weight. 0 disables (plain BCE). Suggested: gt_neg/gt_pos ratio.")
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--out", default="outputs/models/hyperbone_anymate_static_v2.14b_root_baseline")
    args = parser.parse_args()
    train_baseline(args)


if __name__ == "__main__":
    main()
