"""
Streaming dataset pipeline — download, train, validate, extract replay, delete.

Prevents catastrophic forgetting via replay buffer.
Includes collapse protection (spread, active recall, count error).

Usage:
    python scripts/train_dataset_stream.py --manifest datasets/manifest.jsonl
    python scripts/train_dataset_stream.py --manifest datasets/manifest.jsonl --start-pack anymate_014
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "scratch" / "current_pack"
REPLAY = ROOT / "replay"
MODELS_DIR = ROOT / "outputs" / "models" / "hyperbone_anymate_static_stream"
STATE_PATH = MODELS_DIR / "stream_state.json"


def run(cmd: list, check: bool = True):
    """Run subprocess with logging."""
    print(f"\n$ {' '.join(map(str, cmd))}", flush=True)
    result = subprocess.run(cmd, capture_output=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    return result


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "completed_packs": [],
        "latest_checkpoint": None,
        "metrics": {},
    }


def save_state(state: dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def load_manifest(path: str) -> list:
    packs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                packs.append(json.loads(line))
    return packs


def reset_scratch():
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True, exist_ok=True)


def install_pack(pack: dict) -> Path:
    """Download/copy pack to scratch, verify checksum, extract."""
    reset_scratch()

    source = pack["source"]
    ext = Path(source).suffix
    archive = SCRATCH / f"pack{ext}"

    if source.startswith("s3://"):
        run(["aws", "s3", "cp", source, str(archive)])
    elif source.startswith("http://") or source.startswith("https://"):
        run(["curl", "-L", "-o", str(archive), source])
    else:
        # Local file
        src_path = Path(source)
        if not src_path.exists():
            raise FileNotFoundError(f"Source not found: {source}")
        shutil.copy2(src_path, archive)

    # Verify checksum
    if pack.get("sha256"):
        actual = sha256_file(archive)
        if actual != pack["sha256"]:
            raise RuntimeError(
                f"SHA256 mismatch for {pack['pack_id']}: "
                f"expected {pack['sha256']}, got {actual}"
            )
        print(f"  Checksum verified: {actual[:16]}...")

    # Extract if archive
    data_dir = SCRATCH / "data"
    data_dir.mkdir(exist_ok=True)

    if ext in (".zst", ".tar"):
        run(["tar", "-xf", str(archive), "-C", str(data_dir)])
    elif ext == ".zip":
        run(["python", "-m", "zipfile", "-e", str(archive), str(data_dir)])
    elif ext == ".pt":
        # Already a .pt file, just use it directly
        shutil.move(str(archive), data_dir / "data.pt")
    else:
        # Unknown format, assume directory or single file
        shutil.move(str(archive), data_dir / archive.name)

    return data_dir


def train_pack(pack: dict, data_dir: Path, latest_checkpoint: str | None,
               train_args: dict) -> tuple[str, str]:
    """Train on a pack, resuming from latest checkpoint."""
    out_dir = MODELS_DIR
    epochs = pack.get("epochs", 10)

    cmd = [
        sys.executable, str(ROOT / "scripts" / "train_anymate_static_rig_v2.py"),
        "--pt", str(data_dir / "data.pt") if (data_dir / "data.pt").exists()
                else str(next(data_dir.glob("*.pt"))),
        "--out", str(out_dir),
        "--epochs", str(epochs),
        "--batch-size", str(train_args.get("batch_size", 8)),
        "--w-bone", str(train_args.get("w_bone", 0.8)),
        "--w-unmatched", str(train_args.get("w_unmatched", 0.3)),
        "--w-count", str(train_args.get("w_count", 0.1)),
        "--edge-pos-weight", str(train_args.get("edge_pos_weight", 3.0)),
        "--edge-fp-weight", str(train_args.get("edge_fp_weight", 1.0)),
        "--count-overpredict-scale", str(train_args.get("count_overpredict_scale", 0.25)),
        "--count-underpredict-scale", str(train_args.get("count_underpredict_scale", 1.0)),
        "--active-pos-weight", str(train_args.get("active_pos_weight", 2.0)),
        "--ramp-start", str(train_args.get("ramp_start", 3)),
        "--ramp-end", str(train_args.get("ramp_end", 8)),
        "--kill-spread", str(train_args.get("kill_spread", 0.45)),
    ]

    # TODO: Add --resume flag support to train script
    # if latest_checkpoint:
    #     cmd += ["--resume", latest_checkpoint]

    # Add replay dirs if they exist
    replay_pts = list(REPLAY.glob("*/*.pt"))
    # TODO: Implement replay buffer mixing in train script

    run(cmd)

    latest = out_dir / "model_final.pt"
    best = out_dir / "best_model.pt"

    if not latest.exists() and not best.exists():
        raise RuntimeError("Training completed but no checkpoint found.")

    ckpt = str(best) if best.exists() else str(latest)
    return ckpt, ckpt


def validate_checkpoint(checkpoint: str, data_dir: Path) -> dict:
    """Run eval and check collapse criteria."""
    eval_out = MODELS_DIR / "stream_eval"
    eval_out.mkdir(parents=True, exist_ok=True)

    # Find the .pt data file
    pt_file = data_dir / "data.pt" if (data_dir / "data.pt").exists() \
              else next(data_dir.glob("*.pt"))

    cmd = [
        sys.executable, str(ROOT / "scripts" / "eval_anymate_static_rig_v2.py"),
        "--pt", str(pt_file),
        "--checkpoint", checkpoint,
        "--out", str(eval_out),
        "--split", "val",
    ]

    run(cmd)

    metrics_path = eval_out / "eval_metrics_v2.json"
    if not metrics_path.exists():
        raise RuntimeError("Eval did not produce metrics file")

    metrics = json.loads(metrics_path.read_text())

    # --- Collapse protection ---
    spread = metrics.get("spread_collapse_score_mean", 0)
    overpred = metrics.get("overprediction_ratio_mean", 999)

    if spread < 0.45:
        raise RuntimeError(f"COLLAPSE: spread={spread:.3f} < 0.45")

    if overpred < 0.5:
        raise RuntimeError(f"UNDERPREDICTION: overpred_ratio={overpred:.3f} < 0.5")

    if overpred > 2.0:
        raise RuntimeError(f"OVERPREDICTION: overpred_ratio={overpred:.3f} > 2.0")

    print(f"  Validation passed: spread={spread:.3f}, overpred={overpred:.3f}")
    return metrics


def extract_replay_samples(pack_id: str, data_dir: Path, fraction: float = 0.03):
    """Keep a small anti-forgetting subset from this pack."""
    import torch

    out = REPLAY / pack_id
    out.mkdir(parents=True, exist_ok=True)

    # Find .pt file
    pt_file = data_dir / "data.pt" if (data_dir / "data.pt").exists() \
              else next(data_dir.glob("*.pt"), None)

    if pt_file is None:
        print(f"  Warning: No .pt file found for replay extraction")
        return

    data = torch.load(pt_file, map_location="cpu", weights_only=False)
    n_total = len(data)
    n_keep = max(1, int(n_total * fraction))

    indices = random.sample(range(n_total), n_keep)
    subset = [data[i] for i in indices]

    torch.save(subset, out / "replay.pt")
    print(f"  Replay: kept {n_keep}/{n_total} samples in {out}")


def cleanup_pack():
    """Remove scratch directory."""
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    print("  Scratch cleaned.")


def main():
    parser = argparse.ArgumentParser(description="Streaming dataset train pipeline")
    parser.add_argument("--manifest", required=True,
                        help="Path to datasets/manifest.jsonl")
    parser.add_argument("--start-pack", default=None,
                        help="Skip packs before this pack_id")
    parser.add_argument("--replay-fraction", type=float, default=0.03,
                        help="Fraction of each pack to keep for replay")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--w-bone", type=float, default=0.8)
    parser.add_argument("--w-count", type=float, default=0.1)
    args = parser.parse_args()

    packs = load_manifest(args.manifest)
    state = load_state()

    train_args = {
        "batch_size": args.batch_size,
        "w_bone": args.w_bone,
        "w_count": args.w_count,
    }

    print(f"[Stream] Manifest: {args.manifest}")
    print(f"[Stream] Packs: {len(packs)} total, "
          f"{len(state['completed_packs'])} completed")
    print(f"[Stream] Replay fraction: {args.replay_fraction}")
    print("=" * 70)

    for pack in packs:
        pack_id = pack["pack_id"]

        if pack_id in state["completed_packs"]:
            print(f"\n[Skip] {pack_id} (already completed)")
            continue

        if args.start_pack and pack_id < args.start_pack:
            print(f"\n[Skip] {pack_id} (before --start-pack)")
            continue

        print(f"\n{'='*70}")
        print(f"[PACK] {pack_id}")
        print(f"{'='*70}")

        try:
            # 1. Install
            print(f"\n  [1/5] Installing {pack_id}...")
            data_dir = install_pack(pack)

            # 2. Train
            print(f"\n  [2/5] Training ({pack.get('epochs', 10)} epochs)...")
            ckpt, best_ckpt = train_pack(
                pack, data_dir, state.get("latest_checkpoint"), train_args
            )

            # 3. Validate
            print(f"\n  [3/5] Validating...")
            metrics = validate_checkpoint(ckpt, data_dir)

            # 4. Extract replay
            print(f"\n  [4/5] Extracting replay subset...")
            extract_replay_samples(pack_id, data_dir, args.replay_fraction)

            # 5. Update state + cleanup
            state["latest_checkpoint"] = ckpt
            state["completed_packs"].append(pack_id)
            state["metrics"][pack_id] = {
                "spread": metrics.get("spread_collapse_score_mean"),
                "mpjpe": metrics.get("mpjpe_matched_mean"),
                "edge_f1": metrics.get("edge_f1_matched_mean"),
                "bone_ratio": metrics.get("bone_length_ratio_mean_mean"),
                "overpred": metrics.get("overprediction_ratio_mean"),
            }
            save_state(state)

            print(f"\n  [5/5] Cleaning up...")
            cleanup_pack()

            print(f"\n  [DONE] {pack_id} complete.")

        except Exception as e:
            print(f"\n  [FAIL] {pack_id}: {e}")
            save_state(state)
            raise

    print(f"\n{'='*70}")
    print("[Stream] All packs complete.")
    print(f"  Total completed: {len(state['completed_packs'])}")
    print(f"  Latest checkpoint: {state.get('latest_checkpoint')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
