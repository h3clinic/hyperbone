"""SAM2 environment diagnostic — prints installation status and readiness."""

import sys
import os
from pathlib import Path


def main():
    print("=" * 60)
    print("HyperBone SAM2 Environment Check")
    print("=" * 60)
    print()

    # Python
    print(f"Python executable: {sys.executable}")
    print(f"Python version:    {sys.version.split()[0]}")
    print()

    # PyTorch
    try:
        import torch
        print(f"PyTorch version:   {torch.__version__}")
        print(f"CUDA available:    {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device:       {torch.cuda.get_device_name(0)}")
            print(f"CUDA version:      {torch.version.cuda}")
        else:
            print("CUDA device:       N/A (CPU only)")
    except ImportError:
        print("PyTorch:           NOT INSTALLED")
        print("  → pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print()
        print("FATAL: PyTorch is required. Install it first.")
        sys.exit(1)

    print()

    # SAM2
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        print("SAM2 import:       OK")
        import sam2
        sam2_path = Path(sam2.__file__).parent
        print(f"SAM2 location:     {sam2_path}")
    except ImportError as e:
        print(f"SAM2 import:       FAILED ({e})")
        print()
        print("To install SAM2:")
        print("  pip install sam2")
        print("  # or: git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e .")
        sys.exit(1)

    print()

    # Checkpoint
    checkpoint_paths = [
        "checkpoints/sam2.1_hiera_tiny.pt",
        "sam2/checkpoints/sam2.1_hiera_tiny.pt",
    ]
    ckpt_found = None
    for p in checkpoint_paths:
        if Path(p).exists():
            ckpt_found = p
            break

    if ckpt_found:
        size_mb = Path(ckpt_found).stat().st_size / (1024 * 1024)
        print(f"Checkpoint:        {ckpt_found} ({size_mb:.1f} MB)")
    else:
        print("Checkpoint:        NOT FOUND")
        print(f"  Expected at:     checkpoints/sam2.1_hiera_tiny.pt")
        print("  Download:")
        print("    curl -L https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt -o checkpoints/sam2.1_hiera_tiny.pt")

    print()

    # Config
    config_candidates = [
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
    ]
    # Also check inside the installed sam2 package
    try:
        import sam2
        pkg_config = Path(sam2.__file__).parent / "configs" / "sam2.1" / "sam2.1_hiera_t.yaml"
        config_candidates.append(str(pkg_config))
        # SAM2 may use relative config name resolved internally
        pkg_config2 = Path(sam2.__file__).parent.parent / "configs" / "sam2.1" / "sam2.1_hiera_t.yaml"
        config_candidates.append(str(pkg_config2))
    except Exception:
        pass

    cfg_found = None
    for p in config_candidates:
        if Path(p).exists():
            cfg_found = p
            break

    if cfg_found:
        print(f"Model config:      {cfg_found}")
    else:
        print("Model config:      NOT FOUND")
        print(f"  Searched:        {config_candidates}")
        print("  Note: SAM2 build_sam2() may accept config name like 'sam2.1_hiera_t' ")
        print("        and resolve it from the installed package.")

    print()
    print("-" * 60)

    # Summary
    ready = True
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. SAM2 will be extremely slow on CPU.")
    if not ckpt_found:
        print("BLOCKED: Checkpoint not found. Download it first.")
        ready = False

    if ready:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg_arg = cfg_found if cfg_found else "sam2.1_hiera_t"
        ckpt_arg = ckpt_found if ckpt_found else "checkpoints/sam2.1_hiera_tiny.pt"
        print()
        print("READY. Suggested commands:")
        print()
        print(f"  # Frame-level test:")
        print(f"  python scripts/test_sam2_frame.py --image outputs/test_run/frames/frame_000024.png --out outputs/sam2_frame_test --checkpoint {ckpt_arg} --model-cfg {cfg_arg} --device {device}")
        print()
        print(f"  # Full pipeline:")
        print(f"  python scripts/run_pseudo_label_batch.py --video-dir HyperVid --sample-fps 1 --out outputs/sam2_smoke --max-videos 1 --max-frames-per-video 5 --mask-backend sam2 --sam2-checkpoint {ckpt_arg} --sam2-model-cfg {cfg_arg} --device {device}")
    else:
        print()
        print("NOT READY. Fix the issues above first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
