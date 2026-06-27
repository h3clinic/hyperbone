#!/usr/bin/env bash
# HyperBone SAM2 Setup (Linux/macOS)
# Run from HyperBone workspace root: bash scripts/setup_sam2.sh

set -e

echo "============================================================"
echo "HyperBone SAM2 Setup"
echo "============================================================"
echo ""

# 1. Check PyTorch
echo "[1/4] Checking PyTorch..."
if ! python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}')" 2>/dev/null; then
    echo "ERROR: PyTorch not installed."
    echo ""
    echo "Install PyTorch first:"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi
echo ""

# 2. Install SAM2
echo "[2/4] Installing SAM2..."
if python -c "from sam2.build_sam import build_sam2" 2>/dev/null; then
    echo "  SAM2 already installed."
else
    echo "  Installing SAM2..."
    pip install sam2 || {
        echo "  pip install failed. Trying from source..."
        rm -rf sam2_repo
        git clone https://github.com/facebookresearch/sam2.git sam2_repo
        cd sam2_repo && pip install -e . && cd ..
    }
    python -c "from sam2.build_sam import build_sam2; print('  SAM2 import OK')"
fi
echo ""

# 3. Download checkpoint
echo "[3/4] Checking checkpoint..."
mkdir -p checkpoints
CKPT="checkpoints/sam2.1_hiera_tiny.pt"
if [ -f "$CKPT" ]; then
    SIZE=$(du -h "$CKPT" | cut -f1)
    echo "  Checkpoint exists: $CKPT ($SIZE)"
else
    echo "  Downloading sam2.1_hiera_tiny.pt..."
    curl -L "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt" -o "$CKPT"
    if [ -f "$CKPT" ]; then
        SIZE=$(du -h "$CKPT" | cut -f1)
        echo "  Downloaded: $CKPT ($SIZE)"
    else
        echo "ERROR: Download failed."
        exit 1
    fi
fi
echo ""

# 4. Config
echo "[4/4] Locating model config..."
CFG=$(python -c "
import sam2, os
base = os.path.dirname(sam2.__file__)
candidates = [
    os.path.join(base, 'configs', 'sam2.1', 'sam2.1_hiera_t.yaml'),
    os.path.join(base, '..', 'configs', 'sam2.1', 'sam2.1_hiera_t.yaml'),
]
found = [c for c in candidates if os.path.exists(c)]
print(found[0] if found else 'NOT_FOUND')
" 2>/dev/null)
echo "  Config: $CFG"
echo ""

echo "============================================================"
echo "Setup complete. Run: python scripts/check_sam2.py"
echo "============================================================"
