# SAM2 Setup Guide

## Overview

HyperBone supports two mask generation backends:

| Backend | Quality | Speed | GPU Required | Install |
|---------|---------|-------|-------------|---------|
| `threshold` | Low (0% acceptance on real film) | Fast | No | Built-in |
| `sam2` | High (instance segmentation) | Moderate | Recommended | Separate |

The **threshold backend works without SAM2** and requires no extra dependencies.
It exists as a pipeline skeleton and fallback — do not use its outputs for training.

## SAM2 Installation

### Option 1: pip install

```bash
pip install sam2
```

### Option 2: From source

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

### Requirements

- Python 3.10+
- PyTorch 2.3.1+ with CUDA support (recommended)
- torchvision

## Checkpoint Download

Download a SAM2.1 checkpoint:

```bash
# Tiny (fastest, good for smoke tests)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt -P checkpoints/

# Small (balanced speed/quality)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt -P checkpoints/

# Base+ (higher quality, slower)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt -P checkpoints/

# Large (best quality, slowest)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -P checkpoints/
```

## Model Config

SAM2 model configs are included in the `sam2` package. After installing sam2, the config
paths are relative to the sam2 package. Common configs:

| Checkpoint | Config path |
|-----------|-------------|
| `sam2.1_hiera_tiny.pt` | `configs/sam2.1/sam2.1_hiera_t.yaml` |
| `sam2.1_hiera_small.pt` | `configs/sam2.1/sam2.1_hiera_s.yaml` |
| `sam2.1_hiera_base_plus.pt` | `configs/sam2.1/sam2.1_hiera_b+.yaml` |
| `sam2.1_hiera_large.pt` | `configs/sam2.1/sam2.1_hiera_l.yaml` |

## Usage

### Single video

```bash
python scripts/run_pseudo_label.py \
    --video HyperVid/HyperVid.mp4 \
    --sample-fps 1 \
    --out outputs/sam2_run \
    --max-frames 10 \
    --mask-backend sam2 \
    --sam2-checkpoint checkpoints/sam2.1_hiera_tiny.pt \
    --sam2-model-cfg configs/sam2.1/sam2.1_hiera_t.yaml \
    --device cuda
```

### Batch

```bash
python scripts/run_pseudo_label_batch.py \
    --video-dir videos/ \
    --sample-fps 1 \
    --out outputs/sam2_batch \
    --max-videos 5 \
    --max-frames-per-video 10 \
    --mask-backend sam2 \
    --sam2-checkpoint checkpoints/sam2.1_hiera_tiny.pt \
    --sam2-model-cfg configs/sam2.1/sam2.1_hiera_t.yaml \
    --device cuda
```

### Threshold fallback (no SAM2 needed)

```bash
python scripts/run_pseudo_label.py \
    --video video.mp4 \
    --mask-backend threshold \
    --out outputs/threshold_run
```

## Device

- **`cuda`** (recommended): GPU inference. Required for practical batch processing.
- **`cpu`**: Works for smoke tests on 1-2 frames but extremely slow.

## Important Notes

1. **Do not train HyperBone from threshold backend results.** The 0% acceptance
   rate on real film frames is expected and correct.

2. **SAM2 is optional.** The threshold backend lets you develop and test the
   pipeline without SAM2 installed.

3. **Quality gates are not weakened for SAM2.** The same disconnected_graph,
   fragmented_graph, and mask_too_small checks apply. SAM2 should produce masks
   good enough to pass naturally.

4. **First checkpoint recommendation:** Start with `sam2.1_hiera_tiny.pt` for
   speed during development. Switch to `small` or `base+` for production runs.

5. **VRAM usage:** Tiny ~2GB, Small ~4GB, Base+ ~6GB, Large ~10GB.
