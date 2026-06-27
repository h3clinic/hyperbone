# HyperBone Fox Tracking — Synthetic Smoke Test

## Asset

- **Source**: https://github.com/KhronosGroup/glTF-Sample-Assets
- **License**: CC0 1.0 base model, CC-BY 4.0 rigging/animation by PixelMannen
- **Attribution**: Fox model from glTF-Sample-Assets by KhronosGroup. Base mesh: CC0 1.0 Universal. Rigging and animation: CC-BY 4.0 by PixelMannen (@ARebelSpy).
- **Model path**: `C:\Users\ritayan\HyperBone\assets\gltf_samples\Fox\glTF-Binary\Fox.glb`

## Render

- Animation: synthetic_walk
- FPS: 24
- Duration: 5.0s
- Resolution: [640, 480]

## Pipeline

```
Synthetic bbox proposal → HyperBone crop → custom mask → Zhang-Suen thinning → custom graph → quality gate
```

## Results

| Metric | Value |
|--------|-------|
| Video path | `C:\Users\ritayan\HyperBone\outputs\synthetic\fox_5s.mp4` |
| Frames processed | 24 |
| Proposals generated | 24 |
| Graphs generated | 24 |
| **Accepted graphs** | **24** |
| Rejected graphs | 0 |
| **Acceptance rate** | **100.0%** |
| Avg runtime/object | 42.0 ms |

## Tracking Stability

| Metric | Value |
|--------|-------|
| **Topology stability score** | **0.8853** |
| Centroid jump mean | 13.1 px |
| Centroid jump P95 | 28.9 px |
| Node count mean ± std | 353.0 ± 39.1 |
| Edge count mean ± std | 573.1 ± 67.9 |
| Skeleton length mean ± std | 1225.2 ± 92.5 |
| Graph presence rate | 100.00% |

## Verdict

**PASS**

At least 5 accepted graphs (24) and topology stability >= 0.50 (0.885)

## Ownership Claim

> All skeleton graphs produced by HyperBone-owned mapper.
> skeleton_mapper = hyperbone-custom for every record.
> No SAM2, no CoTracker, no Depth Anything.
