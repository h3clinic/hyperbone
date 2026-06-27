# HyperBone Anymate Frame Model — Evaluation

## Verdict: **FRAME_MODEL_FAIL**

## Training
| Metric | Value |
|--------|-------|
| Train loss start | 1.3971 |
| Train loss end | 1.2298 |
| Val MPJPE start | 0.5971 |
| Val MPJPE end | 0.5430 |
| Loss decreases | YES |

## Metrics
| Metric | Value |
|--------|-------|
| Val MPJPE | 0.5368 |
| Val MPJPE (median) | 0.5586 |
| PCK-3D @ 0.05 | 0.000 |
| PCK-3D @ 0.10 | 0.002 |
| PCK-3D @ 0.20 | 0.149 |
| 2D Reproj Error (mean) | 62.6px |
| 2D Reproj Error (median) | 56.4px |
| Bone Length Error | 62.9% |
| Visibility Accuracy | 0.655 |
| Total Joints Evaluated | 1313 |
| Total Samples | 60 |

## Worst 5 Joints
| Joint | MPJPE | Count |
|-------|-------|-------|
| joint_28 | 1.3164 | 18 |
| joint_27 | 1.0829 | 1 |
| joint_20 | 0.9717 | 43 |
| joint_24 | 0.9531 | 42 |
| joint_16 | 0.8447 | 12 |

## Best 3 Samples
| Asset | Frame | MPJPE | Reproj |
|-------|-------|-------|--------|
| fbx_9/d4e938d77c0e2b4225c | 29 | 0.4014 | 41.2px |
| fbx_9/d4e938d77c0e2b4225c | 28 | 0.4289 | 43.5px |
| fbx_9/d4e938d77c0e2b4225c | 25 | 0.4295 | 45.0px |

## Worst 3 Samples
| Asset | Frame | MPJPE | Reproj |
|-------|-------|-------|--------|
| fbx_9/d9e4715c76a789da423 | 18 | 0.6165 | 74.5px |
| fbx_9/d9e4715c76a789da423 | 17 | 0.6167 | 74.4px |
| fbx_9/d9e4715c76a789da423 | 10 | 0.6238 | 74.5px |

## Overlay Folder
`outputs\models\hyperbone_anymate_frame_pilot\eval\overlays`

## Pass/Fail
- Train loss decreases: YES
- Reproj < 25px: NO
- Bone error < 10%: NO
