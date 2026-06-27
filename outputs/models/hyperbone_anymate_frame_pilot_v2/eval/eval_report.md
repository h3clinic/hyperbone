# HyperBone Anymate Frame Model — Evaluation

## Verdict: **FRAME_MODEL_FAIL**

## Training
| Metric | Value |
|--------|-------|
| Train loss start | 1.4157 |
| Train loss end | 1.0781 |
| Val MPJPE start | 0.5068 |
| Val MPJPE end | 0.6401 |
| Loss decreases | YES |

## Metrics
| Metric | Value |
|--------|-------|
| Val MPJPE | 0.5068 |
| Val MPJPE (median) | 0.5326 |
| PCK-3D @ 0.05 | 0.021 |
| PCK-3D @ 0.10 | 0.046 |
| PCK-3D @ 0.20 | 0.133 |
| 2D Reproj Error (mean) | 57.9px |
| 2D Reproj Error (median) | 51.3px |
| Bone Length Error | 54.3% |
| Visibility Accuracy | 0.761 |
| Total Joints Evaluated | 1313 |
| Total Samples | 60 |

## Worst 5 Joints
| Joint | MPJPE | Count |
|-------|-------|-------|
| joint_28 | 1.3951 | 18 |
| joint_27 | 1.2352 | 1 |
| joint_16 | 0.9663 | 12 |
| joint_20 | 0.8500 | 43 |
| joint_23 | 0.7279 | 60 |

## Best 3 Samples
| Asset | Frame | MPJPE | Reproj |
|-------|-------|-------|--------|
| fbx_9/d4e938d77c0e2b4225c | 29 | 0.3532 | 33.9px |
| fbx_9/d4e938d77c0e2b4225c | 18 | 0.3750 | 39.2px |
| fbx_9/d4e938d77c0e2b4225c | 19 | 0.3750 | 38.4px |

## Worst 3 Samples
| Asset | Frame | MPJPE | Reproj |
|-------|-------|-------|--------|
| fbx_9/d9e4715c76a789da423 | 18 | 0.6068 | 74.4px |
| fbx_9/d9e4715c76a789da423 | 17 | 0.6071 | 74.1px |
| fbx_9/d9e4715c76a789da423 | 10 | 0.6188 | 72.2px |

## Overlay Folder
`outputs\models\hyperbone_anymate_frame_pilot_v2\eval\overlays`

## Pass/Fail
- Train loss decreases: YES
- Reproj < 25px: NO
- Bone error < 10%: NO
