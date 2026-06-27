# HyperBone Anymate Frame Model — Evaluation

## Verdict: **FRAME_MODEL_PARTIAL**

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
| Val MPJPE | 0.0734 |
| Val MPJPE (median) | 0.0637 |
| PCK-3D @ 0.05 | 0.336 |
| PCK-3D @ 0.10 | 0.782 |
| PCK-3D @ 0.20 | 0.987 |
| 2D Reproj Error (mean) | 7.9px |
| 2D Reproj Error (median) | 6.6px |
| Bone Length Error | 19.1% |
| Visibility Accuracy | 0.955 |
| Total Joints Evaluated | 5891 |
| Total Samples | 240 |

## Worst 5 Joints
| Joint | MPJPE | Count |
|-------|-------|-------|
| joint_33 | 0.1053 | 68 |
| joint_25 | 0.1052 | 150 |
| joint_11 | 0.1043 | 194 |
| joint_34 | 0.1027 | 103 |
| joint_41 | 0.0984 | 60 |

## Best 3 Samples
| Asset | Frame | MPJPE | Reproj |
|-------|-------|-------|--------|
| fbx_9/60fcea30f2b2d200537 | 29 | 0.0370 | 3.3px |
| fbx_9/2997b0df9206614ee41 | 12 | 0.0402 | 4.1px |
| fbx_9/60fcea30f2b2d200537 | 28 | 0.0413 | 4.4px |

## Worst 3 Samples
| Asset | Frame | MPJPE | Reproj |
|-------|-------|-------|--------|
| fbx_3/f126174205e5be26fae | 22 | 0.1388 | 16.0px |
| fbx_3/f126174205e5be26fae | 21 | 0.1433 | 15.7px |
| fbx_3/f126174205e5be26fae | 23 | 0.1700 | 18.7px |

## Overlay Folder
`outputs\models\hyperbone_anymate_frame_pilot_v2\eval_train\overlays`

## Pass/Fail
- Train loss decreases: YES
- Reproj < 25px: YES
- Bone error < 10%: NO
