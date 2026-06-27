# HyperBone Anymate Frame Model — Evaluation

## Verdict: **FRAME_MODEL_PARTIAL**

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
| Val MPJPE | 0.2566 |
| Val MPJPE (median) | 0.1997 |
| PCK-3D @ 0.05 | 0.033 |
| PCK-3D @ 0.10 | 0.186 |
| PCK-3D @ 0.20 | 0.501 |
| 2D Reproj Error (mean) | 28.8px |
| 2D Reproj Error (median) | 22.6px |
| Bone Length Error | 46.5% |
| Visibility Accuracy | 0.816 |
| Total Joints Evaluated | 5891 |
| Total Samples | 240 |

## Worst 5 Joints
| Joint | MPJPE | Count |
|-------|-------|-------|
| joint_33 | 0.6222 | 68 |
| joint_34 | 0.4615 | 103 |
| joint_11 | 0.3891 | 194 |
| joint_23 | 0.3489 | 136 |
| joint_32 | 0.3359 | 132 |

## Best 3 Samples
| Asset | Frame | MPJPE | Reproj |
|-------|-------|-------|--------|
| 000-076/3de65beaf5484d968 | 3 | 0.0793 | 9.6px |
| 000-076/3de65beaf5484d968 | 19 | 0.0796 | 9.3px |
| 000-076/3de65beaf5484d968 | 21 | 0.0798 | 9.3px |

## Worst 3 Samples
| Asset | Frame | MPJPE | Reproj |
|-------|-------|-------|--------|
| fbx_9/60fcea30f2b2d200537 | 4 | 0.6034 | 61.8px |
| fbx_9/60fcea30f2b2d200537 | 8 | 0.6039 | 58.1px |
| fbx_9/60fcea30f2b2d200537 | 5 | 0.6048 | 60.6px |

## Overlay Folder
`outputs\models\hyperbone_anymate_frame_pilot\eval_train\overlays`

## Pass/Fail
- Train loss decreases: YES
- Reproj < 25px: NO
- Bone error < 10%: NO
