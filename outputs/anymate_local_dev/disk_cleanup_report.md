# Disk Cleanup Report

## Date: 2026-06-15

## Space Before: 9.80 GB free
## Space After: 11.47 GB free
## Space Freed: 1.67 GB

## Files Deleted

| Path | Approx Size | Reason |
|------|-------------|--------|
| HyperVid/trimmed/HyperVid.mp4 | 654 MB | Original movie trimmed copy, no longer needed |
| datasets/BigBuckBunny_320x180.mp4 | 62 MB | Sample video, not needed for training |
| outputs/sam2_m23_repair_100f/ | 519 MB | M2.3 already passed, QA complete |
| outputs/sam2_m23_no_repair/ | 90 MB | Old baseline comparison run |
| outputs/sam2_m23_repair_20f/ | 89 MB | Old 20f repair run |
| outputs/dino_custom_smoke/ | 102 MB | Stale DINO smoke test |
| outputs/labelforge_v0/ | 54 MB | Old labelforge version |
| outputs/sam2_smoke_m22_trimmed/ | 47 MB | Old M2.2 smoke |
| outputs/sam2_smoke_m22_repair/ | 8 MB | Old M2.2 repair smoke |
| outputs/test_run/ | 44 MB | Old test run |
| outputs/grounded_sam2_skeletons/ | 38 MB | Old grounded SAM2 skeletons |
| outputs/grounded_sam2_test/ | 12 MB | Old grounded SAM2 test |
| outputs/batch_smoke/ | 11 MB | Old batch smoke |
| outputs/sam2_smoke/ | 9 MB | Old SAM2 smoke |
| outputs/sam2_smoke_baseline/ | 8 MB | Old baseline |
| outputs/sam2_repair_smoke/ | 8 MB | Old repair smoke |
| outputs/threshold_regression/ | 6 MB | Old threshold test |
| outputs/manual_proposal_custom_smoke/ | 8 MB | Old manual proposal smoke |
| outputs/sam2_frame_test/ | <1 MB | Old frame test |
| outputs/comparison/ | <1 MB | Empty comparison dir |

## Files Preserved

| Path | Size | Reason |
|------|------|--------|
| datasets/anymate/Anymate_test.pt | 4,505 MB | Primary training data |
| checkpoints/sam2.1_hiera_tiny.pt | 149 MB | SAM2 checkpoint for future pipeline |
| outputs/models/ | ~300 MB | All trained model checkpoints |
| outputs/anymate_clips_pilot/ | 91 MB | Rendered clip pilot data |
| outputs/labelforge_v05/ | 198 MB | Latest labelforge graphs |
| Source code (hyperbone/, scripts/, tests/) | ~5 MB | Active codebase |

## Notes

- HyperVid/HyperVid.mp4 was deleted in a prior step (per user directive)
- Per permanent memory rule: all videos must have 10 min trimmed front+back before processing
- Per permanent memory rule: delete full source videos after trimming
