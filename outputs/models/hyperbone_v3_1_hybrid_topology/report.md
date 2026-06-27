# HyperBone v3.1 Hybrid Neural-Cost Optimizer

## Side-by-Side (Same Splits, Same Metrics)
| group | split | method | k | neural_w | dist_w | max_deg | edge_f1 | precision | recall | sel/gt | comp | avg_deg | max_deg_obs | cycles |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| side_by_side | train | v2.16_threshold | 12 | 1.00 | 1.00 | 2.92 | 0.4991 | 0.9579 | 0.3752 | 0.381 | 31.46 | 0.74 | 2.92 | 0.04 |
| side_by_side | train | v2.16_top_e | 12 | 1.00 | 1.00 | 4.875 | 0.7766 | 0.7766 | 0.7766 | 1.000 | 7.24 | 1.94 | 4.88 | 6.09 |
| side_by_side | train | v3_density_normalized_mst | 8 | 0.00 | 1.00 | 3.56 | 0.6291 | 0.6273 | 0.6315 | 1.008 | 1.01 | 1.95 | 3.56 | 0.00 |
| side_by_side | train | v3_degree_capped_mst | 8 | 0.00 | 1.00 | 3.0 | 0.6207 | 0.6189 | 0.6231 | 1.008 | 1.01 | 1.95 | 3.00 | 0.00 |
| side_by_side | train | v3_hybrid_mst_plus_branches | 8 | 0.00 | 1.00 | 5.47 | 0.6033 | 0.5420 | 0.6810 | 1.258 | 1.01 | 2.44 | 5.47 | 12.82 |
| side_by_side | val | v2.16_threshold | 12 | 1.00 | 1.00 | 2.6476868327402134 | 0.4603 | 0.9168 | 0.3365 | 0.351 | 29.27 | 0.68 | 2.65 | 0.08 |
| side_by_side | val | v2.16_top_e | 12 | 1.00 | 1.00 | 4.6263345195729535 | 0.7527 | 0.7527 | 0.7527 | 1.000 | 7.06 | 1.93 | 4.63 | 5.83 |
| side_by_side | val | v3_density_normalized_mst | 8 | 0.00 | 1.00 | 3.5338078291814945 | 0.6405 | 0.6387 | 0.6425 | 1.007 | 1.02 | 1.94 | 3.53 | 0.00 |
| side_by_side | val | v3_degree_capped_mst | 8 | 0.00 | 1.00 | 2.991103202846975 | 0.6326 | 0.6308 | 0.6346 | 1.007 | 1.02 | 1.94 | 2.99 | 0.00 |
| side_by_side | val | v3_hybrid_mst_plus_branches | 8 | 0.00 | 1.00 | 5.186832740213523 | 0.6168 | 0.5542 | 0.6958 | 1.257 | 1.02 | 2.43 | 5.19 | 11.21 |
| side_by_side | test | v2.16_threshold | 12 | 1.00 | 1.00 | 2.620629370629371 | 0.4559 | 0.8964 | 0.3338 | 0.349 | 29.77 | 0.68 | 2.62 | 0.06 |
| side_by_side | test | v2.16_top_e | 12 | 1.00 | 1.00 | 4.603146853146853 | 0.7478 | 0.7478 | 0.7478 | 1.000 | 7.09 | 1.93 | 4.60 | 5.89 |
| side_by_side | test | v3_density_normalized_mst | 8 | 0.00 | 1.00 | 3.554195804195804 | 0.6341 | 0.6323 | 0.6364 | 1.008 | 1.03 | 1.94 | 3.55 | 0.00 |
| side_by_side | test | v3_degree_capped_mst | 8 | 0.00 | 1.00 | 2.9947552447552446 | 0.6238 | 0.6220 | 0.6262 | 1.008 | 1.03 | 1.94 | 2.99 | 0.00 |
| side_by_side | test | v3_hybrid_mst_plus_branches | 8 | 0.00 | 1.00 | 5.155594405594406 | 0.6090 | 0.5468 | 0.6878 | 1.259 | 1.03 | 2.43 | 5.16 | 11.37 |

## Notes
- Sweep skipped by --skip-sweep.
