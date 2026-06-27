# HyperBone v3.1 Hybrid Neural-Cost Optimizer

## Side-by-Side (Same Splits, Same Metrics)
| group | split | method | k | neural_w | dist_w | max_deg | edge_f1 | precision | recall | sel/gt | comp | avg_deg | max_deg_obs | cycles |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| side_by_side | test | v2.16_threshold | 12 | 1.00 | 1.00 | 2.624125874125874 | 0.4551 | 0.8985 | 0.3330 | 0.348 | 29.83 | 0.68 | 2.62 | 0.06 |
| side_by_side | test | v2.16_top_e | 12 | 1.00 | 1.00 | 4.608391608391608 | 0.7479 | 0.7479 | 0.7479 | 1.000 | 7.14 | 1.93 | 4.61 | 5.95 |
| side_by_side | test | v3_density_normalized_mst | 8 | 0.00 | 1.00 | 3.5594405594405596 | 0.6341 | 0.6323 | 0.6365 | 1.008 | 1.03 | 1.94 | 3.56 | 0.00 |
| side_by_side | test | v3_degree_capped_mst | 8 | 0.00 | 1.00 | 2.9947552447552446 | 0.6241 | 0.6223 | 0.6265 | 1.008 | 1.03 | 1.94 | 2.99 | 0.00 |
| side_by_side | test | v3_hybrid_mst_plus_branches | 8 | 0.00 | 1.00 | 5.1625874125874125 | 0.6089 | 0.5467 | 0.6877 | 1.259 | 1.03 | 2.43 | 5.16 | 11.37 |

## Best v3.1 by Split
| group | split | method | k | neural_w | dist_w | max_deg | edge_f1 | precision | recall | sel/gt | comp | avg_deg | max_deg_obs | cycles |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3.1_sweep | test | density_normalized_mst | 12 | 2.00 | 0.50 | 4.277972027972028 | 0.7734 | 0.7714 | 0.7761 | 1.008 | 1.02 | 1.94 | 4.28 | 0.00 |

## Notes
- v2.16 top-E is diagnostic (uses GT edge count budget).
- Deterministic baselines are deployable constrained decoders.
- v3.1 combines deterministic structural costs with neural edge logits.