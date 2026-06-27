# HyperBone v3.0 Deterministic Topology Baselines

## Best by Split (edge_f1)
| split | method | k | edge_f1 | precision | recall | sel/gt | comp | avg_deg | max_deg | cycles |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | density_normalized_mst | 8 | 0.6291 | 0.6272 | 0.6315 | 1.008 | 1.01 | 1.95 | 3.56 | 0.00 |
| val | density_normalized_mst | 8 | 0.6409 | 0.6391 | 0.6430 | 1.007 | 1.02 | 1.94 | 3.54 | 0.00 |
| test | density_normalized_mst | 8 | 0.6339 | 0.6322 | 0.6363 | 1.008 | 1.03 | 1.94 | 3.56 | 0.00 |

## Full Results
| split | method | k | edge_f1 | precision | recall | sel/gt | comp | avg_deg | max_deg | cycles |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | knn_mst | 8 | 0.6252 | 0.6234 | 0.6277 | 1.008 | 1.01 | 1.95 | 3.74 | 0.00 |
| train | knn_mst | 12 | 0.6252 | 0.6233 | 0.6277 | 1.008 | 1.00 | 1.95 | 3.73 | 0.00 |
| train | knn_mst | 16 | 0.6249 | 0.6231 | 0.6274 | 1.008 | 1.00 | 1.95 | 3.73 | 0.00 |
| train | degree_capped_mst | 8 | 0.6208 | 0.6190 | 0.6232 | 1.008 | 1.01 | 1.95 | 3.00 | 0.00 |
| train | degree_capped_mst | 12 | 0.6208 | 0.6190 | 0.6233 | 1.008 | 1.00 | 1.95 | 3.00 | 0.00 |
| train | degree_capped_mst | 16 | 0.6206 | 0.6187 | 0.6230 | 1.008 | 1.00 | 1.95 | 3.00 | 0.00 |
| train | budgeted_knn_forest | 8 | 0.6252 | 0.6234 | 0.6276 | 1.008 | 1.01 | 1.95 | 3.73 | 0.00 |
| train | budgeted_knn_forest | 12 | 0.6251 | 0.6232 | 0.6275 | 1.008 | 1.00 | 1.95 | 3.75 | 0.00 |
| train | budgeted_knn_forest | 16 | 0.6247 | 0.6228 | 0.6272 | 1.008 | 1.00 | 1.95 | 3.74 | 0.00 |
| train | density_normalized_mst | 8 | 0.6291 | 0.6272 | 0.6315 | 1.008 | 1.01 | 1.95 | 3.56 | 0.00 |
| train | density_normalized_mst | 12 | 0.6290 | 0.6273 | 0.6311 | 1.008 | 1.00 | 1.95 | 3.60 | 0.00 |
| train | density_normalized_mst | 16 | 0.6285 | 0.6268 | 0.6307 | 1.008 | 1.00 | 1.95 | 3.65 | 0.00 |
| train | mutual_knn_sparse | 8 | 0.4337 | 0.2889 | 0.8755 | 3.048 | 2.94 | 5.90 | 8.00 | 104.80 |
| train | mutual_knn_sparse | 12 | 0.3347 | 0.2045 | 0.9287 | 4.583 | 2.25 | 8.86 | 11.96 | 180.52 |
| train | mutual_knn_sparse | 16 | 0.2668 | 0.1554 | 0.9522 | 6.186 | 1.97 | 11.97 | 15.85 | 261.63 |
| train | hybrid_mst_plus_branches | 8 | 0.6035 | 0.5421 | 0.6811 | 1.258 | 1.01 | 2.44 | 5.48 | 12.82 |
| train | hybrid_mst_plus_branches | 12 | 0.5994 | 0.5384 | 0.6766 | 1.258 | 1.00 | 2.44 | 5.46 | 12.82 |
| train | hybrid_mst_plus_branches | 16 | 0.5970 | 0.5363 | 0.6739 | 1.258 | 1.00 | 2.44 | 5.46 | 12.82 |
| val | knn_mst | 8 | 0.6351 | 0.6334 | 0.6372 | 1.007 | 1.02 | 1.94 | 3.70 | 0.00 |
| val | knn_mst | 12 | 0.6348 | 0.6331 | 0.6369 | 1.007 | 1.01 | 1.94 | 3.72 | 0.00 |
| val | knn_mst | 16 | 0.6348 | 0.6330 | 0.6369 | 1.007 | 1.01 | 1.94 | 3.71 | 0.00 |
| val | degree_capped_mst | 8 | 0.6323 | 0.6306 | 0.6343 | 1.007 | 1.02 | 1.94 | 2.99 | 0.00 |
| val | degree_capped_mst | 12 | 0.6322 | 0.6305 | 0.6343 | 1.007 | 1.01 | 1.94 | 2.99 | 0.00 |
| val | degree_capped_mst | 16 | 0.6322 | 0.6305 | 0.6343 | 1.007 | 1.01 | 1.94 | 2.99 | 0.00 |
| val | budgeted_knn_forest | 8 | 0.6354 | 0.6337 | 0.6375 | 1.007 | 1.02 | 1.94 | 3.71 | 0.00 |
| val | budgeted_knn_forest | 12 | 0.6348 | 0.6330 | 0.6368 | 1.007 | 1.01 | 1.94 | 3.73 | 0.00 |
| val | budgeted_knn_forest | 16 | 0.6347 | 0.6329 | 0.6368 | 1.007 | 1.01 | 1.94 | 3.72 | 0.00 |
| val | density_normalized_mst | 8 | 0.6409 | 0.6391 | 0.6430 | 1.007 | 1.02 | 1.94 | 3.54 | 0.00 |
| val | density_normalized_mst | 12 | 0.6396 | 0.6379 | 0.6417 | 1.007 | 1.01 | 1.94 | 3.60 | 0.00 |
| val | density_normalized_mst | 16 | 0.6402 | 0.6385 | 0.6423 | 1.007 | 1.01 | 1.94 | 3.61 | 0.00 |
| val | mutual_knn_sparse | 8 | 0.4351 | 0.2884 | 0.8900 | 3.101 | 2.46 | 5.98 | 8.00 | 93.26 |
| val | mutual_knn_sparse | 12 | 0.3321 | 0.2021 | 0.9393 | 4.680 | 1.94 | 9.03 | 11.96 | 160.75 |
| val | mutual_knn_sparse | 16 | 0.2633 | 0.1529 | 0.9582 | 6.325 | 1.73 | 12.20 | 15.81 | 232.06 |
| val | hybrid_mst_plus_branches | 8 | 0.6172 | 0.5545 | 0.6962 | 1.257 | 1.02 | 2.43 | 5.19 | 11.21 |
| val | hybrid_mst_plus_branches | 12 | 0.6136 | 0.5513 | 0.6922 | 1.257 | 1.01 | 2.43 | 5.23 | 11.21 |
| val | hybrid_mst_plus_branches | 16 | 0.6123 | 0.5501 | 0.6907 | 1.257 | 1.01 | 2.43 | 5.22 | 11.21 |
| test | knn_mst | 8 | 0.6279 | 0.6261 | 0.6304 | 1.008 | 1.03 | 1.94 | 3.72 | 0.00 |
| test | knn_mst | 12 | 0.6278 | 0.6258 | 0.6303 | 1.008 | 1.02 | 1.94 | 3.73 | 0.00 |
| test | knn_mst | 16 | 0.6277 | 0.6258 | 0.6302 | 1.008 | 1.01 | 1.95 | 3.74 | 0.00 |
| test | degree_capped_mst | 8 | 0.6238 | 0.6220 | 0.6262 | 1.008 | 1.03 | 1.94 | 2.99 | 0.00 |
| test | degree_capped_mst | 12 | 0.6234 | 0.6215 | 0.6259 | 1.008 | 1.02 | 1.94 | 2.99 | 0.00 |
| test | degree_capped_mst | 16 | 0.6239 | 0.6220 | 0.6265 | 1.008 | 1.01 | 1.95 | 2.99 | 0.00 |
| test | budgeted_knn_forest | 8 | 0.6285 | 0.6266 | 0.6310 | 1.008 | 1.03 | 1.94 | 3.74 | 0.00 |
| test | budgeted_knn_forest | 12 | 0.6278 | 0.6259 | 0.6304 | 1.008 | 1.02 | 1.94 | 3.74 | 0.00 |
| test | budgeted_knn_forest | 16 | 0.6280 | 0.6260 | 0.6306 | 1.008 | 1.01 | 1.95 | 3.75 | 0.00 |
| test | density_normalized_mst | 8 | 0.6339 | 0.6322 | 0.6363 | 1.008 | 1.03 | 1.94 | 3.56 | 0.00 |
| test | density_normalized_mst | 12 | 0.6321 | 0.6302 | 0.6346 | 1.008 | 1.02 | 1.94 | 3.60 | 0.00 |
| test | density_normalized_mst | 16 | 0.6317 | 0.6298 | 0.6343 | 1.008 | 1.01 | 1.95 | 3.64 | 0.00 |
| test | mutual_knn_sparse | 8 | 0.4305 | 0.2855 | 0.8808 | 3.106 | 2.62 | 5.98 | 8.00 | 94.39 |
| test | mutual_knn_sparse | 12 | 0.3290 | 0.2001 | 0.9317 | 4.698 | 2.01 | 9.05 | 11.96 | 163.51 |
| test | mutual_knn_sparse | 16 | 0.2616 | 0.1518 | 0.9549 | 6.359 | 1.75 | 12.25 | 15.83 | 236.39 |
| test | hybrid_mst_plus_branches | 8 | 0.6088 | 0.5467 | 0.6875 | 1.259 | 1.03 | 2.43 | 5.17 | 11.37 |
| test | hybrid_mst_plus_branches | 12 | 0.6055 | 0.5437 | 0.6840 | 1.260 | 1.02 | 2.43 | 5.21 | 11.37 |
| test | hybrid_mst_plus_branches | 16 | 0.6038 | 0.5421 | 0.6821 | 1.260 | 1.01 | 2.43 | 5.21 | 11.37 |

## Notes
- Train split uses the first N samples controlled by --train-subset (default 200).
- All methods are deterministic and run from GT joints/active masks only.