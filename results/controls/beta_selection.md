# Per-domain fixed beta

Unlabeled: mean squared cross-view disagreement on the frozen training pairs, 0.05 grid; no pairs -> 0.50. Labeled reference: training-split GT median (primary) and RMSE; not a test oracle.

| Recording | Train pairs | Unlabeled beta | Labeled beta (median) | Labeled beta (RMSE) | HSG projection check (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ko-PER | 7,340 | 0.35 | 0.50 | 0.55 | 2.74e-06 |
| M0 | 0 | 0.50 (fallback) | 0.60 | 0.60 |  |
| M1 | 0 | 0.50 (fallback) | 0.50 | 0.55 |  |
| M2 | 36,705 | 0.50 | 0.40 | 0.50 | 7.39e-06 |
| M3 | 54,962 | 0.35 | 0.45 | 0.45 | 3.73e-06 |
| M4 | 27,576 | 0.45 | 0.50 | 0.60 | 7.24e-06 |
| M5 | 40,868 | 0.50 | 0.50 | 0.60 | 1.07e-05 |
| M6 | 37,294 | 0.45 | 0.50 | 0.55 | 1.13e-05 |
| Town01 | 1,991 | 0.45 | 0.70 | 0.85 | 3.20e-06 |
| Town02 | 1,317 | 0.40 | 0.60 | 0.75 | 4.59e-06 |
| Town03 | 34,426 | 0.50 | 0.65 | 0.75 | 6.90e-06 |
| Town04 | 1,578 | 0.55 | 0.65 | 0.75 | 5.46e-06 |
| Town05 | 24,244 | 0.55 | 0.60 | 0.70 | 5.07e-06 |
