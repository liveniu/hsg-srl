# Geometry-only TrackEval BEV-HOTA

Built from results/anchor_tracking/*.json. Same detections, test keys, ground-truth clock, reference bias, full overlap graph, and final-tracking flags as the HSG rows. No residual. The replay is seven LUMPI recordings and five Synthehicle towns. Ko-PER is not included.

Recording-macro means on a 0-100 scale. The mean column is the unweighted average of 0.5, 1, 1.5, and 2 m.

| Data | Coordinate rule | 0.5 m | 1 m | 1.5 m | 2 m | Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LUMPI | HSG (per-detection) | 23.84 | 38.56 | 46.01 | 48.33 | 39.18 |
| LUMPI | Box-center projection | 6.65 | 20.48 | 29.66 | 36.16 | 23.24 |
| LUMPI | Fixed interior midpoint | 20.79 | 38.22 | 45.74 | 48.23 | 38.25 |
| LUMPI | STMC-derived fixed anchor | 11.68 | 30.47 | 42.36 | 46.49 | 32.75 |
| LUMPI | Bottom-center projection (IPM) | 2.09 | 7.19 | 19.48 | 34.34 | 15.77 |
| Synthetic | HSG (per-detection) | 24.76 | 43.29 | 51.15 | 55.48 | 43.67 |
| Synthetic | Box-center projection | 3.52 | 10.00 | 15.76 | 20.97 | 12.56 |
| Synthetic | Fixed interior midpoint | 15.03 | 33.77 | 43.43 | 50.14 | 35.59 |
| Synthetic | STMC-derived fixed anchor | 32.39 | 46.24 | 51.54 | 54.38 | 46.14 |
| Synthetic | Bottom-center projection (IPM) | 5.32 | 16.48 | 27.96 | 41.79 | 22.89 |
| Twelve recordings | HSG (per-detection) | 24.22 | 40.53 | 48.15 | 51.31 | 41.05 |
| Twelve recordings | Box-center projection | 5.35 | 16.12 | 23.87 | 29.83 | 18.79 |
| Twelve recordings | Fixed interior midpoint | 18.39 | 36.37 | 44.78 | 49.02 | 37.14 |
| Twelve recordings | STMC-derived fixed anchor | 20.31 | 37.04 | 46.19 | 49.78 | 38.33 |
| Twelve recordings | Bottom-center projection (IPM) | 3.43 | 11.06 | 23.01 | 37.45 | 18.74 |

## Per recording, mean of four gates

| Recording | Box center | Fixed interior midpoint | STMC-derived fixed anchor | Bottom center | HSG |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 | 16.39 | 29.76 | 25.88 | 14.69 | 30.41 |
| M1 | 32.08 | 63.35 | 51.90 | 25.01 | 63.05 |
| M2 | 28.11 | 43.31 | 33.52 | 12.17 | 44.94 |
| M3 | 20.57 | 30.17 | 26.80 | 16.19 | 31.81 |
| M4 | 21.83 | 32.76 | 28.56 | 10.97 | 34.31 |
| M5 | 23.12 | 34.59 | 32.82 | 16.24 | 35.33 |
| M6 | 20.57 | 33.79 | 29.78 | 15.15 | 34.44 |
| Town01 | 11.71 | 26.42 | 39.22 | 19.71 | 34.37 |
| Town02 | 11.10 | 34.00 | 44.01 | 21.89 | 40.17 |
| Town03 | 14.61 | 37.95 | 50.63 | 23.01 | 48.65 |
| Town04 | 13.08 | 38.72 | 46.98 | 26.80 | 45.61 |
| Town05 | 12.30 | 40.88 | 49.85 | 23.02 | 49.55 |
