# Geometry-only TrackEval BEV-HOTA

Built from results/anchor_tracking/*.json. Same detections, test keys, ground-truth clock, stored footprint centers, full overlap graph, and final-tracking flags as the HSG rows. No residual. The replay is seven LUMPI recordings, Ko-PER, and five Synthehicle towns.

Recording-macro means on a 0-100 scale. The mean column is the unweighted average of 0.5, 1, 1.5, and 2 m.

| Data | Coordinate rule | 0.5 m | 1 m | 1.5 m | 2 m | Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LUMPI | HSG (per-detection) | 23.36 | 39.19 | 45.89 | 48.00 | 39.11 |
| LUMPI | Box-center projection | 6.65 | 18.70 | 29.16 | 35.82 | 22.58 |
| LUMPI | Fixed interior midpoint | 21.30 | 38.78 | 45.61 | 47.97 | 38.41 |
| LUMPI | STMC-derived fixed anchor | 13.33 | 31.89 | 43.26 | 46.25 | 33.68 |
| LUMPI | Bottom-center projection (IPM) | 2.06 | 7.51 | 20.42 | 34.77 | 16.19 |
| Ko-PER | HSG (per-detection) | 48.47 | 61.93 | 63.79 | 64.20 | 59.60 |
| Ko-PER | Box-center projection | 6.95 | 12.28 | 23.97 | 38.59 | 20.45 |
| Ko-PER | Fixed interior midpoint | 41.58 | 57.27 | 60.96 | 62.08 | 55.47 |
| Ko-PER | STMC-derived fixed anchor | 26.00 | 53.48 | 59.85 | 60.72 | 50.01 |
| Ko-PER | Bottom-center projection (IPM) | 0.47 | 3.76 | 27.51 | 56.86 | 22.15 |
| Synthehicle | HSG (per-detection) | 24.76 | 43.29 | 51.15 | 55.48 | 43.67 |
| Synthehicle | Box-center projection | 3.52 | 10.00 | 15.76 | 20.97 | 12.56 |
| Synthehicle | Fixed interior midpoint | 15.03 | 33.77 | 43.43 | 50.14 | 35.59 |
| Synthehicle | STMC-derived fixed anchor | 32.39 | 46.24 | 51.54 | 54.38 | 46.14 |
| Synthehicle | Bottom-center projection (IPM) | 5.32 | 16.48 | 27.96 | 41.79 | 22.89 |

## Per recording, mean of four gates

| Recording | Box center | Fixed interior midpoint | STMC-derived fixed anchor | Bottom center | HSG |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 | 15.39 | 30.53 | 27.76 | 15.61 | 31.31 |
| M1 | 27.07 | 65.19 | 57.35 | 31.20 | 61.95 |
| M2 | 28.11 | 43.31 | 33.52 | 12.17 | 44.94 |
| M3 | 21.99 | 28.74 | 25.96 | 11.99 | 31.47 |
| M4 | 21.83 | 32.76 | 28.56 | 10.97 | 34.31 |
| M5 | 23.12 | 34.59 | 32.82 | 16.24 | 35.33 |
| M6 | 20.57 | 33.79 | 29.78 | 15.15 | 34.44 |
| Ko-PER | 20.45 | 55.47 | 50.01 | 22.15 | 59.60 |
| Town01 | 11.71 | 26.42 | 39.22 | 19.71 | 34.37 |
| Town02 | 11.10 | 34.00 | 44.01 | 21.89 | 40.17 |
| Town03 | 14.61 | 37.95 | 50.63 | 23.01 | 48.65 |
| Town04 | 13.08 | 38.72 | 46.98 | 26.80 | 45.61 |
| Town05 | 12.30 | 40.88 | 49.85 | 23.02 | 49.55 |
