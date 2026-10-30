# Tables rebuilt inside the artifact

Localization and tracking use the published deployment: a trained residual
on a supported camera only where the fixed HSG point lies in the frozen overlap union.

## Localization, HSG-SRL is the three-seed mean +/- sample SD

| Stratum | IPM median | HSG median | HSG-SRL median | HSG p90 | HSG-SRL p90 | HSG RMSE | HSG-SRL RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LUMPI | 1.891 | 0.604 | 0.553 +/- 0.005 | 1.324 | 1.285 +/- 0.006 | 0.861 | 0.835 +/- 0.005 |
| LUMPI-pole | 1.672 | 0.542 | 0.542 +/- 0.000 | 1.122 | 1.122 +/- 0.000 | 0.785 | 0.785 +/- 0.000 |
| LUMPI-A | 1.892 | 0.568 | 0.609 +/- 0.003 | 1.591 | 1.577 +/- 0.001 | 0.911 | 0.929 +/- 0.005 |
| LUMPI-B | 2.055 | 0.667 | 0.533 +/- 0.009 | 1.308 | 1.214 +/- 0.013 | 0.876 | 0.806 +/- 0.007 |
| Ko-PER | 1.640 | 0.313 | 0.302 +/- 0.002 | 0.758 | 0.703 +/- 0.002 | 0.579 | 0.561 +/- 0.003 |
| Synthehicle | 1.569 | 0.657 | 0.669 +/- 0.010 | 1.615 | 1.599 +/- 0.033 | 1.063 | 1.040 +/- 0.009 |

## Fixed anchors, median / RMSE (m)

| Anchor | LUMPI | Ko-PER | Synthehicle |
| --- | ---: | ---: | ---: |
| HSG (per-detection) | 0.604 / 0.861 | 0.313 / 0.579 | 0.657 / 1.063 |
| Box-center projection | 1.626 / 2.441 | 1.665 / 2.128 | 3.117 / 4.482 |
| Bottom-center projection (IPM) | 1.891 / 2.046 | 1.640 / 1.740 | 1.569 / 1.671 |
| Sensitivity only | 1.014 / 1.226 | 0.560 / 0.764 | 0.599 / 0.868 |
| Shape only | 0.861 / 1.372 | 0.564 / 0.754 | 1.477 / 2.380 |
| Fixed interior midpoint | 0.645 / 0.986 | 0.381 / 0.688 | 0.792 / 1.590 |
| STMC-derived fixed anchor | 0.966 / 1.172 | 0.742 / 0.904 | 0.513 / 0.960 |

## Per-recording geometric anchors, median / RMSE / p90 (m)

All rules use each recording's identical matched detections and GT references.
No residual is applied. Values are read from results/recorded/ipm_baseline.json.

| Recording | Matched detections | Rule | Median | RMSE | p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 | 46,169 | Bottom-center projection (IPM) | 1.721 | 1.948 | 2.618 |
| M0 | 46,169 | Sensitivity only | 0.879 | 1.169 | 1.686 |
| M0 | 46,169 | Shape only | 1.020 | 1.607 | 2.386 |
| M0 | 46,169 | Fixed interior midpoint | 0.702 | 1.283 | 1.742 |
| M0 | 46,169 | STMC-derived fixed anchor | 0.941 | 1.230 | 1.696 |
| M0 | 46,169 | HSG | 0.748 | 1.138 | 1.654 |
| M1 | 184,442 | Bottom-center projection (IPM) | 1.655 | 1.780 | 2.363 |
| M1 | 184,442 | Sensitivity only | 0.710 | 0.937 | 1.420 |
| M1 | 184,442 | Shape only | 0.731 | 0.986 | 1.439 |
| M1 | 184,442 | Fixed interior midpoint | 0.498 | 0.676 | 1.047 |
| M1 | 184,442 | STMC-derived fixed anchor | 0.836 | 0.985 | 1.453 |
| M1 | 184,442 | HSG | 0.514 | 0.667 | 0.999 |
| M2 | 136,095 | Bottom-center projection (IPM) | 1.869 | 2.012 | 2.592 |
| M2 | 136,095 | Sensitivity only | 0.881 | 1.114 | 1.672 |
| M2 | 136,095 | Shape only | 0.529 | 0.938 | 1.381 |
| M2 | 136,095 | Fixed interior midpoint | 0.580 | 0.771 | 1.060 |
| M2 | 136,095 | STMC-derived fixed anchor | 1.028 | 1.146 | 1.587 |
| M2 | 136,095 | HSG | 0.447 | 0.669 | 1.053 |
| M3 | 85,103 | Bottom-center projection (IPM) | 1.959 | 2.251 | 3.337 |
| M3 | 85,103 | Sensitivity only | 0.969 | 1.315 | 2.150 |
| M3 | 85,103 | Shape only | 1.169 | 1.653 | 2.180 |
| M3 | 85,103 | Fixed interior midpoint | 0.911 | 1.311 | 2.102 |
| M3 | 85,103 | STMC-derived fixed anchor | 1.143 | 1.502 | 2.550 |
| M3 | 85,103 | HSG | 0.971 | 1.200 | 1.833 |
| M4 | 96,411 | Bottom-center projection (IPM) | 2.183 | 2.224 | 2.874 |
| M4 | 96,411 | Sensitivity only | 1.264 | 1.371 | 1.873 |
| M4 | 96,411 | Shape only | 0.994 | 1.673 | 2.832 |
| M4 | 96,411 | Fixed interior midpoint | 0.833 | 1.210 | 1.735 |
| M4 | 96,411 | STMC-derived fixed anchor | 1.056 | 1.272 | 1.923 |
| M4 | 96,411 | HSG | 0.725 | 0.934 | 1.418 |
| M5 | 144,480 | Bottom-center projection (IPM) | 1.971 | 2.075 | 2.724 |
| M5 | 144,480 | Sensitivity only | 1.184 | 1.316 | 1.871 |
| M5 | 144,480 | Shape only | 1.033 | 1.567 | 2.411 |
| M5 | 144,480 | Fixed interior midpoint | 0.654 | 1.025 | 1.412 |
| M5 | 144,480 | STMC-derived fixed anchor | 0.889 | 1.093 | 1.645 |
| M5 | 144,480 | HSG | 0.618 | 0.808 | 1.199 |
| M6 | 134,383 | Bottom-center projection (IPM) | 2.081 | 2.149 | 2.776 |
| M6 | 134,383 | Sensitivity only | 1.224 | 1.419 | 2.041 |
| M6 | 134,383 | Shape only | 0.931 | 1.447 | 2.276 |
| M6 | 134,383 | Fixed interior midpoint | 0.705 | 0.960 | 1.367 |
| M6 | 134,383 | STMC-derived fixed anchor | 1.014 | 1.180 | 1.776 |
| M6 | 134,383 | HSG | 0.701 | 0.903 | 1.350 |
| Ko-PER | 25,122 | Bottom-center projection (IPM) | 1.640 | 1.740 | 1.979 |
| Ko-PER | 25,122 | Sensitivity only | 0.560 | 0.764 | 0.943 |
| Ko-PER | 25,122 | Shape only | 0.564 | 0.754 | 1.021 |
| Ko-PER | 25,122 | Fixed interior midpoint | 0.381 | 0.688 | 0.952 |
| Ko-PER | 25,122 | STMC-derived fixed anchor | 0.742 | 0.904 | 1.231 |
| Ko-PER | 25,122 | HSG | 0.313 | 0.579 | 0.758 |
| Town01 | 8,301 | Bottom-center projection (IPM) | 1.501 | 1.833 | 2.399 |
| Town01 | 8,301 | Sensitivity only | 0.510 | 1.040 | 1.193 |
| Town01 | 8,301 | Shape only | 1.745 | 3.204 | 5.012 |
| Town01 | 8,301 | Fixed interior midpoint | 1.167 | 2.580 | 4.268 |
| Town01 | 8,301 | STMC-derived fixed anchor | 0.542 | 1.280 | 2.111 |
| Town01 | 8,301 | HSG | 0.814 | 1.442 | 2.388 |
| Town02 | 11,536 | Bottom-center projection (IPM) | 1.633 | 1.838 | 2.324 |
| Town02 | 11,536 | Sensitivity only | 0.578 | 0.986 | 1.199 |
| Town02 | 11,536 | Shape only | 1.720 | 2.563 | 3.638 |
| Town02 | 11,536 | Fixed interior midpoint | 0.915 | 1.689 | 2.585 |
| Town02 | 11,536 | STMC-derived fixed anchor | 0.536 | 1.079 | 1.441 |
| Town02 | 11,536 | HSG | 0.740 | 1.208 | 1.757 |
| Town03 | 37,121 | Bottom-center projection (IPM) | 1.538 | 1.609 | 2.039 |
| Town03 | 37,121 | Sensitivity only | 0.624 | 0.827 | 1.030 |
| Town03 | 37,121 | Shape only | 1.484 | 2.165 | 2.901 |
| Town03 | 37,121 | Fixed interior midpoint | 0.743 | 1.351 | 1.876 |
| Town03 | 37,121 | STMC-derived fixed anchor | 0.494 | 0.866 | 1.191 |
| Town03 | 37,121 | HSG | 0.625 | 0.939 | 1.356 |
| Town04 | 8,441 | Bottom-center projection (IPM) | 1.585 | 1.526 | 1.914 |
| Town04 | 8,441 | Sensitivity only | 0.625 | 0.762 | 1.170 |
| Town04 | 8,441 | Shape only | 1.503 | 2.544 | 3.998 |
| Town04 | 8,441 | Fixed interior midpoint | 0.926 | 1.705 | 2.837 |
| Town04 | 8,441 | STMC-derived fixed anchor | 0.713 | 1.004 | 1.503 |
| Town04 | 8,441 | HSG | 0.806 | 1.230 | 2.080 |
| Town05 | 17,962 | Bottom-center projection (IPM) | 1.616 | 1.671 | 2.239 |
| Town05 | 17,962 | Sensitivity only | 0.597 | 0.828 | 1.230 |
| Town05 | 17,962 | Shape only | 1.342 | 2.143 | 3.218 |
| Town05 | 17,962 | Fixed interior midpoint | 0.711 | 1.297 | 1.829 |
| Town05 | 17,962 | STMC-derived fixed anchor | 0.449 | 0.868 | 1.178 |
| Town05 | 17,962 | HSG | 0.562 | 0.901 | 1.323 |

## Recording-macro TrackEval BEV-HOTA

| Data / method | 0.5 m | 1 m | 1.5 m | 2 m | mean of four |
| --- | ---: | ---: | ---: | ---: | ---: |
| LUMPI / HSG | 23.84 | 38.56 | 46.01 | 48.33 | 39.18 |
| LUMPI / HSG-SRL | 25.59 +/- 0.18 | 41.40 +/- 0.08 | 47.90 +/- 0.08 | 50.12 +/- 0.07 | 41.25 +/- 0.09 |
| Ko-PER / HSG | 48.47 | 61.93 | 63.79 | 64.20 | 59.60 |
| Ko-PER / HSG-SRL | 50.32 +/- 0.20 | 64.61 +/- 0.07 | 66.36 +/- 0.08 | 66.68 +/- 0.09 | 61.99 +/- 0.10 |
| Synthehicle / HSG | 24.76 | 43.29 | 51.15 | 55.48 | 43.67 |
| Synthehicle / HSG-SRL | 20.02 +/- 0.90 | 45.39 +/- 0.21 | 51.68 +/- 0.41 | 56.47 +/- 0.17 | 43.39 +/- 0.39 |

## Per-recording median and seed-0 bootstrap interval for HSG-SRL minus HSG

| Recording | HSG | HSG-SRL mean +/- SD | Seed-0 difference [interval] |
| --- | ---: | ---: | ---: |
| ko_per | 0.313 | 0.302 +/- 0.002 | -0.012 [-0.039, -0.005] |
| M0 | 0.748 | 0.748 +/- 0.000 | +0.000 [+0.000, +0.000] |
| M1 | 0.514 | 0.514 +/- 0.000 | +0.000 [+0.000, +0.000] |
| M2 | 0.447 | 0.495 +/- 0.003 | +0.045 [+0.011, +0.079] |
| M3 | 0.971 | 0.990 +/- 0.006 | +0.025 [-0.048, +0.105] |
| M4 | 0.725 | 0.574 +/- 0.008 | -0.142 [-0.243, -0.058] |
| M5 | 0.618 | 0.488 +/- 0.007 | -0.135 [-0.171, -0.091] |
| M6 | 0.701 | 0.551 +/- 0.025 | -0.128 [-0.184, -0.071] |
| Town01 | 0.814 | 0.707 +/- 0.023 | -0.120 [-0.339, +0.047] |
| Town02 | 0.740 | 0.795 +/- 0.005 | +0.049 [-0.131, +0.155] |
| Town03 | 0.625 | 0.645 +/- 0.026 | +0.041 [-0.019, +0.083] |
| Town04 | 0.806 | 0.852 +/- 0.018 | +0.025 [-0.081, +0.180] |
| Town05 | 0.562 | 0.575 +/- 0.024 | +0.006 [-0.085, +0.106] |

## Detection and association accuracy

| Data / gate | HSG DetA | HSG-SRL DetA | HSG AssA | HSG-SRL AssA |
| --- | ---: | ---: | ---: | ---: |
| LUMPI / 0.5 m | 13.60 | 14.77 +/- 0.23 | 42.91 | 45.49 +/- 0.30 |
| LUMPI / 1 m | 28.79 | 30.99 +/- 0.18 | 52.34 | 55.92 +/- 0.36 |
| LUMPI / 1.5 m | 37.25 | 38.47 +/- 0.04 | 57.11 | 59.95 +/- 0.24 |
| LUMPI / 2 m | 39.98 | 41.25 +/- 0.04 | 58.62 | 61.11 +/- 0.24 |
| Ko-PER / 0.5 m | 38.42 | 40.63 +/- 0.22 | 61.15 | 62.32 +/- 0.17 |
| Ko-PER / 1 m | 53.91 | 56.84 +/- 0.11 | 71.16 | 73.43 +/- 0.04 |
| Ko-PER / 1.5 m | 56.12 | 59.00 +/- 0.11 | 72.51 | 74.63 +/- 0.04 |
| Ko-PER / 2 m | 56.60 | 59.44 +/- 0.13 | 72.83 | 74.81 +/- 0.05 |
| Synthehicle / 0.5 m | 14.11 | 11.53 +/- 0.67 | 43.58 | 35.02 +/- 1.41 |
| Synthehicle / 1 m | 31.41 | 33.50 +/- 0.50 | 59.92 | 61.75 +/- 0.42 |
| Synthehicle / 1.5 m | 40.93 | 41.31 +/- 0.49 | 64.15 | 64.97 +/- 0.39 |
| Synthehicle / 2 m | 46.43 | 47.54 +/- 0.36 | 66.53 | 67.33 +/- 0.32 |

## Localization accuracy of retained matches

| Data / gate | HSG LocA | HSG-SRL LocA |
| --- | ---: | ---: |
| LUMPI / 0.5 m | 88.92 | 89.23 +/- 0.12 |
| LUMPI / 1 m | 80.44 | 81.01 +/- 0.11 |
| LUMPI / 1.5 m | 74.74 | 76.08 +/- 0.20 |
| LUMPI / 2 m | 72.31 | 73.58 +/- 0.17 |
| Ko-PER / 0.5 m | 89.77 | 89.77 +/- 0.05 |
| Ko-PER / 1 m | 86.35 | 86.51 +/- 0.08 |
| Ko-PER / 1.5 m | 85.52 | 85.76 +/- 0.04 |
| Ko-PER / 2 m | 85.22 | 85.51 +/- 0.05 |
| Synthehicle / 0.5 m | 88.25 | 86.44 +/- 0.20 |
| Synthehicle / 1 m | 80.75 | 77.66 +/- 0.15 |
| Synthehicle / 1.5 m | 75.18 | 74.10 +/- 0.15 |
| Synthehicle / 2 m | 71.58 | 70.20 +/- 0.20 |

## Per-recording BEV-HOTA

| Recording | HSG 0.5 m | HSG-SRL 0.5 m | HSG 2 m | HSG-SRL 2 m |
| --- | ---: | ---: | ---: | ---: |
| ko_per | 48.47 | 50.32 +/- 0.20 | 64.20 | 66.68 +/- 0.09 |
| M0 | 19.48 | 19.48 +/- 0.00 | 39.36 | 39.36 +/- 0.00 |
| M1 | 39.30 | 39.30 +/- 0.00 | 73.85 | 73.85 +/- 0.00 |
| M2 | 30.27 | 27.00 +/- 0.25 | 52.18 | 53.66 +/- 0.33 |
| M3 | 18.00 | 18.08 +/- 0.14 | 43.18 | 43.12 +/- 0.30 |
| M4 | 20.58 | 23.52 +/- 0.60 | 43.30 | 47.91 +/- 0.48 |
| M5 | 21.42 | 27.01 +/- 1.30 | 42.76 | 48.02 +/- 0.21 |
| M6 | 17.81 | 24.78 +/- 1.80 | 43.65 | 44.93 +/- 0.35 |
| Town01 | 25.64 | 21.81 +/- 1.97 | 41.52 | 43.11 +/- 0.26 |
| Town02 | 20.69 | 12.20 +/- 3.44 | 53.95 | 55.26 +/- 0.01 |
| Town03 | 29.35 | 22.35 +/- 4.26 | 58.14 | 58.27 +/- 1.13 |
| Town04 | 20.62 | 19.70 +/- 1.31 | 62.78 | 64.16 +/- 0.53 |
| Town05 | 27.50 | 24.03 +/- 3.21 | 61.01 | 61.57 +/- 0.55 |
