# Table V with fixed anchors on all datasets (untranslated references)

Ko-PER archive check: HSG rows match published Geo rows (max abs 0.0e+00); paper merge flags present in all logs.

| Dataset / rule | 0.5 m | 1 m | 1.5 m | 2 m | HOTA4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LUMPI / Box-center projection | 6.65 | 18.70 | 29.16 | 35.82 | 22.58 |
| LUMPI / Fixed interior midpoint | 21.30 | 38.78 | 45.61 | 47.97 | 38.41 |
| LUMPI / STMC-derived fixed anchor | 13.33 | 31.89 | 43.26 | 46.25 | 33.68 |
| LUMPI / Bottom-center projection (IPM) | 2.06 | 7.51 | 20.42 | 34.77 | 16.19 |
| LUMPI / HSG | 23.36 | 39.19 | 45.89 | 48.00 | 39.11 |
| LUMPI / HSG-SRL | 24.75 ± 0.30 | 42.14 ± 0.04 | 47.78 ± 0.05 | 49.93 ± 0.02 | 41.15 ± 0.07 |
| Ko-PER / Box-center projection | 6.95 | 12.28 | 23.97 | 38.59 | 20.45 |
| Ko-PER / Fixed interior midpoint | 41.58 | 57.27 | 60.96 | 62.08 | 55.47 |
| Ko-PER / STMC-derived fixed anchor | 26.00 | 53.48 | 59.85 | 60.72 | 50.01 |
| Ko-PER / Bottom-center projection (IPM) | 0.47 | 3.76 | 27.51 | 56.86 | 22.15 |
| Ko-PER / HSG | 48.47 | 61.93 | 63.79 | 64.20 | 59.60 |
| Ko-PER / HSG-SRL | 50.32 ± 0.20 | 64.61 ± 0.07 | 66.36 ± 0.08 | 66.68 ± 0.09 | 61.99 ± 0.10 |
| Synthehicle / Box-center projection | 3.52 | 10.00 | 15.76 | 20.97 | 12.56 |
| Synthehicle / Fixed interior midpoint | 15.03 | 33.77 | 43.43 | 50.14 | 35.59 |
| Synthehicle / STMC-derived fixed anchor | 32.39 | 46.24 | 51.54 | 54.38 | 46.14 |
| Synthehicle / Bottom-center projection (IPM) | 5.32 | 16.48 | 27.96 | 41.79 | 22.89 |
| Synthehicle / HSG | 24.76 | 43.29 | 51.15 | 55.48 | 43.67 |
| Synthehicle / HSG-SRL | 20.02 ± 0.90 | 45.39 ± 0.21 | 51.68 ± 0.41 | 56.47 ± 0.17 | 43.39 ± 0.39 |

## Per recording, HOTA4

| Recording | Box-center projection | Fixed interior midpoint | STMC-derived fixed anchor | Bottom-center projection (IPM) | HSG | HSG-SRL mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 | 15.39 | 30.53 | 27.76 | 15.61 | 31.31 | 31.31 |
| M1 | 27.07 | 65.19 | 57.35 | 31.20 | 61.95 | 61.95 |
| M2 | 28.11 | 43.31 | 33.52 | 12.17 | 44.94 | 45.07 |
| M3 | 21.99 | 28.74 | 25.96 | 11.99 | 31.47 | 31.08 |
| M4 | 21.83 | 32.76 | 28.56 | 10.97 | 34.31 | 38.97 |
| M5 | 23.12 | 34.59 | 32.82 | 16.24 | 35.33 | 41.33 |
| M6 | 20.57 | 33.79 | 29.78 | 15.15 | 34.44 | 38.34 |
| Ko-PER | 20.45 | 55.47 | 50.01 | 22.15 | 59.60 | 61.99 |
| Town01 | 11.71 | 26.42 | 39.22 | 19.71 | 34.37 | 36.04 |
| Town02 | 11.10 | 34.00 | 44.01 | 21.89 | 40.17 | 39.87 |
| Town03 | 14.61 | 37.95 | 50.63 | 23.01 | 48.65 | 46.29 |
| Town04 | 13.08 | 38.72 | 46.98 | 26.80 | 45.61 | 45.96 |
| Town05 | 12.30 | 40.88 | 49.85 | 23.02 | 49.55 | 48.80 |

Differences to state: HSG-SRL minus midpoint and HSG minus midpoint per dataset:
- LUMPI: HSG-SRL − midpoint +2.74; HSG − midpoint +0.69; HSG-SRL − HSG +2.04
- Ko-PER: HSG-SRL − midpoint +6.52; HSG − midpoint +4.13; HSG-SRL − HSG +2.39
- Synthehicle: HSG-SRL − midpoint +7.80; HSG − midpoint +8.08; HSG-SRL − HSG -0.28
