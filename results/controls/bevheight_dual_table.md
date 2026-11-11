# BEVHeight reference, two matching protocols (main-text candidate)

| Recording | Protocol | n | HSG-SRL s0 median / RMSE | BEVHeight median / RMSE | >5 m count SRL / BEVHeight | >5 m rate SRL / BEVHeight |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| M6 | greedy containment | 130,177 | 0.563 / 0.830 | 0.503 / 2.107 | 15 / 7210 | 0.012% / 5.539% |
| M6 | unique containment | 68,569 | 0.575 / 0.825 | 0.352 / 0.903 | 14 / 421 | 0.020% / 0.614% |
| M3 | greedy containment | 80,002 | 0.887 / 1.280 | 0.797 / 1.981 | 41 / 5089 | 0.051% / 6.361% |
| M3 | unique containment | 32,341 | 0.754 / 0.997 | 0.464 / 0.841 | 28 / 33 | 0.087% / 0.102% |
| Town05 | greedy containment | 15,983 | 0.564 / 0.836 | 0.266 / 1.333 | 3 / 443 | 0.019% / 2.772% |
| Town05 | unique containment | 11,021 | 0.544 / 0.782 | 0.292 / 0.835 | 3 / 115 | 0.027% / 1.043% |

Full-input mixed tracking (BEVHeight coordinates on matched observations, HSG elsewhere), BEV-HOTA4: M6 35.3, M3 31.9, Town05 56.8. HSG and seed-0 HSG-SRL full-input scores are in the untranslated rescore.

M3 rates under unique containment are 28 and 33 of 32,341 and are reported as counts; no significance claim.
