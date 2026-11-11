# Training-pair coverage of the projection-sensitivity tertiles

Edges are the training-observation tertiles already used for the conditional-error figure. Alpha is feature index 5. A pair side is one detection in a frozen training pair. The higher-alpha member is the more sensitive view of that pair. Published filter: larger kappa at least 0.20, and kappa gap at least 0.05 unless the edge waives the gap. Kappa = 1 − alpha.

## Datasets

| Dataset | Top-tertile edge | Train observations | Pair sides | Higher-alpha member | Test detections | Test inside overlap | Top-tertile test inside overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LUMPI | 0.802 | 33.3% (n=2,875,341) | 20.1% (n=394,810) | 40.2% (n=197,405) | 28.3% (n=827,083) | 27.4% (n=814,988) | 95.2% |
| Ko-PER | 0.732 | 33.3% (n=77,526) | 0.0% (n=14,680) | 0.0% (n=7,340) | 14.4% (n=25,122) | 0.0% (n=8,287) | 0.0% |
| Synthehicle | 0.818 | 33.3% (n=168,740) | 2.3% (n=127,112) | 4.6% (n=63,556) | 23.4% (n=83,361) | 23.4% (n=83,333) | 99.9% |

Train-observation shares are one third by construction of the tertiles, up to ties.

## Recordings

| Recording | Pairs | Train top | Pair-side top | Higher-alpha top | Test top | P(overlap | test top) | Teacher κ≥0.20 | |Δκ|≥0.05 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 | 0 | 21.7% (n=472,311) | — | — | 7.4% (n=46,169) | 100.0% | — | — |
| M1 | 0 | 22.5% (n=454,417) | — | — | 6.6% (n=184,442) | 100.0% | — | — |
| M2 | 36,705 | 27.0% (n=454,293) | 19.0% (n=73,410) | 37.9% (n=36,705) | 9.7% (n=136,095) | 89.8% | 100.0% | 100.0% |
| M3 | 54,962 | 20.0% (n=515,926) | 0.0% (n=109,924) | 0.0% (n=54,962) | 6.4% (n=85,103) | 99.8% | 100.0% | 100.0% |
| M4 | 27,576 | 54.6% (n=234,054) | 27.0% (n=55,152) | 53.9% (n=27,576) | 49.7% (n=96,411) | 79.4% | 100.0% | 100.0% |
| M5 | 40,868 | 52.3% (n=411,495) | 33.9% (n=81,736) | 67.7% (n=40,868) | 54.4% (n=144,480) | 100.0% | 100.0% | 100.0% |
| M6 | 37,294 | 55.6% (n=332,845) | 30.8% (n=74,588) | 61.5% (n=37,294) | 54.9% (n=134,383) | 100.0% | 100.0% | 100.0% |
| Ko-PER | 7,340 | 33.3% (n=77,526) | 0.0% (n=14,680) | 0.0% (n=7,340) | 14.4% (n=25,122) | 0.0% | 100.0% | 100.0% |
| Town01 | 1,991 | 44.9% (n=22,324) | 0.0% (n=3,982) | 0.0% (n=1,991) | 27.5% (n=8,301) | 100.0% | 100.0% | 100.0% |
| Town02 | 1,317 | 21.2% (n=20,279) | 0.0% (n=2,634) | 0.0% (n=1,317) | 22.9% (n=11,536) | 100.0% | 100.0% | 100.0% |
| Town03 | 34,426 | 36.9% (n=71,748) | 3.1% (n=68,852) | 6.1% (n=34,426) | 25.2% (n=37,121) | 100.0% | 100.0% | 100.0% |
| Town04 | 1,578 | 28.7% (n=10,105) | 5.6% (n=3,156) | 11.2% (n=1,578) | 14.7% (n=8,441) | 99.4% | 100.0% | 100.0% |
| Town05 | 24,244 | 28.4% (n=44,284) | 1.4% (n=48,488) | 2.7% (n=24,244) | 22.4% (n=17,962) | 99.5% | 100.0% | 100.0% |
