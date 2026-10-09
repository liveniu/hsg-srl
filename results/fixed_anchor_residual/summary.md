# Fixed-Anchor Residual Substitution Check

This auxiliary localization check retrains the submitted residual architecture once
(seed 0) on two fixed interior anchors. It uses the same final localization cohorts
as the main paper. Each fixed-anchor run forms its own residual-support set from
the same pair-extraction protocol, so this is a substitution check that complements
the three-seed main-protocol comparison.

| Base anchor | Group | Base median | Base + residual median | Base RMSE | Base + residual RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| \(\beta=0.50\) | Real | 0.645 | 0.671 | 0.986 | 1.036 |
| \(\beta=0.50\) | All 12 | 0.655 | 0.680 | 1.056 | 1.096 |
| \(\beta=0.70\) | Real | 0.966 | 0.923 | 1.172 | 1.140 |
| \(\beta=0.70\) | All 12 | 0.934 | 0.890 | 1.154 | 1.126 |

The check complements the geometry-only tracking replay. The residual improves
the weaker real-data \(\beta=0.70\) localization base in this seed-0 run, and the
submitted HSG-SRL chain retains the strongest real pooled median, 0.553 m over
three seeds.
