# Robustness of HSG versus midpoint to s0 and eta

Pre-registered: the published values (s0 = 60 m, eta = 0.75) are not changed by this table. Same matched test detections, untranslated references; only the anchor rule is re-evaluated. Cells are HSG median / RMSE / p90 in metres; flags mark whether HSG is below the fixed midpoint in that statistic.

## LUMPI (midpoint 0.622 / 0.989 / 1.452)

| s0 (m) \ eta | 0.5 | 0.75 | 1 |
| --- | ---: | ---: | ---: |
| 30 | 0.921 / 1.168 / 1.795 [>>>] | 0.756 / 1.027 / 1.609 [>>>] | 0.661 / 0.930 / 1.472 [><>] |
| 60 | 0.643 / 0.912 / 1.422 [><<] | 0.602 / 0.874 / 1.339 [<<<] * | 0.655 / 0.936 / 1.464 [><>] |
| 120 | 0.582 / 0.875 / 1.364 [<<<] | 0.672 / 1.001 / 1.570 [>>>] | 0.816 / 1.191 / 1.876 [>>>] |

## Ko-PER (midpoint 0.381 / 0.688 / 0.952)

| s0 (m) \ eta | 0.5 | 0.75 | 1 |
| --- | ---: | ---: | ---: |
| 30 | 0.576 / 0.777 / 0.930 [>><] | 0.380 / 0.654 / 0.820 [<<<] | 0.298 / 0.587 / 0.762 [<<<] |
| 60 | 0.302 / 0.572 / 0.714 [<<<] | 0.313 / 0.579 / 0.758 [<<<] * | 0.474 / 0.662 / 0.882 [><<] |
| 120 | 0.385 / 0.603 / 0.855 [><<] | 0.458 / 0.658 / 0.894 [><<] | 0.537 / 0.737 / 0.975 [>>>] |

## Synthehicle (midpoint 0.792 / 1.590 / 2.459)

| s0 (m) \ eta | 0.5 | 0.75 | 1 |
| --- | ---: | ---: | ---: |
| 30 | 0.593 / 0.875 / 1.194 [<<<] | 0.493 / 0.841 / 1.208 [<<<] | 0.501 / 0.897 / 1.341 [<<<] |
| 60 | 0.477 / 0.870 / 1.295 [<<<] | 0.657 / 1.063 / 1.615 [<<<] * | 0.932 / 1.330 / 2.010 [><<] |
| 120 | 0.803 / 1.227 / 1.926 [><<] | 1.079 / 1.527 / 2.358 [><<] | 1.364 / 1.859 / 2.841 [>>>] |

Flags: three characters for median, RMSE, p90; `<` means HSG below the midpoint. `*` marks the published setting.

## Real recordings in which HSG has lower RMSE and p90 than the midpoint (of 8)

| s0 (m) \ eta | 0.5 | 0.75 | 1 |
| --- | ---: | ---: | ---: |
| 30 | RMSE 1, p90 2, median 0 | RMSE 4, p90 4, median 2 | RMSE 6, p90 4, median 4 |
| 60 | RMSE 7, p90 6, median 4 | RMSE 7, p90 7, median 6 | RMSE 6, p90 5, median 4 |
| 120 | RMSE 7, p90 6, median 5 | RMSE 5, p90 3, median 3 | RMSE 0, p90 1, median 1 |

Interpretation rule: report whether the published setting lies inside a region where the pooled ordering is unchanged; do not report which cell is best.
