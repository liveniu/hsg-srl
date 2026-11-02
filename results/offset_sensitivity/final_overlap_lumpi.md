# Final overlap-gated LUMPI reference sensitivity

Same frozen models, support, polygons, matched observations, and clock alignment. Only the scoring-reference translation changes. Errors pool the seven recordings within each seed before averaging the three seed statistics. Tracking is not rescored.

| Reference treatment | Fixed midpoint median / RMSE (m) | HSG-SRL mean median / RMSE (m) | Median / RMSE reduction (%) | Both errors lower in every seed |
| --- | ---: | ---: | ---: | --- |
| HSG-derived offset | 0.645 / 0.986 | 0.553 / 0.835 | 14.3 / 15.3 | Yes |
| Offset removed | 0.622 / 0.989 | 0.559 / 0.846 | 10.1 / 14.4 | Yes |
| Offset from midpoint | 0.601 / 1.001 | 0.580 / 0.885 | 3.5 / 11.6 | Yes |
| Offset from bottom center | 0.905 / 1.269 | 0.753 / 1.214 | 16.7 / 4.3 | Yes |
