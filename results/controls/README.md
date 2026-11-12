# Additional recorded comparisons

These files are the manuscript's additional comparisons. They use the same frozen detections, pairs, checkpoints, and association rules as `results/recorded/`, scored on the stored footprint centers.

Primary localization and tracking are rebuilt by:

```text
python scripts/export_support_localization.py
python scripts/eval_fixed_anchors.py --publish
python scripts/build_tables.py
python scripts/build_supplement_tables.py
```

`scripts/check_release.py` checks the rebuilt primary tables and the strings below against the manuscript.

| File | Manuscript |
| --- | --- |
| `RESULTS.md` | Per-domain anchors, per-camera constants, and their tracking |
| `table_v.md` | Four-gate tracking of the geometric anchors, including the midpoint baseline |
| `beta_selection.md` | Unlabeled and training-GT choices of one beta per domain |
| `conditional_error.md` | Errors by projection-sensitivity tertile |
| `class_breakdown.md` | Errors by detector class |
| `hsg_constants_robustness.md` | Neighboring s0 and eta |
| `time_isolation.json` | Latest pair time against the test-interval start |
| `match_probe_stage_a.md` | Midpoint-coordinate reassignment under the stored clocks, Table S2(a) |
| `clock_check.md` | Clock-source check on M2, M6, and Ko-PER, Table S2(b). Rebuild with `python scripts/assignment/clock_check.py` |
| `match_summary.md` | Per-camera assignment rates and stored clocks |
| `table_s3_closure.md` | Detector-row partition, including unassigned-track rows, Table S3. Rebuild with `python scripts/assignment/table_s3_closure.py` |
| `constraint_split/summary.md` | Bound only, and penalties only |
| `unconstrained/summary.md` | MVC only, with no output bound |
| `pair_alpha_coverage/summary.md` | Where the frozen training pairs sit in the sensitivity bins |
| `constant/` | Fitted per-camera offsets |
| `affine/` | Fitted per-camera affine maps; localization only |
| `track/` | Re-associated constant and unlabeled-beta tracking scores |

The per-sample residual dumps used while fitting the constraint variants are not part of this release. The summaries are the scored results.
