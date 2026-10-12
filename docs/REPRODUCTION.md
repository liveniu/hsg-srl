# Reproduction

Commands run from the artifact root. Work products go to `data/work/sites/<recording>/`.

Manuscript section and table numbers are only in `docs/PAPER_MAP.md`. This file names the operation.

## What calls what

`scripts/run_paper.py` is the ordered chain for one recording. It does not build the summary tables, and it does not run the optional board, Orin, or BEVHeight paths.

| Stage | Script | Reads | Writes | Library |
| --- | --- | --- | --- | --- |
| Import | `scripts/import_recordings.py` | Downloaded video and labels | `origin.db` | `src/importers/import_lumpi.py`, `import_synthehicle.py` |
| Detect | `scripts/detect_and_merge.py` | `origin.db`, `configs/sites/`, `configs/botsort_static.yaml`, `configs/pseudo_identity_overlap/` | `geometry.db` | `src/pipeline/multi_camera_trajectory_fusion.py` |
| Attach | `scripts/attach_ground_truth.py` | `geometry.db`, `origin.db` | `det_gt_match_report.json` | `src/evaluation/eval_fab2_det_gt_match.py` |
| Audit | `scripts/audit_reference.py` | Matched detections | `reference_verdict.txt` | — |
| Split | `scripts/make_time_split.py` | Unique timestamps | `train_keys.csv`, `val_keys.csv`, `test_keys.csv` | `src/evaluation/fab2_manifest.py` |
| Overlap | `scripts/build_overlap.py` | Homographies in `geometry.db` | `data/work/sites/<recording>/overlap.json` | `src/geometry/fov_overlap.py` `candidate_edges` |
| Pairs | `scripts/freeze_pairs.py` | `geometry.db`, `configs/overlap/` | Frozen pair arrays | HSG anchor in `multi_camera_trajectory_fusion.py` |
| Train | `scripts/train_recording.py` | Frozen pairs, `configs/paper_defaults.yaml` | `F0` checkpoint and `deploy_support` JSON | `GeometryGuidedResidualMLP` class only. Do not call that module's `train` command |
| Anchors | `scripts/eval_fixed_anchors.py` | Matched test rows | Fixed-anchor errors | Same anchor helpers |
| Localize | `scripts/eval_localization.py` | Test rows, checkpoints, deploy JSON | Localization arrays | — |
| Track | `scripts/eval_tracking.py` | `configs/overlap/`, checkpoints | 10 Hz bins and HOTA JSON | Fusion remerge, then `src/evaluation/hota_trackeval.py` and `third_party/TrackEval/` |

The checked-in graphs in `configs/overlap/` are the published full overlap graphs. `build_overlap.py` can rewrite a working copy under `data/work/`. Final tracking reads `configs/overlap/`, not `configs/pseudo_identity_overlap/` and not `configs/sites/*/trajectory_merge`.

Published numbers without videos or a geometry database:

| Operation | Script | Reads |
| --- | --- | --- |
| Check saved HOTA | `scripts/replay_trackeval.py` | `results/recorded/bins/`, `identity_hota25.csv`, `identity_hota25_support.csv` |
| Write the score summaries | `scripts/build_tables.py` | `results/recorded/loc_support/`, `ipm_baseline.json`, the two HOTA CSV files |
| Reference-frame rescore | `scripts/eval_offset_sensitivity.py` | Released `geometry.db` plus the recorded localization arrays |
| Geometry-only anchor tracking | `scripts/eval_anchor_tracking.py --summarize` | `results/anchor_tracking/*.json` |
| Board-fit residuals | `scripts/report_board_capture.py` | `results/board_capture/fit_residuals.json` |

`src/pipeline/multi_camera_bev_stitch.py` and `tools/` are the optional local-frame path. `reference/orin/` and `reference/bevheight/` are separate from `run_paper.py`. `scripts/run_smoke_test.py` checks the anchor, the 4,802-parameter bound, and the loss; it does not score a recording.

## Published tables from released databases

YOLO26l detections are not identical across machines. The published tables use the detections, tracks, and identities in the geometry databases at https://doi.org/10.5281/zenodo.23008421. Extract each recording directory to `data/work/sites/<recording>/`, then continue from pair mining:

```text
python scripts/run_paper.py --site lumpi_M6 --from-stage overlap
```

The default `--from-stage import` detects again and replaces `geometry.db`. Do not use it when the released database is already in place.

Sections 1--3 run detection on downloaded video and build a new geometry database. That path follows the same method for a new installation. It does not reproduce the published tables.

## 1. Import

`scripts/import_recordings.py`

- LUMPI: dataset homography, batch `l0hn_gt_001`.
- Synthehicle daytime towns: dataset homography, tight boxes, automatic ground height.
- Official test clips are not part of this protocol.

## 2. Detect

`scripts/detect_and_merge.py`

- YOLO26l, input 384 by 640, classes 2, 5, and 7.
- BoT-SORT with camera-motion compensation off.
- LUMPI frames stop at 441.5 s, and M6 at 360 s.
- Association edges come from `configs/pseudo_identity_overlap`, the same edges as the released geometry databases.
- The geometry remerge passes pseudo-identity gates on the command line. LUMPI uses score 0.70, distance 2.5 m, distance scale 1.5 m, and shape scale 0.50. Synthehicle uses score 0.75, distance 2.0 m, distance scale 1.0 m, and shape scale 0.35. The same numbers sit in `configs/sites/*/trajectory_merge` so a YAML-only call still matches this rebuild. They are not the final-tracking gates. Final tracking does not read them.

## 3. Ground truth and split

`scripts/attach_ground_truth.py` writes the detector-to-ground-truth attachment. It calls `src/evaluation/eval_fab2_det_gt_match.py`. The gates are at least eight overlapping samples, median residual 3 m, runner-up margin 0.75 m, a 0.12 s distance to the nearest annotation, and a 10 m sanity cutoff. The 0.12 s rule rejects a query more than 0.12 s from every annotation. Ground truth is linearly interpolated between annotations, and a query within 0.12 s outside a track's time span uses the nearest endpoint. It does not reject every point inside a longer annotation gap: points near either end of that gap can remain eligible.

`scripts/audit_reference.py` writes `reference_verdict.txt`. It checks stationary attached samples and writes `REFERENCE OFFSET` only when the camera-averaged common displacement is larger than 0.25 m and larger than the between-camera spread. The common bias is applied only when the file starts with `REFERENCE OFFSET`; otherwise any stored `common_bias` line is ignored by scoring.

`scripts/make_time_split.py` splits unique timestamps 60/10/30.

## 4. Pairs and residual

`scripts/build_overlap.py` writes the projected field-of-view graph used for pair mining and tracking. An edge is kept when the intersection is at least 1000 m².

`scripts/freeze_pairs.py` mines unlabeled pairs on HSG positions. The gate is 2 m, the time window is 0.35 s, and the confidence thresholds are 0.20 and 0.05.

`scripts/train_recording.py` is the published trainer. It reads the frozen pairs, fits the 7-64-64-2 residual for 80 epochs with Adam at 10^-3 and batch 256, and writes the F0 checkpoint and deploy file. Seeds are 0, 1, and 2. Each coordinate is bounded by 1 m. The loss weights are 0.10, 1.00, 0.50, and 0.02. A recording with fewer than 32 training pairs stays on HSG. Otherwise every camera with a retained validation pair uses the residual.

```text
python scripts/train_recording.py --site lumpi_M6 --seed 0
```

`src/pipeline/geometry_guided_residual_mlp.py train` cannot replace that command. It does not read the frozen pair file, and the published trainer does not call it. It mines pairs from the geometry database and still exits unless `--split-manifest` is set. Its numeric defaults match the published loss. Supplying those numbers does not make that command the paper fit.

## 5. Score

`scripts/eval_fixed_anchors.py` projects the fixed controls on the matched test rows: box center (beta 0), the interior midpoint (beta 0.50, at y1+0.75h), the STMC-derived anchor (beta 0.70, at y1+0.85h), and bottom center (beta 1).

`scripts/eval_localization.py` reports median, 90th percentile, RMSE, and the 1000-draw paired bootstrap clustered by camera and local track, seed 20260921.

`scripts/eval_tracking.py` writes the evaluated coordinates, uses `configs/overlap`, and reruns fusion without reading the site YAML. The command passes score 0.70, distance 2.5 m, time gap 1.2 s, 5 samples, 12 resamples, scales 0.6 s / 0.75 m / 0.35, weights 0.25 / 0.45 / 0.30, and same-camera stitching off. Similarity is max(0, 1 - d / 2.5 m). Those values are also the fusion library defaults. The command still passes them because a site YAML replaces a flag only when the flag is left at its default, and `configs/sites/*/trajectory_merge` is the pseudo-identity rebuild rather than the final-tracking gates.

`results/recorded/residual_support.csv` and `results/recorded/deploy_support/` record the retained validation-pair support set. `results/recorded/identity_hota25_support.csv` stores the published HSG-SRL tracking rows under that support rule. Checkpoint filenames use the tag `F0`. That tag is this residual, not a second method.

`scripts/build_tables.py` writes the IPM / HSG / HSG-SRL localization chain, the fixed-anchor comparison, recording-macro BEV-HOTA, and the per-recording bootstrap, detection and association breakdown, retained-match localization, and per-recording HOTA from `results/recorded/`.

`scripts/replay_trackeval.py` checks the saved tracking scores against those recordings.

## Stored summaries that are not in build_tables.py

These commands read stored results. They do not retrain.

- Reference-frame translation: `python scripts/eval_offset_sensitivity.py` writes `results/offset_sensitivity/summary.md`. The committed summary is the published rescore. Re-running it needs the released geometry databases.
- Geometry-only anchor tracking: `python scripts/eval_anchor_tracking.py --summarize` rebuilds `results/anchor_tracking/summary.md` from the stored JSON. A new fusion replay is `python scripts/eval_anchor_tracking.py --site lumpi_M6` and needs `geometry.db`. It passes the same final-tracking flags as `scripts/eval_tracking.py`.
- Supervised BEVHeight reference: the archived scores are `results/bevheight/score_three.md` and `results/bevheight/hota_remerge.md`. A new supervised run is `docs/BEVHEIGHT.md`.
- One-seed residual on fixed anchors: `results/fixed_anchor_residual/summary.md`. That file is the stored check. It is not regenerated by the frozen replay.
- Pair counts are `results/recorded/pair_counts/`. The board-fit residuals are the next section.

## Optional board capture

`src/pipeline/multi_camera_bev_stitch.py` is the optional ChArUco local-frame procedure. The main experiments do not call it. The helpers are:

```text
python tools/print_ChArUco.py --square-length 0.40 --format svg,pdf --start-id 0
python tools/diagnose_charuco.py --config path/to/scene.yaml
python tools/mark_charuco_corners.py --config path/to/scene.yaml
```

`python scripts/report_board_capture.py` prints the board-fit residuals from `results/board_capture/fit_residuals.json`. That JSON is 19 KB and is the published fit. Scene 02's second camera moves from 0.459 to 0.456 BEV pixels after refinement; the other cameras stay at the before-refinement RMS.

The six scenes are 14 uncompressed BMP stills, 180,634,368 bytes, about 172 MiB. The rest of the capture directory is unused duplicates and is about 691 MB. Those stills are not in this git repository and do not belong in the geometry-database deposit at https://doi.org/10.5281/zenodo.23008421. The published result is the stored reprojection RMS, so a separate Zenodo record is not required to check the board-fit residuals. Re-fitting from pixels would need those 14 stills plus the scene YAML and manual corners. That image archive is optional and should be its own deposit if it is released later.

## Edge timing and the supervised reference

`reference/orin/` times M6 and Town05. The reported timings are `results/orin/runtime.md`.

`reference/bevheight/` is the supervised reference on M6, M3, and Town05. See `docs/BEVHEIGHT.md`.
