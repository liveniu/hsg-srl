# Reproduction

Commands run from the artifact root. Work products go to `data/work/sites/<recording>/`.

Manuscript section and table numbers are only in `docs/PAPER_MAP.md`. This file names the operation.

## What calls what

`scripts/run_paper.py` is the ordered chain for one recording. It does not build the summary tables, and it does not run the optional board, Orin, or BEVHeight paths.

| Stage | Script | Reads | Writes | Library |
| --- | --- | --- | --- | --- |
| Import | `scripts/import_recordings.py` | Downloaded video and labels | `origin.db` | `src/importers/import_lumpi.py`, `import_synthehicle.py`, `import_koper.py` |
| Detect | `scripts/detect_and_merge.py` | `origin.db`, `configs/sites/`, `configs/botsort_static.yaml`, `configs/pseudo_identity_overlap/` | `geometry.db` | `src/pipeline/multi_camera_trajectory_fusion.py` |
| Attach | `scripts/attach_ground_truth.py` | `geometry.db`, `origin.db` | `det_gt_match_report.json` | `src/evaluation/eval_fab2_det_gt_match.py` |
| Audit | `scripts/audit_reference.py` | Matched detections | `reference_verdict.txt` | — |
| Split | `scripts/make_time_split.py` | Unique timestamps | `train_keys.csv`, `val_keys.csv`, `test_keys.csv` | `src/evaluation/fab2_manifest.py` |
| Overlap | `scripts/build_overlap.py` | Homographies in `geometry.db` | Working copy under `data/work/sites/<recording>/overlap.json` | `src/geometry/fov_overlap.py` `candidate_edges`. Published sites still mine and track on `configs/overlap` |
| Pairs | `scripts/freeze_pairs.py` | `geometry.db`, `configs/overlap/` | Frozen pair arrays | HSG anchor in `multi_camera_trajectory_fusion.py` |
| Train | `scripts/train_recording.py` | Frozen pairs, `configs/paper_defaults.yaml` | `F0` checkpoint and `deploy_support` JSON | `GeometryGuidedResidualMLP` class only. Do not call that module's `train` command |
| Anchors | `scripts/eval_fixed_anchors.py` | Matched test rows | Fixed-anchor errors | Same anchor helpers |
| Localize | `scripts/eval_localization.py` | Test rows, checkpoints, deploy JSON | Localization arrays | — |
| Track | `scripts/eval_tracking.py` | `configs/overlap/`, checkpoints | 10 Hz bins and HOTA JSON | Fusion remerge, then `src/evaluation/hota_trackeval.py` and `third_party/TrackEval/` |

The checked-in graphs in `configs/overlap/` are the published overlap graphs, including Ko-PER. `build_overlap.py` can rewrite a working copy under `data/work/`. Pair mining and final tracking read `configs/overlap/` when that file exists. They do not read `configs/pseudo_identity_overlap/` or `configs/sites/*/trajectory_merge`. The working copy is the from-video graph for a recording that has no frozen file.

Published numbers without videos or a geometry database:

| Operation | Script | Reads |
| --- | --- | --- |
| Check saved HOTA | `scripts/replay_trackeval.py` | `results/recorded/bins/`, `identity_hota25.csv`, `identity_hota25_support.csv`, `identity_hota25_overlap.csv` |
| Write the score summaries | `scripts/build_tables.py` | `results/recorded/loc_support/`, `ipm_baseline.json`, the two HOTA CSV files |
| Reference-frame rescore | `scripts/eval_offset_sensitivity.py` | Released `geometry.db` plus the recorded localization arrays |
| Geometry-only anchor tracking | `scripts/eval_anchor_tracking.py --summarize` | `results/anchor_tracking/*.json` |
| Board-fit residuals | `scripts/report_board_capture.py` | `results/board_capture/fit_residuals.json` |

`src/pipeline/multi_camera_bev_stitch.py` and `tools/` are the optional local-frame path. `reference/orin/` and `reference/bevheight/` are separate from `run_paper.py`. `scripts/run_smoke_test.py` checks the anchor, the 4,802-parameter bound, and the loss; it does not score a recording.

## Published tables from released databases

YOLO26l detections are not identical across machines. The published tables use the detections, tracks, and identities in the geometry databases at https://doi.org/10.5281/zenodo.23008421. The archive root is the contents of `data/work/`. Extract it into `data/work/`, so each recording is at `data/work/sites/<recording>/`, then continue from pair mining:

```text
python scripts/run_paper.py --site lumpi_M6 --from-stage overlap
```

The default `--from-stage import` detects again and replaces `geometry.db`. Do not use it when the released database is already in place.

Sections 1--3 run detection on downloaded video and build a new geometry database. That path follows the same method for a new installation. It does not reproduce the published tables.

## 1. Import

`scripts/import_recordings.py`

- LUMPI: dataset homography, batch `l0hn_gt_001`.
- Synthehicle daytime towns: dataset homography, tight boxes, automatic ground height.
- Ko-PER: one recording, dataset calibration, footprint center stored at import. Detection is `scripts/detect_koper.py` (confidence 0.25, IoU 0.70, three-in-frame admission), then `scripts/remerge_pseudo_identity.py`.
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

`scripts/build_overlap.py` writes a working field-of-view graph. A fresh LUMPI or Synthehicle graph keeps a projected-hull intersection of at least 1000 m². The frozen graphs in `configs/overlap` were built under that cutoff: Town03 keeps 27 of 28 edges because C02–C04 covers 995.58 m², and Town04 keeps 5 of 6 because C02–C04 covers 538.42 m². A fresh Ko-PER graph keeps an intersection of at least 500 m², which retains its 579.35 m² SK1–SK4 edge. With these per-dataset cutoffs the builder regenerates all thirteen frozen edge sets. Published pair mining and tracking read `configs/overlap` and apply no further area filter.

`scripts/freeze_pairs.py` mines unlabeled pairs on HSG positions. The gate is 2 m, the time window is 0.35 s, and the confidence thresholds are 0.20 and 0.05.

`scripts/train_recording.py` is the published trainer. It reads the frozen pairs, fits the 7-64-64-2 residual for 80 epochs with Adam at 10^-3 and batch 256, and writes the F0 checkpoint and deploy file. Seeds are 0, 1, and 2. Each coordinate is bounded by 1 m. The loss weights are 0.10, 1.00, 0.50, and 0.02. A recording with fewer than 32 training pairs stays on HSG. Otherwise every camera with a retained validation pair is in the support set. At scoring, the residual is added only where the stored HSG point also lies in that camera's frozen overlap union. `--symmetric` fits the supplement's pair-symmetric control to a separate checkpoint and does not replace F0.

```text
python scripts/train_recording.py --site lumpi_M6 --seed 0
```

`src/pipeline/geometry_guided_residual_mlp.py train` cannot replace that command. It does not read the frozen pair file, and the published trainer does not call it. It mines pairs from the geometry database and still exits unless `--split-manifest` is set. Its numeric defaults match the published loss. Supplying those numbers does not make that command the paper fit.

## 5. Score

`scripts/eval_fixed_anchors.py` projects the geometric controls on the matched test rows: box center, bottom center, sensitivity only, shape only, the interior midpoint (beta 0.50), and the STMC-derived anchor (beta 0.70, at y1+0.85h). It also checks that a fresh HSG projection matches the stored positions. `--publish` rewrites `results/recorded/ipm_baseline.json`.

`scripts/eval_localization.py` reports median, 90th percentile, RMSE, and the 1000-draw paired bootstrap clustered by camera and local track, seed 20260921.

`scripts/eval_tracking.py` writes the overlap-gated coordinates, uses `configs/overlap`, and reruns fusion without reading the site YAML. `--save-bins` stores the 10 Hz TrackEval inputs. `--no-spatial-gate` is the supplement ablation that keeps camera support and drops the overlap mask. `scripts/export_overlap_bins.py` fills the overlap-row hashes after the replay matches the published scores. `scripts/build_supplement_tables.py` writes Tables S9–S11 and S16. The command passes score 0.70, distance 2.5 m, time gap 1.2 s, 5 samples, 12 resamples, scales 0.6 s / 0.75 m / 0.35, weights 0.25 / 0.45 / 0.30, and same-camera stitching off. Similarity is max(0, 1 - d / 2.5 m). Those values are also the fusion library defaults. The command still passes them because a site YAML replaces a flag only when the flag is left at its default, and `configs/sites/*/trajectory_merge` is the pseudo-identity rebuild rather than the final-tracking gates.

`results/recorded/residual_support.csv` and `results/recorded/deploy_support/` record the retained validation-pair support set. `results/recorded/identity_hota25_overlap.csv` stores the published HSG-SRL tracking rows, with the residual applied only inside the frozen overlap union. `results/recorded/identity_hota25_support.csv` is the control without spatial gating: the same models and camera support with the overlap mask removed. `scripts/build_supplement_tables.py` reads it for the spatial-gating control. Checkpoint filenames use the tag `F0`. That tag is this residual, not a second method.

`scripts/build_tables.py` writes the IPM / HSG / HSG-SRL localization chain, the pooled and per-recording geometric-anchor comparisons, recording-macro BEV-HOTA, and the per-recording bootstrap, detection and association breakdown, retained-match localization, and per-recording HOTA from `results/recorded/`. The geometric-anchor tables read `ipm_baseline.json`; the per-recording table gives matched counts, median, RMSE, and p90 for bottom-center, sensitivity-only, shape-only, the two fixed interior anchors, and HSG on all thirteen domains. It uses the recorded matched cohorts and reference positions and requires no new projection or tracking run.

`scripts/replay_trackeval.py` checks the saved tracking scores against those recordings.

## Stored summaries that are not in build_tables.py

These commands read stored results. They do not retrain.

- Reference-frame translation: `python scripts/eval_offset_sensitivity.py` writes `results/offset_sensitivity/summary.md`. The committed summary is the published rescore. Re-running it needs the released geometry databases.
- Geometry-only anchor tracking: `python scripts/eval_anchor_tracking.py --summarize` rebuilds `results/anchor_tracking/summary.md` from the stored JSON. A new fusion replay is `python scripts/eval_anchor_tracking.py --site lumpi_M6` and needs `geometry.db`. It passes the same final-tracking flags as `scripts/eval_tracking.py`.
- Supervised BEVHeight reference: the archived scores are `results/bevheight/score_three.md` and `results/bevheight/hota_remerge.md`. A new supervised run is `docs/BEVHEIGHT.md`.
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

`reference/orin/` times M6, Town05, and Ko-PER. `results/orin/runtime.md` is that recorded measurement, together with the board configuration. A later run with the same configuration is a new measurement. It is not required to match the stored percentiles. See `docs/ORIN.md`.

These commands rebuild the published localization and tracking tables from released databases. They do not retrain, and they do not time the board:

```text
python scripts/export_support_localization.py
python scripts/assemble_hota_tables.py
python scripts/build_tables.py
python scripts/check_release.py
```

`export_support_localization.py` reads `geometry.db`, the test split, and `results/recorded/checkpoints`. `assemble_hota_tables.py` reads the vendored score files in `results/recorded/hota_sources`; those files carry no bin hashes, so it keeps the `input_sha256` already stored for a row with unchanged scores. `scripts/export_overlap_bins.py` is the only command that writes new hashes, after a fusion replay from the geometry databases. `replay_trackeval.py` checks the geometry rows, the spatial-gating control, and the overlap-gated HOTA rows against `results/recorded/bins`. The overlap-gated rows use `bins/*_overlap_s*.npz`.

`reference/bevheight/` is the supervised reference on M6, M3, and Town05. See `docs/BEVHEIGHT.md`.
