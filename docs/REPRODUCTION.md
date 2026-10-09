# Reproduction

Commands run from the artifact root. Work products go to `data/work/sites/<recording>/`.

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
- Association uses the overlap in `configs/pseudo_identity_overlap`, the same edges as the released geometry databases.

## 3. Ground truth and split

`scripts/attach_ground_truth.py` writes the detector-to-ground-truth attachment. It calls `src/evaluation/eval_fab2_det_gt_match.py`. The gates are at least eight overlapping samples, median residual 3 m, runner-up margin 0.75 m, a 0.12 s distance to the nearest annotation, and a 10 m sanity cutoff. The 0.12 s rule rejects a query more than 0.12 s from every annotation. Ground truth is linearly interpolated between annotations, and a query within 0.12 s outside a track's time span uses the nearest endpoint. It does not reject every point inside a longer annotation gap: points near either end of that gap can remain eligible.

`scripts/audit_reference.py` writes `reference_verdict.txt`. It checks stationary attached samples and writes `REFERENCE OFFSET` only when the camera-averaged common displacement is larger than 0.25 m and larger than the between-camera spread. The common bias is applied only when the file starts with `REFERENCE OFFSET`; otherwise any stored `common_bias` line is ignored by scoring.

`scripts/make_time_split.py` splits unique timestamps 60/10/30.

## 4. Pairs and residual

`scripts/build_overlap.py` writes the projected field-of-view graph used for pair mining and tracking. An edge is kept when the intersection is at least 1000 m².

`scripts/freeze_pairs.py` mines unlabeled pairs on HSG positions. The gate is 2 m, the time window is 0.35 s, and the confidence thresholds are 0.20 and 0.05.

`scripts/train_recording.py` trains the 7-64-64-2 residual for 80 epochs, Adam at 10^-3, batch 256, seeds 0, 1, and 2. Each coordinate is bounded by 1 m. A recording with fewer than 32 training pairs stays on HSG. Otherwise every camera with a retained validation pair uses the residual.

## 5. Score

`scripts/eval_fixed_anchors.py` projects beta 0, 0.50, 0.70, and 1 on the matched test rows.

`scripts/eval_localization.py` reports median, 90th percentile, RMSE, and the 1000-draw paired bootstrap clustered by camera and local track, seed 20260921.

`scripts/eval_tracking.py` writes the evaluated coordinates, uses `configs/overlap`, and reruns fusion at score 0.70 and 2.5 m with same-camera stitching off. Similarity is max(0, 1 - d / 2.5 m).

`results/recorded/residual_support.csv` and `results/recorded/deploy_support/` record the retained validation-pair support set. `results/recorded/identity_hota25_support.csv` stores the published HSG-SRL tracking rows under that support rule. Checkpoint filenames use the tag `F0` for this residual.

`scripts/build_tables.py` writes Tables III--V and supplementary Tables S6 and S9--S11 from `results/recorded/`.

`scripts/replay_trackeval.py` checks the saved tracking scores against those recordings.

## Edge timing and the supervised reference

`reference/orin/` times M6 and Town05. The reported timings are `results/orin/runtime.md`.

`reference/bevheight/` is the supervised reference on M6, M3, and Town05. See `docs/BEVHEIGHT.md`.
