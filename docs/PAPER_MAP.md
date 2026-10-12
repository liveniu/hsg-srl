# Paper pointer

This is the only artifact file that uses manuscript section, table, and figure numbers. When those numbers change, edit this file. Commands, comments, and result headings elsewhere use the function names in the right-hand column.

`docs/REPRODUCTION.md` is the operational chain. It does not repeat these numbers.

The manuscript tree `paper/` is not part of the git release. Scoring does not read it.

## Main text

| Manuscript | What it specifies | Artifact |
| --- | --- | --- |
| I | Deployment setting and the two contributions | `README.md`, `docs/ISOLATION.md` |
| III-A | Deployment domain, single-box inference | `scripts/train_recording.py`, `results/recorded/deploy_support/` |
| III-B | Homography-sensitivity anchor | `src/pipeline/multi_camera_trajectory_fusion.py` anchor rule; `src/geometry/fov_overlap.py` is the overlap graph, not the anchor |
| III-C | Frozen cross-view pairs and the full overlap graph | `configs/overlap/`, `scripts/build_overlap.py`, `scripts/freeze_pairs.py` |
| III-D | Bounded residual and the multi-view loss | `scripts/train_recording.py` loss; `configs/paper_defaults.yaml` |
| III-E | Camera support from retained validation pairs | `scripts/train_recording.py` `support_cameras`; `results/recorded/residual_support.csv` column `camera_output` |
| III-F | Residual bound and the common-shift penalties | `configs/paper_defaults.yaml` `residual` |
| III-G | Fixed association backend | `scripts/protocol.py` `paper_merge_args`; `scripts/eval_tracking.py` |
| IV-A | LUMPI M0–M6 and Synthehicle Town01–Town05 | `scripts/protocol.py` `SITES`, `GROUPS` |
| IV-B | Fixed detector, fixed anchors, three seeds | `scripts/detect_and_merge.py`, `scripts/eval_fixed_anchors.py`, `scripts/train_recording.py` |
| IV-C | 60/10/30 split, frozen matches, reference offset | `scripts/make_time_split.py`, `scripts/attach_ground_truth.py`, `scripts/audit_reference.py` |
| IV-D | Localization errors and TrackEval BEV-HOTA | `scripts/eval_localization.py`, `src/evaluation/hota_trackeval.py`, `third_party/TrackEval/` |
| V-A | Localization chain and per-recording residual effects | `results/tables/computed_tables.md` localization and per-recording median sections; `results/recorded/loc_support/` |
| V-B | Fixed-anchor geometric comparison | `results/tables/computed_tables.md` fixed-anchor section; `results/recorded/ipm_baseline.json` |
| V-C | Tolerance-resolved BEV-HOTA | `results/tables/computed_tables.md` BEV-HOTA sections; `results/recorded/identity_hota25.csv`, `identity_hota25_support.csv` |
| V-D | Supervised BEVHeight reference | `docs/BEVHEIGHT.md`, `results/bevheight/` |
| V-E | Edge latency | `docs/ORIN.md`, `results/orin/runtime.md` |
| V-F | Support rule and optional local frame | `results/recorded/deploy_support/`; board procedure below |
| VI–VII | Applicability and conclusion | No separate program. The support rule and separate real/synthetic summaries are the evaluated behavior |

## Main tables and figures

| Manuscript | Contents | Artifact |
| --- | --- | --- |
| Table I | Method-input comparison | No computed table. The implemented inputs are a 2-D box, a metric homography, and unlabeled overlap pairs |
| Table II | Strata and matched-detection counts | `scripts/protocol.py` `GROUPS`; array lengths in `results/recorded/loc_support/` |
| Table III | IPM / HSG / HSG-SRL localization | `scripts/build_tables.py` localization section; `results/tables/computed_tables.md` |
| Table IV | Fixed-anchor localization | `scripts/build_tables.py` fixed-anchor section; `results/recorded/ipm_baseline.json` |
| Table V | Recording-macro BEV-HOTA | `scripts/build_tables.py` BEV-HOTA section; `results/recorded/identity_hota25.csv`, `identity_hota25_support.csv` |
| Table VI | Component latency | `results/orin/runtime.md` |
| Table VII | Support rule and tolerance guide | `results/recorded/residual_support.csv` `camera_output`; `results/recorded/deploy_support/` |
| Fig. 1 | Domain overview | Manuscript drawing in `paper/fab4_assets/overview.png` when that tree is present. Not used by scoring |
| Fig. 2 | Sensitivity field and anchor rule | Manuscript drawing in `paper/fab4_assets/hsg_sensitivity_anchor.png`. The anchor rule is the code in III-B above |
| Fig. 3 | Per-recording residual effects | Numbers in the per-recording median and BEV-HOTA sections of `results/tables/computed_tables.md`. Drawing: `paper/fab4_assets/recording_effects.png` |
| Fig. 4 | Localization error distributions | `results/recorded/loc_support/` and `results/recorded/ipm_baseline.json`. Drawing: `paper/fab4_assets/error_cdf.png` |

## Supplement

| Manuscript | Contents | Artifact |
| --- | --- | --- |
| S1 | Provenance, split, and detector-to-GT matching | `docs/DATASETS.md`, `docs/REPRODUCTION.md` sections 1–3; `src/evaluation/eval_fab2_det_gt_match.py` |
| S2 | Projection, overlap, loss reductions, association | `configs/paper_defaults.yaml`; `scripts/protocol.py`; `scripts/train_recording.py` |
| S3 | Extra localization checks | Offset, anchor, and fixed-anchor sections below |
| S4 | TrackEval replay and runtime accounting | `scripts/replay_trackeval.py`; `docs/ORIN.md` |
| S5 | Optional ChArUco local frame | `src/pipeline/multi_camera_bev_stitch.py`; `tools/print_ChArUco.py`, `tools/diagnose_charuco.py`, `tools/mark_charuco_corners.py` |
| S6 | HOTA breakdown and BEVHeight | `scripts/build_tables.py` breakdown sections; `docs/BEVHEIGHT.md` |
| S7 | Reference and timing interpretation | `results/offset_sensitivity/summary.md`; `docs/ORIN.md` |
| Table S1 | Fixed configuration | `configs/paper_defaults.yaml`; `scripts/protocol.py` |
| Table S2 | Pair counts and support cameras | `results/recorded/pair_counts/`; `results/recorded/residual_support.csv` |
| Tables S3–S5 | Reference-frame translation | `scripts/eval_offset_sensitivity.py`; `results/offset_sensitivity/summary.md` |
| Table S6 | Per-recording median and seed-0 interval | `scripts/build_tables.py`; `results/tables/computed_tables.md` |
| Table S7 | Geometry-only anchor tracking | `scripts/eval_anchor_tracking.py --summarize`; `results/anchor_tracking/summary.md` |
| Table S8 | Board-fit residuals | `scripts/report_board_capture.py`; `results/board_capture/fit_residuals.json` |
| Tables S9–S11 | DetA, AssA, LocA, and per-recording HOTA | `scripts/build_tables.py`; `results/recorded/identity_hota25.csv`, `identity_hota25_support.csv` |
| Table S12 | BEVHeight on M6, M3, and Town05 | `results/bevheight/score_three.md`, `results/bevheight/hota_remerge.md` |
| S3.E | One-seed residual on fixed anchors | `results/fixed_anchor_residual/summary.md`. Stored only; the frozen replay does not regenerate it |
