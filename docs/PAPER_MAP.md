# Paper pointer

This is the only artifact file that uses manuscript section, table, and figure numbers from `paper/paper_draft_v_fab5B.md` and `paper/paper_draft_v_fab5B_supplement.md`. When those numbers change, edit this file. Commands and result headings elsewhere use function names.

`docs/REPRODUCTION.md` is the operational chain. Scoring does not read `paper/`.

## Main text

| Manuscript | What it specifies | Artifact |
| --- | --- | --- |
| III-A | Domain, single-box inference, overlap gate | `scripts/train_recording.py`; `src/geometry/overlap_gate.py` |
| III-B | Homography-sensitivity anchor | Anchor rule in `src/pipeline/multi_camera_trajectory_fusion.py` |
| III-C | Frozen pairs, loss, and bound | `scripts/freeze_pairs.py`; `scripts/train_recording.py`; executed constants in `scripts/protocol.py` |
| III-D | Eligible cameras, overlap union, association | `results/recorded/deploy_support/`; `scripts/protocol.py` `paper_merge_args`; `scripts/eval_tracking.py` |
| IV-A | Thirteen domains and matched-detection counts | `scripts/protocol.py` `SITES`, `GROUPS`; `results/recorded/loc_support/` |
| IV-B | Frozen detections, anchor controls, camera-wise corrections | `scripts/eval_fixed_anchors.py`; `results/controls/` |
| IV-C | Chronological split, fixed matches, no test-set selection | `scripts/make_time_split.py`; `scripts/attach_ground_truth.py` |
| IV-D is not a separate heading | Localization and TrackEval metrics are in IV-C | `scripts/eval_localization.py`; `src/evaluation/hota_trackeval.py`; `third_party/TrackEval/` |
| V-A | Localization and four-threshold HOTA | `results/tables/computed_tables.md` |
| V-B | Anchor components and sensitivity groups | `results/controls/`; per-recording anchor section of `results/tables/computed_tables.md` |
| V-C | Residual, constants, and Ko-PER spatial gate | `results/recorded/identity_hota25_overlap.csv`; `results/recorded/supplement/` |
| V-D | BEVHeight reference | `docs/BEVHEIGHT.md`; `results/bevheight/` |
| V-E | CPU localization-path latency | `docs/ORIN.md`; `results/orin/runtime.md` |

## Main tables and figures

| Manuscript | Contents | Artifact |
| --- | --- | --- |
| Table I | Method-input comparison | No computed table |
| Table II | Domains and 935,566 matched detections | Lengths of `results/recorded/loc_support/*.npz` |
| Table III | Fixed anchors, HSG, and HSG-SRL | `scripts/build_tables.py`; `results/tables/computed_tables.md` |
| Table IV | HSG versus HSG-SRL, localization and HOTA | Same summary, plus `results/recorded/identity_hota25_overlap.csv` |
| Table V | Sensitivity-only, shape-only, and per-domain anchors | `results/controls/`; fixed-anchor section of `results/tables/computed_tables.md` |
| Table VI | Constant offset, MLP, and constraint ranges | `results/controls/constant/`; `results/controls/constraint_split/` |
| Table VII | BEVHeight under two matching rules | `results/bevheight/` |
| Table VIII | Localization path, 1.12–1.20 ms | `results/orin/runtime.md` |
| Fig. 1–5 | Overview, sensitivity, recording effects, Ko-PER gate | Drawings in `paper/fab5B_assets/` when that tree is present. Not used by scoring |

## Supplement

| Manuscript | Contents | Artifact |
| --- | --- | --- |
| S1 | Provenance, splits, and detector-to-GT matching | `docs/DATASETS.md`; `docs/REPRODUCTION.md`; `src/evaluation/eval_fab2_det_gt_match.py` |
| S2 | Projection, pairs, loss, eligibility, overlap gate | `scripts/protocol.py`; unread record in `configs/paper_defaults.yaml`; `scripts/train_recording.py`; `src/geometry/overlap_gate.py` |
| S3 | Anchor, residual, constant, affine, and gating checks | `results/controls/`; `results/tables/computed_tables.md` |
| S4 | TrackEval representation and tracking tables | `scripts/replay_trackeval.py`; `results/recorded/tracking_decomposition_recording.csv` |
| S5 | BEVHeight training and both matching rules | `docs/BEVHEIGHT.md`; `reference/bevheight/` |
| S6 | CPU slices and the separate detector pass | `reference/orin/`; `results/orin/runtime.md` |
| Table S1 | Pair times versus the test interval | Released pair arrays under each site after `scripts/freeze_pairs.py` |
| Table S2 | Assignment and clock-source checks | `scripts/assignment/clock_check.py`; `results/controls/clock_check.md` |
| Table S3 | Per-camera assignment counts | `scripts/assignment/table_s3_closure.py`; `results/controls/table_s3_closure.md` |
| Table S4 | Neighborhood of the HSG constants | `results/controls/hsg_constants_robustness.md` |
| Table S5 | Pair counts and eligible cameras | `results/recorded/pair_counts/`; `results/recorded/residual_support.csv` |
| Table S6 | Shared settings | Executed constants in `scripts/protocol.py`; unread record in `configs/paper_defaults.yaml` |
| Table S7 | Constraint removals | `results/controls/constraint_split/summary.md`; `results/controls/unconstrained/summary.md` |
| Table S8 | Per-recording geometric anchors | Per-recording section of `results/tables/computed_tables.md` |
| Tables S9–S19 | Sensitivity, class, placement, constants, affine, coverage, gating, symmetric loss | `results/controls/` and `results/recorded/supplement/` |
| Tables S20–S23 | Tracking anchors, DetA, AssA, LocA, per-recording HOTA | `results/anchor_tracking/summary.md`; `results/recorded/tracking_decomposition_recording.csv` |
| Tables S24–S25 | BEVHeight subsets | `results/bevheight/score_three.md` |
| Table S26 | CPU slices of one localization call | `results/orin/runtime.md` |

The optional ChArUco board chain in `src/pipeline/multi_camera_bev_stitch.py`, `tools/`, and `results/board_capture/` is not part of this manuscript.
