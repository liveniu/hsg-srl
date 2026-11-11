# Paper pointer

This is the only artifact file that uses manuscript section, table, and figure numbers. When those numbers change, edit this file. Commands, comments, and result headings elsewhere use the function names in the right-hand column.

`docs/REPRODUCTION.md` is the operational chain. It does not repeat these numbers.

The manuscript is not part of this git release. Scoring does not read it. Published scores use the stored footprint centers in the geometry databases. `reference_verdict.txt` is an audit record and is not added to ground truth.

## Main text

| Manuscript | What it specifies | Artifact |
| --- | --- | --- |
| I | Deployment setting and the two contributions | `README.md`, `docs/ISOLATION.md` |
| III-A | One domain, single-box inference, overlap-gated residual | `scripts/train_recording.py`; `scripts/db_access.py` `positions` |
| III-B | Homography-sensitivity anchor | `src/pipeline/multi_camera_trajectory_fusion.py` anchor rule; `src/geometry/overlap_gate.py` `hsg_terms` |
| III-C | Frozen cross-view pairs, bounded residual, and the multi-view loss | `configs/overlap/`; `scripts/freeze_pairs.py`; `scripts/train_recording.py`; `configs/paper_defaults.yaml` |
| III-D | Eligible cameras, overlap union, and the fixed association backend | `positions(..., spatial=True)`; polygon tolerance `-1e-6` m; `scripts/protocol.py` `paper_merge_args`; `scripts/eval_tracking.py` |
| IV-A | LUMPI, Ko-PER, Synthehicle | `scripts/protocol.py` `SITES`, `GROUPS` |
| IV-B | Fixed detector, anchors, domain-selected beta, constant and affine controls, constraint ablations, three seeds | `scripts/detect_and_merge.py`, `scripts/detect_koper.py`, `scripts/eval_fixed_anchors.py`, `results/controls/` |
| IV-C | 60/10/30 split, frozen matches, stored footprint centers | `scripts/make_time_split.py`, `scripts/attach_ground_truth.py`, `scripts/db_access.py` `load_bias` |
| V-A | Localization and BEV-HOTA | `results/tables/computed_tables.md`; `scripts/build_tables.py` |
| V-B | Geometric-anchor contribution and sensitivity tertiles | `results/recorded/ipm_baseline.json`; `results/controls/conditional_error.md` |
| V-C | Constant, MLP, and constraint variants; spatial support | `results/controls/RESULTS.md`; `results/controls/constraint_split/summary.md`; `scripts/build_supplement_tables.py` |
| V-D | Supervised BEVHeight reference | `docs/BEVHEIGHT.md`, `results/bevheight/` |
| V-E | Edge latency on M6, Town05, and Ko-PER | `docs/ORIN.md`, `results/orin/runtime.md`, `reference/orin/` |
| VI–VII | Applicability and conclusion | The optional board procedure is not a scored localization or tracking table |

## Main tables and figures

| Manuscript | Contents | Artifact |
| --- | --- | --- |
| Table II | Strata and matched-detection counts | `scripts/protocol.py` `GROUPS`; array lengths in `results/recorded/loc_support/` |
| Table III | Fixed anchors, HSG, and HSG-SRL | `scripts/eval_fixed_anchors.py`; `results/recorded/ipm_baseline.json`; `scripts/build_tables.py` |
| Table IV | Group localization and four-gate BEV-HOTA | `results/recorded/loc_support/`; `results/recorded/identity_hota25.csv`; `identity_hota25_overlap.csv` |
| Table V | Single-cue anchors and per-domain fixed anchors | `results/recorded/ipm_baseline.json`; `results/controls/RESULTS.md` |
| Table VI | Constant offset, full MLP, bound-only and penalties-only | `results/controls/RESULTS.md`; `results/controls/constraint_split/summary.md` |
| Table VII | BEVHeight greedy and unique containment, plus mixed tracking | `results/bevheight/score_three.md`; `results/bevheight/unique_containment.json`; `results/bevheight/m3_untranslated.json` |
| Table VIII | Localization-path latency | `results/orin/runtime.md` |
| Fig. 1–5 | Overview, workflow, sensitivity, recording effects, Ko-PER overlap | Manuscript drawings. Scoring does not read them |

## Supplement

| Manuscript | Contents | Artifact |
| --- | --- | --- |
| S1 | Provenance, split timing, assignment statistics, Ko-PER footprint center | `docs/DATASETS.md`; `docs/REPRODUCTION.md`; `results/controls/time_isolation.json`; `results/controls/match_probe_stage_a.md`; `results/controls/match_summary.md`; `src/importers/import_koper.py` |
| S2 | Projection, pairs, loss, association, s0/eta neighborhood, constraint removal | `configs/paper_defaults.yaml`; `results/controls/hsg_constants_robustness.md`; `results/controls/constraint_split/summary.md`; `results/controls/unconstrained/summary.md` |
| S3 | Per-recording anchors, sensitivity and class bins, shared cameras, domain anchors, spatial gate, MVC | `results/tables/computed_tables.md`; `results/tables/supplement_controls.md`; `results/controls/` |
| S4 | Tracking decomposition and per-recording HOTA | `scripts/build_tables.py`; `results/anchor_tracking/summary.md` |
| S5 | BEVHeight greedy and unique containment | `results/bevheight/score_three.md`; `results/bevheight/unique_containment.json` |
| S6 | Orin component slices | `results/orin/runtime.md`; `docs/ORIN.md` |
| Table S1 | Pair times versus the test-interval start | `results/controls/time_isolation.json` |
| Table S2 | Midpoint-coordinate reassignment | `results/controls/match_probe_stage_a.md` |
| Table S3 | Per-camera assignment counts | `results/controls/match_summary.md` |
| Table S4 | Neighboring s0 and eta | `results/controls/hsg_constants_robustness.md` |
| Table S5 | Pair counts and eligible cameras | `results/recorded/pair_counts/`; `results/recorded/residual_support.csv` |
| Table S6 | Fixed configuration | `configs/paper_defaults.yaml` |
| Table S7 | Bound-only, penalties-only, and unconstrained residuals | `results/controls/constraint_split/summary.md`; `results/controls/unconstrained/summary.md` |
| Table S8 | Per-recording geometric anchors | `results/recorded/ipm_baseline.json`; per-recording table in `results/tables/computed_tables.md` |
| Tables S9–S11 | Sensitivity tertiles, detector class, shared cameras C08–C10 | `results/controls/conditional_error.md`; `results/controls/class_breakdown.md` |
| Table S12 | Per-domain fixed anchor | `results/controls/beta_selection.md`; `results/controls/RESULTS.md` |
| Tables S13–S14 | Group localization and per-recording bootstrap | `results/tables/computed_tables.md` |
| Table S15 | Per-camera constant and affine corrections | `results/controls/RESULTS.md`; `results/controls/affine/` |
| Table S16 | Sensitivity coverage of the frozen pairs | `results/controls/pair_alpha_coverage/summary.md` |
| Tables S17–S19 | Spatial gate, regions, and the symmetric MVC control | `scripts/build_supplement_tables.py`; `results/tables/supplement_controls.md` |
| Table S20 | Tracking of geometric anchors and the constant correction | `results/anchor_tracking/summary.md`; `results/controls/table_v.md`; `results/controls/RESULTS.md` |
| Tables S21–S23 | DetA, AssA, LocA, and per-recording HOTA | `results/tables/computed_tables.md` |
| Tables S24–S25 | Greedy BEVHeight and unique containment | `results/bevheight/score_three.md`; `results/bevheight/unique_containment.json`; `results/tables/supplement_controls.md` |
| Table S26 | Orin slices | `results/orin/runtime.md` |

`results/recorded/identity_hota25_support.csv` holds the "Without spatial gating" HOTA rows for LUMPI and Synthehicle; `results/recorded/hota_sources/ko_per__full__s*.json` holds them for Ko-PER. These rows still match the saved bins. They are not the published overlap-gated method.

`results/controls/README.md` states which control tables are recorded summaries and which commands rebuild the primary scores. The optional translation audit in `scripts/eval_offset_sensitivity.py` is not a manuscript table.
