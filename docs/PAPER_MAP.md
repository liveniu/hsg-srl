# Paper pointer

This is the only artifact file that uses manuscript section, table, and figure numbers. When those numbers change, edit this file. Commands, comments, and result headings elsewhere use the function names in the right-hand column.

`docs/REPRODUCTION.md` is the operational chain. It does not repeat these numbers.

The manuscript tree `paper/` is not part of the git release. Scoring does not read it.

## Main text

| Manuscript | What it specifies | Artifact |
| --- | --- | --- |
| I | Deployment setting and the two contributions | `README.md`, `docs/ISOLATION.md` |
| III-A | One domain, single-box inference, overlap-gated residual | `scripts/train_recording.py`; `scripts/db_access.py` `positions` |
| III-B | Homography-sensitivity anchor | `src/pipeline/multi_camera_trajectory_fusion.py` anchor rule; `src/geometry/overlap_gate.py` `hsg_terms` |
| III-C | Frozen cross-view pairs and the overlap graph | `configs/overlap/`; `scripts/freeze_pairs.py` reads that frozen graph |
| III-D | Bounded residual and the multi-view loss | `scripts/train_recording.py`; `configs/paper_defaults.yaml` |
| III-E | Eligible cameras and the overlap union | `positions(..., spatial=True)`; polygon tolerance `-1e-6` m. A camera is eligible when the recording has at least 32 training pairs and the camera appears in a retained validation pair |
| III-F | Fixed association backend | `scripts/protocol.py` `paper_merge_args`; `scripts/eval_tracking.py` |
| IV-A | LUMPI, Ko-PER, Synthehicle | `scripts/protocol.py` `SITES`, `GROUPS` |
| IV-B | Fixed detector, anchors, three seeds, spatial-gate and MVC controls | `scripts/detect_and_merge.py`, `scripts/detect_koper.py`, `scripts/eval_fixed_anchors.py`, `scripts/train_recording.py --symmetric` |
| IV-C | 60/10/30 split, frozen matches, reference offset | `scripts/make_time_split.py`, `scripts/attach_ground_truth.py`, `scripts/audit_reference.py` |
| IV-D | Localization errors and TrackEval BEV-HOTA | `scripts/eval_localization.py`, `src/evaluation/hota_trackeval.py` |
| V-A | Geometric anchor comparison | `results/recorded/ipm_baseline.json`; `scripts/build_tables.py` |
| V-B | Overlap-gated residual and the Ko-PER spatial split | `results/recorded/loc_support/`; `scripts/eval_localization.py` |
| V-C | Overlap-gated BEV-HOTA | `results/recorded/identity_hota25_overlap.csv` |
| V-D | Supervised BEVHeight reference | `docs/BEVHEIGHT.md`, `results/bevheight/` |
| V-E | Edge latency on M6, Town05, and Ko-PER | `docs/ORIN.md`, `results/orin/runtime.md`, `reference/orin/` |
| VI–VII | Applicability, optional local frame, and conclusion | The optional board procedure is Supplementary S7. It is not a scored localization or tracking table |

## Main tables and figures

| Manuscript | Contents | Artifact |
| --- | --- | --- |
| Table II | Strata and matched-detection counts | `scripts/protocol.py` `GROUPS`; array lengths in `results/recorded/loc_support/` |
| Table III | Anchor rules, including sensitivity-only and shape-only | `scripts/eval_fixed_anchors.py`; `results/recorded/ipm_baseline.json` |
| Table IV | Overlap-gated HSG-SRL localization and HOTA | `scripts/build_tables.py` |
| Table V | Four-gate BEV-HOTA | `results/recorded/identity_hota25.csv`, `identity_hota25_overlap.csv` |
| Table VI | Edge latency | `results/orin/runtime.md` |
| Fig. 1–4 | Overview, anchor, per-recording effects, Ko-PER overlap | Manuscript drawings. Scoring does not read them |

## Supplement

| Manuscript | Contents | Artifact |
| --- | --- | --- |
| S1 | Provenance, Ko-PER footprint center, detector admission | `docs/DATASETS.md`; `src/importers/import_koper.py`; `scripts/detect_koper.py` |
| S2 | Projection, frozen edges, loss, association, deployment pseudocode | `configs/paper_defaults.yaml`; `src/geometry/overlap_gate.py` |
| S3 | Eligible cameras, reference translation, per-recording localization | `results/recorded/pair_counts/`; `results/recorded/residual_support.csv`; `scripts/eval_offset_sensitivity.py`; `scripts/eval_final_reference_sensitivity.py`; `scripts/build_tables.py` |
| S4 | Bottom-center, single-cue and interior anchors, spatial-gate ablation, MVC control | `scripts/eval_fixed_anchors.py`; `scripts/eval_tracking.py --no-spatial-gate`; `scripts/train_recording.py --symmetric` |
| S5 | TrackEval replay and runtime accounting | `scripts/replay_trackeval.py`; `docs/ORIN.md` |
| S4 Tables S9–S11 | Spatial gate, regions, and the symmetric MVC control | `scripts/build_supplement_tables.py`; `results/tables/supplement_controls.md` |
| S6 Table S16 | BEVHeight matched subset; M6 eligible cameras are C05, C06, and C07 | `results/bevheight/score_three.md`; `results/bevheight/*_bevheight_subset.npz` |
| S6 Table S17 | Unique-containment localization on that supervised subset | `results/bevheight/unique_containment.json`; `results/tables/supplement_controls.md` |
| S6 | HOTA breakdown and BEVHeight tracking | `scripts/build_tables.py`; `results/bevheight/hota_remerge.md` |
| S7 | Optional ChArUco local-frame procedure; field calibration accuracy is outside the paper evaluation | `src/pipeline/multi_camera_bev_stitch.py`; `tools/` |
| Table S1 | Fixed configuration | `configs/paper_defaults.yaml` |
| Table S2 | Pair counts and eligible cameras | `results/recorded/pair_counts/`; `results/recorded/residual_support.csv` |
| Table S3 | Reference sensitivity of geometric controls and the final overlap-gated LUMPI localizer | `results/offset_sensitivity/offset_sensitivity.json` for geometric controls and translations; `final_overlap_lumpi.json` and `scripts/eval_final_reference_sensitivity.py` for final pooled localization |
| Tables S4–S5 | Reference-treatment strata and per-recording geometric comparisons | `results/offset_sensitivity/offset_sensitivity.json`; the Synthehicle row in S4 shares identical values across all four treatments |
| Table S8 | Per-recording geometric anchors, including bottom-center matched counts, median, RMSE, and p90 | `results/recorded/ipm_baseline.json`; `scripts/build_tables.py`; per-recording geometric-anchor table in `results/tables/computed_tables.md` |
| Table S15 | Per-recording HOTA with shared M0/M1 fallback rows | `results/recorded/identity_hota25.csv`, `identity_hota25_overlap.csv`; the full seed rows remain in `results/tables/computed_tables.md` |

The supplement combines identical deployment rows in Table S9 and identical HSG / HSG-SRL seed outputs for M0/M1 in Table S15. Table S10 keeps all regions and their counts while showing HSG and the residual on eligible cameras without spatial gating: final HSG-SRL equals that residual inside the overlap and HSG outside. The full source tables remain in `results/tables/supplement_controls.md` and `computed_tables.md`. `results/board_capture/` is the optional S7 procedure's engineering record. The paper does not score it.

`results/recorded/identity_hota25_support.csv` holds the "Without spatial gating" HOTA rows of Tables S9–S10 for LUMPI and Synthehicle; `results/recorded/hota_sources/ko_per__full__s*.json` holds them for Ko-PER. These rows still match the saved bins. They are not the published overlap-gated method.
