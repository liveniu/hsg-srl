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
| III-E | Camera support and the overlap union | `positions(..., spatial=True)`; polygon tolerance `-1e-6` m |
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
| V-F | Drop-in coordinate stage and optional local frame | `results/recorded/deploy_support/`; board procedure below |
| VI–VII | Applicability and conclusion | No separate program |

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
| S3 | Support counts, reference translation, per-recording localization | `results/recorded/pair_counts/`; `scripts/eval_offset_sensitivity.py`; `scripts/build_tables.py` |
| S4 | Single-cue anchors, spatial-gate ablation, MVC control | `scripts/eval_fixed_anchors.py`; `scripts/eval_tracking.py --no-spatial-gate`; `scripts/train_recording.py --symmetric` |
| S5 | TrackEval replay and runtime accounting | `scripts/replay_trackeval.py`; `docs/ORIN.md` |
| S4 Tables S9–S11 | Spatial gate, regions, and the symmetric MVC control | `scripts/build_supplement_tables.py`; `results/tables/supplement_controls.md` |
| S6 Table S16 | BEVHeight matched subset with C05 on M6 | `results/bevheight/score_three.md`; `results/bevheight/*_bevheight_subset.npz` |
| S6 | HOTA breakdown and BEVHeight tracking | `scripts/build_tables.py`; `results/bevheight/hota_remerge.md` |
| S7 | Optional ChArUco local frame | `src/pipeline/multi_camera_bev_stitch.py`; `tools/` |
| Table S1 | Fixed configuration | `configs/paper_defaults.yaml` |
| Table S2 | Pair counts and support cameras | `results/recorded/pair_counts/`; `results/recorded/residual_support.csv` |

`results/recorded/identity_hota25_support.csv` is the earlier camera-support ablation. It still matches the saved bins. It is not the published overlap-gated method.
