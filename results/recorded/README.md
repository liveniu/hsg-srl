# Recorded results

These files rebuild the published tables without the videos. The geometry databases are a separate Zenodo deposit: https://doi.org/10.5281/zenodo.23008421.

Checkpoint and support-file names use the tag `F0`. That tag is the HSG-SRL residual, not a second method.

| Path | Use |
| --- | --- |
| `loc/` | Frozen HSG errors for the published matched cohort. Each file contains only `geo`. A recomputation checks against this array. Residual errors are in `loc_support/`. |
| `loc_support/` | Per-detection HSG and overlap-gated residual errors, with track clusters for the bootstrap. `scripts/build_tables.py` reads these. |
| `identity_hota25.csv` | Geometry-only HSG tracking scores. Rows with a bin hash are checked by `scripts/replay_trackeval.py`. The `25` is the 2.5 m similarity scale, not a second protocol. |
| `identity_hota25_overlap.csv` | Published HSG-SRL rows. Each row's `input_sha256` is a 10 Hz bin in `bins/*_overlap_s*.npz`. |
| `supplement/` | Recomputed S9–S11 arrays and the S16 subset summary. `scripts/build_supplement_tables.py` writes them. |
| `tracking_decomposition_recording.csv` | HOTA decomposition and the in-repository bin or score file for each replay. `scripts/build_tracking_sources.py` writes it. |
| `identity_hota25_support.csv` | Camera-support ablation, without the overlap mask. These rows still match `bins/`. They are not the published method. |
| `hota_sources/` | Vendored overlap-gated and Ko-PER geometry score files used by `scripts/assemble_hota_tables.py`. |
| `bins/` | Saved 10 Hz tracking inputs for those rows |
| `checkpoints/` | Residual weights. Filenames contain `_F0_`. |
| `residual_support.csv`, `deploy_support/` | Published retained validation-pair residual support. `in_S` and `camera_output` are the camera rule. `deploy` is the same on every camera of a recording: `model` means that recording has a trained residual, including cameras whose output stays HSG. `D_before`, `D_after`, `D_G`, `D_r`, and `D_star` are stored diagnostics and do not select the camera. `uses_test_gt` is false. |
| `pair_counts/` | Training and validation pair totals |
| `ipm_baseline.json` | Fixed-anchor localization |
| `../orin/runtime.md` | Board timings |
| `../bevheight/` | Supervised reference scores |

`scripts/build_tables.py` reads `loc_support/`, `identity_hota25.csv`, and `identity_hota25_overlap.csv`.
