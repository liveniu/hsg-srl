# Recorded results

These files rebuild the published tables without the videos. The geometry databases are a separate Zenodo deposit: https://doi.org/10.5281/zenodo.23008421.

Checkpoint and support-file names use the tag `F0`. That tag is the HSG-SRL residual, not a second method.

| Path | Use |
| --- | --- |
| `loc/` | Frozen HSG errors for the published matched cohort. Each file contains only `geo`. A recomputation checks against this array. Residual errors are in `loc_support/`. |
| `loc_support/` | Per-detection HSG and residual errors, with track clusters for the bootstrap. `scripts/build_tables.py` reads these. |
| `identity_hota25.csv` | Geometry-only HSG tracking scores checked by `scripts/replay_trackeval.py` |
| `identity_hota25_support.csv` | Published HSG-SRL tracking rows under retained validation-pair support |
| `bins/` | Saved 10 Hz tracking inputs for those rows |
| `checkpoints/` | Residual weights. Filenames contain `_F0_`. |
| `residual_support.csv`, `deploy_support/` | Published retained validation-pair residual support |
| `pair_counts/` | Training and validation pair totals |
| `ipm_baseline.json` | Fixed-anchor localization |
| `../orin/runtime.md` | Board timings |
| `../bevheight/` | Supervised reference scores |

`scripts/build_tables.py` reads `loc_support/`, `identity_hota25.csv`, and `identity_hota25_support.csv`.
