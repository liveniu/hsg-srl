# HSG-SRL

Geometry-Guided Metric Localization With Self-Supervised Residual Learning for Roadside Multi-Camera Vehicle Tracking.

This directory is a standalone package. Licensed LUMPI and Synthehicle videos are not included. The geometry databases are a separate Zenodo deposit, not part of this git repository.

## Contents

| Path | Role |
| --- | --- |
| `src/` | Import, HSG projection, residual network, fusion, and scoring |
| `scripts/` | The reproduction chain |
| `docs/REPRODUCTION.md` | Commands and which module each stage calls |
| `docs/PAPER_MAP.md` | Manuscript section and table numbers; the only file that uses them |
| `configs/` | Recordings, overlap, BoT-SORT, and paper constants |
| `third_party/TrackEval/` | HOTA implementation from TrackEval commit 12c8791; see `NOTICE.txt` |
| `results/recorded/` | Arrays and scores for localization, fixed anchors, and BEV-HOTA |
| `reference/orin/` | Jetson timing for M6 and Town05 |
| `reference/bevheight/` | Supervised reference on M6, M3, and Town05 |

## Install

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-minimal.txt
python scripts/run_smoke_test.py
```

Install `requirements.txt` before detection.

`python scripts/run_smoke_test.py` checks imports, the HSG projection path, residual model construction, and the packaged scoring utilities without running the full experiment.

## Published tables

YOLO26l detections are not identical across machines. The released geometry databases freeze the detections, tracks, identities, splits, and ground-truth attachment used for the published tables:

https://doi.org/10.5281/zenodo.23008421

To reproduce those tables, extract each recording to `data/work/sites/<recording>/` and continue from pair mining:

```text
python scripts/run_paper.py --site lumpi_M6 --from-stage overlap
```

Do not use the default `--from-stage import` on that directory. It detects again and replaces `geometry.db`. See `docs/REPRODUCTION.md`.

```text
python scripts/replay_trackeval.py
python scripts/build_tables.py
```

These two commands check the recorded tracking scores and write the localization chain, the fixed-anchor comparison, recording-macro BEV-HOTA, and the per-recording bootstrap and HOTA breakdowns. They do not retrain.

`python scripts/eval_anchor_tracking.py --summarize` writes the geometry-only anchor-tracking summary from stored JSON. `python scripts/report_board_capture.py` prints the board-fit residuals. `docs/REPRODUCTION.md` names the commands for the reference-frame rescore and the supervised BEVHeight comparison.

`docs/REPRODUCTION.md` gives the stage-by-stage script chain, including GT attachment, reference-offset auditing, pair mining, training, scoring, and table generation.

## Run from downloaded video

See `docs/DATASETS.md`.

```text
python scripts/run_paper.py --site lumpi_M6 --model data/models/yolo26l.pt
```

This imports the media, runs YOLO26l, and builds a new geometry database. Use it for a new installation. It does not reproduce the published tables, because the detections will differ from the released databases.

Train with `scripts/train_recording.py`. It uses the constants in `configs/paper_defaults.yaml`. `src/pipeline/geometry_guided_residual_mlp.py train` cannot replace it. That command mines pairs from the geometry database and still requires `--split-manifest`. See `docs/REPRODUCTION.md`.
