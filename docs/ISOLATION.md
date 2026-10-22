# Package boundary

This directory is the HSG-SRL release. Training goes through `scripts/train_recording.py`, which passes the constants in `configs/paper_defaults.yaml`.

`src/pipeline/geometry_guided_residual_mlp.py train` cannot replace `scripts/train_recording.py`. The published trainer fits the frozen pairs from `configs/paper_defaults.yaml` and does not call that command line. The module command is a database trainer. Its numeric defaults match the published loss, and it still requires `--split-manifest`. Checkpoint filenames use the tag `F0` for this residual.

Same-camera tracklet stitching is implemented and left off. There is no camera-translation stage.

The geometry databases for the published tables are not in this git repository. The manuscript cites https://doi.org/10.5281/zenodo.23008421. This release adds Ko-PER and applies the residual only inside the frozen overlap union.

`reference/orin/` is the board timing bundle. Its timer imports this package's `src/`. Do not replace `src/pipeline/multi_camera_trajectory_fusion.py` with a copy from an earlier timing tree: those files share a name and are not the same program.
