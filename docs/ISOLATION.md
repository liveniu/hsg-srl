# Package boundary

This directory is the HSG-SRL release. Training goes through `scripts/train_recording.py`, which passes the constants in `configs/paper_defaults.yaml`.

`src/pipeline/geometry_guided_residual_mlp.py train` cannot replace `scripts/train_recording.py`. The published trainer fits the frozen pairs from `configs/paper_defaults.yaml` and does not call that command line. The module command is a database trainer. Its numeric defaults match the published loss, and it still requires `--split-manifest`. Checkpoint filenames use the tag `F0` for this residual.

Same-camera tracklet stitching is implemented and left off. There is no camera-translation stage.

The geometry databases for the published tables are https://doi.org/10.5281/zenodo.23008421. They are not in this git repository.
