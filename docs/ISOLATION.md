# Package boundary

This directory is the HSG-SRL release. Training goes through `scripts/train_recording.py`, which passes the constants in `configs/paper_defaults.yaml`.

`src/pipeline/geometry_guided_residual_mlp.py` still accepts older command-line defaults from an earlier residual setup. Do not train by calling that file directly.

Same-camera tracklet stitching is implemented and left off. There is no camera-translation stage.

The geometry databases for the published tables are https://doi.org/10.5281/zenodo.23008421. They are not in this git repository.
