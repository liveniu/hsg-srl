# Orin-plus component latency

Detector, localization, and fusion are separate passes. HSG uses the published projection, evaluated once per camera frame for all boxes together. The resulting milliseconds are for this implementation; they are not a rerun of the earlier per-box timing table. Repeating the run with the same board configuration produces a new measurement. It is not required to match the archived percentiles.

| stage | unit | what is timed |
| --- | --- | --- |
| detector | camera frame | one YOLO26l TensorRT FP16 `track` call, 384×640, BoT-SORT `gmc_method: none`, one clip at a time |
| localization path | camera frame | vectorized HSG for every box in the frame, then the static overlap test, then the seed-0 residual on cameras in S |
| HSG, overlap, MLP | slices of that same frame | reported separately; their percentiles are not the path |
| fusion | sync batch | dry-run remerge, score 0.70, max distance 2.5 m |

Sites are `lumpi_M6`, `synth_Town05_O_day`, and `ko_per`. M6 and Town05 are the original operating points. Ko-PER is included because the overlap gate changes a large share of its boxes. S is the retained validation-pair set in `data/*/support.json`, so M6 includes C05.

Camera edges and overlap polygons are frozen in `data/*/support.json` and `data/*/polygons.json`. The packer and runtime reuse them without an intersection-area filter for any domain.

Build the engine on the board. An engine copied from Windows will not load. Clips go in `data/clips/{site}/*.mp4`, or set `HSG_ORIN_CLIPS`. Ko-PER has no clip in this package; its detector cell stays empty.

`setup_orin.sh` may change the power mode and stop the desktop before the run. It does not write the board record. `run_orin.sh` writes `outputs/board_check.txt` immediately before timing, after those changes, including model, memory, L4T, TensorRT, `nvpmodel -q`, desktop state, engine, and the interpreter used for the timing.

`HSG_LIMIT_FRAMES=30 bash run_orin.sh` is a smoke test and skips fusion.
