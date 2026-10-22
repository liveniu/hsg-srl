# Jetson Orin timing

`reference/orin/` times the overlap-gated localizer on M6, Town05, and Ko-PER. Build the TensorRT engine on the board. Copy each released `geometry.db` to `reference/orin/data/<site>/geometry.db` before a new run. Those databases are not in git. The support file, overlap polygons, calibration, and seed-0 checkpoint in that data directory are.

```text
bash reference/orin/setup_orin.sh
bash reference/orin/build_engine.sh
bash reference/orin/run_orin.sh
```

`run_orin.sh` puts this package's `src/` on `PYTHONPATH`. The timed fusion call is the association backend in that tree. The localization path is vectorized HSG, the overlap test, and the seed-0 residual, with the paper's signed-distance membership rule.

The reported timings are `results/orin/runtime.md`. They are the edge-device component latencies from the overlap-gated measurement, including Ko-PER. They do not enter the localization or tracking summaries. A new end-to-end detector measurement needs the board, the engine built on that board, and the videos, which are not in the geometry-database archive. Ko-PER has no clip in this package, so its detector cell stays empty unless clips are supplied.

`HSG_LIMIT_FRAMES=30 bash reference/orin/run_orin.sh` is a smoke test and skips fusion.
