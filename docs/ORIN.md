# Jetson Orin timing

`reference/orin/` times M6 and Town05. Build the TensorRT engine on the board. Use the released `geometry.db` at `data/work/sites/<site>/geometry.db` and the checkpoints in `results/recorded/checkpoints/`.

```text
bash reference/orin/setup_orin.sh
bash reference/orin/build_engine.sh
bash reference/orin/run_orin.sh
```

HSG and the residual run on the CPU. The reported timings are `results/orin/runtime.md`, and they are the numbers in Table VI. They do not enter the localization or tracking tables.

The timing bundle is the recorded M6 and Town05 measurement in `results/orin/runtime.md`. Tables III--V use the retained validation-pair support in `results/recorded/residual_support.csv`. A new timing run reads `results/recorded/deploy_support/` and does not change those tables. A new end-to-end timing measurement needs the board and the videos, which are not in the geometry-database archive.
