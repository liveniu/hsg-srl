# Orin timing bundle

Component timing for the overlap-gated localizer. Copy this directory to the Orin.

```bash
sudo bash setup_orin.sh
bash build_engine.sh
bash run_orin.sh
```

`setup_orin.sh` creates `./.venv` with `--system-site-packages` and checks that NumPy, OpenCV, PyTorch, Ultralytics, and TensorRT import. It does not save the board record, because it may still change the machine. Set the published power mode only with:

```bash
sudo HSG_ORIN_SET_POWER=1 bash setup_orin.sh
```

When the script is root and a desktop is active, it runs `systemctl isolate multi-user.target` for this boot. Restore it with `sudo systemctl isolate graphical.target`.

`build_engine.sh` writes `data/models/yolo26l.engine` on the board from `data/models/yolo26l.pt`, 384×640, FP16, workspace 512 MiB. Do not copy an engine from Windows. The weight file is not in this directory; copy it to `data/models/yolo26l.pt` or set `HSG_YOLO_PT`.

`run_orin.sh` writes `outputs/board_check.txt` at the start of the timing run. That file is the board model, memory, L4T, TensorRT, power mode, desktop state, YOLO engine, and the Python that runs the timer. It then times three separate passes:

- detector, if the engine and `data/clips/{site}/*.mp4` are present
- one localization frame: vectorized HSG, overlap test, seed-0 residual
- fusion, per synchronized batch

Copy each released `geometry.db` to `data/<site>/geometry.db` before a board run. Those databases are not in git. M6 and Town05 clips can be copied from the earlier Orin bundle. Ko-PER has no clip here, so its detector cell stays empty. A missing engine skips only the detector. `HSG_LIMIT_FRAMES=30 bash run_orin.sh` skips fusion.

`run_orin.sh` imports the package `src/` next to this bundle. Do not replace that pipeline with another file of the same name from an older timing tree.

The path percentile is measured. HSG, overlap, and MLP percentiles are slices and are not added together. The table does not convert those numbers into stream fps.
