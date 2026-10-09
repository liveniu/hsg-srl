# Runtime

Milliseconds. Detector, HSG, and MLP are per camera frame. Fusion is per sync batch. Stream fps uses the sum of the three per-frame p50 times. Those stages were timed separately, so this is not one measured end-to-end loop. Fusion p95 is a batch tail and is not folded into fps.

## Device and environment

Read on the board, 2026-09-24.

| item | value |
| --- | --- |
| board | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| GPU | Orin, compute capability 8.7 |
| power | `MAXN_SUPER` (nvpmodel id 2) |
| memory | 7.4 GiB RAM, no swap |
| L4T | R39.2.1, GCID 46758480, aarch64, userspace date 2026-08-07 |
| TensorRT | 10.16.2.10+cuda13.2 (`libnvinfer10`); package `nvidia-tensorrt` 7.2.1-b49 |
| engine | `data/models/yolo26l.engine`, 52 MB, built on this board 2026-09-24 15:58 |
| Python | 3.12, `/home/jetson/Desktop/hsg_srl/.venv` |
| PyTorch | 2.13.0+cu132. This build does not support CC 8.7 |

HSG and the residual MLP in the table were timed on CPU. The detector is the TensorRT engine, not this PyTorch. The memory line is the board capacity, not the footprint of the timed process.

What this run did fix:

| item | value |
| --- | --- |
| detector | YOLO26l, TensorRT FP16 engine built on the board, workspace 512 MiB |
| input | 384×640, one clip at a time, BoT-SORT, `gmc_method: none` |
| detector call | `track`, `persist=True`; confidence and IoU were not overridden |
| HSG | box center \(x\), \(s_0=60\) m, shape \(\eta=0.75\), aspect ratio in \([0.5, 3]\), CPU |
| residual | HSG-SRL seed 0, \(7\to64\to64\to2\), per-axis bound 1 m, CPU |
| fusion | full-graph remerge, dry run, score 0.70, max distance 2.5 m, CPU |
| streams | M6: 3 cameras, 3 edges, residual timed on C06 and C07. Town05: 6 cameras, 15 edges, residual timed on all six |

The M6 camera pair above is the set timed on the board. Published localization and tracking use `results/recorded/residual_support.csv`. For M6 seed 0 that file also applies the residual on C05. The MLP column is the per-frame cost on cameras where the residual ran, so it is not an average that includes skipped cameras.

Detector frames are sequential single streams. The fps below is one camera, not the three or six streams at once.

| site | detector p50/p95 | HSG p50/p95 | MLP p50/p95 | frame sum p50 | stream fps | fusion p50/p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lumpi_M6 | 30.41/37.55 | 3.21/4.49 | 0.60/0.64 | 34.22 | 29.2 | 0.40/37.16 |
| synth_Town05_O_day | 26.66/31.93 | 1.26/2.37 | 0.59/0.64 | 28.51 | 35.1 | 0.16/22.08 |
