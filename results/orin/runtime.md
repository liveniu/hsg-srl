# Component latency

Milliseconds. The localization path is one measured camera frame: vectorized HSG for every box, then the overlap test, then the seed-0 residual. HSG, overlap, and MLP columns are slices of that same call. Their percentiles are not added to produce the path. Detector and fusion are separate passes. Fusion is one synchronized batch. No stream fps is computed from these percentiles.

Detector, when present, is YOLO26l TensorRT FP16, 384×640, BoT-SORT with gmc_method none, one clip at a time. HSG and the residual run on CPU. Fusion is a dry-run remerge, score 0.70, max distance 2.5 m. Support S is the retained validation-pair set. This file records that measurement and the board configuration. A later run with the same configuration is a new measurement and is not required to match these percentiles. This file does not claim power, memory, or station throughput. The detector column in the table is empty because the package does not contain the clips. The recorded detector percentiles and the sample counts are in the section under the table.

## Board check

```text
=== board ===
NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
=== jetpack / tensorrt packages ===
# R39 (release), REVISION: 2.1, GCID: 46758480, BOARD: generic, EABI: aarch64, DATE: Fri Aug  7 05:54:22 AM UTC 2026
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
ii  libnvinfer-bin                                10.16.2.10-1+cuda13.2                            arm64        TensorRT binaries
ii  libnvinfer-dev                                10.16.2.10-1+cuda13.2                            arm64        TensorRT development libraries
ii  libnvinfer-dispatch-dev                       10.16.2.10-1+cuda13.2                            arm64        TensorRT development dispatch runtime libraries
ii  libnvinfer-dispatch10                         10.16.2.10-1+cuda13.2                            arm64        TensorRT dispatch runtime library
ii  libnvinfer-headers-dev                        10.16.2.10-1+cuda13.2                            arm64        TensorRT development headers
ii  libnvinfer-headers-plugin-dev                 10.16.2.10-1+cuda13.2                            arm64        TensorRT plugin headers
ii  libnvinfer-headers-python-plugin-dev          10.16.2.10-1+cuda13.2                            arm64        TensorRT Python plugin development headers
ii  libnvinfer-lean-dev                           10.16.2.10-1+cuda13.2                            arm64        TensorRT lean runtime libraries
ii  libnvinfer-lean10                             10.16.2.10-1+cuda13.2                            arm64        TensorRT lean runtime library
ii  libnvinfer-plugin-dev                         10.16.2.10-1+cuda13.2                            arm64        TensorRT plugin libraries
ii  libnvinfer-plugin10                           10.16.2.10-1+cuda13.2                            arm64        TensorRT plugin libraries
ii  libnvinfer-safe-headers-dev                   10.16.2.10-1+cuda13.2                            arm64        TensorRT safety development headers
ii  libnvinfer-vc-plugin-dev                      10.16.2.10-1+cuda13.2                            arm64        TensorRT vc-plugin library
ii  libnvinfer-vc-plugin10                        10.16.2.10-1+cuda13.2                            arm64        TensorRT vc-plugin library
ii  libnvinfer10                                  10.16.2.10-1+cuda13.2                            arm64        TensorRT runtime libraries
ii  nvidia-l4t-core                               39.2.1-20260806224157                            arm64        NVIDIA Core Package
ii  nvidia-tensorrt                               7.2.1-b49                                        arm64        NVIDIA TensorRT Meta Package
ii  nvidia-tensorrt-dev                           7.2.1-b49                                        arm64        NVIDIA TensorRT dev Meta Package
ii  python3-libnvinfer                            10.16.2.10-1+cuda13.2                            arm64        Python 3 bindings for TensorRT standard runtime
ii  python3-libnvinfer-dev                        10.16.2.10-1+cuda13.2                            arm64        Python 3 development package for TensorRT standard runtime
ii  python3-libnvinfer-dispatch                   10.16.2.10-1+cuda13.2                            arm64        Python 3 bindings for TensorRT dispatch runtime
ii  python3-libnvinfer-lean                       10.16.2.10-1+cuda13.2                            arm64        Python 3 bindings for TensorRT lean runtime
ii  tensorrt                                      10.16.2.10-1+cuda13.2                            arm64        Meta package for TensorRT
ii  tensorrt-libs                                 10.16.2.10-1+cuda13.2                            arm64        Meta package for TensorRT runtime libraries
=== trtexec ===
trtexec /usr/bin/trtexec
=== memory ===
MemTotal:        7727188 kB
SwapTotal:             0 kB
=== power mode ===
NV Power Mode: MAXN_SUPER
2
=== graphical session ===
graphical.target active
default-target graphical.target
gdm active
gdm3 active
=== yolo engine ===
-rw-rw-r-- 1 jetson jetson 54188692 Oct 21 12:55 /home/jetson/Desktop/experiments_fab4_orin_plus/data/models/yolo26l.engine
=== timing interpreter ===
python python3
exe /home/jetson/Desktop/hsg_srl/.venv/bin/python3
numpy OK
cv2 OK
yaml OK
PIL OK
torch OK
torchvision OK
ultralytics OK
tensorrt OK
numpy 1.26.4
torch 2.13.0+cu132 cuda=True
ultralytics 8.4.127
tensorrt 10.16.2.10
```

| site | edges | S | boxes/frame p50 | activation | detector p50/p95 | path p50/p95 | HSG p50/p95 | overlap p50/p95 | MLP p50/p95 | fusion p50/p95 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lumpi_M6 | 3 | C05,C06,C07 | 16.00 | 0.9998 |  | 1.20/1.55 | 0.37/0.40 | 0.09/0.12 | 0.73/1.07 | 0.42/38.56 |
| synth_Town05_O_day | 15 | C01,C02,C03,C04,C05,C06 | 6.00 | 0.9584 |  | 1.17/1.41 | 0.36/0.39 | 0.09/0.15 | 0.70/0.94 | 0.17/22.53 |
| ko_per | 1 | SK1,SK4 | 7.00 | 0.2572 |  | 1.12/2.01 | 0.36/0.39 | 0.04/0.05 | 0.71/1.60 | 0.09/2.18 |

## Published sample counts and detector pass

Localization frames are the calls remaining after the first ten frames are dropped. Fusion batches are the separate remerge pass. The detector pass is the same-board TensorRT FP16 YOLO26l measurement with persistent BoT-SORT. Ko-PER has no detector clip in that pass.

| Recording | Localization frames | Fusion batches | Detector p50/p95 (ms) | Detector frames |
| --- | ---: | ---: | ---: | ---: |
| M6 | 32,384 | 5,399 | 30.41/37.55 | 13,490 |
| Town05 | 10,778 | 1,798 | 26.66/31.93 | 10,790 |
| Ko-PER | 19,329 | 4,836 |  |  |
