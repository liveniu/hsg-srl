# BEVHeight reference

This is the supervised comparison on M6, M3, and Town05. It uses target-scene 3-D boxes and images. HSG-SRL does not. The published scores are `results/bevheight/score_three.md` and `results/bevheight/hota_remerge.md`. The matched-subset arrays are `results/bevheight/*_bevheight_subset.npz`. On that subset, seed-0 HSG-SRL uses the retained validation-pair support and the frozen overlap mask. M6's support set is C05, C06, and C07. A file that lists only C06 and C07 is an older deployment and is not the published comparison. M6 has 15 HSG-SRL errors above 5 m (0.012%), not zero.

A new run is optional. It needs the videos, the official BEVHeight code, a 4090, and the `detections.jsonl` from that run. None of those is in the geometry-database archive. The steps below are the procedure.

1. Extract M6, M3, and Town05 from https://doi.org/10.5281/zenodo.23008421 so `data/work/sites/<site>/` contains `origin.db`, `geometry.db`, the split manifest, and `det_gt_match_report.json`.
2. Pack KITTI-style folders:

```text
python reference/bevheight/step2_pack/tools/build_kitti_site.py
```

3. On a 4090, with the official BEVHeight tree in `BEVHEIGHT_ROOT`, run `reference/bevheight/remote_gpu_4090/tools/train_site.py` and `infer_site.py`. The configuration is the official R50 140.8 m setup, batch size 2, with early stopping on validation detection loss.
4. Copy `detections.jsonl` back to `reference/bevheight/remote_gpu_4090/outputs/<site>/`.
5. Score localization with `reference/bevheight/score_three.py` and tracking with `reference/bevheight/score_hota_remerge.py`.

The archived scores from the submitted run are in `results/bevheight/`.
