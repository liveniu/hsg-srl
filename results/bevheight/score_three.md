# BEVHeight reference on M6, M3, and Town05

Attachment is detection order: the first unused test box containing the projected ground point.
Origin and z_ground are the import metadata stored with ground-truth batch l0hn_gt_001, subtracted once.
Localization errors are on the attached test subset. HSG and HSG-SRL seed 0 on that subset are the supervised BEVHeight comparison.
Full-set HOTA below is the published recording score. BEVHeight tracking is in hota_remerge.md.

## lumpi_M6
- origin_xy: [4.384, 6.316], z_ground: -2.0
- boxes=755248 token_miss=0 frames_without_test_obs=0 no_uv=0
- test observations: 134383
- hung: 130177 (96.9%)
- HSG on attached subset: n=130177 med=0.689 rmse=0.895 p90=1.340 frac>5m=0.000 frac>10m=0.000
- HSG-SRL seed 0 on attached subset: n=130177 med=0.570 rmse=0.878 p90=1.379 frac>5m=0.000 frac>10m=0.000
- BEVHeight on attached subset: n=130177 med=0.503 rmse=2.107 p90=2.987 frac>5m=0.055 frac>10m=0.003
- BEVHeight minus GT, median dx=0.020 dy=0.047
- full-set HOTA HSG: BEV=34.4 (0.5:17.8, 1.0:34.2, 1.5:42.1, 2.0:43.6)
- full-set HOTA HSG-SRL seed 0: BEV=38.0 (0.5:24.3, 1.0:39.2, 1.5:43.8, 2.0:44.6)

## lumpi_M3
- origin_xy: [18.32, -0.16], z_ground: -2.0
- boxes=475661 token_miss=0 frames_without_test_obs=27 no_uv=0
- test observations: 85103
- hung: 80002 (94.0%)
- HSG on attached subset: n=80002 med=0.946 rmse=1.178 p90=1.834 frac>5m=0.000 frac>10m=0.000
- HSG-SRL seed 0 on attached subset: n=80002 med=0.976 rmse=1.212 p90=1.796 frac>5m=0.000 frac>10m=0.000
- BEVHeight on attached subset: n=80002 med=0.822 rmse=1.958 p90=3.358 frac>5m=0.064 frac>10m=0.001
- BEVHeight minus GT, median dx=0.094 dy=-0.292
- full-set HOTA HSG: BEV=31.8 (0.5:18.0, 1.0:29.0, 1.5:37.0, 2.0:43.2)
- full-set HOTA HSG-SRL seed 0: BEV=31.4 (0.5:18.0, 1.0:27.9, 1.5:36.9, 2.0:43.0)

## synth_Town05_O_day
- origin_xy: [27.134028898371504, -88.01786442918214], z_ground: -0.0033718271459928784
- boxes=29567 token_miss=0 frames_without_test_obs=0 no_uv=0
- test observations: 17962
- hung: 15983 (89.0%)
- HSG on attached subset: n=15983 med=0.559 rmse=0.865 p90=1.306 frac>5m=0.002 frac>10m=0.000
- HSG-SRL seed 0 on attached subset: n=15983 med=0.565 rmse=0.836 p90=1.238 frac>5m=0.000 frac>10m=0.000
- BEVHeight on attached subset: n=15983 med=0.266 rmse=1.333 p90=0.835 frac>5m=0.028 frac>10m=0.000
- BEVHeight minus GT, median dx=0.016 dy=0.012
- full-set HOTA HSG: BEV=49.5 (0.5:27.5, 1.0:49.9, 1.5:59.7, 2.0:61.0)
- full-set HOTA HSG-SRL seed 0: BEV=48.5 (0.5:23.8, 1.0:51.4, 1.5:57.4, 2.0:61.4)
