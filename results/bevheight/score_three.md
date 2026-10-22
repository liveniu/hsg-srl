# BEVHeight reference on M6, M3, and Town05

Attachment is detection order: the first unused test box containing the projected ground point.
Localization is the stored matched subset. HSG-SRL seed 0 uses retained validation-pair
support and the frozen overlap mask. On M6 that support set is C05, C06, and C07.
A deployment file containing only C06 and C07 is not this comparison.
Full-set HOTA is the published overlap-gated recording score. BEVHeight tracking is in hota_remerge.md.

## lumpi_M6
- matched subset: 130177
- HSG on attached subset: n=130177 med=0.689 rmse=0.895 p90=1.340 n>5m=16 (0.012%)
- HSG-SRL seed 0 on attached subset: n=130177 med=0.563 rmse=0.830 p90=1.258 n>5m=15 (0.012%)
- BEVHeight on attached subset: n=130177 med=0.503 rmse=2.107 p90=2.987 n>5m=7210 (5.539%)
- full-set HOTA HSG: 34.4
- full-set HOTA HSG-SRL seed 0: 38.0
- full-set HOTA BEVHeight mixed replay: 35.3

## lumpi_M3
- matched subset: 80002
- HSG on attached subset: n=80002 med=0.946 rmse=1.178 p90=1.834 n>5m=27 (0.034%)
- HSG-SRL seed 0 on attached subset: n=80002 med=0.976 rmse=1.212 p90=1.796 n>5m=38 (0.047%)
- BEVHeight on attached subset: n=80002 med=0.822 rmse=1.958 p90=3.358 n>5m=5130 (6.412%)
- full-set HOTA HSG: 31.8
- full-set HOTA HSG-SRL seed 0: 31.5
- full-set HOTA BEVHeight mixed replay: 31.2

## synth_Town05_O_day
- matched subset: 15983
- HSG on attached subset: n=15983 med=0.559 rmse=0.865 p90=1.306 n>5m=28 (0.175%)
- HSG-SRL seed 0 on attached subset: n=15983 med=0.564 rmse=0.836 p90=1.238 n>5m=3 (0.019%)
- BEVHeight on attached subset: n=15983 med=0.266 rmse=1.333 p90=0.835 n>5m=443 (2.772%)
- full-set HOTA HSG: 49.5
- full-set HOTA HSG-SRL seed 0: 48.5
- full-set HOTA BEVHeight mixed replay: 56.8
