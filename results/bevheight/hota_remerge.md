# BEVHeight HOTA after replay through the paper fusion backend

Hung test boxes use BEVHeight. Other observations keep HSG.
Fusion is the paper call: score 0.70, max distance 2.5 m, full FOV graph.
Readout is TrackEval HOTA, similarity max(0, 1-d/2.5 m).

## lumpi_M6
- BEV-HOTA 35.3
- gate 0.5: HOTA=25.8 DetA=17.5 AssA=38.0 LocA=90.2 tp=15289 fp=35124 fn=36744
- gate 1.0: HOTA=36.0 DetA=28.0 AssA=46.3 LocA=84.4 tp=22394 fp=28019 fn=29639
- gate 1.5: HOTA=39.0 DetA=31.4 AssA=48.4 LocA=81.5 tp=24469 fp=25944 fn=27564
- gate 2.0: HOTA=40.2 DetA=33.4 AssA=48.4 LocA=79.1 tp=25652 fp=24761 fn=26381

## lumpi_M3
- BEV-HOTA 31.2
- gate 0.5: HOTA=17.8 DetA=8.1 AssA=39.0 LocA=88.1 tp=3932 fp=32649 fn=11823
- gate 1.0: HOTA=28.2 DetA=18.5 AssA=43.0 LocA=79.4 tp=8173 fp=28408 fn=7582
- gate 1.5: HOTA=37.5 DetA=27.3 AssA=51.6 LocA=71.7 tp=11210 fp=25371 fn=4545
- gate 2.0: HOTA=41.1 DetA=31.6 AssA=53.5 LocA=67.2 tp=12574 fp=24007 fn=3181

## synth_Town05_O_day
- BEV-HOTA 56.8
- gate 0.5: HOTA=49.8 DetA=39.7 AssA=62.5 LocA=91.5 tp=5715 fp=5430 fn=3266
- gate 1.0: HOTA=57.3 DetA=49.4 AssA=66.4 LocA=88.7 tp=6655 fp=4490 fn=2326
- gate 1.5: HOTA=59.6 DetA=52.1 AssA=68.3 LocA=87.4 tp=6892 fp=4253 fn=2089
- gate 2.0: HOTA=60.7 DetA=53.1 AssA=69.4 LocA=86.7 tp=6977 fp=4168 fn=2004
