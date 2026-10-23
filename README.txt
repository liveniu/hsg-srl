HSG-SRL

Paper: Geometry-Guided Metric Localization With Self-Supervised Residual
Learning for Roadside Multi-Camera Vehicle Tracking.

Repository: https://github.com/liveniu/hsg-srl.git

Quick start
-----------
python -m venv .venv
pip install -r requirements-minimal.txt
python scripts/run_smoke_test.py

Published tables use the released geometry databases, because a new
YOLO26l detection does not match the paper's detections:
https://doi.org/10.5281/zenodo.23008421
The archive root is the contents of data/work/. Extract it into
data/work/, so each recording is at data/work/sites/<recording>/,
and start at pair mining.

A new installation from downloaded video:
python scripts/run_paper.py --site lumpi_M6 --model data/models/yolo26l.pt

See README.md and docs/REPRODUCTION.md.
Licensed videos are not in this package.
