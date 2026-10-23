# Datasets

Videos and labels are not in this package. Download them under their own licenses.

YOLO26l detections are not identical across machines. The geometry databases below freeze the detections used for the published tables:

https://doi.org/10.5281/zenodo.23008421

The archive root is the contents of `data/work/`: `README.txt`, `SHA256SUMS`, and `sites/<recording>/`. Extract it into `data/work/`. The thirteen recordings are LUMPI M0–M6, Ko-PER Sequence1a–1d as one recording, and Synthehicle Town01–Town05. Each directory at `data/work/sites/<recording>/` contains `geometry.db`, `origin.db`, the 60/10/30 split, `det_gt_match_report.json`, and `reference_verdict.txt`.

| Variable | Default directory | Contents |
| --- | --- | --- |
| `HSGSRL_LUMPI_ROOT` | `data/external/LUMPI` | Measurements 0-6 |
| `HSGSRL_SYNTHEHICLE_ROOT` | `data/external/synthehicle_core` | Town01-O-day through Town05-O-day, plus calibration |
| `HSGSRL_KOPER_ROOT` | `data/external/ko_per` | Sequence1a–1d, undistorted SK1/SK4 images, labels, and calibration |

LUMPI: https://data.uni-hannover.de/en/dataset/lumpi

The official test clips do not contain the 3-D boxes used here, so they are not imported.

Ko-PER is one recording. Sequence1a–1d share a camera-clock epoch and one 60/10/30 split. Import shifts the annotated front-bumper bottom center backward by half the vehicle length and stores that footprint center. The published SK1–SK4 overlap edge is frozen in `configs/overlap/ko_per.json` (579.35 m²). A from-video graph builder keeps Ko-PER edges of at least 500 m² and LUMPI/Synthehicle edges of at least 1000 m². Published scoring does not apply that threshold again.

Ko-PER: https://www.uni-ulm.de/in/mrm/forschung/datensaetze/ (Ko-PER intersection dataset).

Place a YOLO26l weight at `data/models/yolo26l.pt`. A TensorRT engine must be built on the machine that runs it. The geometry databases are the Zenodo archive above. They are not part of this git repository.
