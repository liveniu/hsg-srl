#!/usr/bin/env bash
# Build the paper detector on this board: YOLO26l, imgsz 384x640, FP16.
# Do not copy an engine from Windows.
set -euo pipefail
BUNDLE="$(cd "$(dirname "$0")" && pwd)"
PT="${BUNDLE}/data/models/yolo26l.pt"
ONNX="${BUNDLE}/data/models/yolo26l.onnx"
OUT="${BUNDLE}/data/models/yolo26l.engine"
IMGSZ_H=384
IMGSZ_W=640
if [[ -z "${PYTHON:-}" && -x "${BUNDLE}/.venv/bin/python" ]]; then
  PYTHON="${BUNDLE}/.venv/bin/python"
fi
PY="${PYTHON:-python3}"
mkdir -p "$(dirname "${OUT}")"
WS="${HSG_TRT_WORKSPACE_MIB:-512}"

echo "=== tensorrt ==="
if "${PY}" -c "import tensorrt" >/dev/null 2>&1; then
  "${PY}" -c "import tensorrt; print('python tensorrt', tensorrt.__version__)"
else
  echo "python tensorrt MISSING; trtexec may still work"
fi
command -v trtexec && trtexec --version || echo "trtexec MISSING"

export_from_pt() {
  echo "ultralytics export ${PT} imgsz=${IMGSZ_H}x${IMGSZ_W}"
  "${PY}" - <<PY
from pathlib import Path
from ultralytics import YOLO
pt = Path(r"${PT}")
out = Path(r"${OUT}")
if not pt.is_file():
    raise SystemExit(f"missing {pt}")
model = YOLO(str(pt))
engine = Path(model.export(format="engine", half=True, imgsz=[${IMGSZ_H}, ${IMGSZ_W}], device=0, workspace=${WS}))
if engine.resolve() != out.resolve():
    out.write_bytes(engine.read_bytes())
print("engine", out, out.stat().st_size)
PY
}

export_from_onnx() {
  if [[ ! -f "${ONNX}" ]]; then
    echo "missing ${ONNX}"
    return 1
  fi
  if ! command -v trtexec >/dev/null; then
    echo "trtexec not on PATH"
    return 1
  fi
  echo "trtexec ${ONNX} -> ${OUT}"
  trtexec --onnx="${ONNX}" --fp16 --memPoolSize=workspace:${WS}M --saveEngine="${OUT}"
}

if [[ -f "${PT}" ]] && "${PY}" -c "import ultralytics" >/dev/null 2>&1; then
  export_from_pt || export_from_onnx
else
  export_from_onnx
fi

if [[ ! -f "${OUT}" ]]; then
  echo "failed to write ${OUT}"
  exit 1
fi
echo "wrote ${OUT}"
