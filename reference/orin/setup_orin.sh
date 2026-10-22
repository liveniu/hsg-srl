#!/usr/bin/env bash
# Board setup for the localization-path timer.
# Creates .venv with the host packages. Does not pip-install torch or opencv.
# When run as root, stops the graphical session for this boot and can set
# nvpmodel id 2 if HSG_ORIN_SET_POWER=1.
set -euo pipefail
BUNDLE="$(cd "$(dirname "$0")" && pwd)"

VENV="${BUNDLE}/.venv"
if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv --system-site-packages "${VENV}"
fi
PY="${VENV}/bin/python"

echo
echo "=== host packages visible inside ${VENV} ==="
"${PY}" - <<'PY'
import importlib.util, sys
print("exe", sys.executable)
names = (
    "numpy", "cv2", "yaml", "PIL",
    "torch", "torchvision", "ultralytics", "tensorrt",
)
missing = []
for name in names:
    spec = importlib.util.find_spec(name)
    print(name, "OK" if spec else "MISSING")
    if spec is None:
        missing.append(name)
def show(label, fn):
    try:
        print(label, fn())
    except Exception as exc:
        print(label, "FAILED", exc)
show("numpy", lambda: __import__("numpy").__version__)
show("cv2", lambda: __import__("cv2").__version__)
show("cv2 gstreamer", lambda: int(getattr(__import__("cv2"), "CAP_GSTREAMER", 0) != 0))
show("yaml", lambda: __import__("yaml").__version__)
show("PIL", lambda: __import__("PIL").__version__)
show("torch", lambda: "%s cuda=%s" % (__import__("torch").__version__, __import__("torch").cuda.is_available()))
show("torchvision", lambda: __import__("torchvision").__version__)
show("ultralytics", lambda: __import__("ultralytics").__version__)
show("tensorrt", lambda: __import__("tensorrt").__version__)
if missing:
    print("missing", ",".join(missing))
PY
if command -v gst-inspect-1.0 >/dev/null; then
  gst-inspect-1.0 nvv4l2decoder >/dev/null 2>&1 && echo "nvv4l2decoder OK" || echo "nvv4l2decoder MISSING"
else
  echo "gst-inspect MISSING"
fi

if [[ "$(id -u)" -eq 0 ]]; then
  if [[ "${HSG_ORIN_SET_POWER:-0}" == "1" ]] && command -v nvpmodel >/dev/null; then
    echo "setting nvpmodel -m 2 (MAXN_SUPER on the published board)"
    nvpmodel -m 2
    nvpmodel -q || true
  fi
  if command -v systemctl >/dev/null; then
    graphic=0
    if [[ "$(systemctl is-active graphical.target 2>/dev/null || true)" == "active" ]]; then
      graphic=1
    fi
    for svc in gdm gdm3 lightdm; do
      if [[ "$(systemctl is-active "${svc}.service" 2>/dev/null || true)" == "active" ]]; then
        graphic=1
      fi
    done
    if [[ "${graphic}" -eq 1 ]]; then
      echo "stopping the graphical session for this boot"
      systemctl isolate multi-user.target
      echo "restore the desktop with: sudo systemctl isolate graphical.target"
    else
      echo "graphical session already inactive"
    fi
  fi
else
  echo "not root: power mode was not changed and the desktop was left running"
  echo "rerun: sudo bash setup_orin.sh"
fi

echo "next: bash build_engine.sh    # only if data/models/yolo26l.engine is missing"
echo "then: bash run_orin.sh"
echo "board model, power mode, desktop, and package versions are recorded by run_orin.sh"
