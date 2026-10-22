#!/usr/bin/env bash
# Record board identity, power mode, graphical session, and the YOLO engine.
# This script does not change power mode or stop the desktop.
set -euo pipefail
BUNDLE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "${BUNDLE}/outputs"
LOG="${BUNDLE}/outputs/board_check.txt"
exec > >(tee "${LOG}") 2>&1

echo "=== board ==="
if [[ -r /proc/device-tree/model ]]; then
  tr -d '\0' < /proc/device-tree/model || true
  echo
else
  echo "device-tree model MISSING"
fi

echo "=== jetpack / tensorrt packages ==="
if [[ -r /etc/nv_tegra_release ]]; then
  cat /etc/nv_tegra_release
else
  echo "nv_tegra_release MISSING"
fi
if command -v dpkg >/dev/null; then
  dpkg -l | awk '/nvidia-l4t-core|tensorrt|libnvinfer/{print}' || true
fi

echo "=== trtexec ==="
if command -v trtexec >/dev/null; then
  echo "trtexec $(command -v trtexec)"
else
  echo "trtexec MISSING"
fi

echo "=== memory ==="
if [[ -r /proc/meminfo ]]; then
  awk '/^(MemTotal|SwapTotal):/{print}' /proc/meminfo
else
  echo "meminfo MISSING"
fi

echo "=== power mode ==="
if command -v nvpmodel >/dev/null; then
  nvpmodel -q || true
else
  echo "nvpmodel MISSING"
fi

echo "=== graphical session ==="
if command -v systemctl >/dev/null; then
  echo "graphical.target $(systemctl is-active graphical.target 2>/dev/null || true)"
  echo "default-target $(systemctl get-default 2>/dev/null || true)"
  for svc in gdm gdm3 lightdm; do
    state="$(systemctl is-active "${svc}.service" 2>/dev/null || true)"
    if [[ -n "${state}" && "${state}" != "inactive" && "${state}" != "unknown" ]]; then
      echo "${svc} ${state}"
    fi
  done
else
  echo "systemctl MISSING"
fi

echo "=== yolo engine ==="
ENGINE="${HSG_ORIN_ENGINE:-${BUNDLE}/data/models/yolo26l.engine}"
if [[ -f "${ENGINE}" ]]; then
  ls -l "${ENGINE}"
else
  echo "MISSING ${ENGINE}"
  echo "build it on this board with: bash build_engine.sh"
  echo "do not copy an engine from Windows or an x86 host"
fi

echo "=== timing interpreter ==="
PY="${PYTHON:-python3}"
echo "python ${PY}"
if command -v "${PY}" >/dev/null || [[ -x "${PY}" ]]; then
  "${PY}" - <<'PY'
import importlib.util, sys
print("exe", sys.executable)
names = ("numpy", "cv2", "yaml", "PIL", "torch", "torchvision", "ultralytics", "tensorrt")
for name in names:
    spec = importlib.util.find_spec(name)
    print(name, "OK" if spec else "MISSING")
def show(label, fn):
    try:
        print(label, fn())
    except Exception as exc:
        print(label, "FAILED", exc)
show("numpy", lambda: __import__("numpy").__version__)
show("torch", lambda: "%s cuda=%s" % (__import__("torch").__version__, __import__("torch").cuda.is_available()))
show("ultralytics", lambda: __import__("ultralytics").__version__)
show("tensorrt", lambda: __import__("tensorrt").__version__)
PY
else
  echo "python MISSING"
fi
