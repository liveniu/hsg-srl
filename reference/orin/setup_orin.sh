#!/usr/bin/env bash
# Create .venv that sees the host's already installed packages.
# Does not pip-install torch, ultralytics, or opencv.
set -euo pipefail
BUNDLE="$(cd "$(dirname "$0")" && pwd)"

echo "=== board ==="
tr -d '\0' < /proc/device-tree/model || true
echo
echo "=== jetpack / tensorrt packages ==="
cat /etc/nv_tegra_release 2>/dev/null || true
dpkg -l | awk '/nvidia-l4t-core|tensorrt|libnvinfer/{print}' || true
echo
if command -v trtexec >/dev/null; then
  echo "trtexec $(command -v trtexec)"
else
  echo "trtexec MISSING"
fi

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
echo "next: bash build_engine.sh"
