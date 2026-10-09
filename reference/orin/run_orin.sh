#!/usr/bin/env bash
# Time HSG, F0 MLP, and fusion on M6 and Town05.
# Build the TensorRT engine on this board. Do not copy an x86 engine.
set -euo pipefail
BUNDLE="$(cd "$(dirname "$0")" && pwd)"
if [[ -z "${PYTHON:-}" && -x "${BUNDLE}/.venv/bin/python" ]]; then
  PYTHON="${BUNDLE}/.venv/bin/python"
fi
PY="${PYTHON:-python3}"
export PYTHONPATH="${BUNDLE}/../../src:${PYTHONPATH:-}"
LIMIT="${HSG_LIMIT_FRAMES:-0}"
ENGINE="${HSG_ORIN_ENGINE:-${BUNDLE}/data/models/yolo26l.engine}"
CLIPS="${HSG_ORIN_CLIPS:-${BUNDLE}/data/clips}"
ARGS=(--limit-frames "${LIMIT}")
if [[ -f "${ENGINE}" ]]; then
  ARGS+=(--engine "${ENGINE}" --clips "${CLIPS}")
else
  echo "no engine at ${ENGINE}; detector will be skipped. Run: bash build_engine.sh"
fi
echo "python ${PY}"
"${PY}" -u "${BUNDLE}/profile_runtime.py" "${ARGS[@]}"
echo "wrote ${BUNDLE}/outputs/runtime.md"
