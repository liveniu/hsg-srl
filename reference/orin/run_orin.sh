#!/usr/bin/env bash
# Record the board, then time detector, the overlap-gated localization path, and fusion.
# A missing YOLO engine skips the detector. It does not skip HSG, overlap, or fusion.
set -euo pipefail
BUNDLE="$(cd "$(dirname "$0")" && pwd)"
if [[ -z "${PYTHON:-}" && -x "${BUNDLE}/.venv/bin/python" ]]; then
  PYTHON="${BUNDLE}/.venv/bin/python"
fi
PY="${PYTHON:-python3}"
export PYTHON="${PY}"
# Scoring and fusion come from this package, not from a second copy of the pipeline.
export PYTHONPATH="${BUNDLE}/../../src:${PYTHONPATH:-}"
bash "${BUNDLE}/board_check.sh"
LIMIT="${HSG_LIMIT_FRAMES:-0}"
ENGINE="${HSG_ORIN_ENGINE:-${BUNDLE}/data/models/yolo26l.engine}"
CLIPS="${HSG_ORIN_CLIPS:-${BUNDLE}/data/clips}"
ARGS=(--limit-frames "${LIMIT}")
if [[ -f "${ENGINE}" ]]; then
  ARGS+=(--engine "${ENGINE}" --clips "${CLIPS}")
else
  echo "no engine at ${ENGINE}; detector will be skipped. Run: bash build_engine.sh"
fi
if [[ "${LIMIT}" != "0" ]]; then
  echo "HSG_LIMIT_FRAMES=${LIMIT}: fusion is skipped because a partial frame set is not a remerge"
fi
echo "python ${PY}"
"${PY}" -u "${BUNDLE}/profile_runtime.py" "${ARGS[@]}"
echo "wrote ${BUNDLE}/outputs/runtime.md"
echo "board record: ${BUNDLE}/outputs/board_check.txt"
