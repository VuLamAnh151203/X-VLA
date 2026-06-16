#!/usr/bin/env bash
set -euo pipefail

SOURCE_PATH="${BASH_SOURCE[0]}"
if [[ "${SOURCE_PATH}" == */* ]]; then
  SCRIPT_DIR="$(cd "${SOURCE_PATH%/*}" && pwd)"
else
  SCRIPT_DIR="$(pwd)"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/libero_direct_eval.py" --help
  exit 0
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run_libero_direct_eval.sh MODEL_PATH [TASK_SUITES] [TASK_IDS] [extra libero_direct_eval.py args...]" >&2
  echo "" >&2
  echo "Example:" >&2
  echo "  bash run_libero_direct_eval.sh /kaggle/input/my-checkpoint/pretrained_model libero_spatial 0,1 --eval_time 1 --no_video" >&2
  exit 2
fi

MODEL_PATH="$1"
shift

TASK_SUITES=""
if [[ $# -gt 0 && "${1}" != --* ]]; then
  TASK_SUITES="$1"
  shift
fi

TASK_IDS=""
if [[ $# -gt 0 && "${1}" != --* ]]; then
  TASK_IDS="$1"
  shift
fi

COMMAND=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/libero_direct_eval.py"
  "--model_path"
  "${MODEL_PATH}"
)

if [[ -n "${TASK_SUITES}" ]]; then
  COMMAND+=("--task_suites" "${TASK_SUITES}")
fi

if [[ -n "${TASK_IDS}" && "${TASK_IDS}" != "all" ]]; then
  COMMAND+=("--task_ids" "${TASK_IDS}")
fi

COMMAND+=("$@")

exec "${COMMAND[@]}"
