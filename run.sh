#!/usr/bin/env bash
set -euo pipefail

# -------- load .env early --------
ENV_FILE="${ENV_FILE:-.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a; # export everything we source
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

# --- config (overridable via .env) ---
VENV_DIR="${VENV_DIR:-.venv}"
REQ_FILE="${REQ_FILE:-requirements.txt}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_gemma3.py}"
DEFAULT_OUT="${DEFAULT_OUT:-dataset/train.jsonl}"
STATE_FILE="${STATE_FILE:-state/last_sync.json}"

# Prefer Python from .env; else try 3.11 → 3
PY_BIN="${PY_BIN:-}"
if [[ -z "${PY_BIN}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PY_BIN="python3.11"
  elif command -v python3 >/dev/null 2>&1; then
    PY_BIN="python3"
  else
    echo "Python 3 not found. Install Python 3.10+ and retry." >&2
    exit 1
  fi
fi

usage() {
  cat <<EOF
Usage:
  $0 setup                          # create venv and install requirements
  $0 shell                          # open a subshell with venv activated
  $0 convert ARCHIVE [OUT]          # run twitter_to_jsonl.py (default OUT=${DEFAULT_OUT})
  $0 sync [args...]                 # run incremental_sync.py (pass args through or use .env)
  $0 train [args...]                # run ${TRAIN_SCRIPT} (args forwarded)
  $0 daily                          # sync (from .env) -> train (auto-resume)
  $0 infer [BASE] [ADAPTER] [PROMPT]| read prompt from stdin if omitted
  $0 merge_adapter [BASE] [ADAPTER] [MERGED_DIR]
  $0 infer_merged [MERGED_DIR] [PROMPT]| read prompt from stdin if omitted
  $0 clean                          # remove venv

Defaults (overridable via .env):
  MODEL_NAME=google/gemma-3-4b-it
  ADAPTER_DIR=out/gemma3-twitter-lora
  MERGED_DIR=out/gemma3-merged
  EPOCHS=1
EOF
}

ensure_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "No venv found at ${VENV_DIR}. Run: $0 setup" >&2
    exit 1
  fi
}
activate() { source "${VENV_DIR}/bin/activate"; } # shellcheck disable=SC1091

create_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating venv at ${VENV_DIR} with ${PY_BIN}..."
    "${PY_BIN}" -m venv "${VENV_DIR}"
  else
    echo "Venv already exists at ${VENV_DIR}"
  fi
  activate
  python -m pip install --upgrade pip setuptools wheel
  [[ -f "${REQ_FILE}" ]] || { echo "Missing ${REQ_FILE}" >&2; exit 1; }
  pip install -r "${REQ_FILE}"
  echo "Venv ready. To enter later: source ${VENV_DIR}/bin/activate"
}

subshell() { ensure_venv; activate; bash --noprofile --norc; }

find_converter() {
  if [[ -f "twitter_to_jsonl.py" ]]; then echo "twitter_to_jsonl.py"
  elif [[ -f "dataset/twitter_to_jsonl.py" ]]; then echo "dataset/twitter_to_jsonl.py"
  else echo ""; fi
}

convert_archive() {
  ensure_venv; activate
  local ARCHIVE="${1:-}"; local OUT_PATH="${2:-${DEFAULT_OUT}}"
  [[ -n "${ARCHIVE}" ]] || { echo "Missing ARCHIVE path." >&2; usage; exit 1; }
  mkdir -p "$(dirname "${OUT_PATH}")"
  local CONVERTER; CONVERTER="$(find_converter)"
  [[ -n "${CONVERTER}" ]] || { echo "twitter_to_jsonl.py not found." >&2; exit 1; }
  echo "Using converter: ${CONVERTER}"
  python "${CONVERTER}" "${ARCHIVE}" --out "${OUT_PATH}"
}

train() {
  ensure_venv; activate
  [[ -f "${TRAIN_SCRIPT}" ]] || { echo "Missing ${TRAIN_SCRIPT}" >&2; exit 1; }
  python "${TRAIN_SCRIPT}" "$@"
}

sync_cmd() {
  ensure_venv; activate
  [[ -f "incremental_sync.py" ]] || { echo "Missing incremental_sync.py" >&2; exit 1; }
  if [[ "$#" -gt 0 ]]; then python incremental_sync.py "$@"; return; fi

  : "${TWITTER_USERNAME:?Set TWITTER_USERNAME in .env or pass --username}"
  mkdir -p "$(dirname "${DEFAULT_OUT}")" "$(dirname "${STATE_FILE}")"
  args=( --username "${TWITTER_USERNAME}" --out "${DEFAULT_OUT}" --state "${STATE_FILE}" )
  [[ -n "${TWITTER_BEARER_TOKEN:-}" ]] && args+=( --bearer "${TWITTER_BEARER_TOKEN}" )
  [[ -n "${EXCLUDE_SOURCES:-}" ]]     && args+=( --exclude-sources "${EXCLUDE_SOURCES}" )
  [[ -n "${INCLUDE_REPLIES:-}" ]]     && args+=( --include-replies )
  [[ -n "${NO_QUOTES:-}" ]]           && args+=( --no-quotes )
  echo "Running incremental_sync.py ${args[*]/$TWITTER_BEARER_TOKEN/***TOKEN***}"
  python incremental_sync.py "${args[@]}"
}

daily() {
  sync_cmd
  local epochs="${EPOCHS:-1}"
  echo "Starting training (epochs=${epochs})..."
  train --epochs "${epochs}" --resume
}

infer_adapter_cmd() {
  ensure_venv; activate
  [[ -f "infer_adapter.py" ]] || { echo "Missing infer_adapter.py" >&2; exit 1; }

  local BASE="${1:-${MODEL_NAME:-google/gemma-3-4b-it}}"
  local ADAPTER="${2:-${ADAPTER_DIR:-out/gemma3-twitter-lora}}"
  local PROMPT="${3:-}"
  if [[ -z "${PROMPT}" ]]; then
    if [ -t 0 ]; then PROMPT="Write a concise tweet in my signature style about: robotics, SLAM, AR."
    else PROMPT="$(cat)"; fi
  fi
  python infer_adapter.py --base "${BASE}" --adapter "${ADAPTER}" --prompt "${PROMPT}"
}

merge_adapter_cmd() {
  ensure_venv; activate
  [[ -f "merge_adapter.py" ]] || { echo "Missing merge_adapter.py" >&2; exit 1; }

  local BASE="${1:-${MODEL_NAME:-google/gemma-3-4b-it}}"
  local ADAPTER="${2:-${ADAPTER_DIR:-out/gemma3-twitter-lora}}"
  local MERGED="${3:-${MERGED_DIR:-out/gemma3-merged}}"
  mkdir -p "${MERGED}"
  python merge_adapter.py --base "${BASE}" --adapter "${ADAPTER}" --out "${MERGED}"
  echo "Merged model saved to: ${MERGED}"
}

infer_merged_cmd() {
  ensure_venv; activate
  [[ -f "infer_merged.py" ]] || { echo "Missing infer_merged.py" >&2; exit 1; }

  local MERGED="${1:-${MERGED_DIR:-out/gemma3-merged}}"
  local PROMPT="${2:-}"
  if [[ -z "${PROMPT}" ]]; then
    if [ -t 0 ]; then PROMPT="Write a concise tweet in my signature style about: robotics, SLAM, AR."
    else PROMPT="$(cat)"; fi
  fi
  python infer_merged.py --model "${MERGED}" --prompt "${PROMPT}"
}

clean() { rm -rf "${VENV_DIR}"; echo "Removed ${VENV_DIR}"; }

cmd="${1:-}"
case "${cmd}" in
  setup)          shift; create_venv "$@";;
  shell)          shift; subshell;;
  convert)        shift; convert_archive "$@";;
  sync)           shift; sync_cmd "$@";;
  train)          shift; train "$@";;
  daily)          shift; daily "$@";;
  infer)          shift; infer_adapter_cmd "$@";;
  merge_adapter)  shift; merge_adapter_cmd "$@";;
  infer_merged)   shift; infer_merged_cmd "$@";;
  clean)          shift; clean;;
  ""|-h|--help|help) usage;;
  *) echo "Unknown command: ${cmd}"; usage; exit 1;;
esac

