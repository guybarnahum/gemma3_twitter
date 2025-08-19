#!/usr/bin/env bash
set -euo pipefail

# -------- load .env early (so PY_BIN & friends apply) --------
ENV_FILE="${ENV_FILE:-.env}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "$ENV_FILE"
  set +a
fi

# --- config (can be overridden via .env) ---
VENV_DIR="${VENV_DIR:-.venv}"
REQ_FILE="${REQ_FILE:-requirements.txt}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_gemma3.py}"
DEFAULT_OUT="${DEFAULT_OUT:-dataset/train.jsonl}"
STATE_FILE="${STATE_FILE:-state/last_sync.json}"

# Prefer Python from .env if set; else try 3.11 → 3
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
  $0 setup                      # create venv and install requirements
  $0 shell                      # open a subshell with venv activated
  $0 convert ARCHIVE [OUT]      # run twitter_to_jsonl.py (default OUT=${DEFAULT_OUT})
  $0 sync [args...]             # run incremental_sync.py (pass args through or use .env)
  $0 train [args...]            # run ${TRAIN_SCRIPT} (args forwarded)
  $0 daily                      # sync (from .env) -> train (auto-resume)
  $0 clean                      # remove venv

Examples:
  $0 setup
  $0 convert ~/Downloads/twitter-archive ${DEFAULT_OUT}
  $0 sync --username your_handle --out ${DEFAULT_OUT} --state ${STATE_FILE}
  $0 train --epochs 1 --resume
  $0 daily

Notes:
- Reads environment from .env (override path with ENV_FILE=/path/to/.env).
- 'daily' uses .env values unless you pass flags to 'sync'.

Relevant .env keys:
  TWITTER_BEARER_TOKEN   # required for API (or pass --bearer to sync)
  TWITTER_USERNAME       # your @handle (without '@')
  EXCLUDE_SOURCES        # optional CSV/space list (e.g. "MyBotApp AnotherApp")
  INCLUDE_REPLIES=1      # set to include replies
  NO_QUOTES=1            # set to exclude quote-tweets
  EPOCHS=1               # training epochs for daily
  MODEL_NAME             # override model for train_gemma3.py (optional)
  PY_BIN                 # override Python, e.g. python3.11
  VENV_DIR, DEFAULT_OUT, STATE_FILE, TRAIN_SCRIPT  # optional overrides
EOF
}

ensure_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "No venv found at ${VENV_DIR}. Run: $0 setup" >&2
    exit 1
  fi
}

activate() {
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
}

create_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating venv at ${VENV_DIR} with ${PY_BIN}..."
    "${PY_BIN}" -m venv "${VENV_DIR}"
  else
    echo "Venv already exists at ${VENV_DIR}"
  fi
  activate
  python -m pip install --upgrade pip setuptools wheel
  if [[ -f "${REQ_FILE}" ]]; then
    pip install -r "${REQ_FILE}"
  else
    echo "requirements.txt not found in $(pwd)" >&2
    exit 1
  fi
  echo "Venv ready. To enter later: source ${VENV_DIR}/bin/activate"
}

subshell() {
  ensure_venv
  activate
  bash --noprofile --norc
}

find_converter() {
  if [[ -f "twitter_to_jsonl.py" ]]; then
    echo "twitter_to_jsonl.py"
  elif [[ -f "dataset/twitter_to_jsonl.py" ]]; then
    echo "dataset/twitter_to_jsonl.py"
  else
    echo ""
  fi
}

convert_archive() {
  ensure_venv
  activate
  local ARCHIVE="${1:-}"
  local OUT_PATH="${2:-${DEFAULT_OUT}}"
  if [[ -z "${ARCHIVE}" ]]; then
    echo "Missing ARCHIVE path." >&2
    usage; exit 1
  fi
  mkdir -p "$(dirname "${OUT_PATH}")"
  local CONVERTER
  CONVERTER="$(find_converter)"
  if [[ -z "${CONVERTER}" ]]; then
    echo "Could not find twitter_to_jsonl.py (looked in ./ and ./dataset/)." >&2
    exit 1
  fi
  echo "Using converter: ${CONVERTER}"
  python "${CONVERTER}" "${ARCHIVE}" --out "${OUT_PATH}"
}

train() {
  ensure_venv
  activate
  if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "Missing ${TRAIN_SCRIPT} in repo root." >&2
    exit 1
  fi
  # Forward any args to the training script (e.g., --epochs, --resume, --model)
  python "${TRAIN_SCRIPT}" "$@"
}

sync() {
  ensure_venv
  activate
  if [[ ! -f "incremental_sync.py" ]]; then
    echo "Missing incremental_sync.py in repo root." >&2
    exit 1
  fi

  if [[ "$#" -gt 0 ]]; then
    # Pass-through mode: user supplied explicit flags
    python incremental_sync.py "$@"
    return
  fi

  # Env-driven defaults (.env)
  : "${TWITTER_USERNAME:?Set TWITTER_USERNAME in .env or pass --username to 'sync'}"
  mkdir -p "$(dirname "${DEFAULT_OUT}")" "$(dirname "${STATE_FILE}")"

  args=( --username "${TWITTER_USERNAME}" --out "${DEFAULT_OUT}" --state "${STATE_FILE}" )
  if [[ -n "${TWITTER_BEARER_TOKEN:-}" ]]; then
    args+=( --bearer "${TWITTER_BEARER_TOKEN}" )
  fi
  if [[ -n "${EXCLUDE_SOURCES:-}" ]]; then
    args+=( --exclude-sources "${EXCLUDE_SOURCES}" )
  fi
  if [[ -n "${INCLUDE_REPLIES:-}" ]]; then
    args+=( --include-replies )
  fi
  if [[ -n "${NO_QUOTES:-}" ]]; then
    args+=( --no-quotes )
  fi

  echo "Running incremental_sync.py ${args[*]/$TWITTER_BEARER_TOKEN/***TOKEN***}"
  python incremental_sync.py "${args[@]}"
}

daily() {
  # 1) Sync using .env
  sync
  # 2) Train a short pass, auto-resume; allow EPOCHS & MODEL_NAME via .env
  local epochs="${EPOCHS:-1}"
  echo "Starting training (epochs=${epochs})..."
  train --epochs "${epochs}" --resume
}

clean() {
  rm -rf "${VENV_DIR}"
  echo "Removed ${VENV_DIR}"
}

cmd="${1:-}"
case "${cmd}" in
  setup)   shift; create_venv "$@";;
  shell)   shift; subshell;;
  convert) shift; convert_archive "$@";;
  sync)    shift; sync "$@";;
  train)   shift; train "$@";;
  daily)   shift; daily "$@";;
  clean)   shift; clean;;
  ""|-h|--help|help) usage;;
  *) echo "Unknown command: ${cmd}"; usage; exit 1;;
esac

