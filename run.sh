#!/usr/bin/env bash
set -euo pipefail

# -------- load .env early --------
ENV_FILE="${ENV_FILE:-.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

# --- config (overridable via .env) ---
USAGE_FILE="${USAGE_FILE:-usage.txt}"
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

# Use the venv's interpreter explicitly (works whether it's python or python3)
venv_py() {
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    printf '%s' "${VENV_DIR}/bin/python"
  elif [[ -x "${VENV_DIR}/bin/python3" ]]; then
    printf '%s' "${VENV_DIR}/bin/python3"
  else
    return 1
  fi
}

usage() {
  self_base="${0##*/}"
  if [[ -f "${USAGE_FILE}" ]]; then
    txt="$(cat "${USAGE_FILE}")"
    txt="${txt//\$0/${self_base}}"
    txt="${txt//\{\{DEFAULT_OUT\}\}/${DEFAULT_OUT}}"
    txt="${txt//\{\{TRAIN_SCRIPT\}\}/${TRAIN_SCRIPT}}"
    txt="${txt//\{\{STATE_FILE\}\}/${STATE_FILE}}"
    printf '%s\n' "$txt"
  else
    cat <<EOF
Usage:
  ${self_base} setup | shell | convert | sync | docs_sync | sql_threads | unify | split_eval | train | daily | infer | merge_adapter | infer_merged | clean
  (Missing ${USAGE_FILE}. Create one to see full help.)
EOF
  fi
}

ensure_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "No venv found at ${VENV_DIR}. Run: $0 setup" >&2
    exit 1
  fi
}

activate() { source "${VENV_DIR}/bin/activate"; } # shellcheck disable=SC1091

create_venv() {
  # Create or reuse venv
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating venv at ${VENV_DIR} with ${PY_BIN}..."
    "${PY_BIN}" -m venv "${VENV_DIR}"
  else
    echo "Venv already exists at ${VENV_DIR}"
  fi

  # Activate and locate interpreter
  activate
  VPY="$(venv_py)" || { echo "No python in ${VENV_DIR}/bin"; exit 1; }

  # Core tools first
  "${VPY}" -m pip install --upgrade pip setuptools wheel

  # ---- Install Torch first (platform-aware), no embedded Python ----
  TORCH_VER="${TORCH_VER:-2.3.1}"
  FORCE_CPU="${FORCE_CPU:-}"
  if [[ -z "${FORCE_CPU}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    # GPU Linux → CUDA 12.1 wheels
    TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
    echo "[setup] Installing torch==${TORCH_VER} from ${TORCH_INDEX_URL}"
    "${VPY}" -m pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VER}"
  else
    # CPU/macOS
    echo "[setup] Installing torch==${TORCH_VER} (CPU/macOS wheel)"
    "${VPY}" -m pip install "torch==${TORCH_VER}"
  fi

  # ---- Then the pinned HF stack from requirements.txt ----
  [[ -f "${REQ_FILE}" ]] || { echo "Missing ${REQ_FILE}" >&2; exit 1; }
  "${VPY}" -m pip install --upgrade --upgrade-strategy eager -r "${REQ_FILE}"

  # ---- Print versions without running inline Python ----
  echo "[setup] Versions:"
  for pkg in torch transformers trl accelerate peft datasets huggingface_hub; do
    ver="$("${VPY}" -m pip show "$pkg" 2>/dev/null | awk -F': ' '/^Version:/{print $2; exit}')"
    printf '  - %s %s\n' "$pkg" "${ver:-(not installed)}"
  done

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
  local ARCHIVE="${1:-}"
  local OUT_PATH="${2:-${DEFAULT_OUT}}"
  shift 2 || true
  local -a REST=()
  if ((${#@})); then REST=("$@"); fi

  [[ -n "${ARCHIVE}" ]] || { echo "Missing ARCHIVE path." >&2; usage; exit 1; }
  mkdir -p "$(dirname "${OUT_PATH}")"
  local CONVERTER; CONVERTER="$(find_converter)"
  [[ -n "${CONVERTER}" ]] || { echo "twitter_to_jsonl.py not found." >&2; exit 1; }
  echo "Using converter: ${CONVERTER}"
  local VPY; VPY="$(venv_py)" || { echo "No python in venv"; exit 1; }

  if ((${#REST[@]})); then
    "${VPY}" "${CONVERTER}" "${ARCHIVE}" --out "${OUT_PATH}" "${REST[@]}"
  else
    "${VPY}" "${CONVERTER}" "${ARCHIVE}" --out "${OUT_PATH}"
  fi
}

train() {
  ensure_venv; activate
  [[ -f "${TRAIN_SCRIPT}" ]] || { echo "Missing ${TRAIN_SCRIPT}" >&2; exit 1; }
  local VPY; VPY="$(venv_py)" || { echo "No python in venv"; exit 1; }
  "${VPY}" "${TRAIN_SCRIPT}" "$@"
}

sync_cmd() {
  ensure_venv; activate
  [[ -f "incremental_sync.py" ]] || { echo "Missing incremental_sync.py" >&2; exit 1; }
  local VPY; VPY="$(venv_py)" || { echo "No python in venv"; exit 1; }

  if [[ "$#" -gt 0 ]]; then
    "${VPY}" incremental_sync.py "$@"
    return
  fi

  : "${TWITTER_USERNAME:?Set TWITTER_USERNAME in .env or pass --username}"
  mkdir -p "$(dirname "${DEFAULT_OUT}")" "$(dirname "${STATE_FILE}")"
  local -a args=( --username "${TWITTER_USERNAME}" --out "${DEFAULT_OUT}" --state "${STATE_FILE}" )
  [[ -n "${TWITTER_BEARER_TOKEN:-}" ]] && args+=( --bearer "${TWITTER_BEARER_TOKEN}" )
  [[ -n "${EXCLUDE_SOURCES:-}" ]]     && args+=( --exclude-sources "${EXCLUDE_SOURCES}" )
  [[ -n "${INCLUDE_REPLIES:-}" ]]     && args+=( --include-replies )
  [[ -n "${NO_QUOTES:-}" ]]           && args+=( --no-quotes )

  echo "Running incremental_sync.py …"
  "${VPY}" incremental_sync.py "${args[@]}"
}

docs_sync_cmd() {
  ensure_venv; activate
  [[ -f "incremental_docs_sync.py" ]] || { echo "Missing incremental_docs_sync.py" >&2; exit 1; }
  local VPY; VPY="$(venv_py)" || { echo "No python in venv"; exit 1; }

  if [[ "$#" -gt 0 ]]; then
    "${VPY}" incremental_docs_sync.py "$@"
    return
  fi

  : "${DOCS_PATH:?Set DOCS_PATH in .env or pass --path}"
  local OUT_PATH="${DOCS_OUT:-${DEFAULT_OUT}}"
  local STATE_PATH="${DOCS_STATE_FILE:-state/docs_sync.json}"
  local MODE="${DOCS_MODE:-style}"

  local -a args=( --path "${DOCS_PATH}" --out "${OUT_PATH}" --state "${STATE_PATH}" --mode "${MODE}" )
  [[ -n "${DOCS_LANG_HINT:-}" ]]     && args+=( --lang_hint "${DOCS_LANG_HINT}" )
  [[ -n "${DOCS_PROMPT_LANG:-}" ]] && args+=( --prompt_lang "${DOCS_PROMPT_LANG}" )
  [[ -n "${DOCS_MIN_CHARS:-}" ]]     && args+=( --min_chars "${DOCS_MIN_CHARS}" )
  [[ -n "${DOCS_MAX_CHARS:-}" ]]     && args+=( --max_chars "${DOCS_MAX_CHARS}" )
  [[ -n "${DOCS_DEDUP_DATASET:-}" ]] && args+=( --dedup-dataset )
  [[ -n "${DOCS_DELETE_MISSING:-}" ]]&& args+=( --delete-missing )

  echo "Running incremental_docs_sync.py …"
  "${VPY}" incremental_docs_sync.py "${args[@]}"
}

sql_threads_cmd() {
  ensure_venv; activate
  [[ -f "sql_threads_to_jsonl.py" ]] || { echo "Missing sql_threads_to_jsonl.py" >&2; exit 1; }
  local VPY; VPY="$(venv_py)" || { echo "No python in venv"; exit 1; }

  if [[ "$#" -lt 2 ]]; then
    echo "Usage: $0 sql_threads --input forum.sql|forum.db --nick YOUR_NICK [--out ${DEFAULT_OUT}] [--max_context K] [--strip_self_context] [--role_assistant assistant|model]" >&2
    exit 1
  fi
  "${VPY}" sql_threads_to_jsonl.py "$@"
}

unify_cmd() {
  ensure_venv; activate
  local VPY; VPY="$(venv_py)" || { echo "No python in venv"; exit 1; }
  [[ -f "unify_jsonl.py" ]] || { echo "Missing unify_jsonl.py" >&2; exit 1; }

  local OUT=""
  if [[ "${1:-}" != "" && "${1#-}" == "${1}" ]]; then
    OUT="$1"; shift
  else
    OUT="${UNIFY_OUT:-${DEFAULT_OUT}}"
  fi

  if printf '%s\0' "$@" | grep -q -- '--in'; then
    "${VPY}" unify_jsonl.py --out "${OUT}" "$@"
    return
  fi

  local -a files=()
  while IFS= read -r -d '' f; do files+=("$f"); done < <(find dataset -type f -name '*.jsonl' -print0 2>/dev/null || true)

  local -a filtered=()
  for f in "${files[@]}"; do
    [[ "$f" == "$OUT" ]] && continue
    [[ "$(basename "$f")" == "train_eval.jsonl" ]] && continue
    [[ "$f" == *"_eval.jsonl" ]] && continue
    [[ ! -s "$f" ]] && continue
    if [[ -n "${UNIFY_EXCLUDE:-}" ]]; then
      IFS=',' read -r -a _exarr <<<"${UNIFY_EXCLUDE}"
      skip=0
      for pat in "${_exarr[@]}"; do
        [[ -n "$pat" && "$f" == $pat ]] && skip=1 && break
      done
      [[ $skip -eq 1 ]] && continue
    fi
    filtered+=("$f")
  done

  if [[ ${#filtered[@]} -eq 0 ]]; then
    echo "No input .jsonl files found under dataset/. Add files or pass --in ..." >&2
    exit 1
  fi

  local -a args_in=()
  for f in "${filtered[@]}"; do args_in+=( --in "$f" ); done

  local -a extra=()
  if [[ "${UNIFY_SHUFFLE:-1}" != "0" ]]; then
    extra+=( --shuffle --seed "${UNIFY_SEED:-13}" )
  fi

  echo "Unifying ${#filtered[@]} files into ${OUT}:"
  printf '  + %s\n' "${filtered[@]}"

  "${VPY}" unify_jsonl.py --out "${OUT}" "${args_in[@]}" "${extra[@]}" "$@"
}

split_eval_cmd() {
  ensure_venv; activate
  VPY="$(venv_py)" || { echo "No python in venv"; exit 1; }
  [[ -f "make_eval_split.py" ]] || { echo "Missing make_eval_split.py" >&2; exit 1; }
  # pass everything straight through to Python; no pre-parsing
  "$VPY" make_eval_split.py "$@"
}

# split_eval_cmd() {
#  ensure_venv; activate
#  [[ -f "make_eval_split.py" ]] || { echo "Missing make_eval_split.py" >&2; exit 1; }
#  local VPY; VPY="$(venv_py)" || { echo "No python in venv"; exit 1; }
#
#  local INP="${1:-${UNIFY_OUT:-${DEFAULT_OUT}}}"
#  local OUT_TRAIN="${2:-${DEFAULT_OUT}}"
#  local OUT_EVAL="${3:-dataset/train_eval.jsonl}"
#  shift || true; shift || true; shift || true
#
#  echo "Splitting ${INP} -> train=${OUT_TRAIN}, eval=${OUT_EVAL}"
#  "${VPY}" make_eval_split.py --in "${INP}" --out_train "${OUT_TRAIN}" --out_eval "${OUT_EVAL}" "$@"
#}

daily() {
  sync_cmd
  local epochs="${EPOCHS:-1}"
  echo "Starting training (epochs=${epochs})..."
  train --epochs "${epochs}" --resume
}

infer_adapter_cmd() {
  ensure_venv; activate
  [[ -f "infer_adapter.py" ]] || { echo "Missing infer_adapter.py" >&2; exit 1; }
  local VPY; VPY="$(venv_py)" || { echo "No python in venv"; exit 1; }

  local BASE="${1:-${MODEL_NAME:-google/gemma-3-4b-it}}"
  local ADAPTER="${2:-${ADAPTER_DIR:-out/gemma3-twitter-lora}}"
  local PROMPT="${3:-}"
  if [[ -z "${PROMPT}" ]]; then
    if [ -t 0 ]; then PROMPT="Write a concise tweet in my signature style about: robotics, SLAM, AR."
    else PROMPT="$(cat)"; fi
  fi
  "${VPY}" infer_adapter.py --base "${BASE}" --adapter "${ADAPTER}" --prompt "${PROMPT}"
}

merge_adapter_cmd() {
  ensure_venv; activate
  [[ -f "merge_adapter.py" ]] || { echo "Missing merge_adapter.py" >&2; exit 1; }
  local VPY; VPY="$(venv_py)" || { echo "No python in venv"; exit 1; }

  local BASE="${1:-${MODEL_NAME:-google/gemma-3-4b-it}}"
  local ADAPTER="${2:-${ADAPTER_DIR:-out/gemma3-twitter-lora}}"
  local MERGED="${3:-${MERGED_DIR:-out/gemma3-merged}}"
  mkdir -p "${MERGED}"
  "${VPY}" merge_adapter.py --base "${BASE}" --adapter "${ADAPTER}" --out "${MERGED}"
  echo "Merged model saved to: ${MERGED}"
}

infer_merged_cmd() {
  ensure_venv; activate
  [[ -f "infer_merged.py" ]] || { echo "Missing infer_merged.py" >&2; exit 1; }
  local VPY; VPY="$(venv_py)" || { echo "No python in venv"; exit 1; }

  local MERGED="${1:-${MERGED_DIR:-out/gemma3-merged}}"
  local PROMPT="${2:-}"
  if [[ -z "${PROMPT}" ]]; then
    if [ -t 0 ]; then PROMPT="Write a concise tweet in my signature style about: robotics, SLAM, AR."
    else PROMPT="$(cat)"; fi
  fi
  "${VPY}" infer_merged.py --model "${MERGED}" --prompt "${PROMPT}"
}

clean() {
  rm -rf "${VENV_DIR}"
  echo "Removed ${VENV_DIR}"
}

cmd="${1:-}"
case "${cmd}" in
  setup)          shift; create_venv "$@";;
  shell)          shift; subshell;;
  convert)        shift; convert_archive "$@";;
  sync)           shift; sync_cmd "$@";;
  docs_sync)      shift; docs_sync_cmd "$@";;
  sql_threads)    shift; sql_threads_cmd "$@";;
  unify)          shift; unify_cmd "$@";;
  split_eval)     shift; split_eval_cmd "$@";;
  train)          shift; train "$@";;
  daily)          shift; daily "$@";;
  infer)          shift; infer_adapter_cmd "$@";;
  merge_adapter)  shift; merge_adapter_cmd "$@";;
  infer_merged)   shift; infer_merged_cmd "$@";;
  clean)          shift; clean;;
  ""|-h|--help|help) usage;;
  *) echo "Unknown command: ${cmd}"; usage; exit 1;;
esac

