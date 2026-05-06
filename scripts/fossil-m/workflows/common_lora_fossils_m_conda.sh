#!/usr/bin/env bash

set -euo pipefail

# Shared Fossils-M conda workflow helpers for local merged LoRA models.

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${WORKFLOW_DIR}/../../.." && pwd)"

# CUDA / runtime defaults
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_CUDA_INDEX_URL="${TORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_PACKAGE_SPEC="${TORCH_PACKAGE_SPEC:-torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0}"
PIP_PACKAGE_SPEC="${PIP_PACKAGE_SPEC:-sglang transformers accelerate safetensors sentencepiece protobuf datasets pyarrow pandas huggingface_hub}"

# Fossils-M defaults. FrontierScience is intentionally excluded.
BENCHMARKS="${BENCHMARKS:-gpqa pubmedqa bioasq biored scierc mmlu simpleqa sciriff}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/fossil-m}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
LIMIT_PER_BENCHMARK="${LIMIT_PER_BENCHMARK:-}"
RESUME="${RESUME:-1}"
RUN_BASELINE="${RUN_BASELINE:-1}"
DRY_RUN="${DRY_RUN:-0}"

# SGLang defaults
HOST="${HOST:-0.0.0.0}"
TP="${TP:-2}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
SERVER_START_TIMEOUT_SECONDS="${SERVER_START_TIMEOUT_SECONDS:-900}"
if [[ -z "${SGLANG_EXTRA_ARGS+x}" ]]; then
  SGLANG_EXTRA_ARGS="--attention-backend triton --sampling-backend pytorch"
fi

SETUP_VERSION="cu128-v1"
SGLANG_PID=""


log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}


require_command() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "${name}" >&2
    exit 1
  fi
}


require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    printf 'Missing %s directory: %s\n' "${label}" "${path}" >&2
    exit 1
  fi
}


slugify() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
}


load_conda_shell() {
  require_command conda

  local conda_base
  conda_base="$(conda info --base)"
  if [[ -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${conda_base}/etc/profile.d/conda.sh"
  else
    eval "$(conda shell.bash hook)"
  fi
}


conda_env_exists() {
  local env_name="$1"
  conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "${env_name}"
}


ensure_conda_env() {
  local env_name="$1"

  if ! conda_env_exists "${env_name}"; then
    log "Creating conda env ${env_name} with Python ${PYTHON_VERSION}"
    conda create -y -n "${env_name}" "python=${PYTHON_VERSION}"
  fi

  conda activate "${env_name}"
  export CUDA_HOME
  export CUDA_VISIBLE_DEVICES
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export SGLANG_API_KEY="${SGLANG_API_KEY:-EMPTY}"
}


install_workflow_packages() {
  local marker_path="${CONDA_PREFIX}/.scienceeval_fossils_m_${SETUP_VERSION}.complete"

  if [[ "${FORCE_REINSTALL:-0}" != "1" && -f "${marker_path}" ]]; then
    log "Using existing packages in ${CONDA_DEFAULT_ENV}; set FORCE_REINSTALL=1 to reinstall"
    return
  fi

  log "Installing PyTorch CUDA packages from ${TORCH_CUDA_INDEX_URL}"
  python -m pip install --upgrade pip setuptools wheel packaging ninja

  local torch_packages=()
  read -r -a torch_packages <<< "${TORCH_PACKAGE_SPEC}"
  python -m pip install "${torch_packages[@]}" --index-url "${TORCH_CUDA_INDEX_URL}"

  log "Installing Fossils-M serving and evaluation packages"
  local pip_packages=()
  read -r -a pip_packages <<< "${PIP_PACKAGE_SPEC}"
  python -m pip install --upgrade "${pip_packages[@]}"

  if [[ -n "${EXTRA_PIP_PACKAGES:-}" ]]; then
    local extra_packages=()
    read -r -a extra_packages <<< "${EXTRA_PIP_PACKAGES}"
    python -m pip install --upgrade "${extra_packages[@]}"
  fi

  touch "${marker_path}"
}


print_gpu_context() {
  log "CUDA_HOME=${CUDA_HOME}"
  log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | sed 's/^/[nvcc] /'
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi | sed 's/^/[nvidia-smi] /'
  fi
}


verify_torch_cuda() {
  python - <<'PY'
import torch

print(f"[torch] version={torch.__version__}")
print(f"[torch] cuda_runtime={torch.version.cuda}")
print(f"[torch] cuda_available={torch.cuda.is_available()}")
print(f"[torch] device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("Torch cannot see CUDA.")
if torch.cuda.device_count() < 2:
    raise SystemExit("Expected at least 2 CUDA devices for the default TP=2 workflow.")
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"[torch] gpu{index}={props.name} total_memory_gb={props.total_memory / 1024**3:.1f}")
PY
}


cleanup_server() {
  if [[ -n "${SGLANG_PID}" ]] && kill -0 "${SGLANG_PID}" >/dev/null 2>&1; then
    log "Stopping SGLang server pid=${SGLANG_PID}"
    kill "${SGLANG_PID}" >/dev/null 2>&1 || true
    wait "${SGLANG_PID}" 2>/dev/null || true
  fi
  SGLANG_PID=""
}


start_sglang_server() {
  local model_path="$1"
  local served_model_name="$2"
  local port="$3"

  local log_dir="${RESULTS_DIR}/logs"
  local model_slug
  model_slug="$(slugify "${served_model_name}")"
  local log_path="${log_dir}/${model_slug}.sglang.log"
  mkdir -p "${log_dir}"

  local extra_args=()
  if [[ -n "${SGLANG_EXTRA_ARGS}" ]]; then
    read -r -a extra_args <<< "${SGLANG_EXTRA_ARGS}"
  fi

  log "Starting SGLang for ${served_model_name}"
  log "Server log: ${log_path}"
  python -m sglang.launch_server \
    --model-path "${model_path}" \
    --served-model-name "${served_model_name}" \
    --host "${HOST}" \
    --port "${port}" \
    --tp "${TP}" \
    --context-length "${CONTEXT_LENGTH}" \
    --trust-remote-code \
    "${extra_args[@]}" \
    >"${log_path}" 2>&1 &
  SGLANG_PID="$!"

  wait_for_server "${port}" "${log_path}"
}


wait_for_server() {
  local port="$1"
  local log_path="$2"
  local start_time
  start_time="$(date +%s)"

  while true; do
    if ! kill -0 "${SGLANG_PID}" >/dev/null 2>&1; then
      printf 'SGLang server exited before becoming ready. Last log lines:\n' >&2
      tail -n 80 "${log_path}" >&2 || true
      exit 1
    fi

    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      log "SGLang is ready on http://127.0.0.1:${port}/v1"
      return
    fi

    local now
    now="$(date +%s)"
    if (( now - start_time > SERVER_START_TIMEOUT_SECONDS )); then
      printf 'Timed out waiting for SGLang after %s seconds. Last log lines:\n' "${SERVER_START_TIMEOUT_SECONDS}" >&2
      tail -n 80 "${log_path}" >&2 || true
      exit 1
    fi
    sleep 5
  done
}


run_fossils_m_suite() {
  local served_model_name="$1"
  local port="$2"

  local benchmarks=()
  read -r -a benchmarks <<< "${BENCHMARKS}"

  local python_bin="${PYTHON_BIN:-python}"
  if ! command -v "${python_bin}" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
      python_bin="python3"
    else
      printf 'Missing required command: %s\n' "${python_bin}" >&2
      exit 1
    fi
  fi

  local cmd=(
    "${python_bin}"
    "${REPO_ROOT}/scripts/fossil-m/run_fossil_suite.py"
    --model "${served_model_name}"
    --base-url "http://127.0.0.1:${port}/v1"
    --api-key-env SGLANG_API_KEY
    --benchmarks "${benchmarks[@]}"
    --results-dir "${RESULTS_DIR}"
    --max-concurrent "${MAX_CONCURRENT}"
    --max-tokens "${MAX_TOKENS}"
    --timeout-seconds "${TIMEOUT_SECONDS}"
  )

  if [[ "${RESUME}" == "1" ]]; then
    cmd+=(--resume)
  fi
  if [[ -n "${LIMIT_PER_BENCHMARK}" ]]; then
    cmd+=(--limit-per-benchmark "${LIMIT_PER_BENCHMARK}")
  fi
  if [[ -n "${EXTRA_BODY:-}" ]]; then
    cmd+=(--extra-body "${EXTRA_BODY}")
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    cmd+=(--dry-run)
  fi

  log "Running Fossils-M benchmarks for ${served_model_name}: ${BENCHMARKS}"
  (cd "${REPO_ROOT}" && "${cmd[@]}")
}


run_model_variant() {
  local model_path="$1"
  local served_model_name="$2"
  local port="$3"

  if [[ "${DRY_RUN}" == "1" ]]; then
    run_fossils_m_suite "${served_model_name}" "${port}"
    return
  fi

  start_sglang_server "${model_path}" "${served_model_name}" "${port}"
  run_fossils_m_suite "${served_model_name}" "${port}"
  cleanup_server
}


run_family_workflow() {
  local env_name="$1"
  local base_model_path="$2"
  local base_served_model_name="$3"
  local lora_model_path="$4"
  local lora_served_model_name="$5"
  local port="$6"

  if [[ "${DRY_RUN}" == "1" ]]; then
    [[ -d "${lora_model_path}" ]] || log "DRY_RUN=1; merged LoRA path not found locally: ${lora_model_path}"
    if [[ "${RUN_BASELINE}" == "1" && ! -d "${base_model_path}" ]]; then
      log "DRY_RUN=1; base model path not found locally: ${base_model_path}"
    fi
  else
    require_dir "${lora_model_path}" "merged LoRA model"
    if [[ "${RUN_BASELINE}" == "1" ]]; then
      require_dir "${base_model_path}" "base model"
    fi
  fi

  trap cleanup_server EXIT
  trap 'cleanup_server; exit 130' INT
  trap 'cleanup_server; exit 143' TERM

  if [[ "${DRY_RUN}" != "1" ]]; then
    load_conda_shell
    ensure_conda_env "${env_name}"
    install_workflow_packages
    print_gpu_context
    verify_torch_cuda
  else
    log "DRY_RUN=1; skipping conda setup and server launch"
  fi

  if [[ "${RUN_BASELINE}" == "1" ]]; then
    run_model_variant "${base_model_path}" "${base_served_model_name}" "${port}"
  fi
  run_model_variant "${lora_model_path}" "${lora_served_model_name}" "${port}"

  log "Workflow complete. Results are under ${RESULTS_DIR}"
}
