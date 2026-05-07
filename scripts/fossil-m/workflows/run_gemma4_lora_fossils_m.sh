#!/usr/bin/env bash

set -euo pipefail

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

SGLANG_PREINSTALL_PACKAGE_SPEC="${SGLANG_PREINSTALL_PACKAGE_SPEC:-sglang==0.5.10.post1}"
SGLANG_PACKAGE_SPEC="${SGLANG_PACKAGE_SPEC:-git+https://github.com/sgl-project/sglang.git#subdirectory=python}"
SGLANG_INSTALL_NO_DEPS="${SGLANG_INSTALL_NO_DEPS:-1}"
TRANSFORMERS_PACKAGE_SPEC="${TRANSFORMERS_PACKAGE_SPEC:-git+https://github.com/huggingface/transformers.git@91b1ab1fdfa81a552644a92fbe3e8d88de40e167}"
TRANSFORMERS_INSTALL_NO_DEPS="${TRANSFORMERS_INSTALL_NO_DEPS:-1}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:---attention-backend triton --sampling-backend pytorch --disable-custom-all-reduce --cuda-graph-max-bs 16 --mem-fraction-static 0.75 --model-impl transformers}"

# shellcheck source=common_lora_fossils_m_conda.sh
source "${WORKFLOW_DIR}/common_lora_fossils_m_conda.sh"

MODEL_ROOT="${MODEL_ROOT:-/root/models}"
ENV_NAME="${ENV_NAME:-scienceeval-fossils-gemma4-cu128}"
PORT="${PORT:-30001}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-${MODEL_ROOT}/google/gemma-4-E2B-it}"
LORA_MODEL_PATH="${LORA_MODEL_PATH:-${MODEL_ROOT}/google/gemma-4-E2B-it-sciriff4096-merged}"
BASE_SERVED_MODEL_NAME="${BASE_SERVED_MODEL_NAME:-gemma-4-e2b-it-base}"
LORA_SERVED_MODEL_NAME="${LORA_SERVED_MODEL_NAME:-gemma-4-e2b-it-sciriff4096-merged}"

run_family_workflow \
  "${ENV_NAME}" \
  "${BASE_MODEL_PATH}" \
  "${BASE_SERVED_MODEL_NAME}" \
  "${LORA_MODEL_PATH}" \
  "${LORA_SERVED_MODEL_NAME}" \
  "${PORT}"
