#!/usr/bin/env bash

set -euo pipefail

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
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
