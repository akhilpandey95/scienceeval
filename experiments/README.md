# Experiments

## Qwen3.5 and Gemma-4 merged-LoRA Fossils-M workflows

The workflow wrappers below assume a 2xH100 machine, CUDA toolkit installed at
`/usr/local/cuda-12.8`, local model weights under `/root/models`, and conda
available on `PATH`. They create separate conda envs for Qwen3.5 and Gemma-4,
serve each model with SGLang, and run every Fossils-M label benchmark except
FrontierScience:

```text
gpqa pubmedqa bioasq biored scierc mmlu simpleqa sciriff
```

By default each family workflow runs the base model first and the merged LoRA
model second so the trained model has a direct baseline. Set `RUN_BASELINE=0`
to run only the merged LoRA model.

```shell
cd /Users/akhilpandey/code/scienceeval

# Optional first pass: verify commands without launching a server.
DRY_RUN=1 bash scripts/fossil-m/workflows/run_qwen35_lora_fossils_m.sh
DRY_RUN=1 bash scripts/fossil-m/workflows/run_gemma4_lora_fossils_m.sh

# Smoke test, 5 examples per benchmark.
LIMIT_PER_BENCHMARK=5 bash scripts/fossil-m/workflows/run_qwen35_lora_fossils_m.sh
LIMIT_PER_BENCHMARK=5 bash scripts/fossil-m/workflows/run_gemma4_lora_fossils_m.sh

# Full runs.
bash scripts/fossil-m/workflows/run_qwen35_lora_fossils_m.sh
bash scripts/fossil-m/workflows/run_gemma4_lora_fossils_m.sh
```

The Qwen3.5 workflow uses:

```text
env:   scienceeval-fossils-qwen35-cu128
base:  /root/models/Qwen3.5-2B
lora:  /root/models/Qwen3.5-2B-sciriff4096-merged
port:  30000
```

The Gemma-4 workflow uses:

```text
env:   scienceeval-fossils-gemma4-cu128
base:  /root/models/google/gemma-4-E2B-it
lora:  /root/models/google/gemma-4-E2B-it-sciriff4096-merged
port:  30001
```

The wrappers install PyTorch CUDA 12.8 wheels from
`https://download.pytorch.org/whl/cu128` and set `CUDA_HOME=/usr/local/cuda-12.8`.
If `nvidia-smi` reports `CUDA Version: 12.7` while `nvcc` reports 12.8, treat
that as a driver/runtime mismatch to resolve before a full run if Torch cannot
see CUDA.

Useful overrides:

```shell
# Run merged LoRA only.
RUN_BASELINE=0 bash scripts/fossil-m/workflows/run_qwen35_lora_fossils_m.sh

# Increase or reduce request concurrency inside each benchmark.
MAX_CONCURRENT=8 bash scripts/fossil-m/workflows/run_gemma4_lora_fossils_m.sh

# Use one GPU instead of tensor parallelism across two GPUs.
CUDA_VISIBLE_DEVICES=0 TP=1 bash scripts/fossil-m/workflows/run_qwen35_lora_fossils_m.sh

# Reinstall Python packages in an existing conda env.
FORCE_REINSTALL=1 bash scripts/fossil-m/workflows/run_gemma4_lora_fossils_m.sh

# Pass additional SGLang launch args.
SGLANG_EXTRA_ARGS="--attention-backend triton --sampling-backend pytorch" \
  bash scripts/fossil-m/workflows/run_qwen35_lora_fossils_m.sh

# Use SGLang's default kernel choices instead of the conservative fallback.
SGLANG_EXTRA_ARGS= bash scripts/fossil-m/workflows/run_gemma4_lora_fossils_m.sh
```

Results are written under:

```text
results/fossil-m/qwen3-5-2b-base/
results/fossil-m/qwen3-5-2b-sciriff4096-merged/
results/fossil-m/gemma-4-e2b-it-base/
results/fossil-m/gemma-4-e2b-it-sciriff4096-merged/
```
