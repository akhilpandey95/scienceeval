# Fossils-M open-weight evals

First-pass harness for turning an open-weight model into a model fossil sheet.
This suite intentionally skips `MMLU` and `SimpleQA` because those are commonly
reported during model launches. It focuses on:

- `FrontierScience`
- `GPQA`
- `PubMedQA`
- `BioASQ`
- `BioRED`
- `SciERC`
- `SciRIFF`

## Download `gpt-oss-20b` model weights

```shell
python - <<'PY'
from huggingface_hub import login, snapshot_download

repo_id = "openai/gpt-oss-20b"
local_dir = "/root/models/gpt-oss-20b"

login()
snapshot_download(repo_id=repo_id, local_dir=local_dir)
PY
```

## Start an SGLang server

```shell
python -m sglang.launch_server \
  --model-path /root/models/gpt-oss-20b \
  --served-model-name gpt-oss-20b \
  --host 0.0.0.0 \
  --port 30000 \
  --tp 1 \
  --context-length 8192 \
  --trust-remote-code
```

Check that the OpenAI-compatible endpoint is alive:

```shell
curl http://127.0.0.1:30000/v1/models
```

## Smoke-test the suite

Use a small limit first. FrontierScience requires a judge model for scored runs.

```shell
python scripts/fossil-m/run_fossil_suite.py \
  --model gpt-oss-20b \
  --base-url http://127.0.0.1:30000/v1 \
  --judge-model gpt-5.4 \
  --judge-base-url https://api.openai.com/v1 \
  --limit-per-benchmark 5 \
  --max-concurrent 2 \
  --resume
```

For a dry run that only prints the planned commands:

```shell
python scripts/fossil-m/run_fossil_suite.py \
  --model gpt-oss-20b \
  --base-url http://127.0.0.1:30000/v1 \
  --judge-model gpt-5.4 \
  --dry-run
```

## Run the first-pass suite

```shell
python scripts/fossil-m/run_fossil_suite.py \
  --model gpt-oss-20b \
  --base-url http://127.0.0.1:30000/v1 \
  --judge-model gpt-5.4 \
  --judge-base-url https://api.openai.com/v1 \
  --max-concurrent 4 \
  --resume
```

Results are written under:

```text
results/fossil-m/<model-slug>/
```

## Current protocol notes

- `FrontierScience` delegates to `scripts/evaluate_frontierscience.py` and uses
  the paper-style judge prompts already in this repo.
- `GPQA` uses deterministic shuffled multiple choice and exact letter scoring.
- `PubMedQA` uses yes/no/maybe label prediction from the labeled PubMedQA split.
- `BioASQ` is a first-pass answer extraction protocol over `valid_bio.csv`.
- `BioRED` is a first-pass relation-classification protocol over gold entity
  pairs, not yet the final NER-slice metric used in the GPT-4o fossil.
- `SciERC` is a marked-pair relation-classification protocol.
- `SciRIFF` uses exact-output scoring over the 8192-token test split.

The next step is to add the model-fossil summarizer and plot generator that
turns these JSON files into the same visual sheet used by `fossils.html`.

