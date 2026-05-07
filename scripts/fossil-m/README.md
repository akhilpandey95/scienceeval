# Fossils-M open-weight evals

First-pass harness for turning an open-weight model into a model fossil sheet.
It covers:

- `FrontierScience`
- `GPQA`
- `PubMedQA`
- `BioASQ`
- `BioRED`
- `SciERC`
- `MMLU`
- `SimpleQA`
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

## Download benchmark datasets

These commands recreate the local dataset layout read by the Fossils-M
evaluators. GPQA is gated, so accept the terms on Hugging Face and run
`hf auth login` first.

```shell
mkdir -p data/biored/extracted

hf download Idavidrein/gpqa \
  gpqa_diamond.csv gpqa_experts.csv gpqa_extended.csv gpqa_main.csv license.txt \
  --repo-type dataset \
  --local-dir data/gpqa/extracted/dataset
hf download kroshan/BioASQ --repo-type dataset --local-dir data/bioasq
hf download qiaojin/PubMedQA --repo-type dataset --local-dir data/pubmedqa
hf download nsusemiehl/SciERC --repo-type dataset --local-dir data/scierc
hf download allenai/SciRIFF --repo-type dataset --local-dir data/sciriff
hf download cais/mmlu --repo-type dataset --include "all/*" --local-dir data/mmlu
hf download basicv8vc/SimpleQA --repo-type dataset --local-dir data/simpleqa

curl -L -o data/biored/BIORED.zip \
  https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip
curl -L -o data/biored/BioRED_Annotation_Guideline.pdf \
  https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BioRED_Annotation_Guideline.pdf
curl -L -o data/biored/README.txt \
  https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/README.txt
unzip -o data/biored/BIORED.zip -d data/biored/extracted
```

FrontierScience is downloaded automatically into `.cache/frontierscience` by
`scripts/evaluate_frontierscience.py`. To pre-seed that cache:

```shell
hf download openai/frontierscience \
  --repo-type dataset \
  --local-dir .cache/frontierscience
```

Download prior Fossils-M result JSON into the default run output layout when
you want `--resume` to pick up work from another box:

```shell
hf download akhilpandey95/scienceeval-fossil-results \
  --repo-type dataset \
  --include "fossil-m/*" \
  --local-dir results
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

For Qwen3.5 and Gemma-4 merged-LoRA conda workflows, see
`experiments/README.md`.

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

## Generate the model fossil sheets

The generator reads model result folders from `results/fossil-m/` by default.
That directory can include both downloaded public bundles and local experiment
runs.

```shell
python scripts/fossil-m/generate_model_fossil_sheet.py
```

To regenerate from a downloaded Hugging Face cache instead, pass the cache root
explicitly:

```shell
hf download akhilpandey95/scienceeval-fossil-results \
  --repo-type dataset \
  --local-dir data/fossil-results

python scripts/fossil-m/generate_model_fossil_sheet.py \
  --results-dir data/fossil-results/fossil-m
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
- `MMLU` uses the all-subject test split with deterministic multiple-choice
  letter scoring.
- `SimpleQA` uses normalized exact-answer scoring over `simple_qa_test_set.csv`.
- `SciRIFF` uses exact-output scoring over the 8192-token test split.
- `scripts/fossil-m/generate_model_fossil_sheet.py` turns the uploaded result
  bundle into `data/fossils-m-catalog.json` and the Fossils-M plot images
  consumed by `fossils.html`.
