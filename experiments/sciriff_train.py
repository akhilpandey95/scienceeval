#!/usr/bin/env python3

# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/scienceeval/blob/main/LICENSE

from __future__ import annotations

# stdlib
import argparse
import inspect
import json
import logging
import os
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


# set env
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


# data
from datasets import Dataset, load_dataset

# training
import torch
import wandb
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer


# init logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# directory constants
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "output" / "sciriff_sft"


# dataset constants
DEFAULT_DATASET_NAME = "allenai/SciRIFF-train-mix"
DEFAULT_SPLIT = "train"
DEFAULT_MESSAGES_COLUMN = "messages"
DEFAULT_INPUT_COLUMN = "input"
DEFAULT_OUTPUT_COLUMN = "output"
DEFAULT_ID_COLUMN = "_instance_id"
DEFAULT_DATASET_COLUMN = "dataset"


# experiment constants
DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-2B"
DEFAULT_EXPERIMENT_NAME = "qwen35_2b_sciriff_sft"
DEFAULT_WANDB_PROJECT = "scienceeval-sciriff"
DEFAULT_SEED = 2025
DEFAULT_MAX_LENGTH = 4096


# schema class for training config
@dataclass(frozen=True)
class TrainConfig:
    """
    Store configuration for one SciRIFF SFT run.

    Parameters
    ------------
    experiment_name: str
        Stable experiment label used in logs and manifests
    model_name: str
        Hugging Face model id or local model path
    output_dir: Path
        Directory where run artifacts are written

    Returns
    ------------
    TrainConfig
    """

    experiment_name: str
    model_name: str
    output_dir: Path
    dataset_name: str
    data_files: list[Path]
    eval_data_files: list[Path]
    split: str
    eval_split: str
    messages_column: str
    input_column: str
    output_column: str
    id_column: str
    dataset_column: str
    max_length: int
    max_train_examples: int | None
    max_eval_examples: int | None
    eval_ratio: float
    seed: int
    system_prompt: str | None
    trust_remote_code: bool
    attn_implementation: str
    torch_dtype: Literal["auto", "bfloat16", "float16", "float32"]
    device_map: str
    gradient_checkpointing: bool
    load_in_4bit: bool
    load_in_8bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_compute_dtype: Literal["bfloat16", "float16", "float32"]
    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    weight_decay: float
    logging_steps: int
    save_total_limit: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: str
    wandb_enabled: bool
    wandb_project: str
    wandb_run_name: str | None
    dry_run: bool


# schema class for prepared split summary
@dataclass(frozen=True)
class DatasetSplitSummary:
    """
    Store the deterministic SciRIFF train/eval split summary.

    Parameters
    ------------
    train_examples: int
        Number of training examples after filtering
    eval_examples: int
        Number of eval examples after filtering

    Returns
    ------------
    DatasetSplitSummary
    """

    dataset_name: str
    split: str
    source_rows: int
    formatted_rows: int
    filtered_rows: int
    dropped_overlength_rows: int
    train_examples: int
    eval_examples: int
    train_source_counts: dict[str, int]
    eval_source_counts: dict[str, int]
    train_token_max: int
    eval_token_max: int


# helper function to build the CLI parser
def build_argument_parser() -> argparse.ArgumentParser:
    """
    Create the CLI parser for SciRIFF SFT runs.

    Parameters
    ------------
    None

    Returns
    ------------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on SciRIFF instruction-following data.")

    # Experiment and data.
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--data-files", nargs="*", type=Path, default=[])
    parser.add_argument("--eval-data-files", nargs="*", type=Path, default=[])
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--eval-split", default=DEFAULT_SPLIT)
    parser.add_argument("--messages-column", default=DEFAULT_MESSAGES_COLUMN)
    parser.add_argument("--input-column", default=DEFAULT_INPUT_COLUMN)
    parser.add_argument("--output-column", default=DEFAULT_OUTPUT_COLUMN)
    parser.add_argument("--id-column", default=DEFAULT_ID_COLUMN)
    parser.add_argument("--dataset-column", default=DEFAULT_DATASET_COLUMN)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument("--system-prompt")

    # Model and runtime.
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--torch-dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device-map", default="auto", help="Use 'none' to let Trainer place the full model.")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Quantization.
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )

    # Optimizer schedule and batching.
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)

    # LoRA adapter.
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="all-linear")

    # Tracking.
    parser.add_argument(
        "--wandb",
        dest="wandb_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--dry-run", action="store_true")
    return parser


# helper function to parse the training config
def parse_config() -> TrainConfig:
    """
    Parse CLI arguments into a training config.

    Parameters
    ------------
    None

    Returns
    ------------
    TrainConfig
    """
    parser = build_argument_parser()
    namespace = parser.parse_args()
    config = TrainConfig(**vars(namespace))
    validate_config(config)
    return config


# helper function to validate the training config
def validate_config(config: TrainConfig) -> None:
    """
    Validate configuration values before expensive setup.

    Parameters
    ------------
    config: TrainConfig
        Parsed training configuration

    Returns
    ------------
    None
    """
    if config.max_length < 512:
        raise SystemExit("--max-length must be >= 512.")
    if config.eval_ratio < 0.0 or config.eval_ratio >= 1.0:
        raise SystemExit("--eval-ratio must be in [0.0, 1.0).")
    if config.max_train_examples is not None and config.max_train_examples <= 0:
        raise SystemExit("--max-train-examples must be > 0 when provided.")
    if config.max_eval_examples is not None and config.max_eval_examples <= 0:
        raise SystemExit("--max-eval-examples must be > 0 when provided.")
    if config.learning_rate <= 0:
        raise SystemExit("--learning-rate must be > 0.")
    if config.num_train_epochs <= 0:
        raise SystemExit("--num-train-epochs must be > 0.")
    if config.per_device_train_batch_size <= 0:
        raise SystemExit("--per-device-train-batch-size must be > 0.")
    if config.per_device_eval_batch_size <= 0:
        raise SystemExit("--per-device-eval-batch-size must be > 0.")
    if config.gradient_accumulation_steps <= 0:
        raise SystemExit("--gradient-accumulation-steps must be > 0.")
    if config.warmup_ratio < 0.0 or config.warmup_ratio > 1.0:
        raise SystemExit("--warmup-ratio must be in [0.0, 1.0].")
    if config.lora_r <= 0 or config.lora_alpha <= 0:
        raise SystemExit("--lora-r and --lora-alpha must be > 0.")
    if config.lora_dropout < 0.0 or config.lora_dropout >= 1.0:
        raise SystemExit("--lora-dropout must be in [0.0, 1.0).")
    if config.load_in_4bit and config.load_in_8bit:
        raise SystemExit("Choose at most one of --load-in-4bit and --load-in-8bit.")


# helper function to detect the main process
def is_main_process(local_rank: int) -> bool:
    """
    Return whether the current distributed worker is the main process.

    Parameters
    ------------
    local_rank: int
        Distributed local rank

    Returns
    ------------
    bool
    """
    return local_rank in (-1, 0)


# helper function to detect the primary process
def is_primary_process() -> bool:
    """
    Return whether this worker should write process-global artifacts.

    Parameters
    ------------
    None

    Returns
    ------------
    bool
    """
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    return rank in (-1, 0)


# helper function to resolve distributed world size
def get_world_size() -> int:
    """
    Return the number of distributed workers in this run.

    Parameters
    ------------
    None

    Returns
    ------------
    int
    """
    return max(1, int(os.environ.get("WORLD_SIZE", 1)))


# helper function to initialize wandb
def setup_wandb(config: TrainConfig) -> bool:
    """
    Initialize Weights & Biases only on the main process.

    Parameters
    ------------
    config: TrainConfig
        Parsed training configuration

    Returns
    ------------
    bool
    """
    if not config.wandb_enabled or config.dry_run:
        logger.info("Weights & Biases logging disabled.")
        return False

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not is_main_process(local_rank):
        os.environ["WANDB_MODE"] = "disabled"
        logging.getLogger().setLevel(logging.WARNING)
        return False

    run_name = resolve_wandb_run_name(config)
    wandb.init(project=config.wandb_project, name=run_name)
    logger.info("Weights & Biases initialized for %s/%s.", config.wandb_project, run_name)
    return True


# helper function to resolve the wandb run name
def resolve_wandb_run_name(config: TrainConfig) -> str:
    """
    Return a stable W&B run name.

    Parameters
    ------------
    config: TrainConfig
        Parsed training configuration

    Returns
    ------------
    str
    """
    return config.wandb_run_name or config.experiment_name


# helper function to log run config
def log_run_config(config: TrainConfig) -> None:
    """
    Log high-signal settings for repeatable SciRIFF runs.

    Parameters
    ------------
    config: TrainConfig
        Parsed training configuration

    Returns
    ------------
    None
    """
    logger.info(
        "Experiment=%s output_dir=%s dry_run=%s",
        config.experiment_name,
        config.output_dir,
        config.dry_run,
    )
    logger.info(
        "Dataset=%s split=%s data_files=%s eval_data_files=%s",
        config.dataset_name,
        config.split,
        [str(path) for path in config.data_files],
        [str(path) for path in config.eval_data_files],
    )
    logger.info(
        "Model=%s max_length=%s dtype=%s attn=%s device_map=%s",
        config.model_name,
        config.max_length,
        config.torch_dtype,
        config.attn_implementation,
        config.device_map,
    )
    logger.info(
        "Schedule lr=%s epochs=%s warmup_ratio=%.3f batch=%s grad_accum=%s",
        config.learning_rate,
        config.num_train_epochs,
        config.warmup_ratio,
        config.per_device_train_batch_size,
        config.gradient_accumulation_steps,
    )
    logger.info(
        "LoRA r=%s alpha=%s dropout=%.3f targets=%s",
        config.lora_r,
        config.lora_alpha,
        config.lora_dropout,
        config.lora_target_modules,
    )


# helper function to serialize stable JSON
def stable_json(value: Any) -> str:
    """
    Serialize a value with stable formatting for manifests.

    Parameters
    ------------
    value: Any
        JSON-serializable value

    Returns
    ------------
    str
    """
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)


# helper function to serialize config for manifests and tracking
def config_to_dict(config: TrainConfig) -> dict:
    """
    Convert a training config into JSON-serializable values.

    Parameters
    ------------
    config: TrainConfig
        Parsed training configuration

    Returns
    ------------
    dict
    """
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["data_files"] = [str(path) for path in config.data_files]
    payload["eval_data_files"] = [str(path) for path in config.eval_data_files]
    return payload


# helper function to load one SciRIFF dataset split
def load_dataset_split(
    config: TrainConfig,
    *,
    data_files: list[Path],
    split: str,
    label: str,
) -> Dataset:
    """
    Load one SciRIFF split from Hugging Face or local parquet files.

    Parameters
    ------------
    config: TrainConfig
        Parsed training configuration
    data_files: list[Path]
        Optional local parquet files
    split: str
        Dataset split name
    label: str
        Log label for this split

    Returns
    ------------
    Dataset
    """
    load_start = time.perf_counter()
    if data_files:
        dataset = load_dataset("parquet", data_files=[str(path) for path in data_files], split=split)
    else:
        dataset = load_dataset(config.dataset_name, split=split)

    logger.info(
        "Loaded %s %s raw row(s) in %.2f seconds.",
        label,
        len(dataset),
        time.perf_counter() - load_start,
    )
    logger.info("%s raw columns: %s", label, ", ".join(dataset.column_names))
    return dataset


# helper function to load the raw SciRIFF training dataset
def load_raw_dataset(config: TrainConfig) -> Dataset:
    """
    Load raw SciRIFF training data.

    Parameters
    ------------
    config: TrainConfig
        Parsed training configuration

    Returns
    ------------
    Dataset
    """
    return load_dataset_split(
        config,
        data_files=config.data_files,
        split=config.split,
        label="train",
    )


# helper function to load the raw SciRIFF eval dataset
def load_raw_eval_dataset(config: TrainConfig) -> Dataset | None:
    """
    Load raw SciRIFF eval data when an explicit eval source is configured.

    Parameters
    ------------
    config: TrainConfig
        Parsed training configuration

    Returns
    ------------
    Dataset | None
    """
    if not config.eval_data_files:
        return None
    return load_dataset_split(
        config,
        data_files=config.eval_data_files,
        split=config.eval_split,
        label="eval",
    )


# helper function to normalize message content
def normalize_content(value: Any) -> str:
    """
    Normalize message content into a string.

    Parameters
    ------------
    value: Any
        Raw message content

    Returns
    ------------
    str
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(normalize_content(item) for item in value).strip()
    if isinstance(value, dict):
        return stable_json(value)
    return str(value).strip()


# helper function to normalize one chat role
def normalize_role(value: Any) -> str:
    """
    Normalize message role names into chat-template roles.

    Parameters
    ------------
    value: Any
        Raw role value

    Returns
    ------------
    str
    """
    role = str(value).strip().lower()
    role_map = {
        "human": "user",
        "instruction": "user",
        "model": "assistant",
        "gpt": "assistant",
        "bot": "assistant",
    }
    return role_map.get(role, role)


# helper function to normalize messages
def normalize_messages(raw_messages: Any) -> list[dict[str, str]]:
    """
    Normalize raw dataset messages into role/content dicts.

    Parameters
    ------------
    raw_messages: Any
        Raw messages column value

    Returns
    ------------
    list[dict[str, str]]
    """
    if isinstance(raw_messages, dict) and {"role", "content"}.issubset(raw_messages):
        roles = raw_messages["role"]
        contents = raw_messages["content"]
        if isinstance(roles, list) and isinstance(contents, list):
            raw_messages = [{"role": role, "content": content} for role, content in zip(roles, contents, strict=False)]
        else:
            raw_messages = [raw_messages]

    if not isinstance(raw_messages, list):
        raise ValueError("messages column must contain a list of message objects.")

    messages = []
    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            raise ValueError("each message must be a mapping.")
        role = normalize_role(raw_message.get("role"))
        content = normalize_content(raw_message.get("content"))
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported chat role: {role!r}")
        if not content:
            continue
        messages.append({"role": role, "content": content})

    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("training messages must end with an assistant response.")
    return messages


# helper function to build messages from input/output columns
def build_input_output_messages(example: dict, config: TrainConfig) -> list[dict[str, str]]:
    """
    Build a two-turn chat example from SciRIFF input/output fields.

    Parameters
    ------------
    example: dict
        Raw dataset row
    config: TrainConfig
        Parsed training configuration

    Returns
    ------------
    list[dict[str, str]]
    """
    user_text = normalize_content(example.get(config.input_column))
    assistant_text = normalize_content(example.get(config.output_column))
    if not user_text or not assistant_text:
        raise ValueError("input/output examples require non-empty prompt and response.")

    messages = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt.strip()})
    messages.extend(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    )
    return messages


# helper function to compute token count
def count_chat_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    """
    Count tokens after applying the model chat template.

    Parameters
    ------------
    tokenizer: Any
        Loaded tokenizer
    messages: list[dict[str, str]]
        Training messages

    Returns
    ------------
    int
    """
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )
    except Exception:
        rendered = "\n".join(f"{message['role']}: {message['content']}" for message in messages)

    tokenized = tokenizer(
        rendered,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )
    return token_sequence_length(tokenized)


# helper function to compute one tokenized sequence length
def token_sequence_length(tokenized: Any) -> int:
    """
    Return the sequence length from tokenizer output variants.

    Parameters
    ------------
    tokenized: Any
        Token ids, tensor, batch encoding, or mapping with input_ids

    Returns
    ------------
    int
    """
    if hasattr(tokenized, "keys") and "input_ids" in tokenized:
        return token_sequence_length(tokenized["input_ids"])

    if isinstance(tokenized, Mapping):
        if "input_ids" in tokenized:
            return token_sequence_length(tokenized["input_ids"])
        if not tokenized:
            return 0
        return token_sequence_length(next(iter(tokenized.values())))

    if hasattr(tokenized, "shape"):
        shape = list(tokenized.shape)
        if not shape:
            return 1
        return int(shape[-1])

    if isinstance(tokenized, (list, tuple)):
        if not tokenized:
            return 0
        first = tokenized[0]
        if isinstance(first, int):
            return len(tokenized)
        if isinstance(first, (list, tuple, dict)) or hasattr(first, "shape"):
            return token_sequence_length(first)
        return len(tokenized)

    return len(tokenized)


# helper function to format one raw row
def format_training_row(example: dict, config: TrainConfig, tokenizer: Any) -> dict:
    """
    Format one raw SciRIFF row for conversational SFT.

    Parameters
    ------------
    example: dict
        Raw dataset row
    config: TrainConfig
        Parsed training configuration
    tokenizer: Any
        Loaded tokenizer

    Returns
    ------------
    dict
    """
    if config.messages_column in example and example.get(config.messages_column):
        messages = normalize_messages(example[config.messages_column])
    else:
        messages = build_input_output_messages(example, config)

    if config.system_prompt and messages[0]["role"] != "system":
        messages = [{"role": "system", "content": config.system_prompt.strip()}, *messages]

    source_id = normalize_content(example.get(config.id_column)) or "unknown"
    source_dataset = normalize_content(example.get(config.dataset_column)) or "unknown"
    token_count = count_chat_tokens(tokenizer, messages)
    return {
        "messages": messages,
        "source_id": source_id,
        "source_dataset": source_dataset,
        "token_count": token_count,
    }


# helper function to ensure tokenizer chat template
def ensure_chat_template(tokenizer: Any) -> None:
    """
    Ensure a tokenizer has a basic chat template for SFT formatting.

    Parameters
    ------------
    tokenizer: Any
        Loaded tokenizer

    Returns
    ------------
    None
    """
    if tokenizer.chat_template:
        return

    logger.warning("Tokenizer has no chat_template; installing a minimal role/content template.")
    tokenizer.chat_template = """
{% for message in messages %}
{% if message['role'] == 'system' %}
{{ '<|system|>\n' + message['content'].strip() + '\n' }}
{% elif message['role'] == 'user' %}
{{ '<|user|>\n' + message['content'].strip() + '\n' }}
{% elif message['role'] == 'assistant' %}
{{ '<|assistant|>\n' + message['content'].strip() + eos_token + '\n' }}
{% endif %}
{% endfor %}
""".strip()


# helper function to create the tokenizer
def create_tokenizer(config: TrainConfig) -> Any:
    """
    Load the tokenizer and ensure padding works for SFT batches.

    Parameters
    ------------
    config: TrainConfig
        Parsed training configuration

    Returns
    ------------
    Any
    """
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    tokenizer.model_max_length = config.max_length
    ensure_chat_template(tokenizer)
    logger.info(
        "Tokenizer loaded in %.2f seconds (eos=%s pad=%s padding_side=%s).",
        time.perf_counter() - load_start,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.padding_side,
    )
    return tokenizer


# helper function to select at most N rows
def select_limit(dataset: Dataset, limit: int | None) -> Dataset:
    """
    Select a deterministic prefix when a row limit is provided.

    Parameters
    ------------
    dataset: Dataset
        Dataset to subset
    limit: int | None
        Maximum rows to keep

    Returns
    ------------
    Dataset
    """
    if limit is None or len(dataset) <= limit:
        return dataset
    return dataset.select(range(limit))


# helper function to summarize one source column
def source_counts(dataset: Dataset) -> dict[str, int]:
    """
    Count examples by source dataset label.

    Parameters
    ------------
    dataset: Dataset
        Prepared dataset with source_dataset values

    Returns
    ------------
    dict[str, int]
    """
    if len(dataset) == 0 or "source_dataset" not in dataset.column_names:
        return {}
    counts = Counter(str(value) for value in dataset["source_dataset"])
    return dict(sorted(counts.items()))


# helper function to return the maximum token count
def max_token_count(dataset: Dataset) -> int:
    """
    Return the maximum token count for a split.

    Parameters
    ------------
    dataset: Dataset
        Prepared dataset with token_count values

    Returns
    ------------
    int
    """
    if len(dataset) == 0 or "token_count" not in dataset.column_names:
        return 0
    return int(max(dataset["token_count"]))


# helper function to prepare train/eval datasets
def prepare_datasets(config: TrainConfig, tokenizer: Any) -> tuple[Dataset, Dataset | None, DatasetSplitSummary]:
    """
    Load, format, filter, shuffle, and split SciRIFF training examples.

    Parameters
    ------------
    config: TrainConfig
        Parsed training configuration
    tokenizer: Any
        Loaded tokenizer

    Returns
    ------------
    tuple[Dataset, Dataset | None, DatasetSplitSummary]
    """
    raw_dataset = load_raw_dataset(config)
    raw_eval_dataset = load_raw_eval_dataset(config)
    format_start = time.perf_counter()
    dataset_workers = max(1, (os.cpu_count() or 1) // get_world_size())
    logger.info("Using %s dataset worker process(es).", dataset_workers)
    formatted = raw_dataset.map(
        lambda example: format_training_row(example, config, tokenizer),
        remove_columns=raw_dataset.column_names,
        desc="Formatting SciRIFF rows",
        num_proc=dataset_workers,
    )
    filtered = formatted.filter(
        lambda example: example["token_count"] <= config.max_length,
        desc="Filtering overlength rows",
        num_proc=dataset_workers,
    )
    formatted_eval = None
    filtered_eval = None
    if raw_eval_dataset is not None:
        formatted_eval = raw_eval_dataset.map(
            lambda example: format_training_row(example, config, tokenizer),
            remove_columns=raw_eval_dataset.column_names,
            desc="Formatting SciRIFF eval rows",
            num_proc=dataset_workers,
        )
        filtered_eval = formatted_eval.filter(
            lambda example: example["token_count"] <= config.max_length,
            desc="Filtering overlength eval rows",
            num_proc=dataset_workers,
        )
    logger.info(
        "Formatted and filtered data in %.2f seconds: %s -> %s row(s).",
        time.perf_counter() - format_start,
        len(formatted),
        len(filtered),
    )

    shuffled = filtered.shuffle(seed=config.seed)
    if filtered_eval is not None:
        train_dataset = shuffled
        eval_dataset = filtered_eval.shuffle(seed=config.seed)
    elif config.eval_ratio > 0.0 and len(shuffled) >= 2:
        split = shuffled.train_test_split(test_size=config.eval_ratio, seed=config.seed, shuffle=False)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = shuffled
        eval_dataset = None

    train_dataset = select_limit(train_dataset, config.max_train_examples)
    if eval_dataset is not None:
        eval_dataset = select_limit(eval_dataset, config.max_eval_examples)

    split_summary = DatasetSplitSummary(
        dataset_name=config.dataset_name,
        split=config.split,
        source_rows=len(raw_dataset) + (len(raw_eval_dataset) if raw_eval_dataset is not None else 0),
        formatted_rows=len(formatted) + (len(formatted_eval) if formatted_eval is not None else 0),
        filtered_rows=len(filtered) + (len(filtered_eval) if filtered_eval is not None else 0),
        dropped_overlength_rows=(len(formatted) - len(filtered))
        + (
            (len(formatted_eval) - len(filtered_eval))
            if formatted_eval is not None and filtered_eval is not None
            else 0
        ),
        train_examples=len(train_dataset),
        eval_examples=len(eval_dataset) if eval_dataset is not None else 0,
        train_source_counts=source_counts(train_dataset),
        eval_source_counts=source_counts(eval_dataset) if eval_dataset is not None else {},
        train_token_max=max_token_count(train_dataset),
        eval_token_max=max_token_count(eval_dataset) if eval_dataset is not None else 0,
    )
    log_dataset_preview(train_dataset, eval_dataset)
    return strip_training_metadata(train_dataset), strip_training_metadata(eval_dataset), split_summary


# helper function to remove metadata before trainer ingestion
def strip_training_metadata(dataset: Dataset | None) -> Dataset | None:
    """
    Keep only the messages column consumed by SFTTrainer.

    Parameters
    ------------
    dataset: Dataset | None
        Prepared dataset

    Returns
    ------------
    Dataset | None
    """
    if dataset is None:
        return None
    removable = [column for column in ("source_id", "source_dataset", "token_count") if column in dataset.column_names]
    if not removable:
        return dataset
    return dataset.remove_columns(removable)


# helper function to log dataset previews
def log_dataset_preview(train_dataset: Dataset, eval_dataset: Dataset | None) -> None:
    """
    Log a compact preview of prepared SciRIFF examples.

    Parameters
    ------------
    train_dataset: Dataset
        Prepared training split
    eval_dataset: Dataset | None
        Optional eval split

    Returns
    ------------
    None
    """
    if len(train_dataset):
        first = train_dataset[0]
        logger.info(
            "First train example source=%s id=%s tokens=%s roles=%s",
            first.get("source_dataset"),
            first.get("source_id"),
            first.get("token_count"),
            [message["role"] for message in first["messages"]],
        )
    if eval_dataset is not None and len(eval_dataset):
        first = eval_dataset[0]
        logger.info(
            "First eval example source=%s id=%s tokens=%s roles=%s",
            first.get("source_dataset"),
            first.get("source_id"),
            first.get("token_count"),
            [message["role"] for message in first["messages"]],
        )


# helper function to resolve torch dtype
def resolve_torch_dtype(dtype_name: str) -> torch.dtype | str:
    """
    Map a CLI dtype string to a torch dtype object.

    Parameters
    ------------
    dtype_name: str
        CLI dtype name

    Returns
    ------------
    torch.dtype | str
    """
    dtype_map = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_name]


# helper function to build quantization config
def build_quantization_config(config: TrainConfig) -> BitsAndBytesConfig | None:
    """
    Build optional bitsandbytes quantization settings.

    Parameters
    ------------
    config: TrainConfig
        Parsed training configuration

    Returns
    ------------
    BitsAndBytesConfig | None
    """
    if not config.load_in_4bit and not config.load_in_8bit:
        return None
    if config.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=resolve_torch_dtype(config.bnb_4bit_compute_dtype),
    )


# helper function to create the model
def create_model(config: TrainConfig) -> Any:
    """
    Load the base causal model for SciRIFF SFT.

    Parameters
    ------------
    config: TrainConfig
        Parsed training configuration

    Returns
    ------------
    Any
    """
    load_start = time.perf_counter()
    model_kwargs = {
        "attn_implementation": config.attn_implementation,
        "torch_dtype": resolve_torch_dtype(config.torch_dtype),
        "trust_remote_code": config.trust_remote_code,
    }
    if config.device_map != "none":
        model_kwargs["device_map"] = config.device_map
    quantization_config = build_quantization_config(config)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    logger.info(
        "Loading model %s with dtype=%s attn=%s quantized=%s.",
        config.model_name,
        config.torch_dtype,
        config.attn_implementation,
        quantization_config is not None,
    )
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.config.use_cache = not config.gradient_checkpointing
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
    logger.info("Model loaded in %.2f seconds.", time.perf_counter() - load_start)
    return model


# helper function to build SFTConfig
def build_sft_config(training_kwargs: dict[str, Any]) -> SFTConfig:
    """
    Construct SFTConfig with small compatibility shims for TRL versions.

    Parameters
    ------------
    training_kwargs: dict[str, Any]
        Candidate SFTConfig keyword arguments

    Returns
    ------------
    SFTConfig
    """
    supported = set(inspect.signature(SFTConfig.__init__).parameters)
    config_kwargs = dict(training_kwargs)

    if "eval_strategy" not in supported and "evaluation_strategy" in supported:
        config_kwargs["evaluation_strategy"] = config_kwargs.pop("eval_strategy")
    if "max_length" not in supported and "max_seq_length" in supported:
        config_kwargs["max_seq_length"] = config_kwargs.pop("max_length")

    filtered_kwargs = {key: value for key, value in config_kwargs.items() if key in supported}
    skipped = sorted(set(config_kwargs) - set(filtered_kwargs))
    if skipped:
        logger.info("Skipping unsupported SFTConfig keys: %s", ", ".join(skipped))

    return SFTConfig(**filtered_kwargs)


# helper function to build the LoRA config
def build_lora_config(config: TrainConfig) -> LoraConfig:
    """
    Build a LoRA adapter configuration.

    Parameters
    ------------
    config: TrainConfig
        Parsed training configuration

    Returns
    ------------
    LoraConfig
    """
    target_modules: str | list[str]
    if config.lora_target_modules == "all-linear":
        target_modules = "all-linear"
    else:
        target_modules = [item.strip() for item in config.lora_target_modules.split(",") if item.strip()]

    return LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
    )


# helper function to build SFTTrainer
def build_sft_trainer(
    model: Any,
    tokenizer: Any,
    training_args: SFTConfig,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    peft_config: LoraConfig,
) -> SFTTrainer:
    """
    Construct SFTTrainer while handling tokenizer argument naming changes.

    Parameters
    ------------
    model: Any
        Loaded causal language model
    tokenizer: Any
        Loaded tokenizer
    training_args: SFTConfig
        SFT training arguments
    train_dataset: Dataset
        Trainer training dataset
    eval_dataset: Dataset | None
        Optional trainer eval dataset
    peft_config: LoraConfig
        LoRA adapter configuration

    Returns
    ------------
    SFTTrainer
    """
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": peft_config,
    }
    supported = set(inspect.signature(SFTTrainer.__init__).parameters)
    if "processing_class" in supported:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in supported:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        raise RuntimeError("This TRL build supports neither processing_class nor tokenizer.")

    return SFTTrainer(**trainer_kwargs)


# helper function to log trainable parameters
def log_trainable_parameters(model: Any) -> None:
    """
    Log trainable and total parameter counts after adapters are attached.

    Parameters
    ------------
    model: Any
        Trainer model with adapters attached

    Returns
    ------------
    None
    """
    total_parameters = 0
    trainable_parameters = 0
    for parameter in model.parameters():
        total_parameters += parameter.numel()
        if parameter.requires_grad:
            trainable_parameters += parameter.numel()

    trainable_fraction = trainable_parameters / total_parameters if total_parameters else 0.0
    logger.info(
        "Trainable parameters: %s/%s (%.4f%%)",
        trainable_parameters,
        total_parameters,
        100.0 * trainable_fraction,
    )


# helper function to create the trainer
def create_trainer(
    config: TrainConfig,
    tokenizer: Any,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    report_to: list[str],
    run_name: str | None,
) -> SFTTrainer:
    """
    Construct the TRL SFT trainer for SciRIFF supervision.

    Parameters
    ------------
    config: TrainConfig
        Parsed training configuration
    tokenizer: Any
        Loaded tokenizer
    train_dataset: Dataset
        Trainer training dataset
    eval_dataset: Dataset | None
        Optional trainer eval dataset
    report_to: list[str]
        Trainer reporting targets
    run_name: str | None
        Optional run name

    Returns
    ------------
    SFTTrainer
    """
    training_kwargs = {
        "output_dir": str(config.output_dir),
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "logging_steps": config.logging_steps,
        "save_strategy": "epoch",
        "eval_strategy": "epoch" if eval_dataset is not None else "no",
        "save_total_limit": config.save_total_limit,
        "seed": config.seed,
        "bf16": config.torch_dtype == "bfloat16",
        "fp16": config.torch_dtype == "float16",
        "gradient_checkpointing": config.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "max_length": config.max_length,
        "packing": False,
        "completion_only_loss": True,
        "report_to": report_to,
        "remove_unused_columns": False,
        "save_safetensors": True,
    }
    if run_name is not None:
        training_kwargs["run_name"] = run_name

    training_args = build_sft_config(training_kwargs)
    return build_sft_trainer(
        model=create_model(config),
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=build_lora_config(config),
    )


# helper function to write the dataset manifest
def write_manifest(config: TrainConfig, split_summary: DatasetSplitSummary) -> None:
    """
    Write a stable manifest describing the training split and configuration.

    Parameters
    ------------
    config: TrainConfig
        Parsed training configuration
    split_summary: DatasetSplitSummary
        Deterministic train/eval split summary

    Returns
    ------------
    None
    """
    if not is_primary_process():
        return
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": config_to_dict(config),
        "split": asdict(split_summary),
    }
    manifest_path = config.output_dir / "dataset_manifest.json"
    manifest_path.write_text(stable_json(manifest) + "\n", encoding="utf-8")
    logger.info("Wrote dataset manifest to %s.", manifest_path)


# orchestration function
def run_sciriff_training(config: TrainConfig) -> None:
    """
    Run the full SciRIFF SFT preparation, training, and save flow.

    Parameters
    ------------
    config: TrainConfig
        Parsed training configuration

    Returns
    ------------
    None
    """
    run_start = time.perf_counter()
    log_run_config(config)
    wandb_enabled = setup_wandb(config)

    if torch.cuda.is_available():
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
        logger.info("CUDA bf16 supported: %s", torch.cuda.is_bf16_supported())
    elif torch.backends.mps.is_available():
        logger.info("Using MPS device.")
    else:
        logger.info("Using CPU.")

    set_seed(config.seed)
    tokenizer = create_tokenizer(config)
    train_dataset, eval_dataset, split_summary = prepare_datasets(config, tokenizer)
    write_manifest(config, split_summary)

    logger.info(
        "Prepared %s train example(s) and %s eval example(s).",
        split_summary.train_examples,
        split_summary.eval_examples,
    )
    if wandb_enabled and wandb.run is not None:
        wandb.config.update(config_to_dict(config), allow_val_change=True)
        wandb.config.update(asdict(split_summary), allow_val_change=True)

    if config.dry_run:
        logger.info("Dry run complete; skipping model load and training.")
        if wandb_enabled and wandb.run is not None:
            wandb.finish()
        return

    try:
        trainer_start = time.perf_counter()
        trainer = create_trainer(
            config=config,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            report_to=["wandb"] if wandb_enabled else [],
            run_name=resolve_wandb_run_name(config) if wandb_enabled else None,
        )
        logger.info("SFTTrainer initialized in %.2f seconds.", time.perf_counter() - trainer_start)
        log_trainable_parameters(trainer.model)

        train_start = time.perf_counter()
        train_result = trainer.train()
        logger.info("Training completed in %.2f seconds.", time.perf_counter() - train_start)
        logger.info("Training metrics: %s", train_result.metrics)

        final_dir = config.output_dir / "final_model"
        save_start = time.perf_counter()
        trainer.save_model(str(final_dir))
        if is_primary_process():
            tokenizer.save_pretrained(str(final_dir))
            logger.info(
                "Saved adapter and tokenizer to %s in %.2f seconds.",
                final_dir,
                time.perf_counter() - save_start,
            )
    finally:
        if wandb_enabled and wandb.run is not None:
            wandb.finish()
        logger.info("Run finished in %.2f seconds.", time.perf_counter() - run_start)


if __name__ == "__main__":
    run_sciriff_training(parse_config())
