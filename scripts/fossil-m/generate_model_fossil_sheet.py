#!/usr/bin/env python3

# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/scienceeval/blob/main/LICENSE

from __future__ import annotations

# stdlib
import argparse
import json
import re
import unicodedata
from pathlib import Path

# data
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

# local
from fossil_m_common import REPO_ROOT, save_json


# directory constants
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "fossil-m"
DATA_OUTPUT_PATH = REPO_ROOT / "data" / "fossils-m-catalog.json"
IMAGE_OUTPUT_DIR = REPO_ROOT / "images"


# source constants
FOSSIL_RESULTS_ROOT_URL = "https://huggingface.co/datasets/akhilpandey95/scienceeval-fossil-results/tree/main/fossil-m"
GPT4O_FRONTIERSCIENCE_URL = "https://cdn.openai.com/pdf/2fcd284c-b468-4c21-8ee0-7a783933efcc/frontierscience-paper.pdf"
GPT4O_OPENAI_45_URL = "https://openai.com/index/introducing-gpt-4-5/"
GPT4O_OPENAI_41_URL = "https://openai.com/index/gpt-4-1/"
GPT4O_SIMPLEQA_URL = "https://openai.com/index/introducing-simpleqa/"
GPT4O_MEDINST_URL = "https://aclanthology.org/2024.findings-emnlp.482.pdf"
GPT4O_GRAPHJUDGE_URL = "https://aclanthology.org/2025.emnlp-main.554.pdf"
GPT4O_SCIRIFF_URL = "https://aclanthology.org/2025.emnlp-main.310.pdf"


# gpt-oss output constants
FINAL_CHANNEL_MARKER = "<|channel|>final<|message|>"
END_MARKER = "<|end|>"
HYPHEN_TRANSLATION = str.maketrans({char: "-" for char in "‐‑‒–—−"})


# plot constants
PAPER = "#f2ede4"
INK = "#171310"
MUTED = "#6f655d"
ACCENT = "#6f3f2b"
STEM = "#c8b59f"
RING = "#d9c2a7"
LAUNCH = "#b54b4b"


# model constants
MODEL_INTRO_YEARS = {
    "gpt-4o": 2024 + (4 + 13 / 31) / 12,
    "gpt-oss-20b": 2025 + (7 + 5 / 31) / 12,
    "qwen3-5-2b-base": 2026 + (1 + 28 / 28) / 12,
    "qwen3-5-2b-sciriff4096": 2026 + (1 + 28 / 28) / 12,
    "gemma-4-31b-it": 2026 + (3 + 2 / 30) / 12,
    "qwen3-6-27b": 2026 + (3 + 21 / 30) / 12,
}

MODEL_ORDER = (
    "gpt-4o",
    "qwen3-6-27b",
    "qwen3-5-2b-base",
    "qwen3-5-2b-sciriff4096",
    "gpt-oss-20b",
    "gemma-4-31b-it",
)

MODEL_RESULT_DIR_ALIASES = {
    "qwen3-5-2b-sciriff4096": (
        "qwen35-2b-sciriff4096",
        "qwen3-5-2b-sciriff4096-merged",
    ),
}

OPEN_MODEL_CONFIGS = {
    "gpt-oss-20b": {
        "label": "gpt-oss-20b",
        "model_type": "Open-weight model",
    },
    "qwen3-5-2b-base": {
        "label": "Qwen3.5-2B base",
        "model_type": "Open-weight small model before SciRIFF SFT",
        "reading_rule": "Small-base control before SFT",
        "source_label": "local scienceeval Fossils-M run, 2026",
        "source_url": "",
    },
    "qwen3-5-2b-sciriff4096": {
        "label": "Qwen3.5-2B SciRIFF SFT",
        "model_type": "Open-weight small model after SciRIFF SFT",
        "reading_rule": "Compare directly to Qwen3.5-2B base",
        "source_label": "local scienceeval Fossils-M run, 2026",
        "source_url": "",
    },
    "gemma-4-31b-it": {
        "label": "gemma-4-31B-it",
        "model_type": "Open-weight model",
    },
    "qwen3-6-27b": {
        "label": "Qwen3.6-27B",
        "model_type": "Open-weight family reference",
        "reading_rule": "Family reference, not the trained target",
    },
}


# benchmark constants
BENCHMARK_ORDER = (
    "frontierscience",
    "gpqa",
    "pubmedqa",
    "bioasq",
    "biored",
    "scierc",
    "mmlu",
    "simpleqa",
    "sciriff",
)

BENCHMARK_META = {
    "frontierscience": {
        "title": "FrontierScience",
        "origin": "**",
        "href": "benchmark.html#fossil-frontierscience",
        "year": 2026.05,
        "secondary_year": 2026.18,
        "plot_label": "Frontier O",
        "secondary_plot_label": "Frontier R",
    },
    "gpqa": {
        "title": "GPQA",
        "origin": "**",
        "href": "benchmark.html#fossil-gpqa",
        "year": 2023.92,
        "plot_label": "GPQA",
    },
    "pubmedqa": {
        "title": "PubMedQA",
        "origin": "**",
        "href": "benchmark.html#fossil-pubmedqa",
        "year": 2019.92,
        "plot_label": "PubMedQA",
    },
    "bioasq": {
        "title": "BioASQ",
        "origin": "**",
        "href": "benchmark.html#fossil-bioasq",
        "year": 2015.00,
        "plot_label": "BioASQ",
    },
    "biored": {
        "title": "BioRED",
        "origin": "**",
        "href": "benchmark.html#fossil-biored",
        "year": 2022.33,
        "plot_label": "BioRED",
    },
    "scierc": {
        "title": "SciERC",
        "origin": "**",
        "href": "benchmark.html#fossil-scierc",
        "year": 2018.83,
        "plot_label": "SciERC",
    },
    "mmlu": {
        "title": "MMLU",
        "origin": "*",
        "href": "benchmark.html#fossil-mmlu",
        "year": 2020.75,
        "plot_label": "MMLU",
    },
    "simpleqa": {
        "title": "SimpleQA",
        "origin": "*",
        "href": "benchmark.html#fossil-simpleqa",
        "year": 2024.92,
        "plot_label": "SimpleQA",
    },
    "sciriff": {
        "title": "SciRIFF",
        "origin": "**",
        "href": "benchmark.html#fossil-sciriff",
        "year": 2024.50,
        "plot_label": "SciRIFF",
    },
}

MISSING_BENCHMARK_METRICS = {
    "frontierscience": "Olympiad pass rate / Research pass rate",
    "mmlu": "MMLU",
    "simpleqa": "Correct",
}


# GPT-4o public record constants
GPT4O_VALUES = {
    "frontierscience": {
        "value": "12.3 / 0.4",
        "primary_value": 12.3,
        "secondary_value": 0.4,
        "metric": "Olympiad acc. / Research acc.",
        "source_label": "FrontierScience paper, 2026",
        "source_url": GPT4O_FRONTIERSCIENCE_URL,
    },
    "gpqa": {
        "value": "53.6%",
        "primary_value": 53.6,
        "metric": "GPQA (science)",
        "source_label": "OpenAI, 2025",
        "source_url": GPT4O_OPENAI_45_URL,
    },
    "pubmedqa": {
        "value": "53.65%",
        "primary_value": 53.65,
        "metric": "Label-F1, PubMedQA-labeled",
        "source_label": "MedINST, 2024",
        "source_url": GPT4O_MEDINST_URL,
    },
    "bioasq": {
        "value": "86.19%",
        "primary_value": 86.19,
        "metric": "Label-F1, Task-B yes/no",
        "source_label": "MedINST, 2024",
        "source_url": GPT4O_MEDINST_URL,
    },
    "biored": {
        "value": "72.45%",
        "primary_value": 72.45,
        "metric": "Entity-F1, BioRED NER slice",
        "source_label": "MedINST, 2024",
        "source_url": GPT4O_MEDINST_URL,
    },
    "scierc": {
        "value": "70.35%",
        "primary_value": 70.35,
        "metric": "G-BS-F1, plotted on a percent scale",
        "source_label": "GraphJudge, 2025",
        "source_url": GPT4O_GRAPHJUDGE_URL,
    },
    "mmlu": {
        "value": "85.7%",
        "primary_value": 85.7,
        "metric": "MMLU",
        "source_label": "OpenAI, 2025",
        "source_url": GPT4O_OPENAI_41_URL,
    },
    "simpleqa": {
        "value": "38.2%",
        "primary_value": 38.2,
        "metric": "Correct",
        "source_label": "OpenAI, 2024",
        "source_url": GPT4O_SIMPLEQA_URL,
    },
    "sciriff": {
        "value": "60.4",
        "primary_value": 60.4,
        "metric": "Avg., SciRIFF-Eval",
        "source_label": "SciRIFF, 2025",
        "source_url": GPT4O_SCIRIFF_URL,
    },
}


# label placement constants
ABOVE_DOT_LABEL_OFFSETS = {
    "bioasq": (0, 12, "bottom"),
    "scierc": (0, 12, "bottom"),
    "pubmedqa": (0, 12, "bottom"),
    "mmlu": (0, 12, "bottom"),
    "biored": (0, 12, "bottom"),
    "gpqa": (0, 12, "bottom"),
    "sciriff": (0, 12, "bottom"),
    "simpleqa": (0, 12, "bottom"),
    "frontierscience": (0, 12, "bottom"),
    "frontierscience_research": (0, 12, "bottom"),
}

MODEL_LABEL_OFFSETS = {
    "gpt-4o": ABOVE_DOT_LABEL_OFFSETS,
    "gpt-oss-20b": ABOVE_DOT_LABEL_OFFSETS,
    "gemma-4-31b-it": ABOVE_DOT_LABEL_OFFSETS,
}


# helper function to parse CLI args
def parse_args() -> argparse.Namespace:
    """
    Parse model fossil sheet generation arguments.

    Returns
    ------------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Generate Fossils-M site data and model plots.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing Fossils-M model result folders, or one model result folder.",
    )
    parser.add_argument(
        "--data-output",
        type=Path,
        default=DATA_OUTPUT_PATH,
        help="Compact catalog JSON written for fossils.html.",
    )
    parser.add_argument(
        "--image-output-dir",
        type=Path,
        default=IMAGE_OUTPUT_DIR,
        help="Directory for generated plot images.",
    )
    return parser.parse_args()


# helper function to load one JSON file
def load_json(path: Path) -> dict:
    """
    Load a JSON payload.

    Parameters
    ------------
    path: Path
        JSON path

    Returns
    ------------
    dict
    """
    return json.loads(path.read_text(encoding="utf-8"))


# helper function to resolve available model result directories
def resolve_model_result_dirs(results_dir: Path) -> dict[str, Path]:
    """
    Resolve configured Fossils-M model result directories.

    Parameters
    ------------
    results_dir: Path
        Results root or a single model result directory

    Returns
    ------------
    dict[str, Path]
    """
    if (results_dir / "gpqa.json").exists():
        model_id = results_dir.name
        if model_id not in OPEN_MODEL_CONFIGS:
            raise RuntimeError(f"No Fossils-M model config for {model_id}.")
        return {model_id: results_dir}

    resolved = {}
    for model_id in MODEL_ORDER:
        if model_id not in OPEN_MODEL_CONFIGS:
            continue
        model_dir = resolve_model_result_dir(results_dir, model_id)
        if model_dir.exists():
            resolved[model_id] = model_dir

    return resolved


# helper function to resolve canonical and legacy result directory names
def resolve_model_result_dir(results_dir: Path, model_id: str) -> Path:
    """
    Resolve the best available result directory for one configured model.

    Parameters
    ------------
    results_dir: Path
        Results root
    model_id: str
        Canonical model identifier

    Returns
    ------------
    Path
    """
    candidates = (model_id, *MODEL_RESULT_DIR_ALIASES.get(model_id, ()))
    existing = [results_dir / candidate for candidate in candidates if (results_dir / candidate).exists()]
    if not existing:
        return results_dir / model_id

    return max(existing, key=result_dir_completeness_score)


# helper function to score result directories by completed benchmark rows
def result_dir_completeness_score(result_dir: Path) -> tuple[int, int]:
    """
    Score a model result directory by benchmark coverage and example count.

    Parameters
    ------------
    result_dir: Path
        Candidate model result directory

    Returns
    ------------
    tuple[int, int]
    """
    completed_benchmarks = 0
    total_examples = 0

    for benchmark in BENCHMARK_ORDER:
        path = result_dir / f"{benchmark}.json"
        if not path.exists():
            continue
        payload = load_json(path)
        example_count = len(payload.get("results", []))
        total_examples += example_count
        if example_count > 5:
            completed_benchmarks += 1

    return completed_benchmarks, total_examples


# helper function to load all benchmark summaries
def load_result_summaries(results_dir: Path) -> dict[str, dict]:
    """
    Load Fossils-M summary blocks from raw result files.

    Parameters
    ------------
    results_dir: Path
        Directory containing raw result JSON files

    Returns
    ------------
    dict[str, dict]
    """
    summaries = {}

    for benchmark in (
        "bioasq",
        "biored",
        "frontierscience",
        "gpqa",
        "mmlu",
        "pubmedqa",
        "scierc",
        "sciriff",
        "simpleqa",
    ):
        path = results_dir / f"{benchmark}.json"
        if not path.exists():
            continue
        payload = load_json(path)
        summary = dict(payload["summary"])
        if benchmark == "bioasq":
            summary["final_channel_accuracy"] = final_channel_accuracy(payload)
        if benchmark == "scierc":
            summary["final_channel_macro_f1"] = final_channel_macro_f1(payload)
        if benchmark == "sciriff":
            summary["final_channel_canonical_accuracy"] = final_channel_canonical_accuracy(payload)
        summaries[benchmark] = summary

    return summaries


# helper function to extract gpt-oss final-channel text
def extract_final_channel(raw_text: str) -> str:
    """
    Extract assistant final text from gpt-oss channel-formatted responses.

    Parameters
    ------------
    raw_text: str
        Raw model response

    Returns
    ------------
    str
    """
    if FINAL_CHANNEL_MARKER not in raw_text:
        return raw_text.strip()
    final_text = raw_text.rsplit(FINAL_CHANNEL_MARKER, 1)[1]
    final_text = final_text.split(END_MARKER, 1)[0]
    return final_text.strip()


# helper function to normalize exact-match text for display scoring
def normalize_display_answer(value: object) -> str:
    """
    Normalize generated answers for corrected display metrics.

    Parameters
    ------------
    value: object
        Answer-like value

    Returns
    ------------
    str
    """
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).translate(HYPHEN_TRANSLATION)
    text = text.strip().lower()
    text = re.sub(r"^answer\s*:\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n`\"'.,;:")


# helper function to compute final-channel exact accuracy
def final_channel_accuracy(payload: dict) -> float:
    """
    Score final-channel text after normalized exact matching.

    Parameters
    ------------
    payload: dict
        Raw result payload

    Returns
    ------------
    float
    """
    results = payload["results"]
    correct = 0
    for result in results:
        final_text = extract_final_channel(result["raw_response"])
        correct += normalize_display_answer(final_text) == normalize_display_answer(result["gold"])
    return correct / len(results) if results else 0.0


# helper function to compute final-channel macro-F1 for label tasks
def final_channel_macro_f1(payload: dict) -> float:
    """
    Score final-channel labels with macro-F1.

    Parameters
    ------------
    payload: dict
        Raw result payload

    Returns
    ------------
    float
    """
    labels = list(payload["config"]["labels"])
    golds = []
    predictions = []
    for result in payload["results"]:
        golds.append(result["gold"])
        final_text = extract_final_channel(result["raw_response"])
        predictions.append(parse_display_label(final_text, labels))
    return macro_f1(golds, predictions, labels)


# helper function to parse a label from final-channel text
def parse_display_label(raw_text: str, labels: list[str]) -> str:
    normalized_raw = normalize_display_answer(raw_text)
    label_by_normalized = {normalize_display_answer(label): label for label in labels}
    if normalized_raw in label_by_normalized:
        return label_by_normalized[normalized_raw]

    for normalized_label, label in label_by_normalized.items():
        if re.search(rf"\b{re.escape(normalized_label)}\b", normalized_raw):
            return label
    return normalized_raw


# helper function to compute macro-F1
def macro_f1(golds: list[str], predictions: list[str], labels: list[str]) -> float:
    scores = []
    for label in labels:
        true_positive = sum(gold == label and pred == label for gold, pred in zip(golds, predictions, strict=False))
        false_positive = sum(gold != label and pred == label for gold, pred in zip(golds, predictions, strict=False))
        false_negative = sum(gold == label and pred != label for gold, pred in zip(golds, predictions, strict=False))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


# helper function to parse JSON output when present
def parse_json_output(value: object) -> object | None:
    text = str(value).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# helper function to canonicalize JSON-style outputs
def canonicalize_json(value: object) -> object:
    if isinstance(value, dict):
        return {
            normalize_display_answer(key): canonicalize_json(item)
            for key, item in sorted(value.items(), key=lambda pair: normalize_display_answer(pair[0]))
        }
    if isinstance(value, list):
        items = [canonicalize_json(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    if isinstance(value, str):
        return normalize_display_answer(value)
    return value


# helper function to compute corrected SciRIFF display accuracy
def final_channel_canonical_accuracy(payload: dict) -> float:
    """
    Score SciRIFF with final-channel exact matching and canonical JSON equality.

    Parameters
    ------------
    payload: dict
        Raw result payload

    Returns
    ------------
    float
    """
    results = payload["results"]
    correct = 0
    for result in results:
        final_text = extract_final_channel(result["raw_response"])
        gold = result["gold"]
        exact_match = normalize_display_answer(final_text) == normalize_display_answer(gold)
        if exact_match or json_outputs_match(gold, final_text):
            correct += 1
    return correct / len(results) if results else 0.0


# helper function to compare JSON outputs semantically
def json_outputs_match(gold: object, prediction: object) -> bool:
    gold_json = parse_json_output(gold)
    prediction_json = parse_json_output(prediction)
    if gold_json is None or prediction_json is None:
        return False
    return canonicalize_json(gold_json) == canonicalize_json(prediction_json)


# helper function to convert result ratios to percentages
def percent(value: float) -> float:
    return round(float(value) * 100, 2)


# helper function to format one plotted percentage
def format_percent(value: float) -> str:
    return f"{value:.1f}%"


# helper function to create a benchmark entry
def make_entry(benchmark: str, payload: dict) -> dict:
    """
    Create a Fossils-M ledger entry.

    Parameters
    ------------
    benchmark: str
        Benchmark id
    payload: dict
        Metric and source fields

    Returns
    ------------
    dict
    """
    meta = BENCHMARK_META[benchmark]
    return {
        "id": benchmark,
        "title": meta["title"],
        "origin": meta["origin"],
        "href": meta["href"],
        **payload,
    }


# helper function to build the GPT-4o model sheet
def build_gpt4o_model() -> dict:
    entries = [make_entry(benchmark, GPT4O_VALUES[benchmark]) for benchmark in BENCHMARK_ORDER]
    return {
        "label": "GPT-4o",
        "status": "available",
        "page_title": "scienceeval | fossils-m | GPT-4o",
        "description": "Model-first fossil sheet for GPT-4o across science benchmark specimens.",
        "plot": "images/gpt4o-fossil-imprint.png",
        "plot_alt": "A fossil record plot of GPT-4o benchmark values over benchmark introduction year.",
        "heading_label": "Model fossil",
        "coverage": "9 / 9 fossils",
        "model_type": "Private frontier model",
        "reading_rule": "Compare within protocol",
        "ledger_aria_label": "GPT-4o benchmark ledger",
        "entries": entries,
        "plot_points": build_plot_points(entries),
    }


# helper function to build entries from open-model result summaries
def build_open_model_entries(model_id: str, summaries: dict[str, dict], config: dict) -> list[dict]:
    """
    Convert open-model result summaries into ledger entries.

    Parameters
    ------------
    model_id: str
        Model identifier
    summaries: dict[str, dict]
        Benchmark summary blocks
    config: dict
        Model configuration

    Returns
    ------------
    list[dict]
    """
    entries = []
    source_label = config.get("source_label", "scienceeval fossil suite, 2026")
    source_url = config.get("source_url", fossil_results_url(model_id))
    for benchmark in BENCHMARK_ORDER:
        payload = build_open_model_entry_payload(benchmark, summaries)
        if not payload.get("missing"):
            payload["source_label"] = source_label
            payload["source_url"] = source_url
        else:
            payload["source_label"] = "Not included in this Fossils-M run"
            payload["source_url"] = ""
        entries.append(make_entry(benchmark, payload))

    return entries


# helper function to build one open-model entry payload
def build_open_model_entry_payload(benchmark: str, summaries: dict[str, dict]) -> dict:
    """
    Build one open-model ledger payload, including missing benchmarks.

    Parameters
    ------------
    benchmark: str
        Benchmark id
    summaries: dict[str, dict]
        Benchmark summary blocks

    Returns
    ------------
    dict
    """
    if benchmark not in summaries:
        return {
            "value": "Not run",
            "metric": MISSING_BENCHMARK_METRICS.get(benchmark, BENCHMARK_META[benchmark]["title"]),
            "missing": True,
        }

    if benchmark == "frontierscience":
        frontierscience = summaries["frontierscience"]["by_split"]
        return {
            "value": f"{frontierscience['olympiad']['pass_rate'] * 100:.1f} / {frontierscience['research']['pass_rate'] * 100:.1f}",
            "primary_value": percent(frontierscience["olympiad"]["pass_rate"]),
            "secondary_value": percent(frontierscience["research"]["pass_rate"]),
            "metric": "Olympiad pass rate / Research pass rate",
        }
    if benchmark == "gpqa":
        return {
            "value": format_percent(percent(summaries["gpqa"]["accuracy"])),
            "primary_value": percent(summaries["gpqa"]["accuracy"]),
            "metric": "Accuracy, GPQA Diamond",
        }
    if benchmark == "mmlu":
        return {
            "value": format_percent(percent(summaries["mmlu"]["accuracy"])),
            "primary_value": percent(summaries["mmlu"]["accuracy"]),
            "metric": "Accuracy, MMLU all-subject test",
        }
    if benchmark == "pubmedqa":
        return {
            "value": format_percent(percent(summaries["pubmedqa"]["macro_f1"])),
            "primary_value": percent(summaries["pubmedqa"]["macro_f1"]),
            "metric": "Macro-F1, PubMedQA-labeled",
        }
    if benchmark == "bioasq":
        value = summaries["bioasq"].get("final_channel_accuracy", summaries["bioasq"]["accuracy"])
        return {
            "value": format_percent(percent(value)),
            "primary_value": percent(value),
            "metric": "Final-output exact, Task-B answer extraction",
        }
    if benchmark == "biored":
        return {
            "value": format_percent(percent(summaries["biored"]["macro_f1"])),
            "primary_value": percent(summaries["biored"]["macro_f1"]),
            "metric": "Macro-F1, relation classification",
        }
    if benchmark == "scierc":
        value = summaries["scierc"].get("final_channel_macro_f1", summaries["scierc"]["macro_f1"])
        return {
            "value": format_percent(percent(value)),
            "primary_value": percent(value),
            "metric": "Final-output Macro-F1, marked-pair relations",
        }
    if benchmark == "sciriff":
        value = summaries["sciriff"].get("final_channel_canonical_accuracy", summaries["sciriff"]["accuracy"])
        return {
            "value": format_percent(percent(value)),
            "primary_value": percent(value),
            "metric": "Final-output exact / canonical JSON, SciRIFF 8192",
        }
    if benchmark == "simpleqa":
        return {
            "value": format_percent(percent(summaries["simpleqa"]["accuracy"])),
            "primary_value": percent(summaries["simpleqa"]["accuracy"]),
            "metric": "Normalized exact answer, SimpleQA",
        }

    return {
        "value": "Not run",
        "metric": MISSING_BENCHMARK_METRICS.get(benchmark, BENCHMARK_META[benchmark]["title"]),
        "missing": True,
    }


# helper function to build the open model sheet
def build_open_model(model_id: str, summaries: dict[str, dict]) -> dict:
    config = OPEN_MODEL_CONFIGS[model_id]
    entries = build_open_model_entries(model_id, summaries, config)
    measured_count = sum(not entry.get("missing") for entry in entries)
    reading_rule = config.get("reading_rule", "Final-output normalized protocol")
    return {
        "label": config["label"],
        "status": "available",
        "page_title": f"scienceeval | fossils-m | {config['label']}",
        "description": config.get(
            "description",
            f"Model-first fossil sheet for {config['label']} from the local Fossils-M result bundle.",
        ),
        "plot": f"images/{model_id}-fossil-imprint.png",
        "plot_alt": f"A fossil record plot of {config['label']} benchmark values over benchmark introduction year.",
        "heading_label": "Model fossil",
        "coverage": f"{measured_count} / {len(BENCHMARK_ORDER)} fossils",
        "model_type": config["model_type"],
        "reading_rule": reading_rule,
        "ledger_aria_label": f"{config['label']} benchmark ledger",
        "entries": entries,
        "plot_points": build_plot_points(entries),
    }


# helper function to build the Hugging Face result URL for one model
def fossil_results_url(model_id: str) -> str:
    return f"{FOSSIL_RESULTS_ROOT_URL}/{model_id}"


# helper function to make plot point rows
def build_plot_points(entries: list[dict]) -> list[dict]:
    """
    Build plot point rows from measured ledger entries.

    Parameters
    ------------
    entries: list[dict]
        Ledger entries

    Returns
    ------------
    list[dict]
    """
    points = []

    for entry in entries:
        if entry.get("missing") or entry.get("primary_value") is None:
            continue

        meta = BENCHMARK_META[entry["id"]]
        points.append(
            {
                "id": entry["id"],
                "name": meta["plot_label"],
                "year": meta["year"],
                "value": float(entry["primary_value"]),
            }
        )

        if entry["id"] == "frontierscience" and entry.get("secondary_value") is not None:
            points.append(
                {
                    "id": "frontierscience_research",
                    "name": meta["secondary_plot_label"],
                    "year": meta["secondary_year"],
                    "value": float(entry["secondary_value"]),
                }
            )

    return points


# helper function to build compact site catalog
def build_catalog(results_dir: Path) -> dict:
    model_result_dirs = resolve_model_result_dirs(results_dir)
    if not model_result_dirs:
        raise RuntimeError(f"No configured Fossils-M model result directories found in {results_dir}.")

    models = {"gpt-4o": build_gpt4o_model()}
    for model_id in MODEL_ORDER:
        model_dir = model_result_dirs.get(model_id)
        if not model_dir:
            continue
        summaries = load_result_summaries(model_dir)
        models[model_id] = build_open_model(model_id, summaries)

    default_model = "qwen3-6-27b" if "qwen3-6-27b" in models else "gpt-4o"
    return {
        "defaults": {
            "model": default_model,
        },
        "models": models,
    }


# helper function to draw one plot
def render_plot(model_id: str, model: dict, output_dir: Path) -> None:
    """
    Render one model fossil plot.

    Parameters
    ------------
    model_id: str
        Model identifier
    model: dict
        Compact model sheet payload
    output_dir: Path
        Directory for generated images

    Returns
    ------------
    None
    """
    points = [point for point in model["plot_points"] if point.get("value") is not None]
    offsets = MODEL_LABEL_OFFSETS.get(model_id, ABOVE_DOT_LABEL_OFFSETS)

    fig, ax = plt.subplots(figsize=(12.6, 4.3), dpi=300)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    ax.set_xlim(2014.7, 2026.5)
    ax.set_ylim(-3, 100)
    style_axes(ax)
    draw_launch_line(ax, MODEL_INTRO_YEARS[model_id])
    draw_points(ax, points, offsets)
    draw_titles(ax)
    draw_legend(ax)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / Path(model["plot"]).name
    fig.savefig(png_path, facecolor=PAPER, bbox_inches="tight", pad_inches=0.22, dpi=300)
    plt.close(fig)
    print(png_path)


# helper function to style plot axes
def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(STEM)
    ax.spines["bottom"].set_color(STEM)
    ax.spines["left"].set_linewidth(0.95)
    ax.spines["bottom"].set_linewidth(0.95)

    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.grid(axis="y", color=STEM, linewidth=0.8, linestyle=(0, (2, 4)), alpha=0.7)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=0, pad=8)
    ax.tick_params(axis="y", colors=MUTED, labelsize=9, length=0, pad=6)
    ax.set_xticks([2015, 2018, 2020, 2022, 2024, 2026])
    ax.set_xlabel(
        "Benchmark introduction year",
        color=MUTED,
        fontsize=9.5,
        labelpad=14,
        fontfamily="DejaVu Serif",
    )
    ax.set_ylabel(
        "Reported value",
        color=MUTED,
        fontsize=9.5,
        labelpad=14,
        fontfamily="DejaVu Serif",
    )


# helper function to draw model launch line
def draw_launch_line(ax, launch_year: float) -> None:
    ax.axvline(
        launch_year,
        color=LAUNCH,
        linewidth=1.1,
        linestyle=(0, (1.5, 2.6)),
        alpha=0.75,
        zorder=0,
    )


# helper function to draw all measured points
def draw_points(ax, points: list[dict], offsets: dict[str, tuple[int, int, str]]) -> None:
    for point in points:
        offset_x, offset_y, vertical_align = offsets.get(point["id"], (0, 12, "bottom"))
        value = point["value"]

        ax.vlines(
            point["year"],
            0,
            value,
            color=STEM,
            linewidth=1.0,
            alpha=0.8,
            zorder=1,
        )
        ax.scatter(
            point["year"],
            value,
            s=220,
            facecolor="none",
            edgecolor=RING,
            linewidth=1.0,
            alpha=0.5,
            zorder=2,
        )
        ax.scatter(
            point["year"],
            value,
            s=58,
            facecolor="#f8f4ee",
            edgecolor=ACCENT,
            linewidth=1.2,
            zorder=3,
        )
        ax.scatter(
            point["year"],
            value,
            s=12,
            color=ACCENT,
            zorder=4,
        )
        ax.annotate(
            f"{point['name']}\n{value:.1f}",
            xy=(point["year"], value),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            ha="center",
            va=vertical_align,
            color=INK,
            fontsize=7.9,
            linespacing=1.12,
            fontfamily="DejaVu Serif",
            zorder=5,
        )


# helper function to draw plot title text
def draw_titles(ax) -> None:
    ax.text(
        0.0,
        1.09,
        "Fossil Record",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=INK,
        fontsize=16.5,
        fontfamily="DejaVu Serif",
    )
    ax.text(
        0.0,
        1.03,
        "Measured benchmark values against benchmark introduction year. Metric families differ, so the shared axis is for reading the record, not direct equivalence.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=8.5,
        fontfamily="DejaVu Serif",
    )
    ax.text(
        1.0,
        1.09,
        "Metric families: Accuracy, Macro-F1, Pass Rate, Avg.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=8.0,
        fontfamily="DejaVu Serif",
    )


# helper function to draw launch legend
def draw_legend(ax) -> None:
    launch_handle = Line2D(
        [0],
        [0],
        color=LAUNCH,
        linewidth=1.1,
        linestyle=(0, (1.5, 2.6)),
    )
    legend = ax.legend(
        [launch_handle],
        ["Model launch date"],
        loc="upper right",
        bbox_to_anchor=(1.0, 0.98),
        frameon=False,
        handlelength=2.4,
        handletextpad=0.6,
        borderaxespad=0.0,
        fontsize=8.3,
    )
    for text in legend.get_texts():
        text.set_color(MUTED)
        text.set_fontfamily("DejaVu Serif")


# orchestration function
def run_model_fossil_sheet(args: argparse.Namespace) -> None:
    catalog = build_catalog(args.results_dir)
    save_json(args.data_output, catalog)
    print(args.data_output)

    for model_id in MODEL_ORDER:
        if model_id == "gpt-4o" or model_id not in catalog["models"]:
            continue
        render_plot(model_id, catalog["models"][model_id], args.image_output_dir)


if __name__ == "__main__":
    run_model_fossil_sheet(parse_args())
