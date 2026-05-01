#!/usr/bin/env python3

# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/scienceeval/blob/main/LICENSE

from __future__ import annotations

# stdlib
import argparse
import asyncio
import csv
import json
import os
import random
import re
from pathlib import Path

# data
import pyarrow.parquet as pq

# local
from fossil_m_common import (
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    REPO_ROOT,
    generate_async,
    load_dotenv,
    load_json,
    logger,
    normalize_answer,
    parse_json_arg,
    save_json,
    slugify,
    summarize_classification_results,
    utc_now,
)


# directory constants
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results" / "fossil-m"


# benchmark constants
BENCHMARKS = ("bioasq", "biored", "gpqa", "mmlu", "pubmedqa", "scierc", "sciriff", "simpleqa")
GPQA_SPLITS = ("diamond", "main", "experts", "extended")
SCIRIFF_CONTEXT_LENGTH = "8192"
SCIERC_LABELS = (
    "COMPARE",
    "CONJUNCTION",
    "EVALUATE-FOR",
    "FEATURE-OF",
    "HYPONYM-OF",
    "PART-OF",
    "USED-FOR",
)
BIORED_RELATION_LABELS = (
    "Association",
    "Bind",
    "Comparison",
    "Conversion",
    "Cotreatment",
    "Drug_Interaction",
    "Negative_Correlation",
    "Positive_Correlation",
)
PUBMEDQA_LABELS = ("yes", "no", "maybe")
CHOICE_LABELS = ("A", "B", "C", "D")


# prompt templates
GPQA_PROMPT_TEMPLATE = """Answer the multiple-choice science question.

Return only one letter: A, B, C, or D.

Question:
{question}

Choices:
{choices}
"""

MMLU_PROMPT_TEMPLATE = """Answer the multiple-choice question.

Return only one letter: A, B, C, or D.

Subject:
{subject}

Question:
{question}

Choices:
{choices}
"""

PUBMEDQA_PROMPT_TEMPLATE = """Answer the biomedical question using only the provided PubMed context.

Return only one label: yes, no, or maybe.

Question:
{question}

Context:
{context}
"""

BIOASQ_PROMPT_TEMPLATE = """Answer the biomedical question using only the provided context.

Return only the shortest answer span or phrase. Do not explain.

Question:
{question}

Context:
{context}
"""

SCIERC_PROMPT_TEMPLATE = """Classify the semantic relation between the two marked scientific entities.

The first entity is marked with [[ ... ]]. The second entity is marked with << ... >>.
Return only one relation label from this list:
{labels}

Sentence:
{text}
"""

BIORED_PROMPT_TEMPLATE = """Classify the BioRED relation between the two biomedical entities.

Return only one relation label from this list:
{labels}

Entity 1:
{entity_1}

Entity 2:
{entity_2}

Document:
{document}
"""

SCIRIFF_PROMPT_TEMPLATE = """Follow the scientific instruction exactly.

Return only the final answer requested by the instruction. Do not explain.

Instruction:
{input_text}
"""

SIMPLEQA_PROMPT_TEMPLATE = """Answer the question directly.

Return only the shortest answer phrase. Do not explain.

Question:
{question}
"""


# helper function to parse CLI args
def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for one fossil benchmark run.

    Returns
    ------------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Run first-pass Fossils-M label/extraction evaluations.")
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True, help="Benchmark fossil to evaluate.")
    parser.add_argument("--model", required=True, help="Candidate model id served by the endpoint.")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, usually http://host:port/v1.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable holding the API key.")
    parser.add_argument("--output", type=Path, help="Output JSON path.")
    parser.add_argument("--start-index", type=int, default=0, help="Skip the first N examples after loading.")
    parser.add_argument("--limit", type=int, help="Maximum examples to run.")
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT, help="Concurrent requests.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max output tokens per request.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="Per-request timeout.")
    parser.add_argument("--extra-body", help="JSON object merged into each chat completion request.")
    parser.add_argument("--gpqa-split", choices=GPQA_SPLITS, default="diamond", help="GPQA CSV split to use.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output state.")
    parser.add_argument("--dry-run", action="store_true", help="Print dataset counts without calling a model.")
    return parser.parse_args()


# helper function to validate args
def validate_args(args: argparse.Namespace) -> None:
    if args.start_index < 0:
        raise SystemExit("--start-index must be >= 0.")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be > 0 when provided.")
    if args.max_concurrent <= 0:
        raise SystemExit("--max-concurrent must be > 0.")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be > 0.")


# helper function to resolve output path
def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        return args.output
    output_dir = RESULTS_DIR / slugify(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{args.benchmark}.json"


# helper function to load benchmark examples
def load_examples(args: argparse.Namespace) -> tuple[list[dict], list[str]]:
    loaders = {
        "bioasq": load_bioasq_examples,
        "biored": load_biored_examples,
        "gpqa": lambda: load_gpqa_examples(args.gpqa_split),
        "mmlu": load_mmlu_examples,
        "pubmedqa": load_pubmedqa_examples,
        "scierc": load_scierc_examples,
        "sciriff": load_sciriff_examples,
        "simpleqa": load_simpleqa_examples,
    }
    examples, labels = loaders[args.benchmark]()
    examples = examples[args.start_index :]
    if args.limit is not None:
        examples = examples[: args.limit]
    return examples, labels


# helper function to load GPQA examples
def load_gpqa_examples(split_name: str) -> tuple[list[dict], list[str]]:
    path = DATA_DIR / "gpqa" / "extracted" / "dataset" / f"gpqa_{split_name}.csv"
    examples = []
    with path.open(newline="", encoding="utf-8", errors="replace") as file:
        for row_index, row in enumerate(csv.DictReader(file)):
            record_id = row.get("Record ID") or f"gpqa-{split_name}-{row_index}"
            choices = [
                ("correct", row["Correct Answer"].strip()),
                ("incorrect", row["Incorrect Answer 1"].strip()),
                ("incorrect", row["Incorrect Answer 2"].strip()),
                ("incorrect", row["Incorrect Answer 3"].strip()),
            ]
            random.Random(record_id).shuffle(choices)
            formatted_choices = []
            gold = None
            for letter, (kind, answer) in zip(CHOICE_LABELS, choices, strict=False):
                formatted_choices.append(f"{letter}. {answer}")
                if kind == "correct":
                    gold = letter
            examples.append(
                {
                    "id": record_id,
                    "benchmark": "gpqa",
                    "metric": "accuracy",
                    "parse_mode": "choice",
                    "labels": list(CHOICE_LABELS),
                    "gold": gold,
                    "messages": [
                        {
                            "role": "user",
                            "content": GPQA_PROMPT_TEMPLATE.format(
                                question=row["Question"].strip(),
                                choices="\n".join(formatted_choices),
                            ),
                        }
                    ],
                }
            )
    return examples, list(CHOICE_LABELS)


# helper function to load MMLU examples
def load_mmlu_examples() -> tuple[list[dict], list[str]]:
    path = DATA_DIR / "mmlu" / "all" / "test-00000-of-00001.parquet"
    rows = pq.read_table(path).to_pylist()
    examples = []
    for row_index, row in enumerate(rows):
        choices = [str(choice).strip() for choice in row["choices"]]
        formatted_choices = [f"{letter}. {choice}" for letter, choice in zip(CHOICE_LABELS, choices, strict=False)]
        examples.append(
            {
                "id": f"mmlu-test-{row_index}",
                "benchmark": "mmlu",
                "metric": "accuracy",
                "parse_mode": "choice",
                "labels": list(CHOICE_LABELS),
                "gold": CHOICE_LABELS[int(row["answer"])],
                "messages": [
                    {
                        "role": "user",
                        "content": MMLU_PROMPT_TEMPLATE.format(
                            subject=str(row["subject"]).replace("_", " "),
                            question=row["question"].strip(),
                            choices="\n".join(formatted_choices),
                        ),
                    }
                ],
            }
        )
    return examples, list(CHOICE_LABELS)


# helper function to load PubMedQA examples
def load_pubmedqa_examples() -> tuple[list[dict], list[str]]:
    path = DATA_DIR / "pubmedqa" / "pqa_labeled" / "train-00000-of-00001.parquet"
    rows = pq.read_table(path).to_pylist()
    examples = []
    for row in rows:
        context = "\n\n".join(row["context"]["contexts"])
        gold = normalize_answer(row["final_decision"])
        examples.append(
            {
                "id": str(row["pubid"]),
                "benchmark": "pubmedqa",
                "metric": "macro_f1",
                "parse_mode": "label",
                "labels": list(PUBMEDQA_LABELS),
                "gold": gold,
                "messages": [
                    {
                        "role": "user",
                        "content": PUBMEDQA_PROMPT_TEMPLATE.format(
                            question=row["question"].strip(),
                            context=context.strip(),
                        ),
                    }
                ],
            }
        )
    return examples, list(PUBMEDQA_LABELS)


# helper function to load BioASQ answer-extraction examples
def load_bioasq_examples() -> tuple[list[dict], list[str]]:
    path = DATA_DIR / "bioasq" / "valid_bio.csv"
    examples = []
    with path.open(newline="", encoding="utf-8", errors="replace") as file:
        for row_index, row in enumerate(csv.DictReader(file)):
            answer, context = parse_bioasq_text(row["text"])
            examples.append(
                {
                    "id": f"bioasq-valid-{row_index}",
                    "benchmark": "bioasq",
                    "metric": "exact_match",
                    "parse_mode": "exact",
                    "labels": [],
                    "gold": normalize_answer(answer),
                    "messages": [
                        {
                            "role": "user",
                            "content": BIOASQ_PROMPT_TEMPLATE.format(
                                question=row["question"].strip(),
                                context=context.strip(),
                            ),
                        }
                    ],
                }
            )
    return examples, []


# helper function to split BioASQ csv answer/context text
def parse_bioasq_text(raw_text: str) -> tuple[str, str]:
    match = re.search(r"<answer>\s*(.*?)\s*<context>\s*(.*)", raw_text, flags=re.DOTALL)
    if not match:
        return "", raw_text
    return match.group(1).strip(), match.group(2).strip()


# helper function to load SciERC marked-pair relation examples
def load_scierc_examples() -> tuple[list[dict], list[str]]:
    path = DATA_DIR / "scierc" / "test.jsonl"
    examples = []
    with path.open(encoding="utf-8") as file:
        for row_index, line in enumerate(file):
            row = json.loads(line)
            label = row["label"].strip()
            examples.append(
                {
                    "id": f"scierc-test-{row_index}",
                    "benchmark": "scierc",
                    "metric": "macro_f1",
                    "parse_mode": "label",
                    "labels": list(SCIERC_LABELS),
                    "gold": label,
                    "messages": [
                        {
                            "role": "user",
                            "content": SCIERC_PROMPT_TEMPLATE.format(
                                labels=", ".join(SCIERC_LABELS),
                                text=row["text"].strip(),
                            ),
                        }
                    ],
                }
            )
    return examples, list(SCIERC_LABELS)


# helper function to load SciRIFF exact-answer examples
def load_sciriff_examples() -> tuple[list[dict], list[str]]:
    path = DATA_DIR / "sciriff" / SCIRIFF_CONTEXT_LENGTH / "test-00000-of-00001.parquet"
    rows = pq.read_table(path).to_pylist()
    examples = []
    for row in rows:
        examples.append(
            {
                "id": row["_instance_id"],
                "benchmark": "sciriff",
                "metric": "exact_match",
                "parse_mode": "exact",
                "labels": [],
                "gold": normalize_answer(row["output"]),
                "metadata": row.get("metadata", {}),
                "messages": [
                    {
                        "role": "user",
                        "content": SCIRIFF_PROMPT_TEMPLATE.format(input_text=row["input"].strip()),
                    }
                ],
            }
        )
    return examples, []


# helper function to load BioRED relation-classification examples
def load_biored_examples() -> tuple[list[dict], list[str]]:
    path = DATA_DIR / "biored" / "extracted" / "BioRED" / "Test.BioC.JSON"
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for document in payload["documents"]:
        document_text = "\n\n".join(passage["text"] for passage in document["passages"])
        entities = collect_biored_entities(document)
        for relation in document.get("relations", []):
            infons = relation["infons"]
            entity_1 = entities.get(infons.get("entity1"))
            entity_2 = entities.get(infons.get("entity2"))
            label = infons.get("type", "").strip()
            if not entity_1 or not entity_2 or label not in BIORED_RELATION_LABELS:
                continue
            examples.append(
                {
                    "id": f"{document['id']}:{relation['id']}",
                    "benchmark": "biored",
                    "metric": "macro_f1",
                    "parse_mode": "label",
                    "labels": list(BIORED_RELATION_LABELS),
                    "gold": label,
                    "messages": [
                        {
                            "role": "user",
                            "content": BIORED_PROMPT_TEMPLATE.format(
                                labels=", ".join(BIORED_RELATION_LABELS),
                                entity_1=format_biored_entity(entity_1),
                                entity_2=format_biored_entity(entity_2),
                                document=document_text,
                            ),
                        }
                    ],
                }
            )
    return examples, list(BIORED_RELATION_LABELS)


# helper function to load SimpleQA exact-answer examples
def load_simpleqa_examples() -> tuple[list[dict], list[str]]:
    path = DATA_DIR / "simpleqa" / "simple_qa_test_set.csv"
    examples = []
    with path.open(newline="", encoding="utf-8", errors="replace") as file:
        for row_index, row in enumerate(csv.DictReader(file)):
            examples.append(
                {
                    "id": f"simpleqa-test-{row_index}",
                    "benchmark": "simpleqa",
                    "metric": "exact_match",
                    "parse_mode": "exact",
                    "labels": [],
                    "gold": normalize_answer(row["answer"]),
                    "messages": [
                        {
                            "role": "user",
                            "content": SIMPLEQA_PROMPT_TEMPLATE.format(question=row["problem"].strip()),
                        }
                    ],
                }
            )
    return examples, []


# helper function to collect BioRED entities by normalized identifier
def collect_biored_entities(document: dict) -> dict[str, dict]:
    entities = {}
    for passage in document["passages"]:
        for annotation in passage.get("annotations", []):
            infons = annotation.get("infons", {})
            identifier = infons.get("identifier")
            if not identifier or identifier in entities:
                continue
            entities[identifier] = {
                "identifier": identifier,
                "text": annotation.get("text", ""),
                "type": infons.get("type", ""),
            }
    return entities


# helper function to format one BioRED entity
def format_biored_entity(entity: dict) -> str:
    return f"{entity['text']} ({entity['type']}; {entity['identifier']})"


# helper function to load or initialize state
def load_or_init_state(args: argparse.Namespace, output_path: Path, labels: list[str]) -> dict:
    if output_path.exists() and args.resume:
        return load_json(output_path)
    return {
        "benchmark": args.benchmark,
        "candidate": {
            "model": args.model,
            "base_url": args.base_url,
        },
        "config": {
            "start_index": args.start_index,
            "limit": args.limit,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "max_concurrent": args.max_concurrent,
            "labels": labels,
            "protocol": protocol_note(args.benchmark),
        },
        "started_at": utc_now(),
        "completed_at": None,
        "summary": {},
        "results": [],
    }


# helper function to explain benchmark protocol
def protocol_note(benchmark: str) -> str:
    notes = {
        "bioasq": "First-pass BioASQ answer extraction from local valid_bio.csv.",
        "biored": "First-pass BioRED relation classification over gold entity pairs.",
        "gpqa": "Deterministic shuffled multiple choice with exact letter scoring.",
        "mmlu": "MMLU all-subject test split multiple choice with exact letter scoring.",
        "pubmedqa": "PubMedQA labeled yes/no/maybe classification from context.",
        "scierc": "Marked-pair SciERC relation classification.",
        "sciriff": "SciRIFF 8192 test exact-output instruction following.",
        "simpleqa": "SimpleQA local test CSV with normalized exact-answer scoring.",
    }
    return notes[benchmark]


# helper function to get completed ids
def completed_example_ids(state: dict) -> set[str]:
    return {result["example_id"] for result in state.get("results", [])}


# helper function to run one example
async def process_example(
    example: dict,
    config: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        raw = await generate_async(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["model"],
            messages=example["messages"],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            timeout_seconds=config["timeout_seconds"],
            extra_body=config["extra_body"],
        )
        prediction = parse_prediction(raw, example)
        return {
            "example_id": example["id"],
            "gold": example["gold"],
            "prediction": prediction,
            "correct": prediction == example["gold"],
            "raw_response": raw,
            "recorded_at": utc_now(),
        }


# helper function to parse model predictions
def parse_prediction(raw_text: str, example: dict) -> str:
    mode = example["parse_mode"]
    if mode == "choice":
        return parse_choice_prediction(raw_text)
    if mode == "label":
        return parse_label_prediction(raw_text, example["labels"])
    return normalize_answer(raw_text)


# helper function to parse GPQA choice predictions
def parse_choice_prediction(raw_text: str) -> str:
    stripped = raw_text.strip()
    match = re.search(r"\b([ABCD])\b", stripped, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"([ABCD])", stripped, flags=re.IGNORECASE)
    return match.group(1).upper() if match else normalize_answer(stripped).upper()


# helper function to parse closed label predictions
def parse_label_prediction(raw_text: str, labels: list[str]) -> str:
    normalized_raw = normalize_answer(raw_text)
    label_by_normalized = {normalize_answer(label): label for label in labels}
    if normalized_raw in label_by_normalized:
        return label_by_normalized[normalized_raw]

    for normalized_label, label in label_by_normalized.items():
        if re.search(rf"\b{re.escape(normalized_label)}\b", normalized_raw):
            return label
    return normalized_raw


# helper function to run pending examples concurrently
async def run_pending_examples_async(
    examples: list[dict],
    state: dict,
    output_path: Path,
    labels: list[str],
    config: dict,
    max_concurrent: int,
) -> None:
    processed = completed_example_ids(state)
    pending = [example for example in examples if example["id"] not in processed]
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [asyncio.create_task(process_example(example, config, semaphore)) for example in pending]
    completed_count = len(processed)

    try:
        for task in asyncio.as_completed(tasks):
            result = await task
            completed_count += 1
            state["results"].append(result)
            state["summary"] = summarize_classification_results(state["results"], labels)
            state["completed_at"] = utc_now()
            save_json(output_path, state)
            logger.info(
                "[%s/%s] %s -> pred=%s gold=%s correct=%s",
                completed_count,
                len(examples),
                result["example_id"],
                result["prediction"],
                result["gold"],
                result["correct"],
            )
    except Exception:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


# orchestration function
async def run_label_fossil_eval_async(args: argparse.Namespace) -> None:
    extra_body = parse_json_arg(args.extra_body, "--extra-body")
    examples, labels = load_examples(args)
    output_path = resolve_output_path(args)
    state = load_or_init_state(args, output_path, labels)
    processed = completed_example_ids(state)

    if args.dry_run:
        logger.info("Benchmark: %s", args.benchmark)
        logger.info("Examples after filtering: %s", len(examples))
        logger.info("Already present in output: %s", sum(example["id"] in processed for example in examples))
        logger.info("Output: %s", output_path)
        return

    config = {
        "base_url": args.base_url,
        "api_key": os.getenv(args.api_key_env, ""),
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timeout_seconds": args.timeout_seconds,
        "extra_body": extra_body,
    }
    await run_pending_examples_async(examples, state, output_path, labels, config, args.max_concurrent)
    state["summary"] = summarize_classification_results(state["results"], labels)
    state["completed_at"] = utc_now()
    save_json(output_path, state)
    logger.info("Saved results to %s", output_path)
    print(json.dumps(state["summary"], indent=2))


# orchestration function
def run_label_fossil_eval(args: argparse.Namespace) -> None:
    asyncio.run(run_label_fossil_eval_async(args))


if __name__ == "__main__":
    load_dotenv(REPO_ROOT / ".env")
    parsed_args = parse_args()
    validate_args(parsed_args)
    run_label_fossil_eval(parsed_args)
