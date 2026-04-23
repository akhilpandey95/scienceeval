#!/usr/bin/env python3

# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/scienceeval/blob/main/LICENSE

from __future__ import annotations

# stdlib
import asyncio
import argparse
import json
import logging
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


# init logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# directory constants
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_CACHE_DIR = REPO_ROOT / ".cache" / "frontierscience"
RESULTS_DIR = REPO_ROOT / "results" / "frontierscience"


# endpoint defaults
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_CANDIDATE_MAX_TOKENS = 4096
DEFAULT_JUDGE_MAX_TOKENS = 1500
DEFAULT_CANDIDATE_TEMPERATURE = 0.0
DEFAULT_JUDGE_TEMPERATURE = 0.0
DEFAULT_SLEEP_SECONDS = 0.0
DEFAULT_MAX_CONCURRENT = 8


# benchmark constants
DEFAULT_OLYMPIAD_TRIALS = 20
DEFAULT_RESEARCH_TRIALS = 30
RESEARCH_RUBRIC_TOTAL = 10.0
RESEARCH_PASS_THRESHOLD = 7.0
VALID_SPLITS = ("olympiad", "research", "all")
VALID_SUBJECTS = ("physics", "chemistry", "biology")

DATASET_URLS = {
    "olympiad": "https://huggingface.co/datasets/openai/frontierscience/resolve/main/olympiad/test.jsonl",
    "research": "https://huggingface.co/datasets/openai/frontierscience/resolve/main/research/test.jsonl",
}


# judge prompts from FrontierScience Appendix B
OLYMPIAD_JUDGE_PROMPT_TEMPLATE = """You are grading an attempted answer to a science olympiad problem. You will be given the problem, attempted answer, and reference answer. Evaluate the solution against the provided reference solution, ensuring it is complete and matches the reference solution. Pay close attention to detail and grade it strictly, but fairly.

The reference answer is either a single number or expression in latex formatting, a chemical formula, a compound name, or a phrase referring to a specific name, entity, or method.
Mark the attempted answer as correct if it fully matches the reference answer or is otherwise equivalent (e.g., an equivalent algebraic expression, a numerical number within 1 decimal place rounding of the reference answer (e.g., 6.69 ≈ 6.7), an equivalent name for a compound/formula, equivalent when accounting for units, etc.). Mark it as incorrect if it is not equivalent to the reference answer.
***
The problem: {problem}
***
The reference answer: {reference_answer}
***
The attempted answer: {attempted_answer}
***
First, think step-by-step about whether the attempted answer matches the reference answer.
If the attempted answer is correct, write "VERDICT: CORRECT" in the last line of your response, with no other text or formatting. If it is incorrect, write "VERDICT: INCORRECT"."""

RESEARCH_JUDGE_PROMPT_TEMPLATE = """You are grading a science exam.
You will be given the problem, attempted answer, and a rubric to grade the answer. The rubric will total up to 10 points.
Evaluate the attempted answer against the provided rubric. Pay close attention to detail and grade it strictly, but fairly. Only evaluate against the rubric, as you yourself should not make any judgements (e.g., even if you think the answer is correct but rubric is wrong, you should treat the rubric as the gold standard). Return the absolute total number of points earned (it can be a decimal based on the rubric).
***
The problem: {problem}
***
The rubric: {rubric}
***
The attempted answer: {attempted_answer}
***
First, think step-by-step about each rubric item. Explain your reasoning for each rubric item.
Then, tally the points up and write VERDICT: <total_points> in the last line of your response, no other text. For example, VERDICT: 2.5 or VERDICT: 8."""


# helper function to parse cli args
def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the FrontierScience evaluator.

    Returns
    ------------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate FrontierScience using the paper's public gold set and Appendix B judge prompts. "
            "Designed for Qwen candidates on OpenAI-compatible endpoints."
        )
    )
    parser.add_argument("--split", choices=VALID_SPLITS, default="all", help="Which FrontierScience split to evaluate.")
    parser.add_argument("--candidate-model", required=True, help="Candidate model id, e.g. Qwen/Qwen3-32B.")
    parser.add_argument("--candidate-base-url", required=True, help="OpenAI-compatible base URL for the candidate endpoint.")
    parser.add_argument(
        "--candidate-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable holding the candidate API key.",
    )
    parser.add_argument("--judge-model", help="Judge model id for paper-style FrontierScience grading.")
    parser.add_argument(
        "--judge-base-url",
        help="OpenAI-compatible base URL for the judge endpoint. Defaults to the candidate base URL.",
    )
    parser.add_argument(
        "--judge-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable holding the judge API key.",
    )
    parser.add_argument("--subjects", nargs="+", choices=VALID_SUBJECTS, help="Optional subject filter.")
    parser.add_argument("--start-index", type=int, default=0, help="Skip the first N filtered questions.")
    parser.add_argument("--limit", type=int, help="Maximum number of filtered questions to run.")
    parser.add_argument(
        "--olympiad-trials",
        type=int,
        default=DEFAULT_OLYMPIAD_TRIALS,
        help=f"Independent trials per Olympiad question. Default {DEFAULT_OLYMPIAD_TRIALS}.",
    )
    parser.add_argument(
        "--research-trials",
        type=int,
        default=DEFAULT_RESEARCH_TRIALS,
        help=f"Independent trials per Research question. Default {DEFAULT_RESEARCH_TRIALS}.",
    )
    parser.add_argument(
        "--candidate-max-tokens",
        type=int,
        default=DEFAULT_CANDIDATE_MAX_TOKENS,
        help=f"Max tokens for the candidate response. Default {DEFAULT_CANDIDATE_MAX_TOKENS}.",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=DEFAULT_JUDGE_MAX_TOKENS,
        help=f"Max tokens for the judge response. Default {DEFAULT_JUDGE_MAX_TOKENS}.",
    )
    parser.add_argument(
        "--candidate-temperature",
        type=float,
        default=DEFAULT_CANDIDATE_TEMPERATURE,
        help=f"Candidate generation temperature. Default {DEFAULT_CANDIDATE_TEMPERATURE}.",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=DEFAULT_JUDGE_TEMPERATURE,
        help=f"Judge generation temperature. Default {DEFAULT_JUDGE_TEMPERATURE}.",
    )
    parser.add_argument(
        "--candidate-extra-body",
        help="JSON object merged into the candidate request body for vendor-specific controls.",
    )
    parser.add_argument("--judge-extra-body", help="JSON object merged into the judge request body.")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-request timeout for model calls. Default {DEFAULT_TIMEOUT_SECONDS}.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help=f"Optional pause between attempts. Default {DEFAULT_SLEEP_SECONDS}.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help=f"Maximum number of concurrent attempts. Default {DEFAULT_MAX_CONCURRENT}.",
    )
    parser.add_argument("--refresh-dataset", action="store_true", help="Re-download the public FrontierScience files.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output file.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned evaluation without calling any model.")
    parser.add_argument("--output", type=Path, help="Output JSON file. Defaults to results/frontierscience/<model>-<split>.json")
    return parser.parse_args()


# helper function to load repo-local env vars
def load_dotenv(env_path: Path) -> None:
    """
    Load a simple .env file into process environment without overriding existing vars.

    Parameters
    ------------
    env_path: Path
        Path to the .env file

    Returns
    ------------
    None
    """
    if not env_path.exists():
        return

    loaded = 0
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key or key in os.environ:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ[key] = value
        loaded += 1

    if loaded:
        logger.info("Loaded %s environment variable(s) from %s", loaded, env_path)


# helper function to validate args early
def validate_args(args: argparse.Namespace) -> None:
    """
    Validate CLI args before doing any work.

    Parameters
    ------------
    args: argparse.Namespace
        Parsed CLI args

    Returns
    ------------
    None
    """
    if not args.judge_model and not args.dry_run:
        raise SystemExit("--judge-model is required for scored FrontierScience runs.")
    if args.start_index < 0:
        raise SystemExit("--start-index must be >= 0.")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be > 0 when provided.")
    if args.olympiad_trials <= 0:
        raise SystemExit("--olympiad-trials must be > 0.")
    if args.research_trials <= 0:
        raise SystemExit("--research-trials must be > 0.")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be > 0.")
    if args.max_concurrent <= 0:
        raise SystemExit("--max-concurrent must be > 0.")


# helper function to parse extra json args
def parse_json_arg(raw_value: str | None, flag_name: str) -> dict:
    """
    Parse a JSON object passed through a CLI flag.

    Parameters
    ------------
    raw_value: str | None
        Raw JSON string from argparse
    flag_name: str
        CLI flag name for error messages

    Returns
    ------------
    dict
    """
    if not raw_value:
        return {}

    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as error:
        raise SystemExit(f"{flag_name} must be valid JSON: {error}") from error

    if not isinstance(value, dict):
        raise SystemExit(f"{flag_name} must decode to a JSON object.")

    return value


# helper function to resolve split order
def resolve_split_order(split: str) -> list[str]:
    """
    Resolve a CLI split choice into concrete evaluation splits.

    Parameters
    ------------
    split: str
        CLI split argument

    Returns
    ------------
    list[str]
    """
    return ["olympiad", "research"] if split == "all" else [split]


# helper function to resolve output path
def resolve_output_path(candidate_model: str, split: str, output: Path | None) -> Path:
    """
    Resolve the output path for this run.

    Parameters
    ------------
    candidate_model: str
        Candidate model name
    split: str
        Requested split
    output: Path | None
        Optional user-provided output path

    Returns
    ------------
    Path
    """
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        return output

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{slugify(candidate_model)}-{split}.json"


# helper function to read cached dataset files
def load_dataset(split_order: list[str], refresh: bool) -> dict[str, list[dict]]:
    """
    Load the public FrontierScience gold set files for the requested splits.

    Parameters
    ------------
    split_order: list[str]
        Concrete splits to load
    refresh: bool
        Whether to re-download cached files

    Returns
    ------------
    dict[str, list[dict]]
    """
    DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = {}

    for split in split_order:
        cache_path = DATASET_CACHE_DIR / split / "test.jsonl"
        ensure_dataset_file(split, cache_path, refresh)
        dataset[split] = load_jsonl(cache_path, split)

    return dataset


# helper function to download the public dataset
def ensure_dataset_file(split: str, cache_path: Path, refresh: bool) -> None:
    if cache_path.exists() and not refresh:
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading FrontierScience %s gold set...", split)
    payload = curl_request(DATASET_URLS[split], timeout_seconds=300)
    cache_path.write_text(payload, encoding="utf-8")


# helper function to load jsonl rows
def load_jsonl(path: Path, split: str) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["split"] = split
        rows.append(row)
    return rows


# helper function to filter examples
def filter_examples(
    dataset: dict[str, list[dict]],
    split_order: list[str],
    subjects: list[str] | None,
    start_index: int,
    limit: int | None,
) -> list[dict]:
    """
    Filter the loaded dataset by split, subject, and index window.

    Parameters
    ------------
    dataset: dict[str, list[dict]]
        Loaded dataset rows
    split_order: list[str]
        Concrete splits to include
    subjects: list[str] | None
        Optional subject filter
    start_index: int
        Starting offset after filtering
    limit: int | None
        Maximum number of questions to keep

    Returns
    ------------
    list[dict]
    """
    examples = []

    for split in split_order:
        for row in dataset[split]:
            if subjects and row["subject"] not in subjects:
                continue
            examples.append(row)

    examples = examples[start_index:]
    if limit is not None:
        examples = examples[:limit]

    return examples


# helper function to expand questions into repeated trial attempts
def build_attempts(examples: list[dict], args: argparse.Namespace) -> list[dict]:
    """
    Expand each question into repeated attempts matching the paper defaults.

    Parameters
    ------------
    examples: list[dict]
        Filtered question rows
    args: argparse.Namespace
        Parsed CLI args

    Returns
    ------------
    list[dict]
    """
    attempts = []

    for example in examples:
        trial_count = trial_count_for_example(example, args)
        for trial_index in range(1, trial_count + 1):
            attempts.append(
                {
                    "attempt_key": format_attempt_key(example_key(example), trial_index),
                    "example": example,
                    "trial_index": trial_index,
                    "trial_count": trial_count,
                }
            )

    return attempts


# helper function to resolve trial count by split
def trial_count_for_example(example: dict, args: argparse.Namespace) -> int:
    return args.olympiad_trials if example["split"] == "olympiad" else args.research_trials


# helper function to read resumable run state
def load_or_init_state(args: argparse.Namespace, output_path: Path) -> dict:
    """
    Load an existing run state when resuming, otherwise initialize a fresh one.

    Parameters
    ------------
    args: argparse.Namespace
        Parsed CLI args
    output_path: Path
        Output state path

    Returns
    ------------
    dict
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and args.resume:
        return json.loads(output_path.read_text(encoding="utf-8"))

    return {
        "benchmark": "frontierscience",
        "dataset": {
            "source": "openai/frontierscience",
            "splits": resolve_split_order(args.split),
            "dataset_urls": DATASET_URLS,
        },
        "candidate": {
            "model": args.candidate_model,
            "base_url": args.candidate_base_url,
        },
        "judge": {
            "model": args.judge_model,
            "base_url": args.judge_base_url or args.candidate_base_url,
        } if args.judge_model else None,
        "config": {
            "split": args.split,
            "subjects": args.subjects or [],
            "start_index": args.start_index,
            "limit": args.limit,
            "olympiad_trials": args.olympiad_trials,
            "research_trials": args.research_trials,
            "candidate_max_tokens": args.candidate_max_tokens,
            "judge_max_tokens": args.judge_max_tokens,
            "candidate_temperature": args.candidate_temperature,
            "judge_temperature": args.judge_temperature,
            "paper_grounding": "Appendix B judge prompts, raw problem input, 20 Olympiad trials, 30 Research trials",
        },
        "started_at": utc_now(),
        "completed_at": None,
        "summary": {},
        "results": [],
    }


# helper function to capture already-recorded attempt ids
def get_recorded_attempts(state: dict) -> set[str]:
    """
    Extract completed attempt ids from an existing run state.

    Parameters
    ------------
    state: dict
        Loaded run state

    Returns
    ------------
    set[str]
    """
    processed_ids = set()

    for result in state.get("results", []):
        attempt_key = result.get("attempt_key")
        if not attempt_key:
            attempt_key = format_attempt_key(result["example_key"], int(result.get("trial_index", 1)))
        processed_ids.add(attempt_key)

    return processed_ids


# helper function to build candidate config
def build_candidate_config(args: argparse.Namespace, extra_body: dict) -> dict:
    return {
        "base_url": args.candidate_base_url,
        "api_key": os.getenv(args.candidate_api_key_env, ""),
        "extra_body": extra_body,
        "max_tokens": args.candidate_max_tokens,
        "temperature": args.candidate_temperature,
        "timeout_seconds": args.timeout_seconds,
    }


# helper function to build judge config
def build_judge_config(args: argparse.Namespace, extra_body: dict) -> dict:
    return {
        "base_url": args.judge_base_url or args.candidate_base_url,
        "api_key": os.getenv(args.judge_api_key_env, ""),
        "extra_body": extra_body,
        "max_tokens": args.judge_max_tokens,
        "temperature": args.judge_temperature,
        "timeout_seconds": args.timeout_seconds,
        "model": args.judge_model,
    }


# helper function to print a dry-run plan
def print_dry_run(
    examples: list[dict],
    attempts: list[dict],
    args: argparse.Namespace,
    processed_ids: set[str],
    output_path: Path,
) -> None:
    """
    Print the planned question and attempt counts without calling any model.

    Parameters
    ------------
    examples: list[dict]
        Filtered questions
    attempts: list[dict]
        Trial-expanded attempts
    args: argparse.Namespace
        Parsed CLI args
    processed_ids: set[str]
        Attempt ids already present in output state
    output_path: Path
        Output path

    Returns
    ------------
    None
    """
    question_counts = defaultdict(int)
    attempt_counts = defaultdict(int)

    for example in examples:
        question_counts[example["split"]] += 1
    for attempt in attempts:
        attempt_counts[attempt["example"]["split"]] += 1

    logger.info("FrontierScience dry run")
    logger.info("Output: %s", output_path)
    logger.info("Questions after filtering: %s", len(examples))
    logger.info("Attempts after trial expansion: %s", len(attempts))
    logger.info("Already present in output: %s", sum(attempt["attempt_key"] in processed_ids for attempt in attempts))
    print(
        json.dumps(
            {
                "per_split_questions": dict(question_counts),
                "per_split_attempts": dict(attempt_counts),
                "preview": [
                    {
                        "example_key": example_key(example),
                        "subject": example["subject"],
                        "trials": trial_count_for_example(example, args),
                    }
                    for example in examples[:5]
                ],
            },
            indent=2,
        )
    )


# helper function to run one full attempt
async def process_single_attempt(
    attempt: dict,
    candidate_model: str,
    candidate_config: dict,
    judge_config: dict,
    semaphore: asyncio.Semaphore,
    sleep_seconds: float,
) -> dict:
    """
    Run candidate generation and paper-style judging for one attempt.

    Parameters
    ------------
    attempt: dict
        Trial-expanded attempt record
    candidate_model: str
        Candidate model name
    candidate_config: dict
        Candidate endpoint config
    judge_config: dict
        Judge endpoint config

    Returns
    ------------
    dict
    """
    async with semaphore:
        example = attempt["example"]
        candidate_response = await generate_candidate_response(candidate_model, candidate_config, example)
        grading = await grade_example(example, candidate_response, judge_config)

        if sleep_seconds > 0:
            await asyncio.sleep(sleep_seconds)

        return {
            "attempt_key": attempt["attempt_key"],
            "example_key": example_key(example),
            "trial_index": attempt["trial_index"],
            "trial_count": attempt["trial_count"],
            "task_group_id": example["task_group_id"],
            "split": example["split"],
            "subject": example["subject"],
            "candidate_model": candidate_model,
            "response": candidate_response,
            "grading": grading,
            "recorded_at": utc_now(),
        }


# helper function to query the candidate model
async def generate_candidate_response(model: str, config: dict, example: dict) -> str:
    """
    Generate a candidate response from the raw FrontierScience problem text.

    Parameters
    ------------
    model: str
        Candidate model name
    config: dict
        Candidate endpoint config
    example: dict
        FrontierScience question row

    Returns
    ------------
    str
    """
    messages = [{"role": "user", "content": example["problem"]}]
    return await generate_async(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model=model,
        messages=messages,
        max_tokens=config["max_tokens"],
        temperature=config["temperature"],
        timeout_seconds=config["timeout_seconds"],
        extra_body=config["extra_body"],
    )


# helper function to route grading by split
async def grade_example(example: dict, candidate_response: str, judge_config: dict) -> dict:
    if example["split"] == "olympiad":
        return await grade_olympiad(example, candidate_response, judge_config)
    return await grade_research(example, candidate_response, judge_config)


# helper function to grade olympiad attempts
async def grade_olympiad(example: dict, candidate_response: str, judge_config: dict) -> dict:
    judge_payload = await judge_olympiad(example, candidate_response, judge_config)
    correct = bool(judge_payload["correct"])
    return {
        "method": "judge_equivalence",
        "correct": correct,
        "passed": correct,
        "score": 1.0 if correct else 0.0,
        "max_score": 1.0,
        "grade_display": f"{'PASS' if correct else 'FAIL'} (olympiad, judge)",
        "gold_answer": example["answer"],
        "rationale": judge_payload["rationale"],
        "judge_result": judge_payload,
    }


# helper function to grade research attempts
async def grade_research(example: dict, candidate_response: str, judge_config: dict) -> dict:
    max_points = extract_max_points(example["answer"])
    judge_payload = await judge_research(example, candidate_response, max_points, judge_config)
    awarded_points = float(judge_payload["awarded_points"])
    passed = bool(judge_payload["passed"])
    return {
        "method": "rubric_judge",
        "correct": passed,
        "passed": passed,
        "score": awarded_points,
        "max_score": max_points,
        "grade_display": (
            f"{'PASS' if passed else 'FAIL'} "
            f"(research, {awarded_points:.2f}/{max_points:.2f}, threshold {RESEARCH_PASS_THRESHOLD:.2f})"
        ),
        "rationale": judge_payload["summary"],
        "judge_result": judge_payload,
    }


# helper function to run olympiad judge prompt
async def judge_olympiad(example: dict, candidate_response: str, judge_config: dict) -> dict:
    prompt = OLYMPIAD_JUDGE_PROMPT_TEMPLATE.format(
        problem=example["problem"],
        reference_answer=example["answer"],
        attempted_answer=candidate_response,
    )
    raw = await generate_async(
        base_url=judge_config["base_url"],
        api_key=judge_config["api_key"],
        model=judge_config["model"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=judge_config["max_tokens"],
        temperature=judge_config["temperature"],
        timeout_seconds=judge_config["timeout_seconds"],
        extra_body=judge_config["extra_body"],
    )
    verdict = parse_verdict_line(raw)
    return {
        "correct": verdict == "CORRECT",
        "verdict": verdict,
        "rationale": strip_verdict_line(raw),
        "raw_judge_response": raw,
    }


# helper function to run research judge prompt
async def judge_research(example: dict, candidate_response: str, max_points: float, judge_config: dict) -> dict:
    prompt = RESEARCH_JUDGE_PROMPT_TEMPLATE.format(
        problem=example["problem"],
        rubric=example["answer"],
        attempted_answer=candidate_response,
    )
    raw = await generate_async(
        base_url=judge_config["base_url"],
        api_key=judge_config["api_key"],
        model=judge_config["model"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=judge_config["max_tokens"],
        temperature=judge_config["temperature"],
        timeout_seconds=judge_config["timeout_seconds"],
        extra_body=judge_config["extra_body"],
    )
    awarded_points = max(0.0, min(max_points, parse_research_verdict_points(raw)))
    return {
        "awarded_points": awarded_points,
        "passed": awarded_points >= RESEARCH_PASS_THRESHOLD,
        "summary": strip_verdict_line(raw),
        "raw_judge_response": raw,
    }


# helper function to parse rubric totals
def extract_max_points(rubric_text: str) -> float:
    matches = re.findall(r"Points:\s*([0-9]+(?:\.[0-9]+)?)", rubric_text)
    if not matches:
        raise RuntimeError("Could not parse rubric point values from the FrontierScience-Research answer field.")

    total = sum(float(match) for match in matches)
    if abs(total - RESEARCH_RUBRIC_TOTAL) > 0.25:
        raise RuntimeError(
            f"Expected FrontierScience-Research rubric to total about {RESEARCH_RUBRIC_TOTAL:.1f}, got {total:.2f}."
        )
    return total


# helper function to parse olympiad verdicts
def parse_verdict_line(raw_text: str) -> str:
    for line in reversed(raw_text.splitlines()):
        match = re.search(r'VERDICT:\s*(CORRECT|INCORRECT)\s*$', line.strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    raise RuntimeError(f"Could not parse verdict line from judge response: {raw_text}")


# helper function to parse research point verdicts
def parse_research_verdict_points(raw_text: str) -> float:
    for line in reversed(raw_text.splitlines()):
        match = re.search(r'VERDICT:\s*([0-9]+(?:\.[0-9]+)?)\s*$', line.strip(), flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    raise RuntimeError(f"Could not parse point verdict from judge response: {raw_text}")


# helper function to strip the verdict trailer from judge text
def strip_verdict_line(raw_text: str) -> str:
    lines = raw_text.splitlines()
    if lines and re.search(r"VERDICT:\s*", lines[-1], flags=re.IGNORECASE):
        lines = lines[:-1]
    return "\n".join(line.rstrip() for line in lines).strip()


# helper function to summarize results
def summarize_results(results: list[dict]) -> dict:
    """
    Summarize attempt-level results for the full run.

    Parameters
    ------------
    results: list[dict]
        Recorded attempt results

    Returns
    ------------
    dict
    """
    summary = {
        "attempts": len(results),
        "questions": 0,
        "passed": 0,
        "pass_rate": 0.0,
        "score_sum": 0.0,
        "max_score_sum": 0.0,
        "score_rate": 0.0,
        "by_split": defaultdict(lambda: _empty_summary_bucket()),
        "by_subject": defaultdict(lambda: _empty_summary_bucket()),
        "_question_keys": set(),
    }

    for result in results:
        grade = result["grading"]
        split = result["split"]
        subject = result["subject"]
        question_key = result["example_key"]

        summary["_question_keys"].add(question_key)
        summary["passed"] += int(bool(grade["passed"]))
        summary["score_sum"] += float(grade["score"])
        summary["max_score_sum"] += float(grade["max_score"])

        split_bucket = summary["by_split"][split]
        split_bucket["attempts"] += 1
        split_bucket["_question_keys"].add(question_key)
        split_bucket["passed"] += int(bool(grade["passed"]))
        split_bucket["score_sum"] += float(grade["score"])
        split_bucket["max_score_sum"] += float(grade["max_score"])

        subject_bucket = summary["by_subject"][subject]
        subject_bucket["attempts"] += 1
        subject_bucket["_question_keys"].add(question_key)
        subject_bucket["passed"] += int(bool(grade["passed"]))
        subject_bucket["score_sum"] += float(grade["score"])
        subject_bucket["max_score_sum"] += float(grade["max_score"])

    summary["questions"] = len(summary.pop("_question_keys"))
    summary["pass_rate"] = round(summary["passed"] / summary["attempts"], 6) if summary["attempts"] else 0.0
    summary["score_rate"] = round(summary["score_sum"] / summary["max_score_sum"], 6) if summary["max_score_sum"] else 0.0
    summary["by_split"] = finalize_summary_bucket_map(summary["by_split"])
    summary["by_subject"] = finalize_summary_bucket_map(summary["by_subject"])
    return summary


# helper function to initialize summary buckets
def _empty_summary_bucket() -> dict:
    return {
        "attempts": 0,
        "questions": 0,
        "passed": 0,
        "score_sum": 0.0,
        "max_score_sum": 0.0,
        "pass_rate": 0.0,
        "score_rate": 0.0,
        "_question_keys": set(),
    }


# helper function to finalize defaultdict-backed bucket maps
def finalize_summary_bucket_map(bucket_map: dict) -> dict:
    finalized = {}

    for key, bucket in bucket_map.items():
        bucket["questions"] = len(bucket.pop("_question_keys"))
        bucket["pass_rate"] = round(bucket["passed"] / bucket["attempts"], 6) if bucket["attempts"] else 0.0
        bucket["score_rate"] = round(bucket["score_sum"] / bucket["max_score_sum"], 6) if bucket["max_score_sum"] else 0.0
        finalized[key] = bucket

    return finalized


# helper function to save the resumable state atomically
def save_state(output_path: Path, state: dict) -> None:
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temp_path.replace(output_path)


# helper function to create stable question ids
def example_key(example: dict) -> str:
    return f"{example['split']}:{example['task_group_id']}"


# helper function to create stable attempt ids
def format_attempt_key(example_key_value: str, trial_index: int) -> str:
    return f"{example_key_value}:trial-{trial_index}"


# helper function to return a UTC timestamp
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# helper function to slugify model names for filenames
def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


# helper function to resolve the output token field
def resolve_output_token_field(base_url: str) -> str:
    """
    Resolve the output token field for the current endpoint.

    Parameters
    ------------
    base_url: str
        OpenAI-compatible base URL

    Returns
    ------------
    str
    """
    normalized = base_url.rstrip("/").lower()
    if "api.openai.com" in normalized:
        return "max_completion_tokens"
    return "max_tokens"


# helper function to build one chat completion payload
def build_chat_payload(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    extra_body: dict,
) -> dict:
    """
    Build a chat completions payload for one endpoint.

    Parameters
    ------------
    base_url: str
        OpenAI-compatible base URL
    model: str
        Model id
    messages: list[dict]
        Chat messages
    max_tokens: int
        Max output tokens
    temperature: float
        Sampling temperature
    extra_body: dict
        Additional provider-specific request keys

    Returns
    ------------
    dict
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        resolve_output_token_field(base_url): max_tokens,
    }
    payload.update(extra_body)
    return payload


# helper function to generate one chat completion
async def generate_async(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    timeout_seconds: int,
    extra_body: dict,
) -> str:
    """
    Generate one completion from an OpenAI-compatible endpoint.

    Parameters
    ------------
    base_url: str
        OpenAI-compatible base URL
    api_key: str
        API key
    model: str
        Model id
    messages: list[dict]
        Chat messages
    max_tokens: int
        Max output tokens
    temperature: float
        Sampling temperature
    timeout_seconds: int
        Per-request timeout
    extra_body: dict
        Additional provider-specific request keys

    Returns
    ------------
    str
    """
    payload = build_chat_payload(
        base_url=base_url,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=extra_body,
    )

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{base_url.rstrip('/')}/chat/completions"
    response_text, status_code = await curl_json_request_async(
        url,
        method="POST",
        headers=headers,
        body=json.dumps(payload),
        timeout_seconds=timeout_seconds,
    )
    if not 200 <= status_code < 300:
        raise_http_error(url, status_code, response_text)
    response = json.loads(response_text)
    return flatten_message_content(response).strip()


# helper function to raise structured HTTP request errors
def raise_http_error(url: str, status_code: int, body: str) -> None:
    """
    Raise a RuntimeError for a non-2xx HTTP response.

    Parameters
    ------------
    url: str
        Request URL
    status_code: int
        HTTP status code
    body: str
        Response body

    Returns
    ------------
    None
    """
    message = body.strip()
    try:
        decoded = json.loads(body)
        error = decoded.get("error", {})
        if error:
            message = str(error.get("message", "")).strip() or message
    except json.JSONDecodeError:
        pass

    raise RuntimeError(f"HTTP request failed for {url} ({status_code}): {message}")


# helper function to flatten content payloads
def flatten_message_content(response: dict) -> str:
    """
    Flatten content from a chat completions response payload.

    Parameters
    ------------
    response: dict
        Decoded JSON response from an OpenAI-compatible endpoint

    Returns
    ------------
    str
    """
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Model response did not include choices: {json.dumps(response)}")

    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", content))
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif item.get("type") == "output_text" and "text" in item:
                    parts.append(str(item["text"]))
        return "\n".join(part for part in parts if part)
    return str(content)


# helper function to make curl-backed requests
def curl_request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: str | None = None,
    timeout_seconds: int | None = None,
) -> str:
    """
    Run one HTTP request through curl.

    Parameters
    ------------
    url: str
        Request URL
    method: str
        HTTP method
    headers: dict[str, str] | None
        Request headers
    body: str | None
        Optional request body
    timeout_seconds: int | None
        Optional request timeout

    Returns
    ------------
    str
    """
    cmd = ["curl", "-L", "-f", "-sS", "-X", method, url]

    if timeout_seconds is not None:
        cmd.extend(["--max-time", str(timeout_seconds)])

    for key, value in (headers or {}).items():
        cmd.extend(["-H", f"{key}: {value}"])

    if body is not None:
        cmd.extend(["--data-binary", "@-"])

    process = subprocess.run(
        cmd,
        input=body,
        text=True,
        capture_output=True,
        check=False,
    )

    if process.returncode != 0:
        error = process.stderr.strip() or process.stdout.strip() or f"curl exited with code {process.returncode}"
        raise RuntimeError(f"HTTP request failed for {url}: {error}")

    return process.stdout


# helper function to make curl-backed requests while preserving status codes
def curl_json_request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: str | None = None,
    timeout_seconds: int | None = None,
) -> tuple[str, int]:
    """
    Run one HTTP request through curl and return the response body plus status code.

    Parameters
    ------------
    url: str
        Request URL
    method: str
        HTTP method
    headers: dict[str, str] | None
        Request headers
    body: str | None
        Optional request body
    timeout_seconds: int | None
        Optional request timeout

    Returns
    ------------
    tuple[str, int]
    """
    marker = "__HTTP_STATUS__:"
    cmd = ["curl", "-L", "-sS", "-X", method, url, "-w", f"\n{marker}%{{http_code}}"]

    if timeout_seconds is not None:
        cmd.extend(["--max-time", str(timeout_seconds)])

    for key, value in (headers or {}).items():
        cmd.extend(["-H", f"{key}: {value}"])

    if body is not None:
        cmd.extend(["--data-binary", "@-"])

    process = subprocess.run(
        cmd,
        input=body,
        text=True,
        capture_output=True,
        check=False,
    )

    if process.returncode != 0:
        error = process.stderr.strip() or process.stdout.strip() or f"curl exited with code {process.returncode}"
        raise RuntimeError(f"HTTP request failed for {url}: {error}")

    payload = process.stdout
    if marker not in payload:
        raise RuntimeError(f"HTTP response missing status marker for {url}")

    body_text, status_text = payload.rsplit(marker, 1)
    return body_text.rstrip("\n"), int(status_text.strip())


# helper function to make async curl-backed requests while preserving status codes
async def curl_json_request_async(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: str | None = None,
    timeout_seconds: int | None = None,
) -> tuple[str, int]:
    """
    Run one HTTP request through async curl and return the response body plus status code.

    Parameters
    ------------
    url: str
        Request URL
    method: str
        HTTP method
    headers: dict[str, str] | None
        Request headers
    body: str | None
        Optional request body
    timeout_seconds: int | None
        Optional request timeout

    Returns
    ------------
    tuple[str, int]
    """
    marker = "__HTTP_STATUS__:"
    cmd = ["curl", "-L", "-sS", "-X", method, url, "-w", f"\n{marker}%{{http_code}}"]

    if timeout_seconds is not None:
        cmd.extend(["--max-time", str(timeout_seconds)])

    for key, value in (headers or {}).items():
        cmd.extend(["-H", f"{key}: {value}"])

    if body is not None:
        cmd.extend(["--data-binary", "@-"])

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate(body.encode("utf-8") if body is not None else None)

    if process.returncode != 0:
        error = stderr.decode("utf-8", errors="replace").strip() or stdout.decode("utf-8", errors="replace").strip()
        if not error:
            error = f"curl exited with code {process.returncode}"
        raise RuntimeError(f"HTTP request failed for {url}: {error}")

    payload = stdout.decode("utf-8", errors="replace")
    if marker not in payload:
        raise RuntimeError(f"HTTP response missing status marker for {url}")

    body_text, status_text = payload.rsplit(marker, 1)
    return body_text.rstrip("\n"), int(status_text.strip())


# helper function to log one completed attempt
def log_completed_attempt(result: dict, completed_count: int, total_count: int) -> None:
    """
    Log one completed attempt result.

    Parameters
    ------------
    result: dict
        Completed attempt record
    completed_count: int
        Number of completed attempts so far
    total_count: int
        Total attempts in this run

    Returns
    ------------
    None
    """
    logger.info(
        "[%s/%s] completed %s %s %s (trial %s/%s) -> %s",
        completed_count,
        total_count,
        result["split"],
        result["subject"],
        result["task_group_id"],
        result["trial_index"],
        result["trial_count"],
        result["grading"]["grade_display"],
    )


# helper function to cancel pending tasks on failure
async def cancel_pending_tasks(tasks: list[asyncio.Task]) -> None:
    """
    Cancel all pending tasks and wait for them to settle.

    Parameters
    ------------
    tasks: list[asyncio.Task]
        Outstanding asyncio tasks

    Returns
    ------------
    None
    """
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


# helper function to run pending attempts concurrently
async def run_pending_attempts_async(
    *,
    remaining_attempts: list[dict],
    all_attempts_count: int,
    already_recorded_count: int,
    state: dict,
    output_path: Path,
    candidate_model: str,
    candidate_config: dict,
    judge_config: dict,
    max_concurrent: int,
    sleep_seconds: float,
) -> None:
    """
    Run the remaining attempts concurrently and save each completed result incrementally.

    Parameters
    ------------
    remaining_attempts: list[dict]
        Attempts not yet recorded
    all_attempts_count: int
        Total attempts in the full planned run
    already_recorded_count: int
        Attempts already present in state
    state: dict
        Mutable run state
    output_path: Path
        Output path
    candidate_model: str
        Candidate model name
    candidate_config: dict
        Candidate endpoint config
    judge_config: dict
        Judge endpoint config
    max_concurrent: int
        Semaphore limit
    sleep_seconds: float
        Optional pause per attempt

    Returns
    ------------
    None
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        asyncio.create_task(
            process_single_attempt(
                attempt=attempt,
                candidate_model=candidate_model,
                candidate_config=candidate_config,
                judge_config=judge_config,
                semaphore=semaphore,
                sleep_seconds=sleep_seconds,
            )
        )
        for attempt in remaining_attempts
    ]

    completed_count = already_recorded_count
    try:
        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            completed_count += 1
            state["results"].append(result)
            state["summary"] = summarize_results(state["results"])
            state["completed_at"] = utc_now()
            save_state(output_path, state)
            log_completed_attempt(result, completed_count, all_attempts_count)
    except Exception:
        await cancel_pending_tasks(tasks)
        raise


# helper function to run the full evaluation pipeline
async def run_frontierscience_eval_async(args: argparse.Namespace) -> None:
    """
    Run the full FrontierScience evaluation pipeline.

    Parameters
    ------------
    args: argparse.Namespace
        Parsed CLI args

    Returns
    ------------
    None
    """
    candidate_extra_body = parse_json_arg(args.candidate_extra_body, "--candidate-extra-body")
    judge_extra_body = parse_json_arg(args.judge_extra_body, "--judge-extra-body")
    split_order = resolve_split_order(args.split)
    dataset = load_dataset(split_order, refresh=args.refresh_dataset)
    examples = filter_examples(dataset, split_order, args.subjects, args.start_index, args.limit)
    attempts = build_attempts(examples, args)
    output_path = resolve_output_path(args.candidate_model, args.split, args.output)
    state = load_or_init_state(args, output_path)
    processed_ids = get_recorded_attempts(state)

    if args.dry_run:
        print_dry_run(examples, attempts, args, processed_ids, output_path)
        return

    candidate_config = build_candidate_config(args, candidate_extra_body)
    judge_config = build_judge_config(args, judge_extra_body)
    remaining_attempts = [attempt for attempt in attempts if attempt["attempt_key"] not in processed_ids]

    logger.info(
        "FrontierScience plan: %s question(s), %s total attempt(s), %s already recorded, %s remaining.",
        len(examples),
        len(attempts),
        len(attempts) - len(remaining_attempts),
        len(remaining_attempts),
    )

    await run_pending_attempts_async(
        remaining_attempts=remaining_attempts,
        all_attempts_count=len(attempts),
        already_recorded_count=len(attempts) - len(remaining_attempts),
        state=state,
        output_path=output_path,
        candidate_model=args.candidate_model,
        candidate_config=candidate_config,
        judge_config=judge_config,
        max_concurrent=args.max_concurrent,
        sleep_seconds=args.sleep_seconds,
    )

    state["summary"] = summarize_results(state["results"])
    state["completed_at"] = utc_now()
    save_state(output_path, state)

    logger.info("Saved results to %s", output_path)
    print(json.dumps(state["summary"], indent=2))


# helper function to run the async evaluation pipeline from sync entrypoints
def run_frontierscience_eval(args: argparse.Namespace) -> None:
    asyncio.run(run_frontierscience_eval_async(args))


if __name__ == "__main__":
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    validate_args(args)
    run_frontierscience_eval(args)
