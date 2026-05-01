#!/usr/bin/env python3

# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/scienceeval/blob/main/LICENSE

from __future__ import annotations

# stdlib
import argparse
import sys
from pathlib import Path

# local
from fossil_m_common import REPO_ROOT, load_dotenv, logger, run_command, slugify


# directory constants
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results" / "fossil-m"
FRONTIERSCIENCE_SCRIPT = REPO_ROOT / "scripts" / "evaluate_frontierscience.py"
LABEL_SCRIPT = SCRIPT_DIR / "evaluate_label_fossils.py"


# benchmark constants
DEFAULT_BENCHMARKS = (
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
LABEL_BENCHMARKS = ("gpqa", "pubmedqa", "bioasq", "biored", "scierc", "mmlu", "simpleqa", "sciriff")


# helper function to parse CLI args
def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the Fossils-M suite runner.

    Returns
    ------------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Run the local Fossils-M benchmark suite for one model.")
    parser.add_argument("--model", required=True, help="Candidate model id served by the endpoint.")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible candidate base URL.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable holding endpoint key.")
    parser.add_argument("--judge-model", help="Judge model id for FrontierScience.")
    parser.add_argument("--judge-base-url", help="OpenAI-compatible judge base URL.")
    parser.add_argument("--judge-api-key-env", default="OPENAI_API_KEY", help="Environment variable for judge key.")
    parser.add_argument("--benchmarks", nargs="+", default=list(DEFAULT_BENCHMARKS), help="Benchmarks to run.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Root directory for result JSON files.")
    parser.add_argument("--limit-per-benchmark", type=int, help="Smoke-test limit passed to each benchmark.")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Concurrent requests per benchmark.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max output tokens for label fossils.")
    parser.add_argument("--timeout-seconds", type=int, default=600, help="Per-request timeout.")
    parser.add_argument("--extra-body", help="JSON object merged into label fossil requests.")
    parser.add_argument("--gpqa-split", default="diamond", choices=("diamond", "main", "experts", "extended"))
    parser.add_argument("--resume", action="store_true", help="Resume benchmark output files when present.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


# helper function to validate args
def validate_args(args: argparse.Namespace) -> None:
    requested = set(args.benchmarks)
    unsupported = requested - set(DEFAULT_BENCHMARKS)
    if unsupported:
        raise SystemExit(f"Unsupported benchmark(s): {sorted(unsupported)}")
    if "frontierscience" in requested and not args.judge_model and not args.dry_run:
        raise SystemExit("--judge-model is required when running FrontierScience.")
    if args.limit_per_benchmark is not None and args.limit_per_benchmark <= 0:
        raise SystemExit("--limit-per-benchmark must be > 0 when provided.")


# helper function to resolve one model output directory
def model_output_dir(args: argparse.Namespace) -> Path:
    path = args.results_dir / slugify(args.model)
    path.mkdir(parents=True, exist_ok=True)
    return path


# helper function to build FrontierScience command
def build_frontierscience_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(FRONTIERSCIENCE_SCRIPT),
        "--split",
        "all",
        "--candidate-model",
        args.model,
        "--candidate-base-url",
        args.base_url,
        "--candidate-api-key-env",
        args.api_key_env,
        "--judge-model",
        args.judge_model or "JUDGE_MODEL_REQUIRED",
        "--judge-base-url",
        args.judge_base_url or args.base_url,
        "--judge-api-key-env",
        args.judge_api_key_env,
        "--max-concurrent",
        str(args.max_concurrent),
        "--output",
        str(output_dir / "frontierscience.json"),
    ]
    if args.limit_per_benchmark is not None:
        cmd.extend(["--limit", str(args.limit_per_benchmark)])
    if args.resume:
        cmd.append("--resume")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


# helper function to build label benchmark command
def build_label_command(args: argparse.Namespace, benchmark: str, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(LABEL_SCRIPT),
        "--benchmark",
        benchmark,
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--api-key-env",
        args.api_key_env,
        "--max-concurrent",
        str(args.max_concurrent),
        "--max-tokens",
        str(args.max_tokens),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--output",
        str(output_dir / f"{benchmark}.json"),
    ]
    if args.limit_per_benchmark is not None:
        cmd.extend(["--limit", str(args.limit_per_benchmark)])
    if args.extra_body:
        cmd.extend(["--extra-body", args.extra_body])
    if benchmark == "gpqa":
        cmd.extend(["--gpqa-split", args.gpqa_split])
    if args.resume:
        cmd.append("--resume")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


# helper function to build all benchmark commands
def build_commands(args: argparse.Namespace) -> list[list[str]]:
    output_dir = model_output_dir(args)
    commands = []
    for benchmark in args.benchmarks:
        if benchmark == "frontierscience":
            commands.append(build_frontierscience_command(args, output_dir))
        elif benchmark in LABEL_BENCHMARKS:
            commands.append(build_label_command(args, benchmark, output_dir))
    return commands


# orchestration function
def run_fossil_suite(args: argparse.Namespace) -> None:
    for cmd in build_commands(args):
        run_command(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    load_dotenv(REPO_ROOT / ".env")
    parsed_args = parse_args()
    validate_args(parsed_args)
    run_fossil_suite(parsed_args)
