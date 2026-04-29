#!/usr/bin/env python3

# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/scienceeval/blob/main/LICENSE

from __future__ import annotations

# stdlib
import asyncio
import json
import logging
import os
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


# init logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# directory constants
REPO_ROOT = Path(__file__).resolve().parents[2]


# endpoint defaults
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_CONCURRENT = 4


# helper function to load repo-local env vars
def load_dotenv(env_path: Path) -> None:
    """
    Load a simple .env file into process environment without overriding values.

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


# helper function to parse json object CLI args
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


# helper function to return a UTC timestamp
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# helper function to slugify model names for filenames
def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


# helper function to save json state atomically
def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


# helper function to load json state
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# helper function to normalize model outputs for exact matching
def normalize_answer(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower()
    text = re.sub(r"^answer\s*:\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n`\"'.,;:")
    return text


# helper function to compute macro-F1
def macro_f1(golds: list[str], predictions: list[str], labels: list[str]) -> float:
    """
    Compute unweighted macro-F1 over known labels.

    Parameters
    ------------
    golds: list[str]
        Gold labels
    predictions: list[str]
        Predicted labels
    labels: list[str]
        Label set to score

    Returns
    ------------
    float
    """
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


# helper function to summarize exact-label results
def summarize_classification_results(results: list[dict], labels: list[str]) -> dict:
    golds = [result["gold"] for result in results]
    predictions = [result["prediction"] for result in results]
    correct = sum(result["correct"] for result in results)
    return {
        "examples": len(results),
        "correct": int(correct),
        "accuracy": round(correct / len(results), 6) if results else 0.0,
        "macro_f1": round(macro_f1(golds, predictions, labels), 6) if labels else 0.0,
        "label_counts": dict(Counter(golds)),
        "prediction_counts": dict(Counter(predictions)),
    }


# helper function to resolve the output token field
def resolve_output_token_field(base_url: str) -> str:
    normalized = base_url.rstrip("/").lower()
    if "api.openai.com" in normalized:
        return "max_completion_tokens"
    return "max_tokens"


# helper function to build a chat completion payload
def build_chat_payload(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    extra_body: dict,
) -> dict:
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

    response_text, status_code = await curl_json_request_async(
        f"{base_url.rstrip('/')}/chat/completions",
        method="POST",
        headers=headers,
        body=json.dumps(payload),
        timeout_seconds=timeout_seconds,
    )
    if not 200 <= status_code < 300:
        raise_http_error(base_url, status_code, response_text)
    return flatten_message_content(json.loads(response_text)).strip()


# helper function to raise HTTP errors
def raise_http_error(base_url: str, status_code: int, body: str) -> None:
    message = body.strip()
    try:
        decoded = json.loads(body)
        error = decoded.get("error", {})
        if error:
            message = str(error.get("message", "")).strip() or message
    except json.JSONDecodeError:
        pass
    raise RuntimeError(f"HTTP request failed for {base_url} ({status_code}): {message}")


# helper function to flatten chat completion content
def flatten_message_content(response: dict) -> str:
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
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(part for part in parts if part)
    return str(content)


# helper function to run async curl-backed requests
async def curl_json_request_async(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: str | None = None,
    timeout_seconds: int | None = None,
) -> tuple[str, int]:
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


# helper function to run sync shell commands
def run_command(cmd: list[str], dry_run: bool) -> None:
    logger.info("%s", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)

