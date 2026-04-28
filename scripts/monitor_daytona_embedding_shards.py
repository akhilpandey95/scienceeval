#!/usr/bin/env python3

# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/scienceeval/blob/main/LICENSE

from __future__ import annotations

# stdlib
import argparse
import json
import os
import re
from pathlib import Path


# directory constants
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = Path("/tmp/scienceeval-daytona-shards.json")


# monitor constants
DEFAULT_SHARDS = (
    {
        "shard": 0,
        "sandbox_id": "e15f47e5-7660-4c24-8444-7501dcae1e52",
        "session_id": "scienceeval-shard-0",
        "cmd_id": "97507228-c1d1-4da2-9ed5-081c4329ceb9",
    },
    {
        "shard": 1,
        "sandbox_id": "5abd9cc1-8074-453a-8950-dca8b5395b52",
        "session_id": "scienceeval-shard-1",
        "cmd_id": "84fb3b64-10af-432a-aa03-c45221a0a910",
    },
    {
        "shard": 2,
        "sandbox_id": "e41433f6-5fcf-42c6-8234-20525d5926d6",
        "session_id": "scienceeval-shard-2",
        "cmd_id": "99fc7d9c-709a-4061-9447-cc67fb8418af",
    },
    {
        "shard": 3,
        "sandbox_id": "93c66317-9119-49e1-9605-8cbe7c6e840e",
        "session_id": "scienceeval-shard-3",
        "cmd_id": "23fe5960-399e-4b21-832e-881f1f44e636",
    },
)


# helper function to parse CLI args
def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for Daytona shard monitoring.

    Returns
    ------------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Monitor Daytona benchmark embedding shards.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH, help="Shard registry JSON path.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Local repo root containing .env.")
    return parser.parse_args()


# helper function to load env values without printing secrets
def load_env_file(repo_root: Path) -> None:
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# helper function to load shard registry
def load_shards(registry_path: Path) -> list[dict]:
    """
    Load shard metadata from disk or built-in defaults.

    Parameters
    ------------
    registry_path: Path
        JSON registry path

    Returns
    ------------
    list[dict]
    """
    if registry_path.exists():
        return json.loads(registry_path.read_text(encoding="utf-8"))
    return list(DEFAULT_SHARDS)


# helper function to extract progress from remote status text
def parse_progress(text: str) -> tuple[int | None, str]:
    chunks = [int(match.group(1)) for match in re.finditer(r"Embedded (\d+) chunk", text)]
    queued = [match.group(0) for match in re.finditer(r"Queued embeddings for \d+/\d+ record\(s\)", text)]
    if chunks:
        return chunks[-1], queued[-1] if queued else ""
    return None, queued[-1] if queued else ""


# helper function to extract completed checkpoint records
def parse_completed(text: str) -> str:
    match = re.search(r"CHECKPOINT_COMPLETED\s+(\d+/\d+)", text)
    return match.group(1) if match else "-"


# helper function to query one remote shard
def query_shard(client: object, shard: dict) -> dict:
    sandbox = client.get(shard["sandbox_id"])
    shard_index = int(shard["shard"])
    exit_code = None
    try:
        command = sandbox.process.get_session_command(shard["session_id"], shard["cmd_id"])
        exit_code = getattr(command, "exit_code", None)
    except Exception:
        exit_code = None

    remote_command = f"""
out=/workspace/scienceeval/output/full-embedding-shard-{shard_index}
echo PS
ps -eo pid,stat,pcpu,pmem,rss,time,cmd | grep 'shard-index {shard_index}' | grep -v grep || true
echo LOG
tail -30 "$out/run.log" 2>/dev/null || true
echo CHECKPOINT
python - <<'PY2' 2>/dev/null || true
from pathlib import Path
import numpy as np
path = Path("/workspace/scienceeval/output/full-embedding-shard-{shard_index}/completed_mask.npy")
if path.exists():
    mask = np.load(path, mmap_mode="r")
    print(f"CHECKPOINT_COMPLETED {{int(mask.sum())}}/{{len(mask)}}")
PY2
echo FILES
find "$out" -maxdepth 1 -type f -printf '%f %s\\n' 2>/dev/null | sort || true
echo MEMORY_EVENTS
cat /sys/fs/cgroup/memory.events 2>/dev/null || true
echo DISK
df -h /workspace | tail -1
"""
    response = sandbox.process.exec(remote_command, timeout=120)
    result = getattr(response, "result", "") or ""
    chunks, queued = parse_progress(result)
    completed = parse_completed(result)
    running = f"shard-index {shard_index}" in result
    done = "record_embeddings.npy" in result and "metadata.parquet" in result
    oom_kill = re.search(r"^oom_kill\s+([1-9]\d*)", result, re.MULTILINE) is not None
    failed = exit_code not in (None, 0) or oom_kill or any(marker in result for marker in ("Traceback", "Killed", "Error"))
    rss_match = re.search(r"^\s*\d+\s+\S+\s+\S+\s+\S+\s+(\d+)\s+\S+\s+python -u", result, re.MULTILINE)
    rss_mb = int(rss_match.group(1)) / 1024 if rss_match else None
    return {
        "shard": shard_index,
        "sandbox_id": shard["sandbox_id"],
        "state": "done" if done else "running" if running else "oom" if oom_kill else "failed" if failed else "unknown",
        "exit_code": exit_code,
        "chunks": chunks,
        "completed": completed,
        "queued": queued,
        "rss_mb": rss_mb,
        "raw": result,
    }


# helper function to print compact status rows
def print_status(rows: list[dict]) -> None:
    print("shard  state    exit  chunks  completed    rss_mb  sandbox")
    for row in rows:
        chunks = "-" if row["chunks"] is None else str(row["chunks"])
        rss = "-" if row["rss_mb"] is None else f"{row['rss_mb']:.0f}"
        exit_code = "-" if row["exit_code"] is None else str(row["exit_code"])
        print(
            f"{row['shard']:<5}  {row['state']:<7}  {exit_code:<4}  {chunks:<6}  "
            f"{row['completed']:<11}  {rss:<6}  {row['sandbox_id']}"
        )
        if row["queued"]:
            print(f"       {row['queued']}")


# orchestration function
def monitor_daytona_embedding_shards() -> None:
    args = parse_args()
    load_env_file(args.repo_root)
    try:
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
        from daytona_sdk import Daytona
    except ImportError as exc:
        raise SystemExit("Install daytona-sdk/certifi or run with /tmp/scienceeval-daytona-client/bin/python") from exc

    client = Daytona()
    rows = [query_shard(client, shard) for shard in load_shards(args.registry)]
    print_status(rows)


if __name__ == "__main__":
    monitor_daytona_embedding_shards()
