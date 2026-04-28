#!/usr/bin/env python3

# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/scienceeval/blob/main/LICENSE

from __future__ import annotations

# stdlib
import argparse
import csv
import json
import logging
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# data
import numpy as np
import onnxruntime as ort
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors
from tokenizers import Tokenizer

# viz
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt


# init logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# directory constants
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output" / "full-embedding-map"
DEFAULT_MODEL_DIR = Path(
    "/workspace/scienceeval/models/pplx-embed-v1-0.6b"
    if Path("/workspace/scienceeval/models/pplx-embed-v1-0.6b").exists()
    else "/Users/akhilpandey/code/models/pplx-embed-v1-0.6b"
)


# embedding constants
RANDOM_SEED = 7207
ONNX_MODEL_NAME = "model_q4.onnx"
EMBEDDING_OUTPUT_NAME = "pooler_output_int8"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_TOKENS = 2048
DEFAULT_CHUNK_OVERLAP = 128
DEFAULT_METRIC_SAMPLE_SIZE = 10_000
DEFAULT_ONNX_THREADS = 16
DEFAULT_TSNE_PERPLEXITY = 50
DEFAULT_TSNE_MAX_ITER = 1500
CHECKPOINT_METADATA_NAME = "metadata_checkpoint.jsonl"
CHECKPOINT_DONE_NAME = "completed_mask.npy"
CHECKPOINT_SUMS_NAME = "embedding_sums.npy"
CHECKPOINT_WEIGHTS_NAME = "embedding_weights.npy"
CHECKPOINT_LOG_INTERVAL = 500


# plot constants
PAPER_DARK = "#050506"
MUTED_LIGHT = "#aaa59b"
GRID_DARK = "#2f2f32"
LABEL_HALO = [pe.withStroke(linewidth=4.0, foreground=PAPER_DARK, alpha=0.96)]
DENSITY_GRID_SIZE = 230
DENSITY_SMOOTHING = 2.0
DENSITY_LEVELS = (0.18, 0.38, 0.62, 1.01)
PLOT_ASPECT_RATIO = 11.8 / 7.4
PLOT_PADDING_RATIO = 0.18
HOMEPAGE_DPI = 300
HOMEPAGE_SAMPLE_PER_BENCHMARK = 700
HOMEPAGE_BOUND_QUANTILES = (0.006, 0.994)
HOMEPAGE_PADDING_RATIO = 0.07
BENCHMARK_COLORS = {
    "BioASQ": "#fb7185",
    "BioRED": "#f59e0b",
    "FrontierScience": "#38bdf8",
    "GPQA": "#a78bfa",
    "MMLU": "#f8fafc",
    "PubMedQA": "#34d399",
    "SciERC": "#f472b6",
    "SciRIFF": "#60a5fa",
    "SimpleQA": "#facc15",
}
BENCHMARK_ORDER = tuple(BENCHMARK_COLORS)
LABEL_OFFSETS = {
    "BioASQ": (-0.02, 0.08),
    "BioRED": (0.05, -0.03),
    "FrontierScience": (0.04, 0.05),
    "GPQA": (-0.05, 0.05),
    "MMLU": (0.04, 0.04),
    "PubMedQA": (0.02, -0.07),
    "SciERC": (0.01, -0.07),
    "SciRIFF": (0.01, -0.08),
    "SimpleQA": (-0.06, 0.03),
}
HOMEPAGE_LABEL_POSITIONS = {
    "BioASQ": (-0.24, 0.77),
    "BioRED": (0.27, 0.58),
    "FrontierScience": (0.28, -0.04),
    "GPQA": (-0.18, -0.04),
    "MMLU": (-0.55, 0.25),
    "PubMedQA": (0.32, 0.23),
    "SciERC": (-0.31, -0.68),
    "SciRIFF": (0.56, -0.40),
    "SimpleQA": (-0.58, -0.32),
}
HOMEPAGE_LABEL_SIZES = {
    "FrontierScience": 11.7,
    "PubMedQA": 12.2,
    "SimpleQA": 13.8,
}


# helper function to parse CLI args
def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the full benchmark embedding map.

    Returns
    ------------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate full-corpus benchmark-fossil embeddings and projection figures."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Local pplx-embed model directory.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for cached outputs.")
    parser.add_argument(
        "--limit-per-benchmark",
        type=int,
        default=0,
        help="Optional deterministic debug sample size. Use 0 for all loaded benchmark items.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="ONNX embedding batch size.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum token window for each embedded chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Token overlap between adjacent chunks for long inputs.",
    )
    parser.add_argument(
        "--projection-methods",
        default="umap,pacmap,pca",
        help="Comma-separated projection methods to run: umap,pacmap,pca,tsne.",
    )
    parser.add_argument(
        "--metric-sample-size",
        type=int,
        default=DEFAULT_METRIC_SAMPLE_SIZE,
        help="Maximum point count used for projection quality metrics.",
    )
    parser.add_argument("--skip-embeddings", action="store_true", help="Reuse cached embeddings if present.")
    parser.add_argument("--skip-projections", action="store_true", help="Reuse cached projections if present.")
    parser.add_argument("--only-embeddings", action="store_true", help="Stop after writing embeddings and metadata.")
    parser.add_argument(
        "--homepage-image-path",
        type=Path,
        default=None,
        help="Optional polished homepage PNG path to render from the t-SNE projection.",
    )
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based record shard index to embed.")
    parser.add_argument("--shard-count", type=int, default=1, help="Total record shard count.")
    return parser.parse_args()


# helper function to normalize benchmark text
def clean_text(value: object) -> str:
    """
    Collapse markup and whitespace into one embedding-ready string.

    Parameters
    ------------
    value: object
        Raw text-like value

    Returns
    ------------
    str
    """
    text = "" if value is None else str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("[[", "").replace("]]", "").replace("<<", "").replace(">>", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# helper function to load csv benchmark records
def load_csv_records(path: Path) -> list[dict]:
    """
    Load a CSV file into dict rows.

    Parameters
    ------------
    path: Path
        CSV path

    Returns
    ------------
    list[dict]
    """
    with path.open(newline="", encoding="utf-8", errors="replace") as file:
        return list(csv.DictReader(file))


# helper function to load jsonl rows
def load_jsonl_records(path: Path) -> list[dict]:
    """
    Load a JSONL file into dict rows.

    Parameters
    ------------
    path: Path
        JSONL path

    Returns
    ------------
    list[dict]
    """
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# helper function to load parquet rows
def load_parquet_records(path: Path, columns: list[str] | None = None) -> list[dict]:
    """
    Load a parquet file into Python dict rows.

    Parameters
    ------------
    path: Path
        Parquet path
    columns: list[str] | None
        Optional column subset

    Returns
    ------------
    list[dict]
    """
    return pq.read_table(path, columns=columns).to_pylist()


# helper function to parse PubTator title and abstract blocks
def parse_pubtator_abstracts(path: Path) -> list[str]:
    texts = []
    current_title = ""
    current_abstract = ""

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            if current_title or current_abstract:
                texts.append(clean_text(f"{current_title} {current_abstract}"))
            current_title = ""
            current_abstract = ""
            continue
        if "|t|" in line:
            current_title = line.split("|t|", 1)[1]
        elif "|a|" in line:
            current_abstract = line.split("|a|", 1)[1]

    if current_title or current_abstract:
        texts.append(clean_text(f"{current_title} {current_abstract}"))
    return texts


# helper function to make a stable benchmark record
def make_record(benchmark: str, local_index: int, text: str, source: str) -> dict:
    return {
        "record_id": f"{benchmark}:{local_index:07d}",
        "benchmark": benchmark,
        "source": source,
        "text": clean_text(text),
    }


# helper function to collect FrontierScience examples
def load_frontierscience_records() -> list[dict]:
    records = []
    for split in ("olympiad", "research"):
        path = DATA_DIR / "frontierscience" / split / "test.jsonl"
        for row in load_jsonl_records(path):
            text = f"{row.get('subject', '')} problem. {row.get('problem', '')}"
            records.append(make_record("FrontierScience", len(records), text, split))
    return records


# helper function to collect GPQA examples
def load_gpqa_records() -> list[dict]:
    path = DATA_DIR / "gpqa" / "extracted" / "dataset" / "gpqa_main.csv"
    records = []
    for row in load_csv_records(path):
        answer_options = [
            row.get("Correct Answer", ""),
            row.get("Incorrect Answer 1", ""),
            row.get("Incorrect Answer 2", ""),
            row.get("Incorrect Answer 3", ""),
        ]
        text = " ".join(
            [
                row.get("High-level domain", ""),
                row.get("Subdomain", ""),
                row.get("Question", ""),
                "Answer options:",
                " ; ".join(option for option in answer_options if option),
            ]
        )
        records.append(make_record("GPQA", len(records), text, "gpqa_main"))
    return records


# helper function to collect PubMedQA examples
def load_pubmedqa_records() -> list[dict]:
    path = DATA_DIR / "pubmedqa" / "pqa_labeled" / "train-00000-of-00001.parquet"
    records = []
    for row in load_parquet_records(path, columns=["question", "context"]):
        context = row.get("context") or {}
        contexts = " ".join(context.get("contexts") or [])
        text = f"{row.get('question', '')} {contexts}"
        records.append(make_record("PubMedQA", len(records), text, "pqa_labeled"))
    return records


# helper function to collect BioASQ examples
def load_bioasq_records() -> list[dict]:
    records = []
    for split_path in (DATA_DIR / "bioasq" / "train_bio.csv", DATA_DIR / "bioasq" / "valid_bio.csv"):
        source = split_path.stem
        for row in load_csv_records(split_path):
            text = f"{row.get('question', '')} {row.get('text', '')}"
            records.append(make_record("BioASQ", len(records), text, source))
    return records


# helper function to collect BioRED examples
def load_biored_records() -> list[dict]:
    records = []
    for split in ("Train", "Dev", "Test"):
        path = DATA_DIR / "biored" / "extracted" / "BioRED" / f"{split}.PubTator"
        for text in parse_pubtator_abstracts(path):
            records.append(make_record("BioRED", len(records), text, split.lower()))
    return records


# helper function to collect SciERC examples
def load_scierc_records() -> list[dict]:
    records = []
    for split in ("train", "dev", "test"):
        path = DATA_DIR / "scierc" / f"{split}.jsonl"
        for row in load_jsonl_records(path):
            text = row.get("text", "")
            records.append(make_record("SciERC", len(records), text, split))
    return records


# helper function to collect MMLU examples
def load_mmlu_records() -> list[dict]:
    path = DATA_DIR / "mmlu" / "all" / "test-00000-of-00001.parquet"
    records = []
    for row in load_parquet_records(path, columns=["question", "subject", "choices"]):
        choices = " ; ".join(row.get("choices") or [])
        text = f"{row.get('subject', '')}. {row.get('question', '')} Answer options: {choices}"
        records.append(make_record("MMLU", len(records), text, "all/test"))
    return records


# helper function to collect SimpleQA examples
def load_simpleqa_records() -> list[dict]:
    path = DATA_DIR / "simpleqa" / "simple_qa_test_set.csv"
    records = []
    for row in load_csv_records(path):
        text = f"{row.get('metadata', '')} {row.get('problem', '')}"
        records.append(make_record("SimpleQA", len(records), text, "simple_qa_test_set"))
    return records


# helper function to collect SciRIFF examples
def load_sciriff_records() -> list[dict]:
    path = DATA_DIR / "sciriff" / "4096" / "test-00000-of-00001.parquet"
    records = []
    for row in load_parquet_records(path, columns=["input", "metadata"]):
        metadata = row.get("metadata") or {}
        text = " ".join(
            [
                metadata.get("task_family", ""),
                metadata.get("source_type", ""),
                " ".join(metadata.get("domains") or []),
                row.get("input", ""),
            ]
        )
        records.append(make_record("SciRIFF", len(records), text, "4096/test"))
    return records


# helper function to load all benchmark records
def load_benchmark_records(limit_per_benchmark: int) -> list[dict]:
    """
    Load benchmark records without storing answer explanations in outputs.

    Parameters
    ------------
    limit_per_benchmark: int
        Optional deterministic debug limit per benchmark

    Returns
    ------------
    list[dict]
    """
    loader_map = {
        "BioASQ": load_bioasq_records,
        "BioRED": load_biored_records,
        "FrontierScience": load_frontierscience_records,
        "GPQA": load_gpqa_records,
        "MMLU": load_mmlu_records,
        "PubMedQA": load_pubmedqa_records,
        "SciERC": load_scierc_records,
        "SciRIFF": load_sciriff_records,
        "SimpleQA": load_simpleqa_records,
    }

    records = []
    for benchmark in BENCHMARK_ORDER:
        benchmark_records = [record for record in loader_map[benchmark]() if record.get("text")]
        if limit_per_benchmark > 0:
            rng = random.Random(f"{RANDOM_SEED}:{benchmark}")
            rng.shuffle(benchmark_records)
            benchmark_records = benchmark_records[:limit_per_benchmark]
        logger.info("Loaded %s record(s) for %s", len(benchmark_records), benchmark)
        records.extend(benchmark_records)
    return records


# helper function to load the tokenizer for raw counts
def load_raw_tokenizer(model_dir: Path) -> Tokenizer:
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing tokenizer file: {tokenizer_path}")
    return Tokenizer.from_file(str(tokenizer_path))


# helper function to load the tokenizer for ONNX batches
def load_batch_tokenizer(model_dir: Path, max_tokens: int) -> Tokenizer:
    tokenizer = load_raw_tokenizer(model_dir)
    tokenizer_config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    pad_token = tokenizer_config.get("pad_token") or "<|endoftext|>"
    pad_id = tokenizer.token_to_id(pad_token)
    if pad_id is None:
        raise RuntimeError(f"Could not resolve pad token id for {pad_token!r}.")

    tokenizer.enable_truncation(max_length=max_tokens)
    tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)
    return tokenizer


# helper function to split long texts by tokenizer windows
def build_text_chunks(raw_tokenizer: Tokenizer, text: str, max_tokens: int, chunk_overlap: int) -> tuple[list[dict], int]:
    """
    Split one benchmark text into overlapping token windows.

    Parameters
    ------------
    raw_tokenizer: Tokenizer
        Tokenizer without truncation
    text: str
        Benchmark item text
    max_tokens: int
        Maximum tokens per embedded chunk
    chunk_overlap: int
        Overlap between adjacent chunks

    Returns
    ------------
    tuple[list[dict], int]
    """
    token_ids = raw_tokenizer.encode(text).ids
    token_count = len(token_ids)
    if token_count <= max_tokens:
        return [{"text": text, "weight": max(1, token_count)}], token_count

    stride = max(1, max_tokens - chunk_overlap)
    chunks = []
    for start in range(0, token_count, stride):
        end = min(start + max_tokens, token_count)
        chunk_ids = token_ids[start:end]
        chunk_text = raw_tokenizer.decode(chunk_ids, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append({"text": chunk_text, "weight": max(1, end - start)})
        if end >= token_count:
            break
    return chunks, token_count


# helper function to normalize matrix rows
def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-8)


# helper function to create a bounded ONNX CPU session
def create_onnx_session(model_dir: Path) -> ort.InferenceSession:
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = int(os.environ.get("ONNX_INTRA_OP_THREADS", DEFAULT_ONNX_THREADS))
    session_options.inter_op_num_threads = int(os.environ.get("ONNX_INTER_OP_THREADS", "1"))
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_dir / "onnx" / ONNX_MODEL_NAME),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


# helper function to embed texts with the ONNX model
def embed_batch(session: ort.InferenceSession, tokenizer: Tokenizer, output_names: list[str], texts: list[str]) -> np.ndarray:
    encoded = tokenizer.encode_batch(texts)
    input_ids = np.array([item.ids for item in encoded], dtype=np.int64)
    attention_mask = np.array([item.attention_mask for item in encoded], dtype=np.int64)
    outputs = session.run(
        output_names,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
    )
    output_index = output_names.index(EMBEDDING_OUTPUT_NAME)
    return normalize_rows(outputs[output_index].astype(np.float32))


# helper function to write metadata without benchmark raw text
def write_metadata(output_dir: Path, rows: list[dict]) -> None:
    scrubbed = [
        {
            "record_id": row["record_id"],
            "benchmark": row["benchmark"],
            "source": row["source"],
            "token_count": row["token_count"],
            "chunk_count": row["chunk_count"],
        }
        for row in rows
    ]
    table = pa.Table.from_pylist(scrubbed)
    pq.write_table(table, output_dir / "metadata.parquet")
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as file:
        for row in scrubbed:
            file.write(json.dumps(row, sort_keys=True) + "\n")


# helper function to write token audit summaries
def write_token_audit(output_dir: Path, rows: list[dict], max_tokens: int) -> None:
    summaries = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["benchmark"]].append(row)

    for benchmark in BENCHMARK_ORDER:
        values = np.array([row["token_count"] for row in grouped[benchmark]], dtype=np.float64)
        chunks = np.array([row["chunk_count"] for row in grouped[benchmark]], dtype=np.float64)
        if len(values) == 0:
            continue
        summaries.append(
            {
                "benchmark": benchmark,
                "records": int(len(values)),
                "chunks": int(chunks.sum()),
                "token_p50": float(np.percentile(values, 50)),
                "token_p90": float(np.percentile(values, 90)),
                "token_p95": float(np.percentile(values, 95)),
                "token_p99": float(np.percentile(values, 99)),
                "token_max": int(values.max()),
                "over_max_tokens": int((values > max_tokens).sum()),
            }
        )

    with (output_dir / "token_audit.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (output_dir / "token_audit.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


# helper function to load completed metadata checkpoints
def load_metadata_checkpoint(output_dir: Path) -> dict[str, dict]:
    """
    Load completed record metadata from a JSONL checkpoint.

    Parameters
    ------------
    output_dir: Path
        Output directory

    Returns
    ------------
    dict[str, dict]
    """
    path = output_dir / CHECKPOINT_METADATA_NAME
    if not path.exists():
        return {}

    rows = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["record_id"]] = row
    return rows


# helper function to apply completed metadata to records
def apply_metadata_checkpoint(records: list[dict], checkpoint_rows: dict[str, dict]) -> None:
    for record in records:
        checkpoint_row = checkpoint_rows.get(record["record_id"])
        if not checkpoint_row:
            continue
        record["token_count"] = int(checkpoint_row["token_count"])
        record["chunk_count"] = int(checkpoint_row["chunk_count"])


# helper function to append completed metadata to the checkpoint
def append_metadata_checkpoint(output_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path = output_dir / CHECKPOINT_METADATA_NAME
    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


# helper function to open or create checkpoint arrays
def open_checkpoint_arrays(output_dir: Path, record_count: int, embedding_dim: int) -> tuple[np.memmap, np.memmap, np.memmap]:
    """
    Open resumable embedding checkpoint arrays.

    Parameters
    ------------
    output_dir: Path
        Output directory
    record_count: int
        Number of records in the shard
    embedding_dim: int
        Embedding dimension

    Returns
    ------------
    tuple[np.memmap, np.memmap, np.memmap]
    """
    sums_path = output_dir / CHECKPOINT_SUMS_NAME
    weights_path = output_dir / CHECKPOINT_WEIGHTS_NAME
    done_path = output_dir / CHECKPOINT_DONE_NAME
    if sums_path.exists():
        sums = np.load(sums_path, mmap_mode="r+")
        weights = np.load(weights_path, mmap_mode="r+")
        done = np.load(done_path, mmap_mode="r+")
        if sums.shape != (record_count, embedding_dim):
            raise ValueError(f"Checkpoint shape mismatch: {sums.shape} != {(record_count, embedding_dim)}")
        return sums, weights, done

    sums = np.lib.format.open_memmap(sums_path, mode="w+", dtype=np.float32, shape=(record_count, embedding_dim))
    weights = np.lib.format.open_memmap(weights_path, mode="w+", dtype=np.float32, shape=(record_count,))
    done = np.lib.format.open_memmap(done_path, mode="w+", dtype=np.bool_, shape=(record_count,))
    sums[:] = 0
    weights[:] = 0
    done[:] = False
    sums.flush()
    weights.flush()
    done.flush()
    return sums, weights, done


# helper function to finalize checkpoint arrays into normalized embeddings
def finalize_checkpoint_embeddings(
    output_dir: Path,
    embedding_sums: np.ndarray,
    embedding_weights: np.ndarray,
    done_mask: np.ndarray,
) -> Path:
    if not bool(np.all(done_mask)):
        missing = int((~done_mask).sum())
        raise RuntimeError(f"Cannot finalize embeddings; {missing} record(s) are incomplete.")

    embeddings_path = output_dir / "record_embeddings.npy"
    embeddings = np.lib.format.open_memmap(
        embeddings_path,
        mode="w+",
        dtype=np.float32,
        shape=embedding_sums.shape,
    )
    chunk_size = 2048
    for start in range(0, len(embedding_sums), chunk_size):
        end = min(start + chunk_size, len(embedding_sums))
        batch = embedding_sums[start:end] / np.maximum(embedding_weights[start:end, None], 1e-8)
        embeddings[start:end] = normalize_rows(batch.astype(np.float32))
    embeddings.flush()
    return embeddings_path


# helper function to generate and cache full-record embeddings
def generate_embeddings(
    records: list[dict],
    model_dir: Path,
    output_dir: Path,
    batch_size: int,
    max_tokens: int,
    chunk_overlap: int,
) -> Path:
    """
    Generate chunk-aggregated record embeddings with bounded memory.

    Parameters
    ------------
    records: list[dict]
        Benchmark records with raw text
    model_dir: Path
        Local model directory
    output_dir: Path
        Output directory
    batch_size: int
        ONNX batch size
    max_tokens: int
        Maximum tokens per chunk
    chunk_overlap: int
        Token overlap for long records

    Returns
    ------------
    Path
    """
    embeddings_path = output_dir / "record_embeddings.npy"
    metadata_path = output_dir / "metadata.parquet"
    if embeddings_path.exists() and metadata_path.exists():
        logger.info("Embedding output already exists at %s", embeddings_path)
        return embeddings_path

    raw_tokenizer = load_raw_tokenizer(model_dir)
    batch_tokenizer = load_batch_tokenizer(model_dir, max_tokens)
    session = create_onnx_session(model_dir)
    output_names = [output.name for output in session.get_outputs()]
    if EMBEDDING_OUTPUT_NAME not in output_names:
        raise RuntimeError(f"Expected ONNX output {EMBEDDING_OUTPUT_NAME!r}, got {output_names}.")

    checkpoint_rows = load_metadata_checkpoint(output_dir)
    apply_metadata_checkpoint(records, checkpoint_rows)
    embedding_sums = None
    embedding_weights = None
    done_mask = None
    if (output_dir / CHECKPOINT_SUMS_NAME).exists():
        existing_sums = np.load(output_dir / CHECKPOINT_SUMS_NAME, mmap_mode="r")
        embedding_sums, embedding_weights, done_mask = open_checkpoint_arrays(
            output_dir,
            len(records),
            int(existing_sums.shape[1]),
        )
        logger.info("Resuming from checkpoint with %s/%s completed record(s)", int(done_mask.sum()), len(records))

    chunk_texts = []
    chunk_records = []
    chunk_weights = []
    pending_chunk_counts = defaultdict(int)
    pending_token_counts = {}
    completed_rows = []
    processed_chunks = 0
    next_chunk_log = CHECKPOINT_LOG_INTERVAL

    def flush_batch() -> None:
        nonlocal embedding_sums, embedding_weights, done_mask, chunk_texts, chunk_records, chunk_weights
        nonlocal pending_chunk_counts, pending_token_counts, completed_rows, processed_chunks, next_chunk_log
        if not chunk_texts:
            return
        batch_count = len(chunk_texts)
        embeddings = embed_batch(session, batch_tokenizer, output_names, chunk_texts)
        if embedding_sums is None:
            embedding_sums, embedding_weights, done_mask = open_checkpoint_arrays(output_dir, len(records), embeddings.shape[1])

        for row_index, weight, embedding in zip(chunk_records, chunk_weights, embeddings, strict=False):
            embedding_sums[row_index] += embedding * weight
            embedding_weights[row_index] += weight
            pending_chunk_counts[row_index] -= 1
            if pending_chunk_counts[row_index] == 0:
                done_mask[row_index] = True
                record = records[row_index]
                row = {
                    "record_id": record["record_id"],
                    "benchmark": record["benchmark"],
                    "source": record["source"],
                    "token_count": int(pending_token_counts[row_index]),
                    "chunk_count": int(record["chunk_count"]),
                }
                record["token_count"] = row["token_count"]
                completed_rows.append(row)
                del pending_chunk_counts[row_index]
                del pending_token_counts[row_index]

        chunk_texts = []
        chunk_records = []
        chunk_weights = []
        processed_chunks += batch_count
        if processed_chunks >= next_chunk_log:
            logger.info("Embedded %s chunk(s)", processed_chunks)
            next_chunk_log += CHECKPOINT_LOG_INTERVAL
        if completed_rows:
            append_metadata_checkpoint(output_dir, completed_rows)
            completed_rows = []
        embedding_sums.flush()
        embedding_weights.flush()
        done_mask.flush()

    for record_index, record in enumerate(records):
        if done_mask is not None and bool(done_mask[record_index]):
            continue
        chunks, token_count = build_text_chunks(raw_tokenizer, record["text"], max_tokens, chunk_overlap)
        record["token_count"] = token_count
        record["chunk_count"] = len(chunks)
        if embedding_sums is not None:
            embedding_sums[record_index] = 0
            embedding_weights[record_index] = 0
            done_mask[record_index] = False
        pending_chunk_counts[record_index] = len(chunks)
        pending_token_counts[record_index] = token_count
        for chunk in chunks:
            chunk_texts.append(chunk["text"])
            chunk_records.append(record_index)
            chunk_weights.append(float(chunk["weight"]))
            if len(chunk_texts) >= batch_size:
                flush_batch()
        if (record_index + 1) % 1000 == 0:
            completed_count = int(done_mask.sum()) if done_mask is not None else 0
            logger.info(
                "Queued embeddings for %s/%s record(s); completed %s/%s",
                record_index + 1,
                len(records),
                completed_count,
                len(records),
            )

    flush_batch()
    if embedding_sums is None or embedding_weights is None or done_mask is None:
        raise RuntimeError("No embeddings were generated.")

    embeddings_path = finalize_checkpoint_embeddings(output_dir, embedding_sums, embedding_weights, done_mask)
    write_metadata(output_dir, records)
    write_token_audit(output_dir, records, max_tokens)
    return embeddings_path


# helper function to build high-dimensional features for projections
def compute_pca_features(embeddings: np.ndarray, output_dir: Path) -> np.ndarray:
    pca_components = min(50, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=pca_components, random_state=RANDOM_SEED)
    features = pca.fit_transform(embeddings)
    np.save(output_dir / "pca50.npy", features.astype(np.float32))
    explained = {
        "components": int(pca_components),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_[:10]],
    }
    (output_dir / "pca_summary.json").write_text(json.dumps(explained, indent=2), encoding="utf-8")
    return features.astype(np.float32)


# helper function to reduce with UMAP
def reduce_umap(features: np.ndarray) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=35,
        min_dist=0.08,
        metric="cosine",
        random_state=RANDOM_SEED,
        low_memory=True,
        verbose=True,
    )
    return reducer.fit_transform(features)


# helper function to reduce with PaCMAP
def reduce_pacmap(features: np.ndarray) -> np.ndarray:
    import pacmap

    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=35,
        MN_ratio=0.5,
        FP_ratio=2.0,
        random_state=RANDOM_SEED,
        verbose=True,
    )
    return reducer.fit_transform(features, init="pca")


# helper function to reduce with PCA
def reduce_pca2(features: np.ndarray) -> np.ndarray:
    reducer = PCA(n_components=2, random_state=RANDOM_SEED)
    return reducer.fit_transform(features)


# helper function to reduce with t-SNE from PCA features
def reduce_tsne(features: np.ndarray) -> np.ndarray:
    perplexity = min(DEFAULT_TSNE_PERPLEXITY, max(5, (features.shape[0] - 1) // 3))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        metric="euclidean",
        random_state=RANDOM_SEED,
        max_iter=DEFAULT_TSNE_MAX_ITER,
        method="barnes_hut",
        angle=0.5,
        n_jobs=-1,
        verbose=1,
    )
    return reducer.fit_transform(features.astype(np.float32, copy=False))


# helper function to normalize projection coordinates for plotting
def normalize_coords(coords: np.ndarray) -> np.ndarray:
    coords = coords.astype(np.float32)
    coords -= coords.mean(axis=0, keepdims=True)
    max_abs = np.max(np.abs(coords))
    if max_abs > 0:
        coords /= max_abs
    return coords


# helper function to save projection coordinates
def write_projection(output_dir: Path, method: str, records: list[dict], coords: np.ndarray) -> None:
    rows = []
    for record, coord in zip(records, coords, strict=False):
        rows.append(
            {
                "record_id": record["record_id"],
                "benchmark": record["benchmark"],
                "source": record["source"],
                "x": float(coord[0]),
                "y": float(coord[1]),
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), output_dir / f"projection_{method}.parquet")


# helper function to measure projection quality
def evaluate_projection(features: np.ndarray, coords: np.ndarray, metric_sample_size: int) -> dict:
    count = len(coords)
    sample_size = min(metric_sample_size, count)
    rng = np.random.default_rng(RANDOM_SEED)
    sample_indices = np.sort(rng.choice(count, sample_size, replace=False)) if sample_size < count else np.arange(count)
    high = normalize_rows(features[sample_indices].astype(np.float32))
    low = coords[sample_indices].astype(np.float32)
    neighbor_count = min(15, max(1, sample_size // 2 - 1), sample_size - 1)
    if neighbor_count < 2:
        return {"sample_size": int(sample_size)}

    high_neighbors = NearestNeighbors(n_neighbors=neighbor_count + 1, metric="cosine").fit(high)
    low_neighbors = NearestNeighbors(n_neighbors=neighbor_count + 1, metric="euclidean").fit(low)
    _, high_indices = high_neighbors.kneighbors(high)
    _, low_indices = low_neighbors.kneighbors(low)
    overlaps = []
    for high_row, low_row in zip(high_indices, low_indices, strict=False):
        high_set = set(int(value) for value in high_row[1:])
        low_set = set(int(value) for value in low_row[1:])
        overlaps.append(len(high_set & low_set) / neighbor_count)

    return {
        "sample_size": int(sample_size),
        "neighbor_count": int(neighbor_count),
        "trustworthiness": float(trustworthiness(high, low, n_neighbors=neighbor_count, metric="cosine")),
        "knn_overlap": float(np.mean(overlaps)),
    }


# helper function to compute plot bounds
def compute_bounds(coords: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    width = xmax - xmin
    height = ymax - ymin
    target_height = width / PLOT_ASPECT_RATIO
    target_width = height * PLOT_ASPECT_RATIO
    if target_height > height:
        delta = (target_height - height) / 2
        ymin -= delta
        ymax += delta
    elif target_width > width:
        delta = (target_width - width) / 2
        xmin -= delta
        xmax += delta
    pad_x = (xmax - xmin) * PLOT_PADDING_RATIO
    pad_y = (ymax - ymin) * PLOT_PADDING_RATIO
    return (float(xmin - pad_x), float(xmax + pad_x)), (float(ymin - pad_y), float(ymax + pad_y))


# helper function to draw density fields behind each cluster
def draw_density_contours(
    ax: plt.Axes,
    coords: np.ndarray,
    labels: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    for benchmark in BENCHMARK_ORDER:
        cluster = coords[labels == benchmark]
        if len(cluster) < 12:
            continue
        density, x_edges, y_edges = np.histogram2d(
            cluster[:, 0],
            cluster[:, 1],
            bins=DENSITY_GRID_SIZE,
            range=[xlim, ylim],
        )
        density = gaussian_filter(density.T, sigma=DENSITY_SMOOTHING)
        peak = float(density.max())
        if peak <= 0:
            continue
        levels = [peak * level for level in DENSITY_LEVELS[:-1]]
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        ax.contourf(
            x_centers,
            y_centers,
            density,
            levels=levels + [peak * DENSITY_LEVELS[-1]],
            colors=[BENCHMARK_COLORS[benchmark]],
            alpha=0.055,
            antialiased=True,
            zorder=0,
        )
        ax.contour(
            x_centers,
            y_centers,
            density,
            levels=levels,
            colors=[BENCHMARK_COLORS[benchmark]],
            linewidths=[0.45, 0.58, 0.72],
            alpha=0.18,
            zorder=1,
        )


# helper function to expand bounds to the target image aspect
def expand_bounds_to_aspect(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    aspect_ratio: float,
    padding_ratio: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    width = xmax - xmin
    height = ymax - ymin
    target_height = width / aspect_ratio
    target_width = height * aspect_ratio
    if target_height > height:
        delta = (target_height - height) / 2
        ymin -= delta
        ymax += delta
    elif target_width > width:
        delta = (target_width - width) / 2
        xmin -= delta
        xmax += delta
    pad_x = (xmax - xmin) * padding_ratio
    pad_y = (ymax - ymin) * padding_ratio
    return (float(xmin - pad_x), float(xmax + pad_x)), (float(ymin - pad_y), float(ymax + pad_y))


# helper function to compute homepage crop bounds
def compute_homepage_bounds(coords: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    lower_q, upper_q = HOMEPAGE_BOUND_QUANTILES
    lower = np.quantile(coords, lower_q, axis=0)
    upper = np.quantile(coords, upper_q, axis=0)
    label_points = np.array(list(HOMEPAGE_LABEL_POSITIONS.values()), dtype=np.float32)
    xmin = min(float(lower[0]), float(label_points[:, 0].min()))
    xmax = max(float(upper[0]), float(label_points[:, 0].max()))
    ymin = min(float(lower[1]), float(label_points[:, 1].min()))
    ymax = max(float(upper[1]), float(label_points[:, 1].max()))
    return expand_bounds_to_aspect(xmin, xmax, ymin, ymax, PLOT_ASPECT_RATIO, HOMEPAGE_PADDING_RATIO)


# helper function to draw balanced homepage point samples
def sample_homepage_indices(labels: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    sampled = []
    for benchmark in BENCHMARK_ORDER:
        indices = np.flatnonzero(labels == benchmark)
        if len(indices) > HOMEPAGE_SAMPLE_PER_BENCHMARK:
            indices = np.sort(rng.choice(indices, HOMEPAGE_SAMPLE_PER_BENCHMARK, replace=False))
        sampled.append(indices)
    return np.concatenate(sampled)


# helper function to render a polished homepage t-SNE image
def render_homepage_projection(output_path: Path, records: list[dict], coords: np.ndarray) -> Path:
    labels = np.array([record["benchmark"] for record in records])
    coords = normalize_coords(coords)
    xlim, ylim = compute_homepage_bounds(coords)
    fig, ax = plt.subplots(figsize=(11.8, 7.4), dpi=HOMEPAGE_DPI, facecolor=PAPER_DARK)
    ax.set_facecolor(PAPER_DARK)

    for radius in (0.48, 0.82, 1.16):
        ax.add_patch(
            plt.Circle(
                (0, 0),
                radius,
                color=GRID_DARK,
                fill=False,
                linewidth=0.42,
                alpha=0.16,
                zorder=0,
            )
        )
    ax.axhline(0, color=GRID_DARK, linewidth=0.35, alpha=0.1, zorder=0)
    ax.axvline(0, color=GRID_DARK, linewidth=0.35, alpha=0.1, zorder=0)

    draw_density_contours(ax, coords, labels, xlim, ylim)
    sampled_indices = sample_homepage_indices(labels)
    sampled_labels = labels[sampled_indices]
    sampled_coords = coords[sampled_indices]

    for benchmark in BENCHMARK_ORDER:
        cluster = sampled_coords[sampled_labels == benchmark]
        if len(cluster) == 0:
            continue
        color = BENCHMARK_COLORS[benchmark]
        ax.scatter(
            cluster[:, 0],
            cluster[:, 1],
            s=22,
            c=color,
            alpha=0.08,
            linewidths=0,
            zorder=2,
            rasterized=True,
        )
        ax.scatter(
            cluster[:, 0],
            cluster[:, 1],
            s=7.2,
            c=color,
            alpha=0.66,
            linewidths=0,
            zorder=3,
            rasterized=True,
        )
        ax.scatter(
            cluster[:, 0],
            cluster[:, 1],
            s=1.6,
            c="#ffffff",
            alpha=0.5,
            linewidths=0,
            zorder=4,
            rasterized=True,
        )

    for benchmark in BENCHMARK_ORDER:
        cluster = coords[labels == benchmark]
        if len(cluster) == 0:
            continue
        color = BENCHMARK_COLORS[benchmark]
        center = np.median(cluster, axis=0)
        label_xy = np.array(HOMEPAGE_LABEL_POSITIONS[benchmark], dtype=np.float32)
        ax.plot(
            [center[0], label_xy[0]],
            [center[1], label_xy[1]],
            color=color,
            linewidth=0.55,
            alpha=0.34,
            zorder=6,
            path_effects=[pe.withStroke(linewidth=2.0, foreground=PAPER_DARK, alpha=0.82)],
        )
        ax.scatter(
            [center[0]],
            [center[1]],
            s=16,
            facecolors=PAPER_DARK,
            edgecolors=color,
            linewidths=0.9,
            alpha=0.95,
            zorder=7,
        )
        ax.text(
            label_xy[0],
            label_xy[1],
            benchmark,
            color=color,
            fontsize=HOMEPAGE_LABEL_SIZES.get(benchmark, 13.2),
            fontweight="normal",
            fontfamily="serif",
            ha="center",
            va="center",
            path_effects=LABEL_HALO,
            zorder=8,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=PAPER_DARK, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return output_path


# helper function to render the projection figure
def render_projection(output_dir: Path, method: str, records: list[dict], coords: np.ndarray) -> Path:
    labels = np.array([record["benchmark"] for record in records])
    coords = normalize_coords(coords)
    xlim, ylim = compute_bounds(coords)
    fig, ax = plt.subplots(figsize=(11.8, 7.4), dpi=320, facecolor=PAPER_DARK)
    ax.set_facecolor(PAPER_DARK)

    for radius in (0.45, 0.75, 1.05, 1.35):
        ax.add_patch(
            plt.Circle(
                (0, 0),
                radius,
                color=GRID_DARK,
                fill=False,
                linewidth=0.45,
                alpha=0.2,
                zorder=0,
            )
        )
    ax.axhline(0, color=GRID_DARK, linewidth=0.38, alpha=0.14, zorder=0)
    ax.axvline(0, color=GRID_DARK, linewidth=0.38, alpha=0.14, zorder=0)

    draw_density_contours(ax, coords, labels, xlim, ylim)

    for benchmark in BENCHMARK_ORDER:
        indices = np.flatnonzero(labels == benchmark)
        cluster = coords[indices]
        color = BENCHMARK_COLORS[benchmark]
        ax.scatter(
            cluster[:, 0],
            cluster[:, 1],
            s=16,
            c=color,
            alpha=0.09,
            linewidths=0,
            zorder=2,
            rasterized=True,
        )
        ax.scatter(
            cluster[:, 0],
            cluster[:, 1],
            s=5.0,
            c=color,
            alpha=0.58,
            linewidths=0,
            zorder=3,
            rasterized=True,
        )
        ax.scatter(
            cluster[:, 0],
            cluster[:, 1],
            s=1.1,
            c="#ffffff",
            alpha=0.35,
            linewidths=0,
            zorder=4,
            rasterized=True,
        )
        center = np.median(cluster, axis=0)
        offset = np.array(LABEL_OFFSETS.get(benchmark, (0, 0)), dtype=np.float32)
        ax.text(
            center[0] + offset[0],
            center[1] + offset[1],
            benchmark,
            color=color,
            fontsize=13.5,
            fontweight="bold",
            fontfamily="serif",
            ha="center",
            va="center",
            path_effects=LABEL_HALO,
            zorder=8,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    output_path = output_dir / f"benchmark-fossil-embedding-{method}.png"
    fig.savefig(output_path, facecolor=PAPER_DARK, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return output_path


# helper function to generate projections and figures
def generate_projections(
    records: list[dict],
    embeddings_path: Path,
    output_dir: Path,
    projection_methods: Iterable[str],
    metric_sample_size: int,
    skip_projections: bool,
    homepage_image_path: Path | None = None,
) -> None:
    embeddings = np.load(embeddings_path, mmap_mode="r")
    pca_path = output_dir / "pca50.npy"
    if pca_path.exists() and skip_projections:
        features = np.load(pca_path, mmap_mode="r")
    else:
        features = compute_pca_features(np.asarray(embeddings), output_dir)

    reducers = {
        "umap": reduce_umap,
        "pacmap": reduce_pacmap,
        "pca": reduce_pca2,
        "tsne": reduce_tsne,
    }
    metrics_path = output_dir / "projection_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    for method in projection_methods:
        method = method.strip().lower()
        if not method:
            continue
        if method not in reducers:
            raise ValueError(f"Unsupported projection method: {method}")
        coords_path = output_dir / f"projection_{method}.npy"
        if coords_path.exists() and skip_projections:
            coords = np.load(coords_path)
        else:
            logger.info("Running %s projection", method)
            coords = normalize_coords(reducers[method](np.asarray(features)))
            np.save(coords_path, coords.astype(np.float32))
            write_projection(output_dir, method, records, coords)
        metrics[method] = evaluate_projection(np.asarray(features), coords, metric_sample_size)
        figure_path = render_projection(output_dir, method, records, coords)
        logger.info("Wrote %s", figure_path)
        if method == "tsne" and homepage_image_path is not None:
            homepage_path = render_homepage_projection(homepage_image_path, records, coords)
            logger.info("Wrote %s", homepage_path)

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


# orchestration function
def run_full_benchmark_embedding_map() -> None:
    args = parse_args()
    if args.shard_count < 1:
        raise ValueError("--shard-count must be at least 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < shard-count")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_benchmark_records(args.limit_per_benchmark)
    full_record_count = len(records)
    if args.shard_count > 1:
        records = [
            record
            for record_index, record in enumerate(records)
            if record_index % args.shard_count == args.shard_index
        ]
        logger.info(
            "Using shard %s/%s with %s/%s record(s)",
            args.shard_index,
            args.shard_count,
            len(records),
            full_record_count,
        )
    logger.info("Loaded %s total record(s)", len(records))

    embeddings_path = args.output_dir / "record_embeddings.npy"
    if args.skip_embeddings and embeddings_path.exists():
        logger.info("Reusing cached embeddings at %s", embeddings_path)
    else:
        embeddings_path = generate_embeddings(
            records=records,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            chunk_overlap=args.chunk_overlap,
        )

    if args.only_embeddings:
        logger.info("Stopping after embeddings as requested")
        return

    methods = [method.strip() for method in args.projection_methods.split(",") if method.strip()]
    generate_projections(
        records=records,
        embeddings_path=embeddings_path,
        output_dir=args.output_dir,
        projection_methods=methods,
        metric_sample_size=args.metric_sample_size,
        skip_projections=args.skip_projections,
        homepage_image_path=args.homepage_image_path,
    )


if __name__ == "__main__":
    run_full_benchmark_embedding_map()
