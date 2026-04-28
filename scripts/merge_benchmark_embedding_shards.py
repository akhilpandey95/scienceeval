#!/usr/bin/env python3

# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/scienceeval/blob/main/LICENSE

from __future__ import annotations

# stdlib
import argparse
import logging
from pathlib import Path

# data
import numpy as np
import pyarrow.parquet as pq

# project
from generate_full_benchmark_embedding_map import (
    OUTPUT_DIR,
    generate_projections,
    load_benchmark_records,
    write_metadata,
    write_token_audit,
)


# init logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# merge constants
DEFAULT_PROJECTION_METHODS = "umap,pca"
DEFAULT_METRIC_SAMPLE_SIZE = 10_000
DEFAULT_MAX_TOKENS = 2048


# helper function to parse CLI args
def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for shard merging and projection.

    Returns
    ------------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Merge benchmark embedding shards and generate full projections.")
    parser.add_argument(
        "--shard-dir",
        type=Path,
        action="append",
        required=True,
        help="Shard output directory containing metadata.parquet and record_embeddings.npy.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Merged output directory.")
    parser.add_argument(
        "--projection-methods",
        default=DEFAULT_PROJECTION_METHODS,
        help="Comma-separated projection methods to run: umap,pacmap,pca,tsne.",
    )
    parser.add_argument(
        "--metric-sample-size",
        type=int,
        default=DEFAULT_METRIC_SAMPLE_SIZE,
        help="Maximum point count used for projection quality metrics.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Token window used during shard embedding, for token-audit summaries.",
    )
    return parser.parse_args()


# helper function to load shard metadata
def load_shard_metadata(shard_dir: Path) -> list[dict]:
    """
    Load one shard's metadata rows.

    Parameters
    ------------
    shard_dir: Path
        Shard output directory

    Returns
    ------------
    list[dict]
    """
    metadata_path = shard_dir / "metadata.parquet"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing shard metadata: {metadata_path}")
    return pq.read_table(metadata_path).to_pylist()


# helper function to load shard embeddings
def load_shard_embeddings(shard_dir: Path) -> np.ndarray:
    """
    Load one shard's embedding matrix.

    Parameters
    ------------
    shard_dir: Path
        Shard output directory

    Returns
    ------------
    np.ndarray
    """
    embeddings_path = shard_dir / "record_embeddings.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing shard embeddings: {embeddings_path}")
    return np.load(embeddings_path, mmap_mode="r")


# helper function to merge embeddings in canonical record order
def merge_shards(shard_dirs: list[Path], output_dir: Path, max_tokens: int) -> Path:
    """
    Merge sharded embeddings into one full embedding matrix.

    Parameters
    ------------
    shard_dirs: list[Path]
        Shard output directories
    output_dir: Path
        Merged output directory
    max_tokens: int
        Token window used for token-audit summaries

    Returns
    ------------
    Path
    """
    records = load_benchmark_records(limit_per_benchmark=0)
    id_to_index = {record["record_id"]: index for index, record in enumerate(records)}
    embedding_matrix = None
    filled = np.zeros(len(records), dtype=bool)

    for shard_dir in shard_dirs:
        metadata_rows = load_shard_metadata(shard_dir)
        embeddings = load_shard_embeddings(shard_dir)
        if len(metadata_rows) != len(embeddings):
            raise ValueError(f"Shard length mismatch in {shard_dir}: metadata={len(metadata_rows)} embeddings={len(embeddings)}")
        if embedding_matrix is None:
            embedding_matrix = np.zeros((len(records), embeddings.shape[1]), dtype=np.float32)

        logger.info("Merging %s record(s) from %s", len(metadata_rows), shard_dir)
        for shard_row, embedding in zip(metadata_rows, embeddings, strict=False):
            record_id = shard_row["record_id"]
            full_index = id_to_index.get(record_id)
            if full_index is None:
                raise KeyError(f"Shard record_id not found in full corpus: {record_id}")
            if filled[full_index]:
                raise ValueError(f"Duplicate shard record_id: {record_id}")
            records[full_index]["token_count"] = int(shard_row["token_count"])
            records[full_index]["chunk_count"] = int(shard_row["chunk_count"])
            embedding_matrix[full_index] = np.asarray(embedding, dtype=np.float32)
            filled[full_index] = True

    if embedding_matrix is None:
        raise RuntimeError("No shard embeddings were loaded.")
    missing_count = int((~filled).sum())
    if missing_count:
        missing_ids = [records[index]["record_id"] for index in np.flatnonzero(~filled)[:10]]
        raise RuntimeError(f"Missing {missing_count} record embedding(s); first missing ids: {missing_ids}")

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "record_embeddings.npy"
    np.save(embeddings_path, embedding_matrix)
    write_metadata(output_dir, records)
    write_token_audit(output_dir, records, max_tokens)
    return embeddings_path


# orchestration function
def run_merge_benchmark_embedding_shards() -> None:
    args = parse_args()
    embeddings_path = merge_shards(args.shard_dir, args.output_dir, args.max_tokens)
    methods = [method.strip() for method in args.projection_methods.split(",") if method.strip()]
    records = load_benchmark_records(limit_per_benchmark=0)
    generate_projections(
        records=records,
        embeddings_path=embeddings_path,
        output_dir=args.output_dir,
        projection_methods=methods,
        metric_sample_size=args.metric_sample_size,
        skip_projections=False,
    )


if __name__ == "__main__":
    run_merge_benchmark_embedding_shards()
