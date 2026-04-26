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
import random
import re
from pathlib import Path

# data
import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from tokenizers import Tokenizer

# viz
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath


# init logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# directory constants
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
IMAGES_DIR = REPO_ROOT / "images"
DEFAULT_MODEL_DIR = Path("/Users/akhilpandey/code/models/pplx-embed-v1-0.6b")
DEFAULT_OUTPUT_STEM = IMAGES_DIR / "benchmark-fossil-embedding-tsne"


# embedding constants
DEFAULT_LIMIT_PER_BENCHMARK = 48
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_TOKENS = 384
RANDOM_SEED = 7207
ONNX_MODEL_NAME = "model_q4.onnx"
EMBEDDING_OUTPUT_NAME = "pooler_output_int8"


# plot constants
PAPER_DARK = "#050506"
INK_LIGHT = "#f6f1e8"
MUTED_LIGHT = "#aaa59b"
GRID_DARK = "#2f2f32"
LABEL_HALO = [pe.withStroke(linewidth=3.2, foreground=PAPER_DARK, alpha=0.95)]
NEIGHBOR_COUNT = 5
CROSS_NEIGHBOR_COUNT = 3
PLOT_ASPECT_RATIO = 11.8 / 7.4
PLOT_PADDING_RATIO = 0.16
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
    "BioASQ": (-0.04, 0.13),
    "BioRED": (0.08, -0.03),
    "FrontierScience": (0.05, -0.10),
    "GPQA": (-0.10, 0.06),
    "MMLU": (0.06, 0.08),
    "PubMedQA": (0.01, -0.10),
    "SciERC": (0.00, -0.08),
    "SciRIFF": (0.00, -0.12),
    "SimpleQA": (0.04, 0.03),
}


# helper function to parse CLI args
def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the benchmark embedding map.

    Returns
    ------------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate a black-background t-SNE map of benchmark fossil examples."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Local pplx-embed model directory.")
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=DEFAULT_OUTPUT_STEM,
        help="Output path stem; .png, .svg, and -plot.png are written.",
    )
    parser.add_argument(
        "--limit-per-benchmark",
        type=int,
        default=DEFAULT_LIMIT_PER_BENCHMARK,
        help=f"Maximum examples to sample per benchmark. Default {DEFAULT_LIMIT_PER_BENCHMARK}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"ONNX embedding batch size. Default {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Tokenizer truncation length. Default {DEFAULT_MAX_TOKENS}.",
    )
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


# helper function to create deterministic samples
def sample_records(records: list[dict], benchmark: str, limit: int) -> list[dict]:
    """
    Sample records deterministically while preserving benchmark labels.

    Parameters
    ------------
    records: list[dict]
        Loaded benchmark records
    benchmark: str
        Benchmark label
    limit: int
        Maximum records to keep

    Returns
    ------------
    list[dict]
    """
    cleaned = [record for record in records if record.get("text")]
    rng = random.Random(f"{RANDOM_SEED}:{benchmark}")
    rng.shuffle(cleaned)
    return cleaned[:limit]


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


# helper function to collect FrontierScience examples
def load_frontierscience_records() -> list[dict]:
    records = []
    for split in ("olympiad", "research"):
        path = DATA_DIR / "frontierscience" / split / "test.jsonl"
        for row in load_jsonl_records(path):
            text = f"{row.get('subject', '')} problem. {row.get('problem', '')}"
            records.append({"benchmark": "FrontierScience", "text": clean_text(text)})
    return records


# helper function to collect GPQA examples
def load_gpqa_records() -> list[dict]:
    rows = load_csv_records(DATA_DIR / "gpqa" / "extracted" / "dataset" / "gpqa_main.csv")
    records = []
    for row in rows:
        text = " ".join(
            [
                row.get("High-level domain", ""),
                row.get("Subdomain", ""),
                row.get("Question", ""),
                row.get("Correct Answer", ""),
                row.get("Explanation", ""),
            ]
        )
        records.append({"benchmark": "GPQA", "text": clean_text(text)})
    return records


# helper function to collect PubMedQA examples
def load_pubmedqa_records() -> list[dict]:
    path = DATA_DIR / "pubmedqa" / "pqa_labeled" / "train-00000-of-00001.parquet"
    records = []
    for row in load_parquet_records(path, columns=["question", "context", "long_answer", "final_decision"]):
        context = row.get("context") or {}
        contexts = " ".join(context.get("contexts") or [])
        text = f"{row.get('question', '')} {contexts} {row.get('long_answer', '')} {row.get('final_decision', '')}"
        records.append({"benchmark": "PubMedQA", "text": clean_text(text)})
    return records


# helper function to collect BioASQ examples
def load_bioasq_records() -> list[dict]:
    records = []
    for path in (DATA_DIR / "bioasq" / "train_bio.csv", DATA_DIR / "bioasq" / "valid_bio.csv"):
        for row in load_csv_records(path):
            text = f"{row.get('question', '')} {row.get('text', '')}"
            records.append({"benchmark": "BioASQ", "text": clean_text(text)})
    return records


# helper function to collect BioRED examples
def load_biored_records() -> list[dict]:
    records = []
    for split in ("Train", "Dev", "Test"):
        path = DATA_DIR / "biored" / "extracted" / "BioRED" / f"{split}.PubTator"
        records.extend(parse_pubtator_abstracts(path))
    return [{"benchmark": "BioRED", "text": record} for record in records]


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


# helper function to collect SciERC examples
def load_scierc_records() -> list[dict]:
    records = []
    for split in ("train", "dev", "test"):
        path = DATA_DIR / "scierc" / f"{split}.jsonl"
        for row in load_jsonl_records(path):
            text = f"{row.get('label', '')} {row.get('text', '')}"
            records.append({"benchmark": "SciERC", "text": clean_text(text)})
    return records


# helper function to collect MMLU examples
def load_mmlu_records() -> list[dict]:
    path = DATA_DIR / "mmlu" / "all" / "test-00000-of-00001.parquet"
    records = []
    for row in load_parquet_records(path, columns=["question", "subject", "choices"]):
        choices = " ".join(row.get("choices") or [])
        text = f"{row.get('subject', '')} {row.get('question', '')} {choices}"
        records.append({"benchmark": "MMLU", "text": clean_text(text)})
    return records


# helper function to collect SimpleQA examples
def load_simpleqa_records() -> list[dict]:
    path = DATA_DIR / "simpleqa" / "simple_qa_test_set.csv"
    records = []
    for row in load_csv_records(path):
        text = f"{row.get('problem', '')} {row.get('answer', '')} {row.get('metadata', '')}"
        records.append({"benchmark": "SimpleQA", "text": clean_text(text)})
    return records


# helper function to collect SciRIFF examples
def load_sciriff_records() -> list[dict]:
    path = DATA_DIR / "sciriff" / "4096" / "test-00000-of-00001.parquet"
    records = []
    for row in load_parquet_records(path, columns=["input", "output", "metadata"]):
        metadata = row.get("metadata") or {}
        text = " ".join(
            [
                metadata.get("task_family", ""),
                metadata.get("source_type", ""),
                row.get("input", ""),
                row.get("output", ""),
            ]
        )
        records.append({"benchmark": "SciRIFF", "text": clean_text(text)})
    return records


# helper function to load all benchmark records
def load_benchmark_records(limit_per_benchmark: int) -> list[dict]:
    """
    Load and sample records for every benchmark fossil.

    Parameters
    ------------
    limit_per_benchmark: int
        Maximum examples per benchmark

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
        benchmark_records = loader_map[benchmark]()
        sampled = sample_records(benchmark_records, benchmark, limit_per_benchmark)
        logger.info("Loaded %s sampled %s/%s record(s)", benchmark, len(sampled), len(benchmark_records))
        records.extend(sampled)
    return records


# helper function to load the local tokenizer
def load_tokenizer(model_dir: Path, max_tokens: int) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer_config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    pad_token = tokenizer_config.get("pad_token") or "<|endoftext|>"
    pad_id = tokenizer.token_to_id(pad_token)
    if pad_id is None:
        raise RuntimeError(f"Could not resolve pad token id for {pad_token!r}.")

    tokenizer.enable_truncation(max_length=max_tokens)
    tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)
    return tokenizer


# helper function to embed texts with the ONNX model
def embed_texts(model_dir: Path, texts: list[str], batch_size: int, max_tokens: int) -> np.ndarray:
    """
    Generate normalized pplx-embed vectors for benchmark texts.

    Parameters
    ------------
    model_dir: Path
        Local model directory
    texts: list[str]
        Texts to embed
    batch_size: int
        ONNX batch size
    max_tokens: int
        Tokenizer truncation length

    Returns
    ------------
    np.ndarray
    """
    tokenizer = load_tokenizer(model_dir, max_tokens)
    session = ort.InferenceSession(str(model_dir / "onnx" / ONNX_MODEL_NAME), providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]
    if EMBEDDING_OUTPUT_NAME not in output_names:
        raise RuntimeError(f"Expected ONNX output {EMBEDDING_OUTPUT_NAME!r}, got {output_names}.")

    embeddings = []
    output_index = output_names.index(EMBEDDING_OUTPUT_NAME)
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer.encode_batch(batch_texts)
        input_ids = np.array([item.ids for item in encoded], dtype=np.int64)
        attention_mask = np.array([item.attention_mask for item in encoded], dtype=np.int64)
        outputs = session.run(
            output_names,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )
        embeddings.append(outputs[output_index].astype(np.float32))
        logger.info("Embedded %s/%s text(s)", min(start + batch_size, len(texts)), len(texts))

    matrix = np.vstack(embeddings)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-8)


# helper function to reduce embeddings into two dimensions
def reduce_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Reduce normalized embeddings with PCA-initialized t-SNE.

    Parameters
    ------------
    embeddings: np.ndarray
        Normalized embedding matrix

    Returns
    ------------
    np.ndarray
    """
    pca_components = min(50, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=pca_components, random_state=RANDOM_SEED)
    reduced = pca.fit_transform(embeddings)
    perplexity = min(32, max(5, math.floor((embeddings.shape[0] - 1) / 4)))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        metric="cosine",
        random_state=RANDOM_SEED,
        max_iter=1500,
    )
    coords = tsne.fit_transform(reduced)
    coords -= coords.mean(axis=0, keepdims=True)
    coords /= np.max(np.abs(coords))
    return coords


# helper function to build same-benchmark neighbor edges
def build_same_benchmark_edges(coords: np.ndarray, labels: np.ndarray) -> list[tuple[int, int, str]]:
    """
    Build local graph edges inside each benchmark cluster.

    Parameters
    ------------
    coords: np.ndarray
        2D t-SNE coordinates
    labels: np.ndarray
        Benchmark labels for each point

    Returns
    ------------
    list[tuple[int, int, str]]
    """
    edges = []
    for benchmark in BENCHMARK_ORDER:
        indices = np.flatnonzero(labels == benchmark)
        if len(indices) < 2:
            continue
        neighbor_count = min(NEIGHBOR_COUNT, len(indices))
        neighbors = NearestNeighbors(n_neighbors=neighbor_count).fit(coords[indices])
        _, neighbor_indices = neighbors.kneighbors(coords[indices])
        seen = set()
        for local_index, row in enumerate(neighbor_indices):
            for local_neighbor in row[1:]:
                source = int(indices[local_index])
                target = int(indices[local_neighbor])
                edge = tuple(sorted((source, target)))
                if edge in seen:
                    continue
                seen.add(edge)
                edges.append((edge[0], edge[1], BENCHMARK_COLORS[benchmark]))
    return edges


# helper function to build cross-benchmark neighbor edges
def build_cross_benchmark_edges(coords: np.ndarray, labels: np.ndarray) -> list[tuple[int, int]]:
    """
    Build sparse nearest-neighbor edges between benchmark clusters.

    Parameters
    ------------
    coords: np.ndarray
        2D t-SNE coordinates
    labels: np.ndarray
        Benchmark labels for each point

    Returns
    ------------
    list[tuple[int, int]]
    """
    neighbor_count = min(CROSS_NEIGHBOR_COUNT + 1, len(coords))
    neighbors = NearestNeighbors(n_neighbors=neighbor_count).fit(coords)
    distances, neighbor_indices = neighbors.kneighbors(coords)
    edges = []
    seen = set()
    for source, row in enumerate(neighbor_indices):
        for distance, target in zip(distances[source][1:], row[1:], strict=False):
            if labels[source] == labels[target] or distance > 0.42:
                continue
            edge = tuple(sorted((int(source), int(target))))
            if edge in seen:
                continue
            seen.add(edge)
            edges.append(edge)
    return edges


# helper function to draw a quiet curved edge
def draw_curve(ax: plt.Axes, start: np.ndarray, end: np.ndarray, color: str, alpha: float, linewidth: float, seed: int) -> None:
    delta = end - start
    distance = float(np.linalg.norm(delta))
    if distance == 0:
        return

    normal = np.array([-delta[1], delta[0]]) / distance
    phase = ((seed * 1103515245 + 12345) % 1000) / 1000
    direction = -1 if seed % 2 else 1
    control = (start + end) / 2 + normal * direction * distance * (0.06 + 0.14 * phase)
    path = MplPath([start, control, end], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
    patch = PathPatch(
        path,
        facecolor="none",
        edgecolor=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=1,
        capstyle="round",
    )
    ax.add_patch(patch)


# helper function to draw cluster graph strands
def draw_network_edges(ax: plt.Axes, coords: np.ndarray, labels: np.ndarray) -> None:
    """
    Draw fine local and cross-cluster strands for the embedding map.

    Parameters
    ------------
    ax: plt.Axes
        Matplotlib axes
    coords: np.ndarray
        2D t-SNE coordinates
    labels: np.ndarray
        Benchmark labels for each point

    Returns
    ------------
    None
    """
    for source, target in build_cross_benchmark_edges(coords, labels):
        seed = source * 10_000 + target
        draw_curve(ax, coords[source], coords[target], "#b9c7d9", 0.028, 0.38, seed)

    for source, target, color in build_same_benchmark_edges(coords, labels):
        seed = source * 10_000 + target
        draw_curve(ax, coords[source], coords[target], color, 0.035, 2.2, seed)
        draw_curve(ax, coords[source], coords[target], color, 0.16, 0.46, seed)


# helper function to draw luminous points for each benchmark
def draw_cluster_points(ax: plt.Axes, coords: np.ndarray, labels: np.ndarray) -> None:
    for benchmark in BENCHMARK_ORDER:
        mask = labels == benchmark
        benchmark_coords = coords[mask]
        color = BENCHMARK_COLORS[benchmark]
        ax.scatter(benchmark_coords[:, 0], benchmark_coords[:, 1], s=130, color=color, alpha=0.035, edgecolors="none", zorder=2)
        ax.scatter(benchmark_coords[:, 0], benchmark_coords[:, 1], s=34, color=color, alpha=0.18, edgecolors="none", zorder=3)
        ax.scatter(benchmark_coords[:, 0], benchmark_coords[:, 1], s=8.5, color=color, alpha=0.9, edgecolors="none", zorder=4)
        ax.scatter(benchmark_coords[:, 0], benchmark_coords[:, 1], s=1.5, color=INK_LIGHT, alpha=0.38, edgecolors="none", zorder=5)


# helper function to draw compact benchmark labels
def draw_cluster_labels(ax: plt.Axes, coords: np.ndarray, labels: np.ndarray, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    for benchmark in BENCHMARK_ORDER:
        mask = labels == benchmark
        benchmark_coords = coords[mask]
        color = BENCHMARK_COLORS[benchmark]
        centroid = np.median(benchmark_coords, axis=0)
        label_offset = np.array(LABEL_OFFSETS.get(benchmark, (0.0, 0.0)))
        label_position = np.clip(
            centroid + label_offset,
            [xlim[0] + 0.05, ylim[0] + 0.05],
            [xlim[1] - 0.05, ylim[1] - 0.05],
        )
        ax.plot(
            [centroid[0], label_position[0]],
            [centroid[1], label_position[1]],
            color=color,
            linewidth=0.45,
            alpha=0.48,
            zorder=5,
        )
        ax.scatter(
            [centroid[0]],
            [centroid[1]],
            s=24,
            facecolors=PAPER_DARK,
            edgecolors=color,
            linewidths=0.9,
            alpha=0.9,
            zorder=6,
        )
        name_text = ax.text(
            label_position[0],
            label_position[1] + 0.018,
            benchmark,
            color=color,
            fontsize=9.6,
            fontfamily="DejaVu Serif",
            fontweight="bold",
            ha="center",
            va="center",
            zorder=7,
        )
        name_text.set_path_effects(LABEL_HALO)
        count_text = ax.text(
            label_position[0],
            label_position[1] - 0.032,
            f"{mask.sum()} samples",
            color=INK_LIGHT,
            fontsize=5.2,
            fontfamily="DejaVu Serif",
            ha="center",
            va="center",
            alpha=0.72,
            zorder=7,
        )
        count_text.set_path_effects(LABEL_HALO)


# helper function to compute tight art-directed plot limits
def compute_plot_limits(coords: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Compute padded limits that fill the image at a fixed aspect ratio.

    Parameters
    ------------
    coords: np.ndarray
        2D t-SNE coordinates

    Returns
    ------------
    tuple[tuple[float, float], tuple[float, float]]
    """
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    width = xmax - xmin
    height = ymax - ymin
    xpad = max(width * PLOT_PADDING_RATIO, 0.05)
    ypad = max(height * PLOT_PADDING_RATIO, 0.05)
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    width = xmax - xmin
    height = ymax - ymin
    current_aspect = width / height
    if current_aspect > PLOT_ASPECT_RATIO:
        target_height = width / PLOT_ASPECT_RATIO
        midpoint = (ymin + ymax) / 2
        ymin = midpoint - target_height / 2
        ymax = midpoint + target_height / 2
    else:
        target_width = height * PLOT_ASPECT_RATIO
        midpoint = (xmin + xmax) / 2
        xmin = midpoint - target_width / 2
        xmax = midpoint + target_width / 2
    return (float(xmin), float(xmax)), (float(ymin), float(ymax))


# helper function to render the t-SNE plot
def render_plot(records: list[dict], coords: np.ndarray, output_stem: Path) -> list[Path]:
    """
    Render a black-background benchmark fossil embedding plot.

    Parameters
    ------------
    records: list[dict]
        Records with benchmark labels
    coords: np.ndarray
        2D t-SNE coordinates
    output_stem: Path
        Output path stem

    Returns
    ------------
    list[Path]
    """
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    labels = np.array([record["benchmark"] for record in records])
    xlim, ylim = compute_plot_limits(coords)
    center = ((xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2)
    max_radius = max(xlim[1] - xlim[0], ylim[1] - ylim[0]) * 0.48

    fig, ax = plt.subplots(figsize=(11.8, 7.4), dpi=320)
    fig.patch.set_facecolor(PAPER_DARK)
    ax.set_facecolor(PAPER_DARK)
    ax.set_position([0, 0, 1, 1])

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    for radius in np.linspace(max_radius * 0.32, max_radius, 4):
        circle = plt.Circle(
            center,
            radius,
            fill=False,
            color=GRID_DARK,
            linewidth=0.35,
            alpha=0.18,
            zorder=0,
        )
        ax.add_patch(circle)

    draw_network_edges(ax, coords, labels)
    draw_cluster_points(ax, coords, labels)
    draw_cluster_labels(ax, coords, labels, xlim, ylim)

    png_path = output_stem.with_suffix(".png")
    svg_path = output_stem.with_suffix(".svg")
    page_plot_path = output_stem.with_name(f"{output_stem.name}-plot").with_suffix(".png")
    fig.savefig(png_path, facecolor=PAPER_DARK, pad_inches=0, dpi=320)
    fig.savefig(svg_path, facecolor=PAPER_DARK, pad_inches=0)
    fig.savefig(page_plot_path, facecolor=PAPER_DARK, pad_inches=0, dpi=320)
    plt.close(fig)
    return [png_path, svg_path, page_plot_path]


# helper function to run the full generation pipeline
def run_embedding_map(args: argparse.Namespace) -> None:
    """
    Generate the benchmark embedding t-SNE map.

    Parameters
    ------------
    args: argparse.Namespace
        Parsed CLI args

    Returns
    ------------
    None
    """
    records = load_benchmark_records(args.limit_per_benchmark)
    texts = [record["text"] for record in records]
    embeddings = embed_texts(args.model_dir, texts, args.batch_size, args.max_tokens)
    coords = reduce_embeddings(embeddings)
    outputs = render_plot(records, coords, args.output_stem)
    for path in outputs:
        logger.info("Saved %s", path)
        print(path)


if __name__ == "__main__":
    run_embedding_map(parse_args())
