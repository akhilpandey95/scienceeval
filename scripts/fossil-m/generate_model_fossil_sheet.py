#!/usr/bin/env python3

# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/scienceeval/blob/main/LICENSE

from __future__ import annotations

# stdlib
import argparse
import json
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
DEFAULT_RESULTS_DIR = REPO_ROOT / "data" / "fossil-results" / "fossil-m" / "gpt-oss-20b"
DATA_OUTPUT_PATH = REPO_ROOT / "data" / "fossils-m-catalog.json"
IMAGE_OUTPUT_DIR = REPO_ROOT / "images"


# source constants
FOSSIL_RESULTS_URL = "https://huggingface.co/datasets/akhilpandey95/scienceeval-fossil-results/tree/main/fossil-m/gpt-oss-20b"
GPT4O_FRONTIERSCIENCE_URL = "https://cdn.openai.com/pdf/2fcd284c-b468-4c21-8ee0-7a783933efcc/frontierscience-paper.pdf"
GPT4O_OPENAI_45_URL = "https://openai.com/index/introducing-gpt-4-5/"
GPT4O_OPENAI_41_URL = "https://openai.com/index/gpt-4-1/"
GPT4O_SIMPLEQA_URL = "https://openai.com/index/introducing-simpleqa/"
GPT4O_MEDINST_URL = "https://aclanthology.org/2024.findings-emnlp.482.pdf"
GPT4O_GRAPHJUDGE_URL = "https://aclanthology.org/2025.emnlp-main.554.pdf"
GPT4O_SCIRIFF_URL = "https://aclanthology.org/2025.emnlp-main.310.pdf"


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
}

MODEL_ORDER = ("gpt-4o", "gpt-oss-20b")


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
DEFAULT_LABEL_OFFSETS = {
    "bioasq": (6, 8, "bottom"),
    "scierc": (6, 8, "bottom"),
    "pubmedqa": (2, -12, "top"),
    "mmlu": (6, 8, "bottom"),
    "biored": (6, -12, "top"),
    "gpqa": (6, 8, "bottom"),
    "sciriff": (6, -12, "top"),
    "simpleqa": (6, 8, "bottom"),
    "frontierscience": (6, 8, "bottom"),
    "frontierscience_research": (6, -10, "top"),
}

GPT_OSS_LABEL_OFFSETS = {
    "bioasq": (6, 8, "bottom"),
    "scierc": (6, 8, "bottom"),
    "pubmedqa": (6, 8, "bottom"),
    "biored": (6, -12, "top"),
    "gpqa": (6, 8, "bottom"),
    "sciriff": (6, 8, "bottom"),
    "frontierscience": (6, -12, "top"),
    "frontierscience_research": (6, 8, "bottom"),
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
        help="Directory containing gpt-oss-20b Fossils-M JSON results.",
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

    for benchmark in ("bioasq", "biored", "frontierscience", "gpqa", "pubmedqa", "scierc", "sciriff"):
        path = results_dir / f"{benchmark}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing required result file: {path}")
        payload = load_json(path)
        summaries[benchmark] = payload["summary"]

    return summaries


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


# helper function to build gpt-oss-20b entries from result summaries
def build_gpt_oss_entries(summaries: dict[str, dict]) -> list[dict]:
    """
    Convert gpt-oss-20b result summaries into ledger entries.

    Parameters
    ------------
    summaries: dict[str, dict]
        Benchmark summary blocks

    Returns
    ------------
    list[dict]
    """
    frontierscience = summaries["frontierscience"]["by_split"]
    values = {
        "frontierscience": {
            "value": f"{frontierscience['olympiad']['pass_rate'] * 100:.1f} / {frontierscience['research']['pass_rate'] * 100:.1f}",
            "primary_value": percent(frontierscience["olympiad"]["pass_rate"]),
            "secondary_value": percent(frontierscience["research"]["pass_rate"]),
            "metric": "Olympiad pass rate / Research pass rate",
        },
        "gpqa": {
            "value": format_percent(percent(summaries["gpqa"]["accuracy"])),
            "primary_value": percent(summaries["gpqa"]["accuracy"]),
            "metric": "Accuracy, GPQA Diamond",
        },
        "pubmedqa": {
            "value": format_percent(percent(summaries["pubmedqa"]["macro_f1"])),
            "primary_value": percent(summaries["pubmedqa"]["macro_f1"]),
            "metric": "Macro-F1, PubMedQA-labeled",
        },
        "bioasq": {
            "value": format_percent(percent(summaries["bioasq"]["accuracy"])),
            "primary_value": percent(summaries["bioasq"]["accuracy"]),
            "metric": "Exact match, Task-B answer extraction",
        },
        "biored": {
            "value": format_percent(percent(summaries["biored"]["macro_f1"])),
            "primary_value": percent(summaries["biored"]["macro_f1"]),
            "metric": "Macro-F1, relation classification",
        },
        "scierc": {
            "value": format_percent(percent(summaries["scierc"]["macro_f1"])),
            "primary_value": percent(summaries["scierc"]["macro_f1"]),
            "metric": "Macro-F1, marked-pair relations",
        },
        "mmlu": {
            "value": "Not run",
            "metric": "MMLU",
            "missing": True,
        },
        "simpleqa": {
            "value": "Not run",
            "metric": "Correct",
            "missing": True,
        },
        "sciriff": {
            "value": format_percent(percent(summaries["sciriff"]["accuracy"])),
            "primary_value": percent(summaries["sciriff"]["accuracy"]),
            "metric": "Exact match, SciRIFF 8192",
        },
    }

    entries = []
    for benchmark in BENCHMARK_ORDER:
        payload = values[benchmark]
        if not payload.get("missing"):
            payload["source_label"] = "scienceeval fossil suite, 2026"
            payload["source_url"] = FOSSIL_RESULTS_URL
        else:
            payload["source_label"] = "Not included in this Fossils-M run"
            payload["source_url"] = ""
        entries.append(make_entry(benchmark, payload))

    return entries


# helper function to build the gpt-oss-20b model sheet
def build_gpt_oss_model(summaries: dict[str, dict]) -> dict:
    entries = build_gpt_oss_entries(summaries)
    return {
        "label": "gpt-oss-20b",
        "status": "available",
        "page_title": "scienceeval | fossils-m | gpt-oss-20b",
        "description": "Model-first fossil sheet for gpt-oss-20b from the uploaded Fossils-M result bundle.",
        "plot": "images/gpt-oss-20b-fossil-imprint.png",
        "plot_alt": "A fossil record plot of gpt-oss-20b benchmark values over benchmark introduction year.",
        "heading_label": "Model fossil",
        "coverage": "7 / 9 fossils",
        "model_type": "Open-weight model",
        "reading_rule": "First-pass suite protocol",
        "ledger_aria_label": "gpt-oss-20b benchmark ledger",
        "entries": entries,
        "plot_points": build_plot_points(entries),
    }


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
    summaries = load_result_summaries(results_dir)
    return {
        "defaults": {
            "model": "gpt-4o",
        },
        "models": {
            "gpt-4o": build_gpt4o_model(),
            "gpt-oss-20b": build_gpt_oss_model(summaries),
        },
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
    offsets = GPT_OSS_LABEL_OFFSETS if model_id == "gpt-oss-20b" else DEFAULT_LABEL_OFFSETS

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
        offset_x, offset_y, vertical_align = offsets.get(point["id"], (6, 8, "bottom"))
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
            ha="left",
            va=vertical_align,
            color=INK,
            fontsize=9.3,
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

    render_plot("gpt-oss-20b", catalog["models"]["gpt-oss-20b"], args.image_output_dir)


if __name__ == "__main__":
    run_model_fossil_sheet(parse_args())
