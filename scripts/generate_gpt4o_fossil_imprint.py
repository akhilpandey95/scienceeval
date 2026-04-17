#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


PAPER = "#f2ede4"
INK = "#171310"
MUTED = "#6f655d"
ACCENT = "#6f3f2b"
STEM = "#c8b59f"
RING = "#d9c2a7"
LAUNCH = "#b54b4b"
MODEL_INTRO_YEAR = 2024 + (4 + 13 / 31) / 12

DATA = [
    {
        "name": "BioASQ",
        "year": 2015.00,
        "value": 86.19,
        "offset": (6, 8),
        "va": "bottom",
    },
    {
        "name": "SciERC",
        "year": 2018.83,
        "value": 70.35,
        "offset": (6, 8),
        "va": "bottom",
    },
    {
        "name": "PubMedQA",
        "year": 2019.92,
        "value": 53.65,
        "offset": (2, -12),
        "va": "top",
    },
    {
        "name": "MMLU",
        "year": 2020.75,
        "value": 85.70,
        "offset": (6, 8),
        "va": "bottom",
    },
    {
        "name": "BioRED",
        "year": 2022.33,
        "value": 72.45,
        "offset": (6, -12),
        "va": "top",
    },
    {
        "name": "GPQA",
        "year": 2023.92,
        "value": 53.60,
        "offset": (6, 8),
        "va": "bottom",
    },
    {
        "name": "SciRIFF",
        "year": 2024.50,
        "value": 60.40,
        "offset": (6, -12),
        "va": "top",
    },
    {
        "name": "SimpleQA",
        "year": 2024.92,
        "value": 38.20,
        "offset": (6, 8),
        "va": "bottom",
    },
    {
        "name": "Frontier O",
        "year": 2026.05,
        "value": 12.30,
        "offset": (6, 8),
        "va": "bottom",
    },
    {
        "name": "Frontier R",
        "year": 2026.18,
        "value": 0.40,
        "offset": (6, -10),
        "va": "top",
    },
]


def render():
    fig, ax = plt.subplots(figsize=(12.6, 4.3), dpi=300)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    ax.set_xlim(2014.7, 2026.5)
    ax.set_ylim(0, 100)

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

    ax.axvline(
        MODEL_INTRO_YEAR,
        color=LAUNCH,
        linewidth=1.1,
        linestyle=(0, (1.5, 2.6)),
        alpha=0.75,
        zorder=0,
    )

    for point in DATA:
        ax.vlines(
            point["year"],
            0,
            point["value"],
            color=STEM,
            linewidth=1.0,
            alpha=0.8,
            zorder=1,
        )
        ax.scatter(
            point["year"],
            point["value"],
            s=220,
            facecolor="none",
            edgecolor=RING,
            linewidth=1.0,
            alpha=0.5,
            zorder=2,
        )
        ax.scatter(
            point["year"],
            point["value"],
            s=58,
            facecolor="#f8f4ee",
            edgecolor=ACCENT,
            linewidth=1.2,
            zorder=3,
        )
        ax.scatter(
            point["year"],
            point["value"],
            s=12,
            color=ACCENT,
            zorder=4,
        )
        ax.annotate(
            f"{point['name']}\n{point['value']:.1f}",
            xy=(point["year"], point["value"]),
            xytext=point["offset"],
            textcoords="offset points",
            ha="left",
            va=point["va"],
            color=INK,
            fontsize=9.3,
            linespacing=1.12,
            fontfamily="DejaVu Serif",
            zorder=5,
        )

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
        "Metric families: Accuracy, Label-F1, Entity-F1, G-BS-F1, Avg.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=8.0,
        fontfamily="DejaVu Serif",
    )

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

    output_dir = Path(__file__).resolve().parents[1] / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "gpt4o-fossil-imprint.svg"
    png_path = output_dir / "gpt4o-fossil-imprint.png"
    fig.savefig(svg_path, facecolor=PAPER, bbox_inches="tight", pad_inches=0.22)
    fig.savefig(png_path, facecolor=PAPER, bbox_inches="tight", pad_inches=0.22, dpi=300)
    plt.close(fig)
    print(svg_path)
    print(png_path)


if __name__ == "__main__":
    render()
