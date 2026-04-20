#!/usr/bin/env python3

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.ticker import MultipleLocator


PAPER = "#f2ede4"
INK = "#171310"
MUTED = "#6f655d"
LINE = "#cdbdab"
SOIL = "#e8ddcf"
HALO = "#efe5d7"
STEM = "#7d6655"

LINEAGE_COLORS = {
    "qwen2": "#8d735f",
    "qwen2inst": "#9a7a63",
    "qwenapi": "#8d6a53",
    "qwen25": "#8a766f",
    "qwen25inst": "#7f8072",
    "qwen32507i": "#7c8669",
    "qwen32507t": "#72846b",
    "qwen3nexti": "#7b8d7a",
    "qwen3nextt": "#6e857e",
    "qwen35": "#72818f",
    "qwen35moe": "#7a7791",
    "qwen36": "#8b7395",
}

ROOT_X = 0.0
ROOT_Y = 0.0
ROOT_JOIN_Y = 12.0

TEXT_HALO = [pe.withStroke(linewidth=2.6, foreground=PAPER, alpha=0.96)]


def load_rows() -> list[dict]:
    data_path = Path(__file__).resolve().parents[1] / "data" / "gpqa-qwen-models.json"
    rows = json.loads(data_path.read_text())
    for row in rows:
        row["release_dt"] = datetime.strptime(row["release_date"], "%Y-%m-%d").date()
        row["release_short"] = row["release_dt"].strftime("%m-%y")
    rows.sort(key=lambda item: (item["release_dt"], item["model"]))
    return rows


def value(row: dict) -> float:
    return row.get("gpqa_value", row.get("gpqa_diamond"))


def assign_tip_positions(rows: list[dict]) -> list[dict]:
    release_clusters: dict = {}
    for row in rows:
        release_clusters.setdefault(row["release_dt"], []).append(row)

    sorted_dates = sorted(release_clusters)
    cluster_centers = np.linspace(-12.8, 6.1, len(sorted_dates))
    cluster_meta = []

    for center_x, release_dt in zip(cluster_centers, sorted_dates):
        cluster_rows = sorted(
            release_clusters[release_dt],
            key=lambda item: (value(item), item["model"]),
        )
        size = len(cluster_rows)
        if size == 1:
            offsets = [0.0]
        elif size == 2:
            offsets = [-0.5, 0.5]
        elif size == 3:
            offsets = [-0.76, 0.0, 0.76]
        elif size == 4:
            offsets = [-1.0, -0.34, 0.34, 1.0]
        else:
            span = min(4.8, 0.42 * (size - 1))
            offsets = np.linspace(-span / 2, span / 2, size)

        for row, offset in zip(cluster_rows, offsets):
            row["tip_x"] = float(center_x + offset)
            row["release_cluster_center_x"] = float(center_x)

        cluster_meta.append(
            {
                "release_dt": release_dt,
                "center_x": float(center_x),
                "label": release_dt.strftime("%m-%y"),
                "size": size,
            }
        )

    return cluster_meta


def assign_lineage_structure(rows: list[dict]) -> list[dict]:
    groups = []
    current_rows = []
    current_lineage = None

    for row in rows:
        if row["lineage"] != current_lineage and current_rows:
            groups.append(current_rows)
            current_rows = []
        current_rows.append(row)
        current_lineage = row["lineage"]

    if current_rows:
        groups.append(current_rows)

    structures = []
    for group in groups:
        x_values = [row["tip_x"] for row in group]
        scores = [value(row) for row in group]
        join_y = ROOT_JOIN_Y if len(group) == 1 else max(ROOT_JOIN_Y + 5, min(min(scores) - 6.5, min(scores) * 0.72))
        center_x = float(np.mean(x_values))
        structure = {
            "lineage": group[0]["lineage"],
            "rows": group,
            "x_min": min(x_values),
            "x_max": max(x_values),
            "center_x": center_x,
            "join_y": join_y,
        }
        for row in group:
            row["group_center_x"] = center_x
            row["group_join_y"] = join_y
        structures.append(structure)
    return structures


def draw_shared_root(ax: plt.Axes) -> None:
    ax.plot(
        [ROOT_X, ROOT_X],
        [ROOT_Y, ROOT_JOIN_Y],
        color=STEM,
        linewidth=2.1,
        alpha=0.75,
        zorder=1,
        solid_capstyle="round",
    )


def draw_root_join(ax: plt.Axes, groups: list[dict]) -> None:
    if not groups:
        return
    left = min(group["center_x"] for group in groups)
    right = max(group["center_x"] for group in groups)
    ax.plot(
        [left, right],
        [ROOT_JOIN_Y, ROOT_JOIN_Y],
        color=STEM,
        linewidth=1.1,
        alpha=0.56,
        zorder=1,
        solid_capstyle="round",
    )


def draw_group(ax: plt.Axes, group: dict) -> None:
    center_x = group["center_x"]
    join_y = group["join_y"]
    rows = group["rows"]

    if len(rows) == 1:
        row = rows[0]
        ax.plot(
            [row["tip_x"], row["tip_x"]],
            [ROOT_JOIN_Y, value(row)],
            color=STEM,
            linewidth=1.14,
            alpha=0.7,
            zorder=1,
            solid_capstyle="round",
        )
        return

    ax.plot(
        [center_x, center_x],
        [ROOT_JOIN_Y, join_y],
        color=STEM,
        linewidth=1.1,
        alpha=0.66,
        zorder=1,
        solid_capstyle="round",
    )
    ax.plot(
        [group["x_min"], group["x_max"]],
        [join_y, join_y],
        color=STEM,
        linewidth=1.06,
        alpha=0.62,
        zorder=1,
        solid_capstyle="round",
    )


def draw_branch(ax: plt.Axes, row: dict) -> None:
    tip_x = row["tip_x"]
    tip_y = value(row)
    branch_color = LINEAGE_COLORS.get(row["lineage"], STEM)
    join_y = row["group_join_y"]

    ax.plot(
        [tip_x, tip_x],
        [join_y, tip_y],
        color=STEM,
        linewidth=1.14,
        alpha=0.7,
        zorder=1,
        solid_capstyle="round",
    )

    ax.scatter(tip_x, tip_y, s=170, facecolor="none", edgecolor=HALO, linewidth=1.2, alpha=0.9, zorder=4)
    ax.scatter(tip_x, tip_y, s=38, facecolor="#f8f3eb", edgecolor=STEM, linewidth=0.9, zorder=5)
    ax.scatter(tip_x, tip_y, s=10, color=branch_color, zorder=6)

    score_text = ax.annotate(
        f"{tip_y:.1f}",
        xy=(tip_x, tip_y),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color=INK,
        fontsize=6.3,
        fontfamily="DejaVu Serif",
        zorder=7,
    )
    score_text.set_path_effects(TEXT_HALO)


def assign_right_side_labels(rows: list[dict]) -> list[dict]:
    label_rows = sorted(rows, key=lambda item: value(item))
    min_gap = 2.28
    bottom_cap = 17.2
    top_cap = 102.2

    placed = []
    for row in label_rows:
        desired_y = value(row)
        if not placed:
            placed_y = max(desired_y, bottom_cap)
        else:
            placed_y = max(desired_y, placed[-1]["label_y"] + min_gap)
        placed.append({"row": row, "label_y": placed_y})

    overflow = placed[-1]["label_y"] - top_cap
    if overflow > 0:
        for item in placed:
            item["label_y"] -= overflow

    underflow = bottom_cap - placed[0]["label_y"]
    if underflow > 0:
        for item in placed:
            item["label_y"] += underflow

    label_x = 13.7
    elbow_x = 11.25
    for item in placed:
        item["label_x"] = label_x
        item["elbow_x"] = elbow_x
        item["text"] = format_model_label(item["row"]["model"])
    return placed


def format_model_label(text: str) -> str:
    return text


def draw_model_callouts(ax: plt.Axes, label_rows: list[dict]) -> None:
    for item in label_rows:
        row = item["row"]
        tip_x = row["tip_x"]
        tip_y = value(row)
        elbow_x = item["elbow_x"]
        label_x = item["label_x"]
        label_y = item["label_y"]
        branch_color = LINEAGE_COLORS.get(row["lineage"], STEM)

        ax.plot(
            [tip_x + 0.12, elbow_x],
            [tip_y, tip_y],
            color=branch_color,
            linewidth=0.72,
            alpha=0.32,
            zorder=2,
            solid_capstyle="round",
        )
        ax.plot(
            [elbow_x, elbow_x],
            [tip_y, label_y],
            color=branch_color,
            linewidth=0.72,
            alpha=0.32,
            zorder=2,
            solid_capstyle="round",
        )
        ax.plot(
            [elbow_x, label_x - 0.16],
            [label_y, label_y],
            color=branch_color,
            linewidth=0.72,
            alpha=0.32,
            zorder=2,
            solid_capstyle="round",
        )

        label_text = ax.text(
            label_x,
            label_y,
            item["text"],
            ha="left",
            va="center",
            color=INK,
            fontsize=5.05,
            fontfamily="DejaVu Serif",
            linespacing=1.12,
            zorder=7,
        )
        label_text.set_path_effects(TEXT_HALO)


def draw_release_labels(ax: plt.Axes, rows: list[dict]) -> None:
    month_groups: dict = {}
    for row in rows:
        month_groups.setdefault(row["release_short"], []).append(row["tip_x"])

    for month_label, x_values in month_groups.items():
        x = float(np.mean(x_values))
        ax.plot([x, x], [0, -1.7], color=LINE, linewidth=0.7, alpha=0.58, zorder=0)
        date_label = ax.text(
            x,
            -4.6,
            month_label,
            ha="center",
            va="top",
            color=MUTED,
            fontsize=6.2,
            fontfamily="DejaVu Serif",
            zorder=7,
        )
        date_label.set_path_effects(TEXT_HALO)


def render(rows: list[dict]) -> None:
    assign_tip_positions(rows)
    groups = assign_lineage_structure(rows)
    label_rows = assign_right_side_labels(rows)

    fig, ax = plt.subplots(figsize=(18.2, 10.8), dpi=300)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    ax.set_xlim(-14.8, 19.6)
    ax.set_ylim(-14, 108)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color(LINE)
    ax.spines["left"].set_linewidth(0.95)

    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.grid(axis="y", color=LINE, linewidth=0.82, linestyle=(0, (2, 4)), alpha=0.72)
    ax.tick_params(axis="y", colors=MUTED, labelsize=9, length=0, pad=6)
    ax.set_xticks([])

    ax.set_ylabel(
        "GPQA value (%)",
        color=MUTED,
        fontsize=10,
        labelpad=14,
        fontfamily="DejaVu Serif",
    )

    draw_shared_root(ax)
    draw_root_join(ax, groups)
    for group in groups:
        draw_group(ax, group)
    for row in rows:
        draw_branch(ax, row)
    draw_model_callouts(ax, label_rows)
    draw_release_labels(ax, rows)

    ax.plot([-12.9, 7.1], [0, 0], color=LINE, linewidth=0.78, alpha=0.34, zorder=0)

    ax.text(
        0.0,
        1.09,
        "GPQA Fossil",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=INK,
        fontsize=17.5,
        fontfamily="DejaVu Serif",
    )
    ax.text(
        0.0,
        1.03,
        "Benchmark specimen values following model release order.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=8.7,
        fontfamily="DejaVu Serif",
    )

    output_dir = Path(__file__).resolve().parents[1] / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "gpqa-fossil-grove-qwen.png"
    fig.savefig(png_path, facecolor=PAPER, bbox_inches="tight", pad_inches=0.24, dpi=300)
    plt.close(fig)
    print(png_path)


if __name__ == "__main__":
    render(load_rows())
