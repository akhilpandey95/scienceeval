#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.ticker import MultipleLocator


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "data" / "fossils-b-catalog.json"

PAPER = "#f2ede4"
INK = "#171310"
MUTED = "#6f655d"
LINE = "#cdbdab"
HALO = "#efe5d7"
STEM = "#7d6655"

ROOT_X = 0.0
ROOT_Y = 0.0
ROOT_JOIN_Y = 12.0

TEXT_HALO = [pe.withStroke(linewidth=2.6, foreground=PAPER, alpha=0.96)]


@dataclass(frozen=True)
class FamilyStyle:
    lineage_colors: dict[str, str]
    label_field: str = "model"
    wrap_long_labels: bool = False
    wrap_tokens: tuple[str, ...] = ()
    cluster_range: tuple[float, float] = (-9.5, 4.7)
    wide_cluster_step: float | None = None
    wide_cluster_max_span: float | None = None
    label_min_gap: float = 3.0
    label_bottom_cap: float = 26.0
    label_top_cap: float = 98.8
    label_x: float = 10.55
    elbow_x: float = 8.7
    score_fontsize: float = 7.25
    label_fontsize: float = 6.6
    label_linespacing: float = 1.0
    date_fontsize: float = 6.85
    figure_size: tuple[float, float] = (16.2, 9.2)
    x_limits: tuple[float, float] = (-11.3, 14.2)
    y_limits: tuple[float, float] = (-14, 108)
    baseline_x: tuple[float, float] = (-9.2, 5.3)
    save_svg: bool = False


FAMILY_STYLES = {
    "openai": FamilyStyle(
        lineage_colors={
            "gpt35": "#a38366",
            "gpt4": "#8d735f",
            "gpt4o": "#8a766f",
            "gpt45": "#a56f56",
            "gpt41": "#7f8072",
            "gpt5": "#7c8669",
            "gpt51": "#72846b",
            "gpt52": "#6a7f67",
            "gpt53": "#6c7e70",
            "gpt54": "#72818f",
        },
        save_svg=True,
    ),
    "anthropic": FamilyStyle(
        lineage_colors={
            "claude3": "#8d735f",
            "claude35": "#8a766f",
            "claude37": "#9a745f",
            "claude4": "#7f8072",
            "claude41": "#7c8669",
            "claude45": "#6f7f6a",
            "claude46": "#72818f",
            "claude47": "#7c7a86",
        },
        save_svg=True,
    ),
    "meta": FamilyStyle(
        lineage_colors={
            "llama2": "#8d735f",
            "llama3": "#8a766f",
            "llama31": "#7f8072",
            "llama32": "#7c8669",
            "llama33": "#72818f",
            "llama4": "#8b7395",
        },
        cluster_range=(-10.8, 4.8),
        wide_cluster_step=0.44,
        wide_cluster_max_span=3.4,
        label_min_gap=3.0,
        label_bottom_cap=22.0,
        label_top_cap=100.0,
        label_x=11.2,
        elbow_x=9.1,
        score_fontsize=6.45,
        label_fontsize=6.05,
        date_fontsize=6.2,
        figure_size=(15.9, 9.8),
        x_limits=(-12.0, 14.9),
        baseline_x=(-10.7, 5.4),
    ),
    "qwen": FamilyStyle(
        lineage_colors={
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
        },
        wrap_long_labels=True,
        wrap_tokens=("-Instruct", "-Thinking"),
        cluster_range=(-12.8, 6.1),
        wide_cluster_step=0.42,
        wide_cluster_max_span=4.8,
        label_min_gap=2.55,
        label_bottom_cap=16.2,
        label_top_cap=106.0,
        label_x=12.6,
        elbow_x=10.5,
        score_fontsize=6.45,
        label_fontsize=5.45,
        label_linespacing=1.08,
        date_fontsize=6.2,
        figure_size=(17.8, 12.2),
        x_limits=(-14.4, 18.2),
        y_limits=(-15, 112),
        baseline_x=(-12.8, 6.2),
    ),
    "google": FamilyStyle(
        lineage_colors={
            "gemini15": "#8d735f",
            "gemini20": "#9a745f",
            "gemma3it": "#8a766f",
            "gemini25": "#7f8072",
            "gemma3n": "#7c8669",
            "gemini3": "#72846b",
            "gemini31": "#72818f",
            "gemma4it": "#8b7395",
        },
        label_field="model",
        cluster_range=(-11.6, 5.6),
        wide_cluster_step=0.42,
        wide_cluster_max_span=4.0,
        label_min_gap=4.4,
        label_bottom_cap=18.0,
        label_top_cap=112.0,
        label_x=16.5,
        elbow_x=13.2,
        score_fontsize=6.45,
        label_fontsize=5.8,
        label_linespacing=1.14,
        date_fontsize=6.2,
        figure_size=(20.0, 12.4),
        x_limits=(-13.6, 24.0),
        y_limits=(-15, 122),
        baseline_x=(-11.7, 6.5),
    ),
}


def load_catalog() -> dict[str, dict]:
    catalog = json.loads(CATALOG_PATH.read_text())
    return catalog["specimens"]["gpqa"]["families"]


def resolve_family_ids(requested: list[str], catalog_families: dict[str, dict]) -> list[str]:
    available_ids = [family_id for family_id in catalog_families if family_id in FAMILY_STYLES]
    if not requested or requested == ["all"]:
        return available_ids

    if "all" in requested:
        raise ValueError("Use either explicit family ids or 'all', not both.")

    unknown = [family_id for family_id in requested if family_id not in available_ids]
    if unknown:
        raise ValueError(f"Unknown family ids: {', '.join(unknown)}")

    return requested


def detect_score_key(rows: list[dict]) -> str:
    if any(row.get("gpqa_diamond") is not None for row in rows):
        return "gpqa_diamond"
    if any(row.get("gpqa_value") is not None for row in rows):
        return "gpqa_value"
    raise ValueError("Could not find GPQA score field in rows.")


def load_rows(data_path: Path) -> tuple[list[dict], str]:
    raw_rows = json.loads(data_path.read_text())
    score_key = detect_score_key(raw_rows)
    score_axis_label = "GPQA Diamond (%)" if score_key == "gpqa_diamond" else "GPQA value (%)"

    rows = []
    for raw_row in raw_rows:
        score = raw_row.get(score_key)
        if score is None:
            continue

        row = dict(raw_row)
        release_dt = datetime.strptime(row["release_date"], "%Y-%m-%d").date()
        row["score"] = float(score)
        row["release_dt"] = release_dt
        row["release_short"] = release_dt.strftime("%m-%y")
        rows.append(row)

    rows.sort(key=lambda item: (item["release_dt"], item["model"]))
    return rows, score_axis_label


def cluster_offsets(size: int, style: FamilyStyle) -> list[float]:
    if size == 1:
        return [0.0]
    if size == 2:
        return [-0.5, 0.5]
    if size == 3:
        return [-0.76, 0.0, 0.76]
    if size == 4:
        return [-1.0, -0.34, 0.34, 1.0]
    if style.wide_cluster_step is None or style.wide_cluster_max_span is None:
        return np.linspace(-1.05, 1.05, size).tolist()

    span = min(style.wide_cluster_max_span, style.wide_cluster_step * (size - 1))
    return np.linspace(-span / 2, span / 2, size).tolist()


def assign_tip_positions(rows: list[dict], style: FamilyStyle) -> None:
    release_clusters: dict = {}
    for row in rows:
        release_clusters.setdefault(row["release_dt"], []).append(row)

    sorted_dates = sorted(release_clusters)
    cluster_centers = np.linspace(style.cluster_range[0], style.cluster_range[1], len(sorted_dates))

    for center_x, release_dt in zip(cluster_centers, sorted_dates):
        cluster_rows = sorted(
            release_clusters[release_dt],
            key=lambda item: (item["score"], item["model"]),
        )
        offsets = cluster_offsets(len(cluster_rows), style)

        for row, offset in zip(cluster_rows, offsets):
            row["tip_x"] = float(center_x + offset)


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
        scores = [row["score"] for row in group]
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
            [ROOT_JOIN_Y, row["score"]],
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


def wrap_label(text: str, style: FamilyStyle) -> str:
    if not style.wrap_long_labels or "\n" in text:
        return text

    for token in style.wrap_tokens:
        if token in text:
            return text.replace(token, f"\n{token.lstrip('-')}", 1)

    break_chars = [index for index, char in enumerate(text) if char in {"-", " "}]
    if not break_chars or len(text) < 18:
        return text

    midpoint = len(text) / 2
    split_at = min(break_chars, key=lambda index: abs(index - midpoint))
    if text[split_at] == " ":
        return f"{text[:split_at]}\n{text[split_at + 1:]}"

    return f"{text[:split_at]}\n{text[split_at + 1:]}"


def model_label(row: dict, style: FamilyStyle) -> str:
    label = row.get(style.label_field) or row["model"]
    return wrap_label(label, style)


def draw_branch(ax: plt.Axes, row: dict, style: FamilyStyle) -> None:
    tip_x = row["tip_x"]
    tip_y = row["score"]
    branch_color = style.lineage_colors.get(row["lineage"], STEM)
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
        fontsize=style.score_fontsize,
        fontfamily="DejaVu Serif",
        zorder=7,
    )
    score_text.set_path_effects(TEXT_HALO)


def assign_right_side_labels(rows: list[dict], style: FamilyStyle) -> list[dict]:
    label_rows = sorted(rows, key=lambda item: item["score"])
    placed = []

    for row in label_rows:
        desired_y = row["score"]
        if not placed:
            placed_y = max(desired_y, style.label_bottom_cap)
        else:
            placed_y = max(desired_y, placed[-1]["label_y"] + style.label_min_gap)
        placed.append({"row": row, "label_y": placed_y})

    overflow = placed[-1]["label_y"] - style.label_top_cap
    if overflow > 0:
        for item in placed:
            item["label_y"] -= overflow

    underflow = style.label_bottom_cap - placed[0]["label_y"]
    if underflow > 0:
        for item in placed:
            item["label_y"] += underflow

    for item in placed:
        item["label_x"] = style.label_x
        item["elbow_x"] = style.elbow_x
        item["text"] = model_label(item["row"], style)

    return placed


def draw_model_callouts(ax: plt.Axes, label_rows: list[dict], style: FamilyStyle) -> None:
    for item in label_rows:
        row = item["row"]
        tip_x = row["tip_x"]
        tip_y = row["score"]
        elbow_x = item["elbow_x"]
        label_x = item["label_x"]
        label_y = item["label_y"]
        branch_color = style.lineage_colors.get(row["lineage"], STEM)

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
            fontsize=style.label_fontsize,
            fontfamily="DejaVu Serif",
            linespacing=style.label_linespacing,
            zorder=7,
        )
        label_text.set_path_effects(TEXT_HALO)


def draw_release_labels(ax: plt.Axes, rows: list[dict], style: FamilyStyle) -> None:
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
            fontsize=style.date_fontsize,
            fontfamily="DejaVu Serif",
            zorder=7,
        )
        date_label.set_path_effects(TEXT_HALO)


def render_family(rows: list[dict], style: FamilyStyle, score_axis_label: str, output_stem: Path) -> list[Path]:
    assign_tip_positions(rows, style)
    groups = assign_lineage_structure(rows)
    label_rows = assign_right_side_labels(rows, style)

    fig, ax = plt.subplots(figsize=style.figure_size, dpi=300)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    ax.set_xlim(*style.x_limits)
    ax.set_ylim(*style.y_limits)

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
        score_axis_label,
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
        draw_branch(ax, row, style)
    draw_model_callouts(ax, label_rows, style)
    draw_release_labels(ax, rows, style)

    ax.plot(list(style.baseline_x), [0, 0], color=LINE, linewidth=0.78, alpha=0.34, zorder=0)

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

    outputs = []
    png_path = output_stem.with_suffix(".png")
    fig.savefig(png_path, facecolor=PAPER, bbox_inches="tight", pad_inches=0.24, dpi=300)
    outputs.append(png_path)

    if style.save_svg:
        svg_path = output_stem.with_suffix(".svg")
        fig.savefig(svg_path, facecolor=PAPER, bbox_inches="tight", pad_inches=0.24)
        outputs.insert(0, svg_path)

    plt.close(fig)
    return outputs


def generate_family(family_id: str, family_meta: dict, style: FamilyStyle, output_dir: Path) -> list[Path]:
    data_path = REPO_ROOT / family_meta["data"]
    plot_path = Path(family_meta["plot"])
    output_stem = output_dir / plot_path.stem
    rows, score_axis_label = load_rows(data_path)
    return render_family(rows, style, score_axis_label, output_stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GPQA fossil grove plots.")
    parser.add_argument(
        "families",
        nargs="*",
        help="Family ids to generate (openai, anthropic, meta, qwen) or 'all'. Defaults to all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "images",
        help="Directory to write generated assets into.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog_families = load_catalog()
    try:
        family_ids = resolve_family_ids(args.families, catalog_families)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for family_id in family_ids:
        outputs = generate_family(
            family_id=family_id,
            family_meta=catalog_families[family_id],
            style=FAMILY_STYLES[family_id],
            output_dir=args.output_dir,
        )
        for path in outputs:
            print(path)


if __name__ == "__main__":
    main()
