"""Create compact precision and rank-relevance visuals for audited retrieval runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


COLORS = {"old_gate": "#2563eb", "anchored_gate": "#16a34a", "no_gate": "#d97706"}


def font(size: int, bold: bool = False):
    try:
        return ImageFont.truetype(
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def parse_run(value: str) -> tuple[str, Path]:
    label, path = value.split("=", 1)
    return label, Path(path)


def load_runs(values: list[str]) -> dict[str, pd.DataFrame]:
    runs = {}
    for value in values:
        label, path = parse_run(value)
        frame = pd.read_csv(path).sort_values("rank")
        if frame["query"].nunique() != 1:
            raise ValueError(f"expected exactly one query in {path}")
        runs[label] = frame
    return runs


def summary_rows(runs: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for label, frame in runs.items():
        relevant = frame["relevant"].astype(bool).to_numpy()
        discounts = 1.0 / np.log2(np.arange(2, len(relevant) + 2))
        rows.append(
            {
                "label": label,
                "gate": frame["gate_global_weight"].iloc[0],
                "p10": relevant[:10].mean(),
                "p25": relevant[:25].mean(),
                "p50": relevant[:50].mean(),
                "p100": relevant.mean(),
                "discounted": (relevant * discounts).sum() / discounts.sum(),
                "relevant": int(relevant.sum()),
            }
        )
    return rows


def render_strips(runs: dict[str, pd.DataFrame], output: Path) -> None:
    rows = summary_rows(runs)
    margin, header, row_height, bar_left, block, gap = 34, 112, 62, 260, 12, 2
    width = bar_left + 100 * (block + gap) + margin
    height = header + len(rows) * row_height + margin
    canvas = Image.new("RGB", (width, height), "#f7f8fa")
    draw = ImageDraw.Draw(canvas)
    query = str(next(iter(runs.values()))["query"].iloc[0])
    draw.text((margin, 22), f"Luna pixel audit: {query} Top-100", fill="#17202a", font=font(24, True))
    draw.text((margin, 56), "Each block is one rank: green relevant, red not relevant. Labels show prefix precision.", fill="#4b5563", font=font(14))
    for rank in range(10, 101, 10):
        x = bar_left + (rank - 1) * (block + gap)
        draw.text((x - 8, 88), str(rank), fill="#6b7280", font=font(11))
    for index, (row, (label, frame)) in enumerate(zip(rows, runs.items())):
        top = header + index * row_height
        gate = "fixed 0.35/0.65" if pd.isna(row["gate"]) else f"global gate {row['gate']:.3f}"
        draw.text((margin, top), label.replace("_", " "), fill=COLORS.get(label, "#111827"), font=font(16, True))
        draw.text((margin, top + 23), f"{row['relevant']}/100 | {gate}", fill="#374151", font=font(12))
        for rank, relevant in enumerate(frame["relevant"].astype(bool), start=1):
            left = bar_left + (rank - 1) * (block + gap)
            color = "#18803d" if relevant else "#c7362f"
            draw.rectangle((left, top + 7, left + block, top + 41), fill=color)
        draw.text((bar_left, top + 45), f"P@10 {row['p10']:.2f}   P@25 {row['p25']:.2f}   P@50 {row['p50']:.2f}   P@100 {row['p100']:.2f}", fill="#17202a", font=font(12, True))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94)


def render_curve(runs: dict[str, pd.DataFrame], output: Path) -> None:
    width, height, margin = 1200, 680, 90
    canvas = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 25), "River retrieval: precision at each prefix", fill="#17202a", font=font(25, True))
    left, top, right, bottom = margin, 100, width - 50, height - margin
    draw.rectangle((left, top, right, bottom), outline="#9ca3af", width=2)
    for value in np.linspace(0.0, 1.0, 6):
        y = bottom - int(value * (bottom - top))
        draw.line((left, y, right, y), fill="#e5e7eb", width=1)
        draw.text((25, y - 8), f"{value:.1f}", fill="#6b7280", font=font(13))
    for rank in range(0, 101, 10):
        x = left + int(rank / 100 * (right - left))
        draw.line((x, top, x, bottom), fill="#f1f5f9", width=1)
        draw.text((x - 7, bottom + 12), str(rank), fill="#6b7280", font=font(13))
    for index, (label, frame) in enumerate(runs.items()):
        relevant = frame["relevant"].astype(bool).to_numpy()
        precision = np.cumsum(relevant) / np.arange(1, len(relevant) + 1)
        points = [
            (
                left + int((rank + 1) / len(relevant) * (right - left)),
                bottom - int(value * (bottom - top)),
            )
            for rank, value in enumerate(precision)
        ]
        draw.line(points, fill=COLORS.get(label, "#111827"), width=4)
        y = 55 + index * 25
        draw.rectangle((right - 295, y + 3, right - 275, y + 17), fill=COLORS.get(label, "#111827"))
        draw.text((right - 267, y), label.replace("_", " "), fill="#17202a", font=font(14, True))
    draw.text((left, bottom + 43), "rank k", fill="#374151", font=font(15, True))
    draw.text((10, top - 25), "Precision@k", fill="#374151", font=font(15, True))
    canvas.save(output, quality=94)


def render_markdown(runs: dict[str, pd.DataFrame], output: Path) -> None:
    rows = summary_rows(runs)
    lines = [
        "# River Top-100 High-Resolution Retrieval Audit",
        "",
        "Luna judged each candidate solely from pixels. Every method searched the same 39,344-image test pool for `river`; each Top-100 list was sent in ten 10-image batches.",
        "",
        "| Method | Global/local rule | Relevant / 100 | P@10 | P@25 | P@50 | P@100 | Discounted relevance@100 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    rules = {
        "old_gate": "text sigmoid gate",
        "anchored_gate": "anchored text gate, local-first",
        "no_gate": "fixed 0.35 global / 0.65 local",
    }
    for row in rows:
        lines.append(
            f"| {row['label']} | {rules.get(row['label'], '')} | {row['relevant']} | {row['p10']:.4f} | {row['p25']:.4f} | {row['p50']:.4f} | {row['p100']:.4f} | {row['discounted']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The anchored gate improves the first ten river candidates, but its relevance falls below the old gate as the candidate budget grows. The old gate is therefore stronger for a large candidate pool; the anchored gate is stronger for a short coarse-search first page. The fixed no-gate high-resolution baseline trails both at Top-100.",
        "",
        "Visuals: `relevance_strips.jpg` shows every judged rank; `precision_curve.jpg` shows P@k; the `top100_grids/` directory contains the complete 100-image grids.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="label=audited CSV")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    runs = load_runs(args.run)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    render_strips(runs, output / "relevance_strips.jpg")
    render_curve(runs, output / "precision_curve.jpg")
    render_markdown(runs, output / "RIVER_TOP100_LUNA_COMPARISON.md")


if __name__ == "__main__":
    main()
