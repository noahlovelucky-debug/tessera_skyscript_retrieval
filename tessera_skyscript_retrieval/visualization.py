from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .config import ensure_dir
from .indexing import retrieve_by_modality

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _metric_rows(report: dict[str, Any]) -> list[tuple[str, float, float, float, float]]:
    rows = []
    labels = {
        "text_to_highres_teacher": "Text -> SkyCLIP teacher",
        "text_to_highres": "Text -> high-res",
        "text_to_highres_fine": "Text -> high-res fine",
        "text_to_tessera": "Text -> TESSERA",
        "highres_to_tessera_semantic": "High-res -> TESSERA",
    }
    for split, prefix in (("test", "Closed"), ("oov_test", "OOV")):
        for key, label in labels.items():
            if key not in report["splits"][split]:
                continue
            values = report["splits"][split][key]
            rows.append(
                (
                    f"{prefix}\n{label}",
                    float(values["Precision@10"]),
                    float(values["Precision@100"]),
                    float(values["nDCG@10"]),
                    float(values["nDCG@100"]),
                )
            )
    return rows


def _plot_metrics(report: dict[str, Any], output: Path) -> None:
    rows = _metric_rows(report)
    labels = [row[0] for row in rows]
    precision_10 = np.array([row[1] for row in rows]) * 100
    precision_100 = np.array([row[2] for row in rows]) * 100
    ndcg_10 = np.array([row[3] for row in rows]) * 100
    ndcg_100 = np.array([row[4] for row in rows]) * 100
    positions = np.arange(len(rows))
    width = 0.36

    fig, axes = plt.subplots(2, 1, figsize=(15.5, 10.5), dpi=160, sharex=True)
    fig.patch.set_facecolor("#f7f8fa")
    panels = (
        (axes[0], precision_10, precision_100, "Precision", "Relevant results among the top K"),
        (axes[1], ndcg_10, ndcg_100, "nDCG", "Relevance quality with higher ranks weighted more"),
    )
    for ax, values_10, values_100, metric, subtitle in panels:
        ax.set_facecolor("#f7f8fa")
        bars_10 = ax.bar(positions - width / 2, values_10, width, label=f"{metric}@10", color="#087e8b")
        bars_100 = ax.bar(positions + width / 2, values_100, width, label=f"{metric}@100", color="#d1495b")
        ax.bar_label(bars_10, fmt="%.1f", padding=3, fontsize=8)
        ax.bar_label(bars_100, fmt="%.1f", padding=3, fontsize=8)
        ax.set_title(metric, fontsize=13, pad=14, loc="left")
        ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=9, color="#555b66")
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, min(105, max(10, float(max(values_10.max(), values_100.max())) + 10)))
        ax.grid(axis="y", color="#d9dde3", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, ncols=2, loc="upper left")
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("TESSERA-SkyCLIP retrieval quality", fontsize=17, x=0.08, ha="left")
    axes[1].set_xticks(positions, labels, fontsize=8)
    fig.text(
        0.08,
        0.005,
        "Binary relevance uses exact title_id matches; broader semantic synonyms require a concept-level label map.",
        fontsize=9,
        color="#555b66",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _retrieval_rows(config: dict, query_text: str, top_k: int) -> dict[str, list[dict[str, Any]]]:
    return retrieve_by_modality(
        config,
        query_text,
        ("highres", "tessera"),
        top_k,
    )


def _draw_retrievals(query_text: str, rows: dict[str, list[dict[str, Any]]], output: Path) -> None:
    count = max(len(values) for values in rows.values())
    tile_width, image_height, caption_height = 250, 185, 104
    gap, margin, section_header = 16, 30, 34
    header_height = 86
    width = margin * 2 + count * tile_width + (count - 1) * gap
    height = header_height + 2 * (section_header + image_height + caption_height + gap) + margin
    canvas = Image.new("RGB", (width, height), "#f4f6f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), f'Text query: "{query_text}"', fill="#17202a", font=_font(25, True))
    draw.text(
        (margin, 52),
        "TESSERA matches use their geographically paired high-resolution image as the preview.",
        fill="#58616d",
        font=_font(13),
    )

    styles = {
        "highres": ("HIGH-RES IMAGE RETRIEVAL", "#087e8b"),
        "tessera": ("TESSERA RETRIEVAL / PAIRED IMAGE PREVIEW", "#d1495b"),
    }
    top = header_height
    for modality in ("highres", "tessera"):
        label, color = styles[modality]
        draw.rectangle((margin, top, width - margin, top + section_header), fill=color)
        draw.text((margin + 12, top + 7), label, fill="white", font=_font(15, True))
        image_top = top + section_header
        for column, row in enumerate(rows[modality]):
            left = margin + column * (tile_width + gap)
            with Image.open(str(row["image_path"])) as source:
                preview = ImageOps.fit(
                    source.convert("RGB"),
                    (tile_width, image_height),
                    method=Image.Resampling.LANCZOS,
                )
            canvas.paste(preview, (left, image_top))
            caption_top = image_top + image_height
            draw.rectangle(
                (left, caption_top, left + tile_width, caption_top + caption_height),
                fill="white",
            )
            draw.text(
                (left + 8, caption_top + 7),
                f'#{row["rank"]}  score {row["score"]:.3f}  {row["source"]} {row["year"]}',
                fill="#17202a",
                font=_font(13, True),
            )
            title_lines = textwrap.wrap(str(row["title"]), width=36)[:2]
            draw.multiline_text(
                (left + 8, caption_top + 29),
                "\n".join(title_lines),
                fill="#3e4854",
                font=_font(11),
                spacing=2,
            )
            draw.text(
                (left + 8, caption_top + 78),
                f'lon {float(row["center_lon"]):.4f}  lat {float(row["center_lat"]):.4f}',
                fill=color,
                font=_font(11, True),
            )
        top = image_top + image_height + caption_height + gap
    canvas.save(output, quality=95)


def visualize_results(
    config: dict,
    query_text: str = "quarry area",
    top_k: int = 5,
    output_dir: str | None = None,
) -> Path:
    destination = ensure_dir(output_dir or Path(config["training"]["output_dir"]).parents[1] / "visualizations")
    report_path = Path(config["evaluation"]["report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    slug = re.sub(r"[^a-z0-9]+", "_", query_text.lower()).strip("_") or "query"
    metrics_path = destination / "precision_ndcg_at_10_100.png"
    retrieval_path = destination / f"{slug}_retrieval.png"
    results_path = destination / f"{slug}_results.json"
    _plot_metrics(report, metrics_path)
    rows = _retrieval_rows(config, query_text, top_k)
    _draw_retrievals(query_text, rows, retrieval_path)
    results_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {metrics_path}")
    print(f"wrote {retrieval_path}")
    print(f"wrote {results_path}")
    return destination
