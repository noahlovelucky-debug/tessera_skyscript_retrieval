"""Render coarse and Sol-reranked Top-10 image grids using Terra labels."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


def font(size: int, bold: bool = False):
    try:
        return ImageFont.truetype(
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def render_row(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    rows: pd.DataFrame,
    label: str,
    top: int,
    tile: int,
    gap: int,
    margin: int,
) -> None:
    draw.text((margin, top - 31), label, fill="#17202a", font=font(18, True))
    for offset, row in enumerate(rows.itertuples(index=False)):
        left = margin + offset * (tile + gap)
        with Image.open(row.image_path) as image:
            preview = ImageOps.fit(
                image.convert("RGB"), (tile, tile), method=Image.Resampling.LANCZOS
            )
        canvas.paste(preview, (left, top))
        relevant = bool(row.relevant)
        color = "#16803c" if relevant else "#c7362f"
        draw.rectangle((left, top, left + tile - 1, top + tile - 1), outline=color, width=6)
        label_top = top + tile
        draw.rectangle((left, label_top, left + tile, label_top + 38), fill="#ffffff")
        draw.text(
            (left + 7, label_top + 5),
            f"#{int(row.display_rank)}  {'relevant' if relevant else 'not relevant'}",
            fill=color,
            font=font(12, True),
        )
        evidence = textwrap.shorten(str(row.terra_visible_evidence), width=42, placeholder="...")
        draw.text((left + 7, label_top + 22), evidence, fill="#4b5563", font=font(8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sol-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query", help="Select one query when the CSV contains multiple runs")
    parser.add_argument("--title", default="Caption-OOV query: container ship")
    args = parser.parse_args()
    frame = pd.read_csv(args.sol_csv)
    if args.query:
        frame = frame[frame["query"].eq(args.query)].copy()
    if frame["query"].nunique() != 1 or len(frame) != 100:
        raise ValueError("expected one query with exactly 100 Sol-reranked candidates")
    if "terra_visible_evidence" not in frame or "relevant" not in frame:
        raise ValueError("Sol/Terra CSV is missing Terra judgment columns")
    coarse = frame.sort_values("coarse_rank").head(10).copy()
    coarse["display_rank"] = coarse["coarse_rank"]
    reranked = frame.sort_values("rank").head(10).copy()
    reranked["display_rank"] = reranked["rank"]
    tile, gap, margin = 205, 10, 26
    width = margin * 2 + 10 * tile + 9 * gap
    header, row_height = 120, tile + 78
    canvas = Image.new("RGB", (width, header + 2 * row_height + 24), "#f4f6f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 15), args.title, fill="#17202a", font=font(24, True))
    draw.text(
        (margin, 52),
        "Green/red borders are independent Terra pixel judgments. The query phrase is absent from training captions.",
        fill="#4b5563",
        font=font(12),
    )
    render_row(canvas, draw, coarse, "High-resolution gated coarse Top-10", header, tile, gap, margin)
    render_row(
        canvas,
        draw,
        reranked,
        "Sol reranked Top-10",
        header + row_height,
        tile,
        gap,
        margin,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94)


if __name__ == "__main__":
    main()
