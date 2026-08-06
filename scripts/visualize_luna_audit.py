"""Render Top-K Luna-audited image grids with relevance-colored borders."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


def font(size: int, bold: bool = False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render(rows: pd.DataFrame, output: Path, columns: int, tile: int) -> None:
    rows = rows.sort_values("rank")
    caption = 42
    gap, margin, header = 10, 22, 52
    row_count = (len(rows) + columns - 1) // columns
    width = margin * 2 + columns * tile + (columns - 1) * gap
    height = header + row_count * (tile + caption) + (row_count - 1) * gap + margin
    canvas = Image.new("RGB", (width, height), "#f4f6f8")
    draw = ImageDraw.Draw(canvas)
    query = str(rows["query"].iloc[0])
    system = str(rows["system"].iloc[0])
    relevant = int(rows["relevant"].astype(bool).sum())
    draw.text((margin, 14), f"{system} | query: {query} | Luna relevant: {relevant}/{len(rows)}", fill="#17202a", font=font(16, True))
    for offset, row in enumerate(rows.itertuples(index=False)):
        col, grid_row = offset % columns, offset // columns
        left = margin + col * (tile + gap)
        top = header + grid_row * (tile + caption + gap)
        with Image.open(row.image_path) as image:
            preview = ImageOps.fit(image.convert("RGB"), (tile, tile), method=Image.Resampling.LANCZOS)
        canvas.paste(preview, (left, top))
        color = "#16803c" if bool(row.relevant) else "#c7362f"
        draw.rectangle((left, top, left + tile - 1, top + tile - 1), outline=color, width=6)
        draw.rectangle((left, top + tile, left + tile, top + tile + caption), fill="white")
        draw.text((left + 7, top + tile + 5), f"#{row.rank}  {'relevant' if row.relevant else 'not relevant'}", fill=color, font=font(12, True))
        evidence = textwrap.shorten(str(row.visible_evidence), width=44, placeholder="...")
        draw.text((left + 7, top + tile + 22), evidence, fill="#4b5563", font=font(8))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--tile", type=int, default=220)
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    output_dir = Path(args.output_dir)
    for (system, query), rows in frame.groupby(["system", "query"], sort=False):
        slug = "".join(character if character.isalnum() else "_" for character in str(query).lower()).strip("_")
        render(rows, output_dir / f"{system}__{slug}.jpg", args.columns, args.tile)


if __name__ == "__main__":
    main()
