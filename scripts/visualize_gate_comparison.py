"""Render old/new gated retrieval candidates in one visual comparison."""

from __future__ import annotations

import argparse
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


def slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")


def draw_ranked_row(
    canvas: Image.Image,
    rows: pd.DataFrame,
    top: int,
    label: str,
    tile: int,
    gap: int,
    margin: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    relevant = int(rows["relevant"].astype(bool).sum())
    gate = float(rows["gate_global_weight"].iloc[0])
    draw.text(
        (margin, top - 25),
        f"{label}: Luna relevant {relevant}/{len(rows)} | global gate {gate:.3f}",
        fill="#17202a",
        font=font(15, True),
    )
    for offset, row in enumerate(rows.sort_values("rank").itertuples(index=False)):
        left = margin + offset * (tile + gap)
        with Image.open(row.image_path) as image:
            preview = ImageOps.fit(
                image.convert("RGB"), (tile, tile), method=Image.Resampling.LANCZOS
            )
        canvas.paste(preview, (left, top))
        color = "#16803c" if bool(row.relevant) else "#c7362f"
        draw.rectangle((left, top, left + tile - 1, top + tile - 1), outline=color, width=5)
        draw.text(
            (left + 5, top + 5),
            f"#{row.rank}",
            fill="white",
            stroke_width=2,
            stroke_fill="#111827",
            font=font(14, True),
        )


def render(query: str, old: pd.DataFrame, new: pd.DataFrame, output: Path) -> None:
    tile, gap, margin = 180, 8, 20
    header, row_label, row_gap = 55, 28, 26
    width = margin * 2 + len(old) * tile + (len(old) - 1) * gap
    first_top = header + row_label
    second_top = first_top + tile + row_gap + row_label
    height = second_top + tile + margin
    canvas = Image.new("RGB", (width, height), "#f4f6f8")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 15),
        f"Query: {query} | Top-10 retrieval comparison (green=relevant, red=not relevant)",
        fill="#17202a",
        font=font(17, True),
    )
    draw_ranked_row(canvas, old, first_top, "Old sigmoid gate", tile, gap, margin)
    draw_ranked_row(canvas, new, second_top, "Anchored local-first gate", tile, gap, margin)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=93)


def overview(paths: list[tuple[str, Path]], output: Path) -> None:
    columns, preview_width, preview_height = 2, 820, 230
    margin, gap, caption = 20, 16, 26
    rows = (len(paths) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (
            margin * 2 + columns * preview_width + (columns - 1) * gap,
            margin * 2 + rows * (preview_height + caption) + (rows - 1) * gap,
        ),
        "#f4f6f8",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (query, path) in enumerate(paths):
        left = margin + (index % columns) * (preview_width + gap)
        top = margin + (index // columns) * (preview_height + caption + gap)
        with Image.open(path) as image:
            preview = ImageOps.contain(
                image.convert("RGB"), (preview_width, preview_height), method=Image.Resampling.LANCZOS
            )
        preview_left = left + (preview_width - preview.width) // 2
        preview_top = top + (preview_height - preview.height) // 2
        canvas.paste(preview, (preview_left, preview_top))
        draw.text((left, top + preview_height + 5), query, fill="#17202a", font=font(15, True))
    canvas.save(output, quality=93)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-csv", required=True)
    parser.add_argument("--new-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--overview-queries",
        nargs="*",
        default=["farmland", "school", "airport", "river", "bridge", "industrial_building"],
    )
    args = parser.parse_args()
    old = pd.read_csv(args.old_csv)
    new = pd.read_csv(args.new_csv)
    output_dir = Path(args.output_dir)
    paths = []
    for query in old["query"].drop_duplicates():
        old_rows = old[old["query"].eq(query)]
        new_rows = new[new["query"].eq(query)]
        if len(old_rows) != len(new_rows) or len(old_rows) == 0:
            raise ValueError(f"incompatible candidate rows for query={query!r}")
        path = output_dir / f"{slug(str(query))}.jpg"
        render(str(query), old_rows, new_rows, path)
        if query in args.overview_queries:
            paths.append((str(query), path))
    order = {query: index for index, query in enumerate(args.overview_queries)}
    paths.sort(key=lambda item: order[item[0]])
    overview(paths, output_dir / "overview.jpg")


if __name__ == "__main__":
    main()
