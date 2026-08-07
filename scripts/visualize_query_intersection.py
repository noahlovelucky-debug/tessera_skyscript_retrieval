"""Build and render a hard intersection of two ranked image-retrieval result sets."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--first-query", required=True)
    parser.add_argument("--second-query", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-image", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    frame = pd.read_csv(args.candidates)
    first = frame[frame["query"].eq(args.first_query)].set_index("sample_id")
    second = frame[frame["query"].eq(args.second_query)].set_index("sample_id")
    shared = first.join(second, lsuffix="_first", rsuffix="_second", how="inner")
    shared["balanced_rank"] = shared[["rank_first", "rank_second"]].max(axis=1)
    shared["rank_sum"] = shared["rank_first"] + shared["rank_second"]
    shared = shared.sort_values(["balanced_rank", "rank_sum"]).reset_index()
    shared["intersection_rank"] = range(1, len(shared) + 1)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    shared.to_csv(args.output_csv, index=False)

    selected = shared.head(args.top_k)
    tile, gap, margin, caption = 220, 12, 26, 42
    width = margin * 2 + len(selected) * tile + max(0, len(selected) - 1) * gap
    canvas = Image.new("RGB", (width, tile + caption + 104), "#f4f6f8")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 16),
        f"Hard intersection: {args.first_query} AND {args.second_query}",
        fill="#17202a",
        font=font(23, True),
    )
    draw.text(
        (margin, 49),
        f"Top-{args.top_k} of {len(shared)} shared Top-100 images, ordered by the worse of the two ranks.",
        fill="#4b5563",
        font=font(12),
    )
    for offset, row in enumerate(selected.itertuples(index=False)):
        left = margin + offset * (tile + gap)
        with Image.open(row.image_path_first) as image:
            preview = ImageOps.fit(image.convert("RGB"), (tile, tile), method=Image.Resampling.LANCZOS)
        canvas.paste(preview, (left, 90))
        draw.rectangle((left, 90, left + tile - 1, 90 + tile - 1), outline="#2563eb", width=5)
        draw.rectangle((left, 90 + tile, left + tile, 90 + tile + caption), fill="#ffffff")
        draw.text(
            (left + 7, 90 + tile + 6),
            f"#{row.intersection_rank}  ship #{row.rank_first}",
            fill="#17202a",
            font=font(11, True),
        )
        draw.text(
            (left + 7, 90 + tile + 23),
            f"container #{row.rank_second}",
            fill="#4b5563",
            font=font(10),
        )
    Path(args.output_image).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output_image, quality=94)


if __name__ == "__main__":
    main()
