"""Select a reproducible, frequency-stratified set of title-derived audit queries."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd


_BOILERPLATE = re.compile(
    r"^\s*(?:an aerial image|a satellite image)\.\s*(?:it shows:\s*)?",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/prepared_manifest.parquet")
    parser.add_argument("--split", default="test")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--min-examples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--output",
        default="artifacts/audits/protocols/luna_100_stratified_title_queries.csv",
    )
    return parser.parse_args()


def clean_title(value: str) -> str:
    value = _BOILERPLATE.sub("", value).strip()
    return value.rstrip(".").strip()


def stable_order(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    if args.count < 3:
        raise ValueError("count must be at least 3 for frequency stratification")
    frame = pd.read_parquet(args.manifest, columns=["split", "title"])
    counts = (
        frame.loc[frame["split"].eq(args.split), "title"]
        .value_counts()
        .rename_axis("source_title")
        .rename("test_examples")
        .reset_index()
    )
    counts["query"] = counts["source_title"].map(clean_title)
    counts = counts[
        counts["test_examples"].ge(args.min_examples)
        & counts["query"].str.len().between(3, 120)
    ].copy()
    # Different prompts can reduce to the same search string; retain the better
    # represented one so each audited row is a distinct text query.
    counts = counts.sort_values(["test_examples", "source_title"], ascending=[False, True])
    counts = counts.drop_duplicates("query", keep="first").reset_index(drop=True)
    if len(counts) < args.count:
        raise ValueError(f"only {len(counts)} eligible queries for requested {args.count}")
    ordered = counts.sort_values("test_examples", ascending=False).reset_index(drop=True)
    bins = [chunk.copy() for chunk in np.array_split(ordered, 3)]
    target_sizes = [args.count // 3, args.count // 3, args.count - 2 * (args.count // 3)]
    selected = []
    for stratum, (chunk, size) in enumerate(zip(bins, target_sizes), start=1):
        chunk["_order"] = chunk["query"].map(lambda value: stable_order(value, args.seed))
        chosen = chunk.sort_values("_order").iloc[:size].copy()
        chosen["frequency_stratum"] = stratum
        selected.append(chosen)
    result = pd.concat(selected, ignore_index=True)
    result = result.sort_values(["frequency_stratum", "query"]).drop(columns="_order")
    result.insert(0, "protocol", "test-title-frequency-stratified-v1")
    result.insert(1, "seed", args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result[["query", "test_examples", "frequency_stratum"]].to_string(index=False))
    print(f"wrote {len(result)} queries to {output}")


if __name__ == "__main__":
    main()
