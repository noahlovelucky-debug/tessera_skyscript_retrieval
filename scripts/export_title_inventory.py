"""Export the complete fine-grained title inventory and split counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/prepared_manifest.parquet")
    parser.add_argument("--output", default="artifacts/audits/result/title_inventory.csv")
    args = parser.parse_args()
    frame = pd.read_parquet(args.manifest, columns=["title_id", "title", "split"])
    values = (
        frame.assign(samples=1)
        .pivot_table(
            index=["title_id", "title"], columns="split", values="samples", aggfunc="sum", fill_value=0
        )
        .reset_index()
    )
    for split in ("train", "val", "test", "oov_test", "excluded_oov"):
        if split not in values:
            values[split] = 0
    values["total"] = values[["train", "val", "test", "oov_test", "excluded_oov"]].sum(axis=1)
    values = values.sort_values(["total", "title"], ascending=[False, True])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    values.to_csv(output, index=False)
    print(f"wrote {len(values)} titles to {output}")


if __name__ == "__main__":
    main()
