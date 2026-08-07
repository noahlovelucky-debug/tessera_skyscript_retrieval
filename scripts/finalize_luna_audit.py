"""Validate incrementally saved Luna judgments and write final audit artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate-pool-size", type=int, default=39344)
    args = parser.parse_args()
    candidates = pd.read_csv(args.candidates)
    judgments = pd.read_csv(args.judgments)
    keys = ["system", "query", "rank"]
    if judgments.duplicated(keys).any():
        raise ValueError("duplicate Luna judgment keys")
    completed = judgments[["system", "query"]].drop_duplicates()
    selected = candidates.merge(completed, on=["system", "query"], validate="many_to_one")
    if len(selected) != len(judgments):
        raise ValueError("candidate and judgment row counts do not match")
    audited = selected.merge(judgments, on=keys, validate="one_to_one")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audited.to_csv(output_dir / "candidates_luna_judged.csv", index=False)
    top_k = int(audited.groupby(["system", "query"])["rank"].size().max())
    discounts = 1.0 / np.log2(np.arange(2, top_k + 2))
    systems = {}
    for system, system_rows in audited.groupby("system", sort=False):
        per_query = {}
        for query, rows in system_rows.groupby("query", sort=False):
            ranked = rows.sort_values("rank")
            relevant = ranked["relevant"].astype(bool).to_numpy()
            if len(relevant) != top_k:
                raise ValueError(f"incomplete ranked judgments for {system}:{query}")
            per_query[query] = {
                "luna_visual_precision_at_k": float(relevant.mean()),
                "luna_visual_hit_at_k": float(relevant.any()),
                "luna_discounted_relevance_at_k": float(
                    (relevant * discounts).sum() / discounts.sum()
                ),
                "luna_relevant_count": int(relevant.sum()),
                "gate_global_weight": float(ranked["gate_global_weight"].iloc[0]),
            }
        systems[system] = {
            "mean_luna_visual_precision_at_k": float(
                np.mean([row["luna_visual_precision_at_k"] for row in per_query.values()])
            ),
            "mean_luna_visual_hit_at_k": float(
                np.mean([row["luna_visual_hit_at_k"] for row in per_query.values()])
            ),
            "mean_luna_discounted_relevance_at_k": float(
                np.mean([row["luna_discounted_relevance_at_k"] for row in per_query.values()])
            ),
            "queries": per_query,
        }
    summary = {
        "model": args.model,
        "retrieval_checkpoints": {system: args.checkpoint for system in systems},
        "candidate_split": "test",
        "candidate_count": args.candidate_pool_size,
        "top_k": top_k,
        "systems": systems,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
