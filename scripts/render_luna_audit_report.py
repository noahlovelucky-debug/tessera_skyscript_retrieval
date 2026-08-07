"""Render reproducible Markdown tables from Luna retrieval-audit summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol")
    parser.add_argument("--comparison-summary")
    return parser.parse_args()


def system_summary(path: str) -> tuple[dict, pd.DataFrame]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    systems = payload["systems"]
    if len(systems) != 1:
        raise ValueError("report renderer expects one audited retrieval system")
    name, values = next(iter(systems.items()))
    rows = [
        {
            "query": query,
            "p10": item["luna_visual_precision_at_k"],
            "ndcg10": item["luna_discounted_relevance_at_k"],
            "hit10": item["luna_visual_hit_at_k"],
            "relevant": item["luna_relevant_count"],
            "gate": item["gate_global_weight"],
        }
        for query, item in values["queries"].items()
    ]
    return name, pd.DataFrame(rows), values


def number(value: float) -> str:
    return f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    name, rows, values = system_summary(args.summary)
    protocol = pd.read_csv(args.protocol) if args.protocol else None
    if protocol is not None:
        rows = protocol.merge(rows, on="query", validate="one_to_one")
        rows = rows.sort_values(["frequency_stratum", "query"])
    else:
        rows = rows.sort_values("query")
    lines = [
        "# Anchored High-Resolution Gate: Luna Audit",
        "",
        "Luna judges retrieval relevance only from pixels. Each query is sent with its ranked "
        "Top-10 images in a single request; the response contains one boolean judgment per rank.",
        "",
        "## 100-Class Result",
        "",
        f"- System: `{name}`",
        f"- Checkpoint: `{args.summary}`",
        f"- Mean P@10: `{number(values['mean_luna_visual_precision_at_k'])}`",
        f"- Mean discounted relevance@10: `{number(values['mean_luna_discounted_relevance_at_k'])}`",
        f"- Mean Hit@10: `{number(values['mean_luna_visual_hit_at_k'])}`",
        "",
    ]
    if protocol is not None:
        grouped = rows.groupby("frequency_stratum")[["p10", "ndcg10", "hit10"]].mean()
        lines += [
            "Frequency strata use the fixed title-query protocol in the linked CSV: 1 is high "
            "test support, 2 is medium, and 3 is lower support.",
            "",
            "| Stratum | Mean P@10 | Mean discounted relevance@10 | Mean Hit@10 |",
            "|---:|---:|---:|---:|",
        ]
        for stratum, item in grouped.iterrows():
            lines.append(
                f"| {int(stratum)} | {number(item.p10)} | {number(item.ndcg10)} | {number(item.hit10)} |"
            )
        lines += ["", "## Per-Class Results", ""]
        lines += [
            "| Query | Test examples | Stratum | P@10 | Discounted relevance@10 | Relevant / 10 | Global gate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for item in rows.itertuples(index=False):
            query = str(item.query).replace("|", "\\|")
            lines.append(
                f"| {query} | {int(item.test_examples)} | {int(item.frequency_stratum)} | "
                f"{number(item.p10)} | {number(item.ndcg10)} | {int(item.relevant)} | {number(item.gate)} |"
            )
    else:
        lines += ["## Per-Class Results", "", "| Query | P@10 | Discounted relevance@10 | Relevant / 10 | Global gate |", "|---|---:|---:|---:|---:|"]
        for item in rows.itertuples(index=False):
            query = str(item.query).replace("|", "\\|")
            lines.append(
                f"| {query} | {number(item.p10)} | {number(item.ndcg10)} | "
                f"{int(item.relevant)} | {number(item.gate)} |"
            )
    if args.comparison_summary:
        old_name, old_rows, old_values = system_summary(args.comparison_summary)
        joined = old_rows.merge(rows[["query", "p10", "ndcg10"]], on="query", suffixes=("_old", "_new"))
        lines += [
            "",
            "## 12-Class Gate Comparison",
            "",
            f"Baseline `{old_name}`: P@10 `{number(old_values['mean_luna_visual_precision_at_k'])}`, "
            f"discounted relevance `{number(old_values['mean_luna_discounted_relevance_at_k'])}`.",
            "",
            "| Query | Baseline P@10 | Anchored P@10 | Baseline discounted | Anchored discounted |",
            "|---|---:|---:|---:|---:|",
        ]
        for item in joined.sort_values("query").itertuples(index=False):
            query = str(item.query).replace("|", "\\|")
            lines.append(
                f"| {query} | {number(item.p10_old)} | {number(item.p10_new)} | "
                f"{number(item.ndcg10_old)} | {number(item.ndcg10_new)} |"
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
