"""Rerank retrieved aerial images with Sol and independently audit them with Terra."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="CSV with query, rank, and image_path columns")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--queries", nargs="+", help="Defaults to every query in --candidates")
    parser.add_argument("--base-url", default="http://ai.spacebus.org.cn/v1")
    parser.add_argument("--sol-model", default="gpt-5.6-sol")
    parser.add_argument("--terra-model", default="gpt-5.6-terra")
    parser.add_argument("--stage1-batch-size", type=int, default=10)
    parser.add_argument("--stage2-top-n", type=int, default=30)
    parser.add_argument("--terra-batch-size", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=768)
    return parser.parse_args()


def image_payload(path: str, max_size: int) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_size, max_size))
        stream = BytesIO()
        image.save(stream, format="JPEG", quality=85, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def extract_text(response: dict) -> str:
    values = [
        content["text"]
        for item in response.get("output", [])
        for content in item.get("content", [])
        if content.get("type") == "output_text" and isinstance(content.get("text"), str)
    ]
    if not values and isinstance(response.get("output_text"), str):
        values = [response["output_text"]]
    if not values:
        raise ValueError("response contains no output text")
    return "\n".join(values).strip().removeprefix("```json").removesuffix("```").strip()


def call_json(api_key: str, base_url: str, model: str, content: list[dict], max_output_tokens: int) -> tuple[dict, dict]:
    payload = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": max_output_tokens,
    }
    request = Request(
        f"{base_url.rstrip('/')}/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=300) as stream:
                response = json.loads(stream.read().decode("utf-8"))
            return json.loads(extract_text(response)), response
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"{model} request failed after retries: {error}")


def ranked_content(prompt: str, rows: pd.DataFrame, image_size: int) -> list[dict]:
    content: list[dict] = [{"type": "input_text", "text": prompt}]
    for row in rows.sort_values("coarse_rank").itertuples(index=False):
        content.append({"type": "input_text", "text": f"Candidate rank {int(row.coarse_rank)}:"})
        content.append({"type": "input_image", "image_url": image_payload(str(row.image_path), image_size)})
    return content


def sol_stage1(
    query: str, rows: pd.DataFrame, api_key: str, args: argparse.Namespace
) -> tuple[list[dict], dict]:
    ranks = rows["coarse_rank"].astype(int).tolist()
    prompt = (
        f"You are reranking aerial images for the query {query!r}. Judge only visible pixels; "
        "do not infer hidden metadata or titles. Return JSON only as "
        '{"scores":[{"rank":1,"relevance":0,"confidence":0.0,"visible_evidence":"..."}]}. '
        f"Return exactly these ranks: {json.dumps(ranks)}. relevance is an integer: 4=direct, "
        "unambiguous match; 3=clear match; 2=plausible but weak or mixed; 1=tangential; 0=absent."
    )
    parsed, raw = call_json(
        api_key, args.base_url, args.sol_model, ranked_content(prompt, rows, args.image_size), 2200
    )
    values = parsed.get("scores") if isinstance(parsed, dict) else None
    if not isinstance(values, list) or sorted(int(x.get("rank", -1)) for x in values) != sorted(ranks):
        raise ValueError(f"Sol stage 1 returned invalid ranks for {query!r}")
    return values, raw


def sol_stage2(
    query: str, rows: pd.DataFrame, api_key: str, args: argparse.Namespace
) -> tuple[list[int], dict]:
    ranks = rows["coarse_rank"].astype(int).tolist()
    prompt = (
        f"You are doing the final aerial-image rerank for query {query!r}. Rank every candidate "
        "from most to least visually relevant based only on pixels. Do not use hidden metadata or titles. "
        'Return JSON only as {"ordered_ranks":[1,2,...]}. Include every provided rank exactly once: '
        + json.dumps(ranks)
    )
    parsed, raw = call_json(
        api_key, args.base_url, args.sol_model, ranked_content(prompt, rows, args.image_size), 2600
    )
    ordered = parsed.get("ordered_ranks") if isinstance(parsed, dict) else None
    if not isinstance(ordered, list) or sorted(int(value) for value in ordered) != sorted(ranks):
        raise ValueError(f"Sol stage 2 returned invalid ranks for {query!r}")
    return [int(value) for value in ordered], raw


def terra_judge(
    query: str, rows: pd.DataFrame, api_key: str, args: argparse.Namespace
) -> tuple[list[dict], dict]:
    ranks = rows["rank"].astype(int).tolist()
    prompt = (
        f"You are independently auditing aerial-image retrieval for the query {query!r}. "
        "Judge every image solely from visible pixels. Never infer metadata, titles, prior rankings, "
        "or another model's score. An image is relevant when the object or land-use is visibly present. "
        'Return JSON only as {"judgments":[{"rank":1,"relevant":true,"confidence":0.0,'
        '"visible_evidence":"..."}]}. Include exactly these ranks: '
        + json.dumps(ranks)
    )
    content = [{"type": "input_text", "text": prompt}]
    for row in rows.sort_values("rank").itertuples(index=False):
        content.append({"type": "input_text", "text": f"Candidate final rank {int(row.rank)}:"})
        content.append({"type": "input_image", "image_url": image_payload(str(row.image_path), args.image_size)})
    parsed, raw = call_json(api_key, args.base_url, args.terra_model, content, 2200)
    values = parsed.get("judgments") if isinstance(parsed, dict) else None
    if not isinstance(values, list) or sorted(int(x.get("rank", -1)) for x in values) != sorted(ranks):
        raise ValueError(f"Terra returned invalid ranks for {query!r}")
    return values, raw


def metric_summary(frame: pd.DataFrame, order_column: str = "rank") -> dict[str, float]:
    relevant = frame.sort_values(order_column)["relevant"].astype(bool).to_numpy()
    total = int(relevant.sum())
    values: dict[str, float] = {"relevant_in_pool": float(total)}
    for cutoff in (10, 25, 50, 100):
        selected = relevant[:cutoff]
        discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))
        ideal = discounts[: min(total, cutoff)].sum()
        values[f"precision_at_{cutoff}"] = float(selected.mean())
        values[f"pool_ndcg_at_{cutoff}"] = float((selected * discounts).sum() / max(ideal, 1e-8))
    return values


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("LUNA_API_KEY")
    if not api_key:
        raise RuntimeError("set LUNA_API_KEY in the process environment")
    source = pd.read_csv(args.candidates)
    required = {"query", "rank", "image_path"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"candidate CSV missing columns: {sorted(missing)}")
    queries = args.queries or source["query"].drop_duplicates().tolist()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_ranked: list[pd.DataFrame] = []
    raw_responses: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}
    for query in queries:
        rows = source[source["query"].eq(query)].drop(
            columns=["relevant", "confidence", "visible_evidence"], errors="ignore"
        ).copy()
        if len(rows) != 100:
            raise ValueError(f"expected 100 candidates for {query!r}, got {len(rows)}")
        rows = rows.sort_values("rank").reset_index(drop=True)
        rows["coarse_rank"] = rows["rank"].astype(int)
        stage1_values: list[dict] = []
        raw_responses[query] = []
        for start in range(0, len(rows), args.stage1_batch_size):
            batch = rows.iloc[start : start + args.stage1_batch_size]
            values, raw = sol_stage1(query, batch, api_key, args)
            stage1_values.extend(values)
            raw_responses[query].append({"sol_stage1": raw})
        stage1 = pd.DataFrame(stage1_values).rename(
            columns={
                "rank": "coarse_rank",
                "confidence": "sol_confidence",
                "visible_evidence": "sol_visible_evidence",
            }
        )
        rows = rows.merge(stage1, on="coarse_rank", validate="one_to_one")
        rows = rows.sort_values(
            ["relevance", "sol_confidence", "coarse_rank"], ascending=[False, False, True]
        )
        finalists = rows.head(args.stage2_top_n).copy()
        ordered, raw = sol_stage2(query, finalists, api_key, args)
        raw_responses[query].append({"sol_stage2": raw})
        selected_ranks = set(ordered)
        final_order = ordered + [
            int(value) for value in rows["coarse_rank"] if int(value) not in selected_ranks
        ]
        final_positions = {coarse_rank: rank for rank, coarse_rank in enumerate(final_order, start=1)}
        rows["rank"] = rows["coarse_rank"].map(final_positions).astype(int)
        rows = rows.sort_values("rank").reset_index(drop=True)
        judgments: list[dict] = []
        for start in range(0, len(rows), args.terra_batch_size):
            batch = rows.iloc[start : start + args.terra_batch_size]
            values, raw = terra_judge(query, batch, api_key, args)
            judgments.extend(values)
            raw_responses[query].append({"terra": raw})
        judged = pd.DataFrame(judgments).rename(
            columns={"confidence": "terra_confidence", "visible_evidence": "terra_visible_evidence"}
        )
        rows = rows.merge(judged, on="rank", validate="one_to_one")
        all_ranked.append(rows)
        summaries[query] = {
            "coarse_order": metric_summary(rows, "coarse_rank"),
            "sol_rerank": metric_summary(rows),
        }
        pd.concat(all_ranked, ignore_index=True).to_csv(output_dir / "sol_terra_progress.csv", index=False)
        (output_dir / "raw_responses_progress.json").write_text(
            json.dumps(raw_responses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"completed {query}", flush=True)
    ranked = pd.concat(all_ranked, ignore_index=True)
    ranked.to_csv(output_dir / "sol_reranked_terra_judged.csv", index=False)
    summary = {
        "sol_model": args.sol_model,
        "terra_model": args.terra_model,
        "candidate_source": str(Path(args.candidates).resolve()),
        "stage1": "Sol assigns 0-4 relevance and confidence to every image in batches.",
        "stage2": f"Sol globally orders the stage-1 Top-{args.stage2_top_n}; remaining candidates retain stage-1 order.",
        "per_query": summaries,
        "mean": {
            method: {
                key: float(np.mean([value[method][key] for value in summaries.values()]))
                for key in next(iter(summaries.values()))[method]
            }
            for method in next(iter(summaries.values()))
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "raw_responses.json").write_text(
        json.dumps(raw_responses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
