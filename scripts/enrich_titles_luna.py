"""Create a non-destructive Luna semantic-label version of the prepared manifest."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

SYSTEM_PROMPT = """You curate remote-sensing retrieval labels from supplied title text.
Return JSON only. Do not infer visual content that is not explicitly present in a title.

For every input item, produce exactly one object with the same title_id and these fields:
- canonical_label: concise lowercase noun phrase for the primary subject.
- retrieval_group: a broader concise noun phrase used to group semantically related
  search results. Examples: "school building", "primary school", and "schoolyard"
  should all use "school"; "vineyard", "orchard", and "crop field" may use
  "farmland" only when that title explicitly describes agricultural land.
- supercategory: exactly one of:
  agriculture, aviation, commercial, education, energy, government, healthcare,
  heritage, industrial, natural_landscape, recreation, residential, retail,
  transportation, utility, water, other.
- concepts: 1 to 4 lowercase concrete concepts explicitly stated by the title.
- confidence: number from 0 to 1 reflecting how unambiguous the title is.

Remove boilerplate such as "an aerial image" and "it shows". Preserve meaningful
distinctions in canonical_label, but use retrieval_group to merge variants that a
user searching a broad object class would want to retrieve together."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="artifacts/prepared_manifest.parquet",
        help="Immutable source manifest.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/label_enrichment/luna_v1",
        help="New versioned output directory. Existing source data is never edited.",
    )
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--base-url", default="http://ai.spacebus.org.cn/v1")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def _extract_response_text(response: dict) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    values: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str):
                    values.append(text)
    if values:
        return "\n".join(values)
    raise ValueError(f"responses payload has no output text: {response.keys()}")


def _parse_json(text: str) -> list[dict]:
    value = text.strip()
    value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE)
    parsed = json.loads(value)
    if isinstance(parsed, dict):
        parsed = parsed.get("labels", parsed.get("items"))
    if not isinstance(parsed, list):
        raise TypeError("model response must be a JSON list or an object containing labels")
    return parsed


def _request_labels(
    batch: list[dict],
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    max_retries: int,
) -> tuple[list[dict], dict]:
    user_text = (
        "Label the following title records. Return a JSON list with one result per input.\n"
        + json.dumps(batch, ensure_ascii=False)
    )
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
        ],
        "max_output_tokens": 12000,
    }
    request = Request(
        f"{base_url.rstrip('/')}/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(max_retries):
        try:
            with urlopen(request, timeout=timeout) as stream:
                response = json.loads(stream.read().decode("utf-8"))
            return _parse_json(_extract_response_text(response)), response
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError):
            if attempt + 1 == max_retries:
                raise
            time.sleep(min(30.0, 2.0 ** attempt))
    raise AssertionError("unreachable")


def _clean_record(record: dict, expected_id: int) -> dict:
    if int(record.get("title_id", -1)) != expected_id:
        raise ValueError(f"unexpected title_id in response: {record.get('title_id')}")
    required = ("canonical_label", "retrieval_group", "supercategory", "concepts", "confidence")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"missing fields for title_id={expected_id}: {missing}")
    result = {"title_id": expected_id}
    for key in ("canonical_label", "retrieval_group", "supercategory"):
        result[f"luna_{key}"] = " ".join(str(record[key]).lower().split())
    concepts = record["concepts"]
    if not isinstance(concepts, list) or not concepts:
        raise ValueError(f"concepts must be a non-empty list for title_id={expected_id}")
    concepts = [" ".join(str(value).lower().split()) for value in concepts]
    result["luna_concepts"] = json.dumps(concepts, ensure_ascii=False)
    # A concise deterministic vocabulary keeps LLM responses compact while
    # retaining broad and fine-grained terms for category-level retrieval.
    result["luna_search_terms"] = json.dumps(
        list(dict.fromkeys([result["luna_retrieval_group"], result["luna_canonical_label"], *concepts])),
        ensure_ascii=False,
    )
    result["luna_confidence"] = float(record["confidence"])
    if not 0.0 <= result["luna_confidence"] <= 1.0:
        raise ValueError(f"confidence outside [0, 1] for title_id={expected_id}")
    return result


def _load_completed(output_dir: Path) -> dict[int, dict]:
    completed: dict[int, dict] = {}
    for path in sorted((output_dir / "batches").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["labels"]:
            completed[int(row["title_id"])] = row
    return completed


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("LUNA_API_KEY")
    if not api_key:
        raise RuntimeError("set LUNA_API_KEY in the process environment; it is never written to disk")
    source_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    batch_dir = output_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source_path)
    titles = frame[["title_id", "title"]].drop_duplicates().sort_values("title_id")
    if args.limit is not None:
        titles = titles.iloc[: args.limit]
    completed = _load_completed(output_dir)
    pending = [
        {"title_id": int(row.title_id), "title": str(row.title)}
        for row in titles.itertuples(index=False)
        if int(row.title_id) not in completed
    ]
    batches = [pending[start : start + args.batch_size] for start in range(0, len(pending), args.batch_size)]
    metadata = {
        "source_manifest": str(source_path),
        "output_dir": str(output_dir),
        "model": args.model,
        "base_url": args.base_url,
        "title_count": len(titles),
        "batch_size": args.batch_size,
        "source_columns": list(frame.columns),
        "note": "Original manifest, images, chips, and cached embeddings are unchanged.",
    }
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if batches:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _request_labels,
                    batch,
                    api_key,
                    args.base_url,
                    args.model,
                    args.timeout,
                    args.max_retries,
                ): batch
                for batch in batches
            }
            for count, future in enumerate(as_completed(futures), start=1):
                batch = futures[future]
                try:
                    labels, raw_response = future.result()
                    expected = {int(item["title_id"]) for item in batch}
                    if {int(item.get("title_id", -1)) for item in labels} != expected:
                        raise ValueError("response title IDs do not exactly match request")
                    cleaned = [_clean_record(item, int(item["title_id"])) for item in labels]
                    batch_id = min(expected)
                    path = batch_dir / f"batch_{batch_id:06d}.json"
                    path.write_text(
                        json.dumps({"labels": cleaned, "raw_response": raw_response}, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    print(f"completed {count}/{len(batches)}: {path.name}", flush=True)
                except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
                    raise RuntimeError(f"Luna enrichment failed for title IDs {[x['title_id'] for x in batch]}: {error}") from error
    completed = _load_completed(output_dir)
    expected_ids = set(titles["title_id"].astype(int))
    if set(completed) != expected_ids:
        raise RuntimeError(f"incomplete enrichment: expected {len(expected_ids)}, got {len(completed)}")
    labels = pd.DataFrame([completed[index] for index in sorted(completed)])
    labels = titles.merge(labels, on="title_id", validate="one_to_one")
    labels["luna_model"] = args.model
    labels.to_parquet(output_dir / "title_labels.parquet", index=False)
    enriched = frame.merge(labels.drop(columns="title"), on="title_id", how="left", validate="many_to_one")
    enriched.to_parquet(output_dir / "prepared_manifest_luna_v1.parquet", index=False)
    summary = {
        "titles": len(labels),
        "rows": len(enriched),
        "supercategories": labels["luna_supercategory"].value_counts().to_dict(),
        "retrieval_groups": int(labels["luna_retrieval_group"].nunique()),
        "outputs": {
            "title_labels": str(output_dir / "title_labels.parquet"),
            "enriched_manifest": str(output_dir / "prepared_manifest_luna_v1.parquet"),
            "batch_audit": str(batch_dir),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
