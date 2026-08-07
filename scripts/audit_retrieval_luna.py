"""Audit whether text-retrieval candidates are visually relevant, independent of titles."""

from __future__ import annotations

import argparse
import base64
import gc
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from tessera_skyscript_retrieval.config import load_config
from tessera_skyscript_retrieval.data import load_prepared
from tessera_skyscript_retrieval.features import load_latent_feature_arrays
from tessera_skyscript_retrieval.metrics import gated_coarse_topk
from tessera_skyscript_retrieval.metrics import late_interaction_prefilter_topk
from tessera_skyscript_retrieval.model import build_latent_model
from tessera_skyscript_retrieval.skyclip import load_skyclip
from tessera_skyscript_retrieval.tessera_box import bbox_features, encode_tessera_box
from tessera_skyscript_retrieval.model import build_tessera_box_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gated_coarse_v3.yaml")
    parser.add_argument("--baseline-config", default="configs/latent_v2.yaml")
    parser.add_argument("--box-config", default="configs/tessera_box_v1.yaml")
    parser.add_argument("--queries", nargs="+", default=None)
    parser.add_argument(
        "--queries-file",
        help="CSV containing a query column, or a newline-delimited query file.",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=(
            "gated_highres",
            "latent_v2_highres_fine",
            "latent_v2_tessera",
            "tessera_v1_box_mlp",
        ),
        default=(
            "gated_highres",
            "latent_v2_highres_fine",
            "latent_v2_tessera",
            "tessera_v1_box_mlp",
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", default="artifacts/audits/luna_retrieval_v1")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--base-url", default="http://ai.spacebus.org.cn/v1")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--judge-workers", type=int, default=1)
    parser.add_argument(
        "--judge-batch-size",
        type=int,
        default=10,
        help="Number of ranked images submitted to Luna in each request.",
    )
    return parser.parse_args()


def load_queries(args: argparse.Namespace) -> list[str]:
    if args.queries and args.queries_file:
        raise ValueError("use either --queries or --queries-file, not both")
    if args.queries_file:
        path = Path(args.queries_file)
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            column = "query" if "query" in frame.columns else frame.columns[0]
            values = frame[column].astype(str).tolist()
        else:
            values = path.read_text(encoding="utf-8").splitlines()
    else:
        values = args.queries or ["river", "school", "farmland"]
    queries = list(dict.fromkeys(value.strip() for value in values if value.strip()))
    if not queries:
        raise ValueError("at least one non-empty query is required")
    return queries


@torch.inference_mode()
def encode_highres(
    model, teacher: np.ndarray, tokens: np.ndarray, device: torch.device, batch_size: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    global_features = np.empty((len(teacher), teacher.shape[1]), dtype=np.float32)
    latents = np.empty(
        (len(teacher), model.highres_adapter.latent_count, model.highres_adapter.latent_dim),
        dtype=np.float16,
    )
    model.eval()
    for start in tqdm(range(0, len(teacher), batch_size), desc="encode-audit-highres"):
        stop = min(start + batch_size, len(teacher))
        teacher_batch = F.normalize(
            torch.from_numpy(np.asarray(teacher[start:stop], dtype=np.float32)).to(device), dim=-1
        )
        token_batch = torch.from_numpy(np.asarray(tokens[start:stop], dtype=np.float32)).to(device)
        global_batch, latent_batch = model.encode_highres(token_batch, teacher_batch)
        global_features[start:stop] = global_batch.cpu().numpy()
        latents[start:stop] = latent_batch.cpu().numpy().astype(np.float16)
    return global_features, latents


@torch.inference_mode()
def encode_tessera(model, descriptors: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    output_dim = model.tessera_adapter.output_projection[-1].out_features
    output = np.empty((len(descriptors), output_dim), dtype=np.float32)
    model.eval()
    for start in tqdm(range(0, len(descriptors), batch_size), desc="encode-audit-tessera"):
        stop = min(start + batch_size, len(descriptors))
        batch = torch.from_numpy(np.asarray(descriptors[start:stop], dtype=np.float32)).to(device)
        output[start:stop] = model.encode_tessera(batch).cpu().numpy()
    return output


@torch.inference_mode()
def encode_queries(config: dict, queries: list[str], device: torch.device) -> np.ndarray:
    model, _, tokenizer, _ = load_skyclip(config, device)
    prompts = [
        template.format(query=query)
        for query in queries
        for template in config["index"]["prompt_templates"]
    ]
    tokens = tokenizer(prompts).to(device)
    values = model.encode_text(tokens, normalize=True).float()
    values = values.reshape(len(queries), len(config["index"]["prompt_templates"]), -1).mean(dim=1)
    return F.normalize(values, dim=-1).cpu().numpy()


def extract_text(response: dict) -> str:
    values = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                values.append(content["text"])
    if not values and isinstance(response.get("output_text"), str):
        values.append(response["output_text"])
    if not values:
        raise ValueError("Luna response has no output text")
    return "\n".join(values).strip().removeprefix("```json").removesuffix("```").strip()


def image_payload(path: str, max_size: int) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_size, max_size))
        from io import BytesIO

        stream = BytesIO()
        image.save(stream, format="JPEG", quality=85, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def judge_query(
    query: str,
    rows: pd.DataFrame,
    api_key: str,
    base_url: str,
    model: str,
    image_size: int,
    workers: int,
) -> tuple[list[dict], list[dict]]:
    del workers  # Each batch is sent in one request for a consistent judgment context.
    ranks = [int(row.rank) for row in rows.sort_values("rank").itertuples(index=False)]
    prompt = (
        f"You are auditing aerial-image retrieval for the query: {query!r}. "
        f"Judge each of the {len(ranks)} numbered candidate images solely from its pixels. An image is "
        "relevant when the queried object or land-use is visibly present, including visually "
        "clear subtypes. Do not rely on or infer hidden metadata, titles, or ranking. Return "
        "JSON only as {\"judgments\":[{\"rank\":1,\"relevant\":true,\"confidence\":0.0,"
        "\"visible_evidence\":\"...\"}, ...]}. Include exactly these ranks: "
        + json.dumps(ranks)
    )
    content = [{"type": "input_text", "text": prompt}]
    for row in rows.sort_values("rank").itertuples(index=False):
        content.append(
            {
                "type": "input_text",
                "text": f"Candidate rank {int(row.rank)}:",
            }
        )
        content.append(
            {
                "type": "input_image",
                "image_url": image_payload(str(row.image_path), image_size),
            }
        )
    payload = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 1800,
    }
    request = Request(
        f"{base_url.rstrip('/')}/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    error = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=300) as stream:
                response = json.loads(stream.read().decode("utf-8"))
            parsed = json.loads(extract_text(response))
            judgments = parsed.get("judgments") if isinstance(parsed, dict) else None
            if not isinstance(judgments, list):
                raise ValueError(f"Luna returned no judgments for query={query!r}")
            returned_ranks = [int(item.get("rank", -1)) for item in judgments]
            if sorted(returned_ranks) != ranks or len(judgments) != len(ranks):
                raise ValueError(
                    f"Luna returned invalid ranks for query={query!r}: {returned_ranks}"
                )
            return judgments, [response]
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Luna audit failed for query={query!r}: {error}")


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("LUNA_API_KEY")
    if not api_key:
        raise RuntimeError("set LUNA_API_KEY in the process environment")
    queries = load_queries(args)
    requested = set(args.systems)
    needs_gated = "gated_highres" in requested
    needs_baseline = bool(
        requested & {"latent_v2_highres_fine", "latent_v2_tessera"}
    )
    needs_box = "tessera_v1_box_mlp" in requested
    config = load_config(args.config)
    frame = load_prepared(config)
    baseline_config = load_config(args.baseline_config) if needs_baseline else None
    box_config = load_config(args.box_config) if needs_box else None
    if baseline_config is not None:
        baseline_frame = load_prepared(baseline_config)
        if not frame[["sample_id", "split"]].equals(
            baseline_frame[["sample_id", "split"]]
        ):
            raise ValueError("latest and baseline configs must use the same prepared manifest")
    positions = frame.index[frame["split"].eq(args.split)].to_numpy(np.int64)
    if not len(positions):
        raise ValueError(f"no rows in split={args.split!r}")
    descriptors, teacher, tokens, _ = load_latent_feature_arrays(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_global = encode_queries(config, queries, device).astype(np.float32)
    top_indices = query_gates = None
    if needs_gated:
        checkpoint = torch.load(
            config["evaluation"]["checkpoint"], map_location="cpu", weights_only=False
        )
        model = build_latent_model(config).to(device)
        model.load_state_dict(checkpoint["model_state"])
        highres_global, highres_latents = encode_highres(
            model, np.asarray(teacher[positions]), np.asarray(tokens[positions]), device
        )
        with torch.inference_mode():
            values = torch.from_numpy(query_global).to(device)
            query_latents = model.encode_text_latent(values).cpu().numpy()
            query_gates = model.encode_text_gate(values).cpu().numpy()
        top_indices = gated_coarse_topk(
            query_global, query_latents, query_gates, highres_global, highres_latents,
            args.top_k, device, query_batch_size=len(queries),
            candidate_chunk_size=int(config["evaluation"]["gated_candidate_chunk_size"]),
        )

    baseline_indices = baseline_highres_indices = None
    if needs_baseline:
        assert baseline_config is not None
        baseline_checkpoint = torch.load(
            baseline_config["evaluation"]["checkpoint"], map_location="cpu", weights_only=False
        )
        baseline_model = build_latent_model(baseline_config).to(device)
        incompatible = baseline_model.load_state_dict(
            baseline_checkpoint["model_state"], strict=False
        )
        allowed_missing = {
            key for key in incompatible.missing_keys
            if key.startswith("highres_adapter.gate_projection.")
        }
        if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"unexpected baseline checkpoint mismatch: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        if "latent_v2_tessera" in requested:
            tessera_features = encode_tessera(
                baseline_model, np.asarray(descriptors[positions]), device
            )
            candidate_tessera = F.normalize(
                torch.from_numpy(tessera_features).to(device), dim=-1
            )
            query_tessera = F.normalize(torch.from_numpy(query_global).to(device), dim=-1)
            baseline_indices = (query_tessera @ candidate_tessera.T).topk(
                args.top_k, dim=1
            ).indices.cpu().numpy()
        if "latent_v2_highres_fine" in requested:
            baseline_highres_global, baseline_highres_latents = encode_highres(
                baseline_model, np.asarray(teacher[positions]), np.asarray(tokens[positions]), device
            )
            with torch.inference_mode():
                baseline_text_latents = baseline_model.encode_text_latent(
                    torch.from_numpy(query_global).to(device)
                ).cpu().numpy()
            baseline_highres_indices = late_interaction_prefilter_topk(
                baseline_text_latents, baseline_highres_latents, query_global,
                baseline_highres_global, args.top_k,
                int(baseline_config["index"].get("fine_prefilter", 1000)),
                float(baseline_config["evaluation"]["fine_weight"]), device,
                query_batch_size=min(8, len(queries)),
            )

    box_indices = None
    if needs_box:
        assert box_config is not None
        box_checkpoint = torch.load(
            box_config["evaluation"]["checkpoint"], map_location="cpu", weights_only=False
        )
        box_model = build_tessera_box_adapter(box_config).to(device)
        box_model.load_state_dict(box_checkpoint["model_state"])
        normalization = box_checkpoint["box_normalization"]
        all_boxes = bbox_features(frame)
        normalized_boxes = (
            (all_boxes - np.asarray(normalization["mean"], dtype=np.float32))
            / np.asarray(normalization["std"], dtype=np.float32)
        ).astype(np.float32)
        box_features = encode_tessera_box(
            box_model, descriptors[positions], normalized_boxes[positions], device
        )
        with torch.inference_mode():
            box_queries = box_model.encode_text(torch.from_numpy(query_global).to(device))
        box_indices = (box_queries @ torch.from_numpy(box_features).to(device).T).topk(
            args.top_k, dim=1
        ).indices.cpu().numpy()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows: list[dict] = []
    test_frame = frame.iloc[positions].reset_index(drop=True)
    available_systems = (
        ("gated_highres", top_indices, query_gates),
        ("latent_v2_highres_fine", baseline_highres_indices, np.full(len(queries), np.nan)),
        ("latent_v2_tessera", baseline_indices, np.full(len(queries), np.nan)),
        ("tessera_v1_box_mlp", box_indices, np.full(len(queries), np.nan)),
    )
    systems = tuple(item for item in available_systems if item[0] in args.systems)
    for system, system_indices, system_gates in systems:
        for query, gate, indices in zip(queries, system_gates, system_indices):
            for rank, index in enumerate(indices, start=1):
                row = test_frame.iloc[int(index)]
                candidate_rows.append(
                    {
                        "system": system,
                        "query": query,
                        "rank": rank,
                        "gate_global_weight": float(gate),
                        "sample_id": row["sample_id"],
                        "source_title": row["title"],
                        "image_path": row["image_path"],
                        "center_lon": row["center_lon"],
                        "center_lat": row["center_lat"],
                    }
                )
    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(output_dir / "candidates_pending_judgment.csv", index=False)
    judgments: list[dict] = []
    raw_responses = {}
    for system, query in candidates[["system", "query"]].drop_duplicates().itertuples(index=False):
        rows = candidates[candidates["system"].eq(system) & candidates["query"].eq(query)].copy()
        for start in range(0, len(rows), args.judge_batch_size):
            batch = rows.iloc[start : start + args.judge_batch_size]
            result, raw = judge_query(
                query, batch, api_key, args.base_url, args.model, args.image_size, args.judge_workers
            )
            first_rank = int(batch["rank"].min())
            last_rank = int(batch["rank"].max())
            raw_responses[f"{system}:{query}:{first_rank}-{last_rank}"] = raw
            judgments.extend({"system": system, "query": query, **item} for item in result)
            pd.DataFrame(judgments).to_csv(
                output_dir / "judgments_progress.csv", index=False
            )
            (output_dir / "luna_raw_responses_progress.json").write_text(
                json.dumps(raw_responses, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"Luna judged {system}:{query}: ranks {first_rank}-{last_rank}",
                flush=True,
            )
    judgment_frame = pd.DataFrame(judgments)
    audited = candidates.merge(judgment_frame, on=["system", "query", "rank"], validate="one_to_one")
    audited.to_csv(output_dir / "candidates_luna_judged.csv", index=False)
    retrieval_checkpoints = {}
    if needs_gated:
        retrieval_checkpoints["gated_highres"] = config["evaluation"]["checkpoint"]
    if baseline_config is not None:
        retrieval_checkpoints["latent_v2_highres_fine"] = baseline_config["evaluation"]["checkpoint"]
        retrieval_checkpoints["latent_v2_tessera"] = baseline_config["evaluation"]["checkpoint"]
    if box_config is not None:
        retrieval_checkpoints["tessera_v1_box_mlp"] = box_config["evaluation"]["checkpoint"]
    summary = {
        "model": args.model,
        "retrieval_checkpoints": retrieval_checkpoints,
        "candidate_split": args.split,
        "candidate_count": len(test_frame),
        "top_k": args.top_k,
        "systems": {},
    }
    discounts = 1.0 / np.log2(np.arange(2, args.top_k + 2))
    for system, system_rows in audited.groupby("system", sort=False):
        query_summary = {}
        for query, rows in system_rows.groupby("query", sort=False):
            ranked = rows.sort_values("rank")
            relevant = ranked["relevant"].astype(bool).to_numpy()
            query_summary[query] = {
                "luna_visual_precision_at_k": float(relevant.mean()),
                "luna_visual_hit_at_k": float(relevant.any()),
                "luna_discounted_relevance_at_k": float((relevant * discounts).sum() / discounts.sum()),
                "luna_relevant_count": int(relevant.sum()),
                "gate_global_weight": float(ranked["gate_global_weight"].iloc[0]),
            }
        summary["systems"][system] = {
            "mean_luna_visual_precision_at_k": float(np.mean([x["luna_visual_precision_at_k"] for x in query_summary.values()])),
            "mean_luna_visual_hit_at_k": float(np.mean([x["luna_visual_hit_at_k"] for x in query_summary.values()])),
            "mean_luna_discounted_relevance_at_k": float(np.mean([x["luna_discounted_relevance_at_k"] for x in query_summary.values()])),
            "queries": query_summary,
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "luna_raw_responses.json").write_text(
        json.dumps(raw_responses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
