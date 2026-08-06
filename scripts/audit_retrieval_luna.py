"""Audit whether text-retrieval candidates are visually relevant, independent of titles."""

from __future__ import annotations

import argparse
import base64
import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
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
    parser.add_argument("--queries", nargs="+", default=["river", "school", "farmland"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", default="artifacts/audits/luna_retrieval_v1")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--base-url", default="http://ai.spacebus.org.cn/v1")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--judge-workers", type=int, default=1)
    return parser.parse_args()


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


def judge_candidate(
    query: str,
    rank: int,
    image_path: str,
    api_key: str,
    base_url: str,
    model: str,
    image_size: int,
) -> tuple[dict, dict]:
    prompt = (
        f"You are auditing aerial-image retrieval for the query: {query!r}. "
        "Judge this candidate solely from the image pixels. The candidate is relevant when the "
        "queried object or land-use is visibly present, including visually clear subtypes. "
        "Do not rely on or infer hidden metadata, titles, or ranking. Return JSON only as "
        f'{{"rank":{rank},"relevant":true,"confidence":0.0,"visible_evidence":"..."}}.'
    )
    payload = {
        "model": model,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": image_payload(image_path, image_size)},
        ]}],
        "max_output_tokens": 300,
    }
    request = Request(
        f"{base_url.rstrip('/')}/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180) as stream:
        response = json.loads(stream.read().decode("utf-8"))
    parsed = json.loads(extract_text(response))
    if int(parsed.get("rank", -1)) != rank:
        raise ValueError(f"Luna returned invalid rank for query={query!r}: {parsed}")
    return parsed, response


def judge_query(
    query: str,
    rows: pd.DataFrame,
    api_key: str,
    base_url: str,
    model: str,
    image_size: int,
    workers: int,
) -> tuple[list[dict], list[dict]]:
    judgments: list[dict] = []
    raw_responses: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                judge_candidate,
                query,
                int(row.rank),
                str(row.image_path),
                api_key,
                base_url,
                model,
                image_size,
            ): int(row.rank)
            for row in rows.itertuples(index=False)
        }
        for future in as_completed(futures):
            judgment, raw = future.result()
            judgments.append(judgment)
            raw_responses.append(raw)
    return judgments, raw_responses


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("LUNA_API_KEY")
    if not api_key:
        raise RuntimeError("set LUNA_API_KEY in the process environment")
    config = load_config(args.config)
    baseline_config = load_config(args.baseline_config)
    box_config = load_config(args.box_config)
    frame = load_prepared(config)
    baseline_frame = load_prepared(baseline_config)
    if not frame[["sample_id", "split"]].equals(baseline_frame[["sample_id", "split"]]):
        raise ValueError("latest and baseline configs must use the same prepared manifest")
    positions = frame.index[frame["split"].eq(args.split)].to_numpy(np.int64)
    if not len(positions):
        raise ValueError(f"no rows in split={args.split!r}")
    descriptors, teacher, tokens, _ = load_latent_feature_arrays(config)
    checkpoint = torch.load(config["evaluation"]["checkpoint"], map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_latent_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    highres_global, highres_latents = encode_highres(
        model, np.asarray(teacher[positions]), np.asarray(tokens[positions]), device
    )
    query_global = encode_queries(config, args.queries, device).astype(np.float32)
    with torch.inference_mode():
        values = torch.from_numpy(query_global).to(device)
        query_latents = model.encode_text_latent(values).cpu().numpy()
        query_gates = model.encode_text_gate(values).cpu().numpy()
    top_indices = gated_coarse_topk(
        query_global,
        query_latents,
        query_gates,
        highres_global,
        highres_latents,
        args.top_k,
        device,
        query_batch_size=len(args.queries),
        candidate_chunk_size=int(config["evaluation"]["gated_candidate_chunk_size"]),
    )
    baseline_checkpoint = torch.load(
        baseline_config["evaluation"]["checkpoint"], map_location="cpu", weights_only=False
    )
    baseline_model = build_latent_model(baseline_config).to(device)
    incompatible = baseline_model.load_state_dict(baseline_checkpoint["model_state"], strict=False)
    allowed_missing = {
        key for key in incompatible.missing_keys if key.startswith("highres_adapter.gate_projection.")
    }
    if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected baseline checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    tessera_features = encode_tessera(
        baseline_model, np.asarray(descriptors[positions]), device
    )
    candidate_tessera = F.normalize(torch.from_numpy(tessera_features).to(device), dim=-1)
    query_tessera = F.normalize(torch.from_numpy(query_global).to(device), dim=-1)
    baseline_indices = (query_tessera @ candidate_tessera.T).topk(args.top_k, dim=1).indices.cpu().numpy()
    baseline_highres_global, baseline_highres_latents = encode_highres(
        baseline_model, np.asarray(teacher[positions]), np.asarray(tokens[positions]), device
    )
    with torch.inference_mode():
        baseline_text_latents = baseline_model.encode_text_latent(
            torch.from_numpy(query_global).to(device)
        ).cpu().numpy()
    baseline_highres_indices = late_interaction_prefilter_topk(
        baseline_text_latents,
        baseline_highres_latents,
        query_global,
        baseline_highres_global,
        args.top_k,
        int(baseline_config["index"].get("fine_prefilter", 1000)),
        float(baseline_config["evaluation"]["fine_weight"]),
        device,
        query_batch_size=min(8, len(args.queries)),
    )
    box_checkpoint = torch.load(box_config["evaluation"]["checkpoint"], map_location="cpu", weights_only=False)
    box_model = build_tessera_box_adapter(box_config).to(device)
    box_model.load_state_dict(box_checkpoint["model_state"])
    normalization = box_checkpoint["box_normalization"]
    all_boxes = bbox_features(frame)
    normalized_boxes = ((all_boxes - np.asarray(normalization["mean"], dtype=np.float32)) / np.asarray(normalization["std"], dtype=np.float32)).astype(np.float32)
    box_features = encode_tessera_box(box_model, descriptors[positions], normalized_boxes[positions], device)
    with torch.inference_mode():
        box_queries = box_model.encode_text(torch.from_numpy(query_global).to(device))
    box_indices = (box_queries @ torch.from_numpy(box_features).to(device).T).topk(args.top_k, dim=1).indices.cpu().numpy()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows: list[dict] = []
    test_frame = frame.iloc[positions].reset_index(drop=True)
    systems = (
        ("gated_highres", top_indices, query_gates),
        ("latent_v2_highres_fine", baseline_highres_indices, np.full(len(args.queries), np.nan)),
        ("latent_v2_tessera", baseline_indices, np.full(len(args.queries), np.nan)),
        ("tessera_v1_box_mlp", box_indices, np.full(len(args.queries), np.nan)),
    )
    for system, system_indices, system_gates in systems:
        for query, gate, indices in zip(args.queries, system_gates, system_indices):
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
    judgments: list[dict] = []
    raw_responses = {}
    for system, query in candidates[["system", "query"]].drop_duplicates().itertuples(index=False):
        rows = candidates[candidates["system"].eq(system) & candidates["query"].eq(query)].copy()
        result, raw = judge_query(
            query, rows, api_key, args.base_url, args.model, args.image_size, args.judge_workers
        )
        raw_responses[f"{system}:{query}"] = raw
        judgments.extend({"system": system, "query": query, **item} for item in result)
        print(f"Luna judged {system}:{query}", flush=True)
    judgment_frame = pd.DataFrame(judgments)
    audited = candidates.merge(judgment_frame, on=["system", "query", "rank"], validate="one_to_one")
    audited.to_csv(output_dir / "candidates_luna_judged.csv", index=False)
    summary = {
        "model": args.model,
        "retrieval_checkpoints": {
            "gated_highres": config["evaluation"]["checkpoint"],
            "latent_v2_highres_fine": baseline_config["evaluation"]["checkpoint"],
            "latent_v2_tessera": baseline_config["evaluation"]["checkpoint"],
            "tessera_v1_box_mlp": box_config["evaluation"]["checkpoint"],
        },
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
    del model, baseline_model, box_model, highres_global, highres_latents, baseline_highres_global, baseline_highres_latents, tessera_features, box_features
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
