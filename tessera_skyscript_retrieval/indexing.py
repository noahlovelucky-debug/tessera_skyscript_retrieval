from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import ensure_dir, uses_latent_tokens, uses_text_gated_coarse
from .data import load_prepared
from .evaluation import encode_tessera
from .features import load_feature_arrays, load_latent_feature_arrays
from .latent import encode_latent_features
from .model import build_adapter, build_latent_model
from .skyclip import encode_query


def build_index(config: dict, limit: int | None = None, checkpoint_path: str | None = None) -> Path:
    if uses_latent_tokens(config):
        return _build_latent_index(config, limit, checkpoint_path)
    frame = load_prepared(config)
    if limit is not None:
        frame = frame.iloc[:limit].copy()
    descriptors, highres, _ = load_feature_arrays(config, limit)
    checkpoint_value = Path(checkpoint_path or config["evaluation"]["checkpoint"])
    if limit is not None and checkpoint_path is None:
        checkpoint_value = Path(config["training"]["output_dir"]) / f"best_{limit}.pt"
    checkpoint = torch.load(checkpoint_value, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_adapter(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    tessera = encode_tessera(model, descriptors, device)
    output_dir = ensure_dir(config["index"]["output_dir"])
    if limit is not None:
        output_dir = ensure_dir(output_dir / f"limit_{limit}")
    np.save(output_dir / "tessera_features.npy", tessera.astype(np.float16))
    np.save(output_dir / "highres_features.npy", np.asarray(highres, dtype=np.float16))
    frame.to_parquet(output_dir / "metadata.parquet", index=False)
    (output_dir / "index.json").write_text(json.dumps({
        "rows": len(frame), "dimension": int(config["model"]["common_dim"]),
        "checkpoint": str(checkpoint_value), "normalized": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"built highres+tessera index with {len(frame)} paired rows at {output_dir}")
    return output_dir


def _build_latent_index(
    config: dict,
    limit: int | None = None,
    checkpoint_path: str | None = None,
) -> Path:
    frame = load_prepared(config)
    if limit is not None:
        frame = frame.iloc[:limit].copy()
    descriptors, teacher_highres, region_tokens, _ = load_latent_feature_arrays(
        config, limit
    )
    checkpoint_value = Path(checkpoint_path or config["evaluation"]["checkpoint"])
    if limit is not None and checkpoint_path is None:
        checkpoint_value = Path(config["training"]["output_dir"]) / f"best_{limit}.pt"
    checkpoint = torch.load(checkpoint_value, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_latent_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    tessera, highres, highres_latents = encode_latent_features(
        model, descriptors, teacher_highres, region_tokens, device
    )
    output_dir = ensure_dir(config["index"]["output_dir"])
    if limit is not None:
        output_dir = ensure_dir(output_dir / f"limit_{limit}")
    np.save(output_dir / "tessera_features.npy", tessera.astype(np.float16))
    np.save(output_dir / "highres_features.npy", highres.astype(np.float16))
    np.save(output_dir / "highres_latents.npy", highres_latents)
    frame.to_parquet(output_dir / "metadata.parquet", index=False)
    (output_dir / "index.json").write_text(
        json.dumps(
            {
                "rows": len(frame),
                "dimension": int(config["model"]["common_dim"]),
                "highres_latent_shape": [
                    int(config["model"]["latent_count"]),
                    int(config["model"]["latent_dim"]),
                ],
                "checkpoint": str(checkpoint_value),
                "normalized": True,
                "retrieval_mode": (
                    "text_gated_full_scan"
                    if uses_text_gated_coarse(config)
                    else "global_prefilter_then_late_interaction"
                ),
                "fine_weight": (
                    None
                    if uses_text_gated_coarse(config)
                    else float(config["evaluation"]["fine_weight"])
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    architecture = config["model"].get("architecture", "latent_v2")
    print(f"built {architecture} highres+tessera index with {len(frame)} rows at {output_dir}")
    return output_dir


def _top_scores(features: np.ndarray, query: np.ndarray, top_k: int, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    best_scores = np.empty(0, dtype=np.float32)
    best_indices = np.empty(0, dtype=np.int64)
    for start in range(0, len(features), chunk_size):
        values = np.asarray(features[start:start + chunk_size], dtype=np.float32)
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
        scores = values @ query
        local_k = min(top_k, len(scores))
        local = np.argpartition(scores, -local_k)[-local_k:]
        best_scores = np.concatenate([best_scores, scores[local]])
        best_indices = np.concatenate([best_indices, local + start])
        if len(best_scores) > top_k:
            keep = np.argpartition(best_scores, -top_k)[-top_k:]
            best_scores, best_indices = best_scores[keep], best_indices[keep]
    order = np.argsort(best_scores)[::-1]
    return best_scores[order], best_indices[order]


def _gated_top_scores(
    global_features: np.ndarray,
    local_features: np.ndarray,
    global_query: np.ndarray,
    local_query: np.ndarray,
    global_weight: float,
    top_k: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact single-query gated coarse ranking across an on-disk token index."""
    best_scores = np.empty(0, dtype=np.float32)
    best_indices = np.empty(0, dtype=np.int64)
    best_global = np.empty(0, dtype=np.float32)
    best_local = np.empty(0, dtype=np.float32)
    for start in range(0, len(global_features), chunk_size):
        stop = min(start + chunk_size, len(global_features))
        globals_chunk = np.asarray(global_features[start:stop], dtype=np.float32)
        globals_chunk /= np.maximum(
            np.linalg.norm(globals_chunk, axis=1, keepdims=True), 1e-8
        )
        latents_chunk = np.asarray(local_features[start:stop], dtype=np.float32)
        latents_chunk /= np.maximum(
            np.linalg.norm(latents_chunk, axis=-1, keepdims=True), 1e-8
        )
        global_scores = globals_chunk @ global_query
        local_scores = np.max(latents_chunk @ local_query, axis=1)
        scores = global_weight * global_scores + (1.0 - global_weight) * local_scores
        local_k = min(top_k, len(scores))
        local = np.argpartition(scores, -local_k)[-local_k:]
        best_scores = np.concatenate([best_scores, scores[local]])
        best_indices = np.concatenate([best_indices, local + start])
        best_global = np.concatenate([best_global, global_scores[local]])
        best_local = np.concatenate([best_local, local_scores[local]])
        if len(best_scores) > top_k:
            keep = np.argpartition(best_scores, -top_k)[-top_k:]
            best_scores = best_scores[keep]
            best_indices = best_indices[keep]
            best_global = best_global[keep]
            best_local = best_local[keep]
    order = np.argsort(best_scores)[::-1]
    return (
        best_scores[order],
        best_indices[order],
        best_global[order],
        best_local[order],
    )


def retrieve_by_modality(
    config: dict,
    query_text: str,
    modalities: tuple[str, ...] = ("highres", "tessera"),
    top_k: int = 10,
    index_dir: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    unsupported = set(modalities) - {"highres", "tessera"}
    if unsupported:
        raise ValueError(f"unsupported modalities: {sorted(unsupported)}")
    root = Path(index_dir or config["index"]["output_dir"])
    metadata = pd.read_parquet(root / "metadata.parquet")
    index_metadata = json.loads((root / "index.json").read_text(encoding="utf-8"))
    query = encode_query(config, query_text, config["index"]["prompt_templates"])
    query = query / max(float(np.linalg.norm(query)), 1e-8)
    latent_model = None
    text_latent = None
    if uses_latent_tokens(config):
        checkpoint = torch.load(
            index_metadata.get("checkpoint", config["evaluation"]["checkpoint"]),
            map_location="cpu",
            weights_only=False,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        latent_model = build_latent_model(config).to(device)
        latent_model.load_state_dict(checkpoint["model_state"])
        latent_model.eval()
        with torch.inference_mode():
            text_latent = latent_model.encode_text_latent(
                torch.from_numpy(query.astype(np.float32))[None].to(device)
            )[0].cpu().numpy()
            text_gate = float(
                latent_model.encode_text_gate(
                    torch.from_numpy(query.astype(np.float32))[None].to(device)
                )[0].cpu()
            )
    fields = [
        "sample_id", "title", "source", "year", "image_path", "chip_path",
        "bbox_west", "bbox_south", "bbox_east", "bbox_north", "center_lon", "center_lat",
    ]
    results: dict[str, list[dict[str, Any]]] = {}
    for name in modalities:
        features = np.load(root / f"{name}_features.npy", mmap_mode="r")
        score_components: tuple[np.ndarray, np.ndarray] | None = None
        if name == "highres" and latent_model is not None and uses_text_gated_coarse(config):
            latents = np.load(root / "highres_latents.npy", mmap_mode="r")
            scores, indices, global_scores, local_scores = _gated_top_scores(
                features,
                latents,
                query,
                text_latent,
                text_gate,
                top_k,
                int(config["evaluation"].get("gated_candidate_chunk_size", 8192)),
            )
            score_components = (global_scores, local_scores)
        else:
            requested_k = top_k
            if name == "highres" and latent_model is not None:
                requested_k = max(
                    requested_k, int(config["index"].get("fine_prefilter", 1000))
                )
            scores, indices = _top_scores(
                features,
                query,
                requested_k,
                int(config["evaluation"]["candidate_chunk_size"]),
            )
            if name == "highres" and text_latent is not None:
                latents = np.asarray(
                    np.load(root / "highres_latents.npy", mmap_mode="r")[indices],
                    dtype=np.float32,
                )
                latents /= np.maximum(
                    np.linalg.norm(latents, axis=-1, keepdims=True), 1e-8
                )
                fine_scores = np.max(latents @ text_latent, axis=1)
                fine_weight = float(config["evaluation"]["fine_weight"])
                scores = fine_weight * fine_scores + (1.0 - fine_weight) * scores
                order = np.argsort(scores)[::-1][:top_k]
                scores, indices = scores[order], indices[order]
        rows = []
        for rank, (score, index) in enumerate(zip(scores[:top_k], indices[:top_k]), start=1):
            source = metadata.iloc[int(index)]
            result = {"rank": rank, "score": float(score), "modality": name}
            if score_components is not None:
                global_scores, local_scores = score_components
                result.update(
                    {
                        "global_score": float(global_scores[rank - 1]),
                        "local_score": float(local_scores[rank - 1]),
                        "gate_global_weight": text_gate,
                    }
                )
            result.update({field: source[field] for field in fields})
            rows.append(result)
        results[name] = rows
    return results


def search(
    config: dict,
    query_text: str,
    modality: str = "both",
    top_k: int = 10,
    index_dir: str | None = None,
) -> dict[str, Any]:
    if modality not in {"highres", "tessera", "both"}:
        raise ValueError(f"unsupported modality: {modality}")
    modalities = (modality,) if modality != "both" else ("highres", "tessera")
    per_modality_k = top_k if modality != "both" else max(top_k * 4, 50)
    grouped = retrieve_by_modality(
        config, query_text, modalities, per_modality_k, index_dir
    )
    candidates = sorted(
        (row for rows in grouped.values() for row in rows),
        key=lambda row: float(row["score"]),
        reverse=True,
    )
    seen = set()
    rows = []
    for candidate in candidates:
        sample_id = str(candidate["sample_id"])
        if sample_id in seen:
            continue
        seen.add(sample_id)
        result = dict(candidate)
        result["rank"] = len(rows) + 1
        rows.append(result)
        if len(rows) == top_k:
            break
    output = {"query": query_text, "modality": modality, "results": rows}
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    return output
